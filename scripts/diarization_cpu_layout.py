"""Opt-in, reversible channels-last layout for CPU Community-1 WeSpeaker.

Only standard-library imports occur here. The caller must have loaded Torch
and the pipeline already, and must close the handle after synchronous inference.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import sys
import time
from typing import Any, Callable


class LayoutError(RuntimeError):
    """Unsupported layout experiment or failed state-preservation check."""


_ACTIVE: dict[int, "Handle"] = {}
_INPUT_LIMIT = 16


def _kind(value: Any, module: str, name: str) -> bool:
    return any(
        cls.__module__ == module and cls.__name__ == name
        for cls in type(value).__mro__
    )


def _logical_hash(tensor: Any, torch: Any) -> str:
    # Hash logical-order bytes, not storage order; signed zeros remain distinct.
    data = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(data.tobytes(order="C")).hexdigest()


def _metadata(tensor: Any) -> dict:
    storage = tensor.untyped_storage()
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": tensor.requires_grad,
        "stride": tuple(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "storage_ptr": storage.data_ptr(),
        "storage_bytes": storage.nbytes(),
        "data_ptr": tensor.data_ptr(),
    }


def _registered(model: Any) -> dict:
    return {
        **{("parameter", name): tensor for name, tensor in
           model.named_parameters(remove_duplicate=False)},
        **{("buffer", name): tensor for name, tensor in
           model.named_buffers(remove_duplicate=False)},
    }


@dataclass
class _State:
    tensor: Any
    metadata: dict
    digest: str


class Handle:
    def __init__(self, model: Any, network: Any, torch: Any, emit: Callable):
        self._model = model
        self._network = network
        self._torch = torch
        self._emit = emit
        self._baseline: dict = {}
        self._converted: dict = {}
        self._modules: dict = {}
        self._weights: list[tuple[Any, Any, Any]] = []
        self._hook = None
        self._closed = False
        self._active_inputs = 0
        self._started = time.monotonic()
        self._input_calls = 0
        self._input_failed = 0
        self._input_wall = 0.0
        self._input_cpu = 0.0
        self._inputs: dict[tuple, int] = {}
        self._unlisted_inputs = 0

    def _measure(self, phase: str, action: Callable, **fields: Any) -> Any:
        started = time.monotonic()
        cpu_started = time.process_time()
        try:
            result = action()
        except BaseException:
            self._emit(
                phase, started, status="failed", layout_version=1,
                elapsed_seconds=time.monotonic() - started,
                cpu_seconds=time.process_time() - cpu_started, **fields,
            )
            raise
        self._emit(
            phase, started, status="completed", layout_version=1,
            elapsed_seconds=time.monotonic() - started,
            cpu_seconds=time.process_time() - cpu_started, **fields,
            **(result if isinstance(result, dict) else {}),
        )
        return result

    def _check_modules(self) -> dict:
        modules = dict(self._model.named_modules(remove_duplicate=False))
        if len({id(module) for module in modules.values()}) != len(modules):
            raise LayoutError("Shared embedding submodules are unsupported")
        for name, module in modules.items():
            if module.training:
                raise LayoutError(f"Embedding module {name!r} is not in eval mode")
            if (
                getattr(module, "_compiled_call_impl", None) is not None
                or hasattr(module.forward, "_torchdynamo_orig_callable")
            ):
                raise LayoutError(f"Compiled embedding module {name!r} is unsupported")
        if self._modules and (
            modules.keys() != self._modules.keys()
            or any(module is not self._modules[name] for name, module in modules.items())
        ):
            raise LayoutError("Embedding module identities changed")
        return modules

    def _snapshot(self) -> dict:
        self._check_modules()
        tensors = _registered(self._model)
        identities = set()
        storages = set()
        result = {}
        for key, tensor in tensors.items():
            if tensor.layout != self._torch.strided or tensor.device.type != "cpu":
                raise LayoutError(f"{key} must be a dense CPU tensor")
            if (
                key[0] == "parameter" or tensor.is_floating_point()
            ) and tensor.dtype != self._torch.float32:
                raise LayoutError(f"{key} must retain FP32 dtype")
            if tensor.is_complex() or tensor.grad is not None:
                raise LayoutError(f"{key} has complex values or an existing gradient")
            metadata = _metadata(tensor)
            if id(tensor) in identities:
                raise LayoutError(f"Registered tensor alias at {key}")
            identities.add(id(tensor))
            if metadata["storage_bytes"]:
                storage_key = (metadata["device"], metadata["storage_ptr"])
                if storage_key in storages:
                    raise LayoutError(f"Registered storage alias at {key}")
                storages.add(storage_key)
            result[key] = _State(tensor, metadata, _logical_hash(tensor, self._torch))
        return result

    @staticmethod
    def _digest(states: dict) -> str:
        values = [
            (key, state.digest, state.metadata["shape"], state.metadata["dtype"])
            for key, state in sorted(states.items())
        ]
        return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()

    def _verify(self, expected: dict, allow_weight_layout: bool = False) -> dict:
        current = self._snapshot()
        if current.keys() != expected.keys():
            raise LayoutError("Registered embedding tensor names changed")
        weights = {id(parameter) for _module, parameter, _data in self._weights}
        for key, before in expected.items():
            after = current[key]
            if after.tensor is not before.tensor:
                raise LayoutError(f"Tensor/Parameter identity changed at {key}")
            if after.digest != before.digest:
                raise LayoutError(f"Logical bytes changed at {key}")
            fields = (
                ("shape", "dtype", "device", "requires_grad")
                if allow_weight_layout and id(after.tensor) in weights
                else tuple(before.metadata)
            )
            if any(after.metadata[field] != before.metadata[field] for field in fields):
                raise LayoutError(f"Tensor metadata changed at {key}")
            if allow_weight_layout and id(after.tensor) in weights:
                if not after.tensor.is_contiguous(memory_format=self._torch.channels_last):
                    raise LayoutError(f"Conv2d weight is not channels-last at {key}")
        self._converted = current if allow_weight_layout else self._converted
        return {"state_sha256": self._digest(current), "tensor_count": len(current)}

    def _prepare(self) -> dict:
        self._modules = self._check_modules()
        self._baseline = self._snapshot()
        convs = []
        for module in self._network.modules():
            if isinstance(module, self._torch.nn.ConvTranspose2d):
                raise LayoutError("ConvTranspose2d is outside this pinned ResNet factor")
            if isinstance(module, self._torch.nn.Conv2d):
                if module.weight.ndim != 4:
                    raise LayoutError("Conv2d weight must be four-dimensional")
                convs.append(module)
        if not convs or convs[0] is not self._network.conv1:
            raise LayoutError("The pinned first Conv2d entry could not be identified")
        self._weights = [(module, module.weight, module.weight.data) for module in convs]
        if not all(any(state.tensor is parameter for state in self._baseline.values())
                   for _module, parameter, _data in self._weights):
            raise LayoutError("Conv2d weight is not registered in the embedding model")
        return {
            "state_sha256": self._digest(self._baseline),
            "tensor_count": len(self._baseline),
            "conv_weights": len(self._weights),
        }

    def _input_hook(self, module: Any, args: tuple, kwargs: dict) -> tuple:
        if self._closed:
            raise LayoutError("Closed layout hook was unexpectedly invoked")
        started = time.monotonic()
        cpu_started = time.process_time()
        self._active_inputs += 1
        self._input_calls += 1
        try:
            if self._torch.is_grad_enabled():
                raise LayoutError("Layout factor requires inference without gradient recording")
            if args:
                if "input" in kwargs:
                    raise LayoutError("Ambiguous Conv2d input")
                tensor = args[0]
            elif "input" in kwargs:
                tensor = kwargs["input"]
            else:
                raise LayoutError("Missing Conv2d input")
            if (
                tensor.ndim != 4 or tensor.shape[1] != 1
                or tensor.device.type != "cpu" or tensor.dtype != self._torch.float32
                or tensor.layout != self._torch.strided or tensor.requires_grad
                or module.training
            ):
                raise LayoutError("First Conv2d input must be dense CPU FP32 (B,1,F,T), eval/no-grad")
            before = (tuple(tensor.shape), tuple(tensor.stride()),
                      tensor.is_contiguous(memory_format=self._torch.channels_last))
            converted = tensor.contiguous(memory_format=self._torch.channels_last)
            if (
                tuple(converted.shape) != before[0] or converted.dtype != tensor.dtype
                or converted.device != tensor.device or converted.requires_grad != tensor.requires_grad
                or not converted.is_contiguous(memory_format=self._torch.channels_last)
            ):
                raise LayoutError("Input layout conversion changed metadata or failed")
            key = (*before, tuple(converted.stride()),
                   converted.is_contiguous(memory_format=self._torch.channels_last),
                   converted is tensor)
            if key in self._inputs or len(self._inputs) < _INPUT_LIMIT:
                self._inputs[key] = self._inputs.get(key, 0) + 1
            else:
                self._unlisted_inputs += 1
            if args:
                return (converted, *args[1:]), kwargs
            return args, {**kwargs, "input": converted}
        except BaseException:
            self._input_failed += 1
            raise
        finally:
            self._input_wall += time.monotonic() - started
            self._input_cpu += time.process_time() - cpu_started
            self._active_inputs -= 1

    def _restore(self) -> None:
        errors = []
        if self._hook is not None:
            try:
                self._hook.remove()
                self._hook = None
            except BaseException as exc:
                errors.append(exc)
        for module, parameter, original_data in self._weights:
            try:
                if module.weight is not parameter:
                    module.weight = parameter
                parameter.data = original_data
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise LayoutError("Failed to remove layout hook or restore original weight views") from errors[0]
        self._closed = True
        _ACTIVE.pop(id(self._network), None)

    def close(self) -> None:
        """Verify inference state, restore exact views, then verify restoration."""
        if self._closed:
            return
        if self._active_inputs:
            raise LayoutError("Cannot close during input conversion")
        errors = []
        try:
            self._measure(
                "layout_verification", lambda: self._verify(self._converted),
                checkpoint="after_inference",
            )
        except BaseException as exc:
            errors.append(exc)
        try:
            self._measure("layout_restoration", self._restore)
        except BaseException as exc:
            errors.append(exc)
        try:
            self._measure(
                "layout_verification", lambda: self._verify(self._baseline),
                checkpoint="after_restore",
            )
        except BaseException as exc:
            errors.append(exc)
        try:
            self._emit(
                "layout_input", self._started, layout_version=1,
                status=("not_observed" if not self._input_calls else
                        "partial_failure" if 0 < self._input_failed < self._input_calls else
                        "failed" if self._input_failed else "completed"),
                measurement="sum_of_pre_hook_calls",
                elapsed_seconds=self._input_wall if self._input_calls else None,
                cpu_seconds=self._input_cpu if self._input_calls else None,
                cpu_scope="process_all_threads",
                calls=self._input_calls,
                completed_calls=self._input_calls - self._input_failed,
                failed_calls=self._input_failed,
                input_layouts=[
                    {"shape": key[0], "before_stride": key[1],
                     "before_channels_last": key[2], "after_stride": key[3],
                     "after_channels_last": key[4], "same_tensor": key[5], "calls": calls}
                    for key, calls in self._inputs.items()
                ],
                layout_limit=_INPUT_LIMIT, unlisted_input_calls=self._unlisted_inputs,
            )
        except BaseException as exc:
            errors.append(exc)
        if errors:
            for error in errors[1:]:
                if hasattr(errors[0], "add_note"):
                    errors[0].add_note(f"Additional layout cleanup error: {type(error).__name__}: {error}")
            raise errors[0]


def install(pipeline: Any, emit: Callable[..., Any]) -> Handle:
    """Install the explicit factor using already-loaded Torch; never import it."""
    if not callable(emit):
        raise LayoutError("Layout factor requires emit(phase, started, **values)")
    torch = sys.modules.get("torch")
    if torch is None:
        raise LayoutError("Torch must already be loaded by the CPU pipeline")
    embedding = getattr(pipeline, "_embedding", None)
    model = getattr(embedding, "model_", None)
    network = getattr(model, "resnet", None)
    if not (
        _kind(embedding, "pyannote.audio.pipelines.speaker_verification",
              "PyannoteAudioPretrainedSpeakerEmbedding")
        and _kind(model, "pyannote.audio.models.embedding.wespeaker", "BaseWeSpeakerResNet")
        and _kind(network, "pyannote.audio.models.embedding.wespeaker.resnet", "ResNet")
        and getattr(getattr(embedding, "device", None), "type", None) == "cpu"
    ):
        raise LayoutError("Layout factor supports only the CPU PyTorch WeSpeaker ResNet backend")
    if id(network) in _ACTIVE:
        raise LayoutError("Layout factor is already installed on this network")
    if not isinstance(getattr(network, "conv1", None), torch.nn.Conv2d):
        raise LayoutError("Pinned first Conv2d entry is unsupported")
    utility = getattr(torch.nn.utils, "convert_conv2d_weight_memory_format", None)
    if not callable(utility):
        raise LayoutError("Torch lacks convert_conv2d_weight_memory_format")
    handle = Handle(model, network, torch, emit)
    _ACTIVE[id(network)] = handle
    try:
        handle._measure("layout_verification", handle._prepare, checkpoint="before_conversion")
        handle._measure(
            "layout_conversion", lambda: utility(network, torch.channels_last),
            factor="conv2d_weights_channels_last",
        )
        handle._measure(
            "layout_verification", lambda: handle._verify(handle._baseline, True),
            checkpoint="after_conversion",
        )
        handle._hook = network.conv1.register_forward_pre_hook(
            handle._input_hook, with_kwargs=True, prepend=False,
        )
    except BaseException as original:
        try:
            handle._restore()
            if handle._baseline:
                handle._measure(
                    "layout_verification", lambda: handle._verify(handle._baseline),
                    checkpoint="after_rollback",
                )
        except BaseException as cleanup:
            raise LayoutError("Layout installation failed and rollback did not verify") from cleanup
        raise original
    return handle
