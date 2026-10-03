"""Channels-last factor tests using byte-aware stand-ins, not Torch/models."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import importlib.util
import inspect
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest


PATH = Path(__file__).resolve().parents[3] / "scripts" / "diarization_cpu_layout.py"
SPEC = importlib.util.spec_from_file_location("diarization_cpu_layout", PATH)
assert SPEC and SPEC.loader
layout = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = layout
SPEC.loader.exec_module(layout)


@dataclass(frozen=True)
class Device:
    type: str = "cpu"

    def __str__(self):
        return self.type


class Storage:
    next_ptr = 1000

    def __init__(self, size):
        self.ptr = Storage.next_ptr
        Storage.next_ptr += 1000
        self.size = size

    def data_ptr(self):
        return self.ptr

    def nbytes(self):
        return self.size


def strides(shape, channels_last=False):
    if channels_last:
        n, c, h, w = shape
        return c * h * w, 1, w * c, c
    result = []
    stride = 1
    for size in reversed(shape):
        result.insert(0, stride)
        stride *= size
    return tuple(result)


class Tensor:
    def __init__(self, values, shape, *, dtype="float32", stride=None, offset=0,
                 storage=None, device=None, requires_grad=False):
        self.values = list(values)
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device or Device()
        self.layout = "strided"
        self.requires_grad = requires_grad
        self.grad = None
        self._stride = stride or strides(self.shape)
        self.offset = offset
        self.storage = storage or Storage((len(self.values) + offset) * self.itemsize)

    @property
    def itemsize(self):
        return 1 if self.dtype == "uint8" else 8 if self.dtype == "int64" else 4

    @property
    def ndim(self):
        return len(self.shape)

    def stride(self):
        return self._stride

    def storage_offset(self):
        return self.offset

    def data_ptr(self):
        return self.storage.data_ptr() + self.offset * self.itemsize

    def untyped_storage(self):
        return self.storage

    def detach(self):
        return self

    def is_floating_point(self):
        return self.dtype in ("float32", "float16")

    def is_complex(self):
        return False

    def is_contiguous(self, memory_format=None):
        return self._stride == strides(self.shape, memory_format == "channels_last")

    def contiguous(self, memory_format=None):
        if self.is_contiguous(memory_format):
            return self
        return Tensor(self.values, self.shape, dtype=self.dtype, device=self.device,
                      stride=strides(self.shape, memory_format == "channels_last"))

    def clone(self, memory_format=None):
        return Tensor(self.values, self.shape, dtype=self.dtype, device=self.device,
                      stride=strides(self.shape, memory_format == "channels_last"))

    def reshape(self, *shape):
        if shape == (-1,):
            shape = (len(self.values),)
        return Tensor(self.values, shape, dtype=self.dtype, device=self.device)

    def view(self, dtype):
        assert dtype == "uint8"
        if self.ndim == 0:
            raise RuntimeError("A scalar cannot change element size through a dtype view")
        assert self.ndim == 1 and self.is_contiguous()
        code = "q" if self.dtype == "int64" else "f"
        raw = struct.pack("<" + code * len(self.values), *self.values)
        return Tensor(raw, (len(raw),), dtype="uint8")

    def numpy(self):
        assert self.dtype == "uint8"
        return SimpleNamespace(tobytes=lambda order: bytes(self.values))


class Parameter:
    def __init__(self, data):
        self.data = data
        self.requires_grad = True
        self.grad = None

    def __getattr__(self, name):
        return getattr(self.data, name)


class Module:
    def __init__(self):
        self.training = False
        self.children_by_name = OrderedDict()
        self.parameters_by_name = OrderedDict()
        self.buffers_by_name = OrderedDict()

    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        if memo is None:
            memo = set()
        if remove_duplicate and self in memo:
            return
        if remove_duplicate:
            memo.add(self)
        yield prefix, self
        for name, child in self.children_by_name.items():
            child_prefix = prefix + ("." if prefix else "") + name
            yield from child.named_modules(
                memo=memo, prefix=child_prefix, remove_duplicate=remove_duplicate,
            )

    def modules(self):
        return (module for _name, module in self.named_modules())

    def _named_members(self, attribute, prefix, recurse, remove_duplicate):
        modules = (
            self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate)
            if recurse else [(prefix, self)]
        )
        seen = set()
        for module_prefix, module in modules:
            for name, tensor in getattr(module, attribute).items():
                if tensor is None or (remove_duplicate and id(tensor) in seen):
                    continue
                seen.add(id(tensor))
                yield (module_prefix + "." if module_prefix else "") + name, tensor

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        yield from self._named_members(
            "parameters_by_name", prefix, recurse, remove_duplicate,
        )

    def named_buffers(self, prefix="", recurse=True, remove_duplicate=True):
        yield from self._named_members(
            "buffers_by_name", prefix, recurse, remove_duplicate,
        )

    def forward(self, *args, **kwargs):
        return args, kwargs


class Conv2d(Module):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == "weight" and hasattr(self, "parameters_by_name"):
            self.parameters_by_name["weight"] = value

    def __init__(self):
        super().__init__()
        self.weight = Parameter(Tensor([-0.0] + [float(i) for i in range(1, 18)],
                                       (2, 1, 3, 3), stride=(40, 20, 6, 2), offset=3))
        self.parameters_by_name["weight"] = self.weight
        self.hooks = OrderedDict()
        self.next_hook = 0
        self.seen = []
        self.failure = None

    def register_forward_pre_hook(self, hook, *, with_kwargs=False, prepend=False):
        assert with_kwargs is True
        key = self.next_hook
        self.next_hook += 1
        self.hooks[key] = hook
        if prepend:
            self.hooks.move_to_end(key, last=False)
        return SimpleNamespace(remove=lambda: self.hooks.pop(key, None))

    def __call__(self, *args, **kwargs):
        for hook in list(self.hooks.values()):
            replacement = hook(self, args, kwargs)
            if replacement is not None:
                args, kwargs = replacement
        self.seen.append((args, kwargs))
        if self.failure:
            raise self.failure
        return args[0] if args else kwargs["input"]


class ConvTranspose2d(Module):
    pass


class ResNet(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d()
        self.conv2 = Conv2d()
        self.children_by_name.update(conv1=self.conv1, conv2=self.conv2)
        self.buffers_by_name["running_mean"] = Tensor([1.0, 2.0], (2,))
        self.buffers_by_name["num_batches_tracked"] = Tensor([0], (), dtype="int64")


ResNet.__module__ = "pyannote.audio.models.embedding.wespeaker.resnet"


class BaseWeSpeakerResNet(Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNet()
        self.children_by_name["resnet"] = self.resnet


BaseWeSpeakerResNet.__module__ = "pyannote.audio.models.embedding.wespeaker"


class PyannoteAudioPretrainedSpeakerEmbedding:
    def __init__(self):
        self.model_ = BaseWeSpeakerResNet()
        self.device = Device()


PyannoteAudioPretrainedSpeakerEmbedding.__module__ = (
    "pyannote.audio.pipelines.speaker_verification"
)


@pytest.fixture
def setup(monkeypatch):
    converted = []

    def utility(module, memory_format):
        assert memory_format == "channels_last"
        converted.append(module)
        for child in module.modules():
            if isinstance(child, Conv2d):
                child.weight.data = child.weight.detach().clone(memory_format=memory_format)
        return module

    torch = SimpleNamespace(
        float32="float32", uint8="uint8", strided="strided", channels_last="channels_last",
        is_grad_enabled=lambda: False,
        nn=SimpleNamespace(
            Conv2d=Conv2d, ConvTranspose2d=ConvTranspose2d,
            utils=SimpleNamespace(convert_conv2d_weight_memory_format=utility),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    clock = [100.0]

    def now():
        clock[0] += 0.01
        return clock[0]

    monkeypatch.setattr(layout.time, "monotonic", now)
    monkeypatch.setattr(layout.time, "process_time", now)
    pipeline = SimpleNamespace(_embedding=PyannoteAudioPretrainedSpeakerEmbedding())
    network = pipeline._embedding.model_.resnet
    original = [(conv.weight, conv.weight.data, layout._metadata(conv.weight))
                for conv in (network.conv1, network.conv2)]
    records = []

    def emit(phase, started, **values):
        records.append({"phase": phase, "elapsed_seconds": now() - started, **values})

    result = SimpleNamespace(
        torch=torch, pipeline=pipeline, network=network, original=original,
        records=records, emit=emit, converted=converted,
    )
    yield result
    for handle in list(layout._ACTIVE.values()):
        handle._restore()


def assert_restored(setup):
    assert len(setup.network.conv1.hooks) == 0
    for parameter, original_data, original_metadata in setup.original:
        assert parameter.data is original_data
        assert layout._metadata(parameter) == original_metadata
    assert id(setup.network) not in layout._ACTIVE


def test_values_metadata_parameter_identity_and_exact_storage_restoration(setup):
    initial_hashes = [layout._logical_hash(parameter, setup.torch)
                      for parameter, _data, _metadata in setup.original]
    handle = layout.install(setup.pipeline, setup.emit)
    assert setup.converted == [setup.network]
    for index, (parameter, original, metadata) in enumerate(setup.original):
        assert parameter.data is not original
        assert parameter.dtype == "float32" and parameter.device.type == "cpu"
        assert parameter.requires_grad is True and parameter.grad is None
        assert parameter.is_contiguous(memory_format="channels_last")
        assert layout._logical_hash(parameter, setup.torch) == initial_hashes[index]
        assert parameter.shape == metadata["shape"]
    input = Tensor(range(12), (1, 1, 3, 4), stride=(12, 12, 1, 3))
    before = layout._logical_hash(input, setup.torch)
    result = setup.network.conv1(input)
    assert result.shape == input.shape
    assert result.is_contiguous(memory_format="channels_last")
    assert result.dtype == input.dtype and result.device == input.device
    assert layout._logical_hash(result, setup.torch) == before
    assert input.stride() == (12, 12, 1, 3)
    handle.close()
    assert_restored(setup)
    checkpoints = [record for record in setup.records if record["phase"] == "layout_verification"]
    assert [record["checkpoint"] for record in checkpoints] == [
        "before_conversion", "after_conversion", "after_inference", "after_restore",
    ]
    assert len({record["state_sha256"] for record in checkpoints}) == 1
    assert all(record["status"] == "completed" for record in checkpoints)
    assert all(record["elapsed_seconds"] > 0 and record["cpu_seconds"] > 0
               for record in checkpoints)
    inputs = setup.records[-1]
    assert inputs["phase"] == "layout_input"
    assert inputs["calls"] == inputs["completed_calls"] == 1
    assert inputs["input_layouts"] == [{
        "shape": (1, 1, 3, 4), "before_stride": (12, 12, 1, 3),
        "before_channels_last": False, "after_stride": (12, 1, 4, 1),
        "after_channels_last": True, "same_tensor": False, "calls": 1,
    }]
    count = len(setup.records)
    handle.close()
    assert len(setup.records) == count


def test_hook_preserves_existing_order_args_kwargs_and_other_tensor_identity(setup):
    seen = []
    conv = setup.network.conv1
    sentinel = object()
    existing = conv.register_forward_pre_hook(
        lambda _module, args, kwargs: seen.append((args, kwargs)),
        with_kwargs=True,
    )
    original_hooks = dict(conv.hooks)
    handle = layout.install(setup.pipeline, setup.emit)
    tensor = Tensor(range(12), (1, 1, 3, 4))
    assert conv(tensor, sentinel, token=sentinel).shape == tensor.shape
    args, kwargs = conv.seen[-1]
    assert args[1] is sentinel and kwargs["token"] is sentinel
    assert seen[0][0][0] is tensor
    assert conv(input=tensor, token=sentinel).shape == tensor.shape
    args, kwargs = conv.seen[-1]
    assert args == () and kwargs["token"] is sentinel
    assert kwargs["input"].shape == tensor.shape
    handle.close()
    assert dict(conv.hooks) == original_hooks
    existing.remove()
    assert_restored(setup)


def test_signed_zero_hash_distinguishes_values_equal_under_numeric_comparison(setup):
    positive = Tensor([0.0], (1,))
    negative = Tensor([-0.0], (1,))
    assert positive.values == negative.values
    assert math.copysign(1.0, negative.values[0]) == -1.0
    assert layout._logical_hash(positive, setup.torch) != layout._logical_hash(negative, setup.torch)


@pytest.mark.parametrize("value,dtype,code", [
    (0, "int64", "q"),
    (-1, "int64", "q"),
    (2**63 - 1, "int64", "q"),
    (-0.0, "float32", "f"),
])
def test_scalar_hash_flattens_before_dtype_view_and_preserves_raw_bytes(setup, value, dtype, code):
    scalar = Tensor([value], (), dtype=dtype)
    before = layout._metadata(scalar)
    with pytest.raises(RuntimeError, match="scalar"):
        scalar.view("uint8")
    expected = hashlib.sha256(struct.pack("<" + code, value)).hexdigest()
    assert layout._logical_hash(scalar, setup.torch) == expected
    assert layout._metadata(scalar) == before
    assert scalar.shape == ()


@pytest.mark.parametrize("alias", ["parameter", "storage", "buffer", "module"])
def test_aliases_are_rejected_before_conversion(setup, alias):
    network = setup.network
    if alias == "parameter":
        network.conv2.weight = network.conv1.weight
        network.conv2.parameters_by_name["weight"] = network.conv1.weight
    elif alias == "storage":
        network.conv2.weight.data.storage = network.conv1.weight.data.storage
    elif alias == "buffer":
        network.buffers_by_name["alias"] = network.buffers_by_name["running_mean"]
    else:
        network.children_by_name["again"] = network.conv1
    with pytest.raises(layout.LayoutError, match="alias|Shared"):
        layout.install(setup.pipeline, setup.emit)
    assert setup.converted == []
    assert not network.conv1.hooks
    assert not layout._ACTIVE


@pytest.mark.parametrize("failure", [
    "partial_conversion", "bytes", "dtype", "identity", "alias", "hook", "emit",
])
def test_failed_install_restores_partial_conversion(setup, failure):
    original_utility = setup.torch.nn.utils.convert_conv2d_weight_memory_format
    original_error = RuntimeError("injected install failure")

    def utility(network, memory_format):
        if failure == "partial_conversion":
            network.conv1.weight.data = network.conv1.weight.data.clone(memory_format)
            raise original_error
        original_utility(network, memory_format)
        if failure == "bytes":
            network.conv1.weight.data.values[0] = 0.0
        if failure == "dtype":
            network.conv1.weight.data.dtype = "float16"
        if failure == "identity":
            network.conv1.weight = Parameter(network.conv1.weight.data)
        if failure == "alias":
            network.conv2.weight.data.storage = network.conv1.weight.data.storage
        return network

    setup.torch.nn.utils.convert_conv2d_weight_memory_format = utility
    if failure == "hook":
        def failed_hook(*args, **kwargs):
            raise original_error
        setup.network.conv1.register_forward_pre_hook = failed_hook

    def emit(phase, started, **values):
        setup.emit(phase, started, **values)
        if failure == "emit" and phase == "layout_conversion":
            raise original_error

    with pytest.raises((RuntimeError, layout.LayoutError)) as caught:
        layout.install(setup.pipeline, emit)
    if failure in ("partial_conversion", "hook", "emit"):
        assert caught.value is original_error
    assert_restored(setup)


def test_changed_inference_weight_is_detected_and_original_bytes_restored(setup):
    handle = layout.install(setup.pipeline, setup.emit)
    setup.network.conv1.weight.data.values[0] = 0.0
    with pytest.raises(layout.LayoutError, match="Logical bytes changed"):
        handle.close()
    assert_restored(setup)
    assert math.copysign(1.0, setup.network.conv1.weight.data.values[0]) == -1.0
    assert any(record.get("checkpoint") == "after_inference" and record["status"] == "failed"
               for record in setup.records)
    assert any(record.get("checkpoint") == "after_restore" and record["status"] == "completed"
               for record in setup.records)
    handle.close()


@pytest.mark.parametrize("change", ["identity", "alias", "storage"])
def test_inference_identity_and_storage_changes_are_detected_and_restored(setup, change):
    handle = layout.install(setup.pipeline, setup.emit)
    if change == "identity":
        setup.network.conv1.weight = Parameter(setup.network.conv1.weight.data)
    elif change == "alias":
        setup.network.conv2.weight.data.storage = setup.network.conv1.weight.data.storage
    else:
        setup.network.conv1.weight.data = setup.network.conv1.weight.data.clone("channels_last")
    with pytest.raises(layout.LayoutError, match="identity|alias|metadata"):
        handle.close()
    assert setup.network.conv1.weight is setup.original[0][0]
    assert setup.network.conv2.weight is setup.original[1][0]
    assert_restored(setup)


def test_changed_buffer_fails_verification_without_silent_repair(setup):
    handle = layout.install(setup.pipeline, setup.emit)
    setup.network.buffers_by_name["running_mean"].values[0] = 9.0
    with pytest.raises(layout.LayoutError, match="Logical bytes changed"):
        handle.close()
    assert_restored(setup)
    assert setup.network.buffers_by_name["running_mean"].values[0] == 9.0


def test_original_forward_exception_survives_failure_cleanup(setup):
    error = RuntimeError("network error")
    setup.network.conv1.failure = error
    handle = layout.install(setup.pipeline, setup.emit)
    with pytest.raises(RuntimeError) as caught:
        try:
            setup.network.conv1(Tensor(range(12), (1, 1, 3, 4)))
        finally:
            handle.close()
    assert caught.value is error
    assert_restored(setup)


@pytest.mark.parametrize("unsupported", [
    "backend", "device", "training", "compiled", "dtype", "gradient", "utility", "torch",
])
def test_unsupported_paths_fail_without_changes(setup, monkeypatch, unsupported):
    if unsupported == "backend":
        setup.pipeline._embedding = SimpleNamespace(model_=object())
    elif unsupported == "device":
        setup.pipeline._embedding.device = Device("cuda")
    elif unsupported == "training":
        setup.network.conv2.training = True
    elif unsupported == "compiled":
        setup.network._compiled_call_impl = object()
    elif unsupported == "dtype":
        setup.network.conv2.weight.data.dtype = "float16"
    elif unsupported == "gradient":
        setup.network.conv1.weight.grad = object()
    elif unsupported == "utility":
        setup.torch.nn.utils.convert_conv2d_weight_memory_format = None
    else:
        monkeypatch.delitem(sys.modules, "torch")
    with pytest.raises(layout.LayoutError):
        layout.install(setup.pipeline, setup.emit)
    assert setup.converted == []
    assert not setup.network.conv1.hooks
    assert not layout._ACTIVE


@pytest.mark.parametrize("invalid", ["shape", "dtype", "device", "grad", "grad_mode"])
def test_invalid_input_fails_explicitly_and_close_restores(setup, invalid):
    handle = layout.install(setup.pipeline, setup.emit)
    tensor = Tensor(range(12), (1, 1, 3, 4))
    if invalid == "shape":
        tensor.shape = (1, 3, 4)
    elif invalid == "dtype":
        tensor.dtype = "float16"
    elif invalid == "device":
        tensor.device = Device("cuda")
    elif invalid == "grad":
        tensor.requires_grad = True
    else:
        setup.torch.is_grad_enabled = lambda: True
    with pytest.raises(layout.LayoutError):
        setup.network.conv1(tensor)
    handle.close()
    assert_restored(setup)
    assert setup.records[-1]["status"] == "failed"
    assert setup.records[-1]["failed_calls"] == 1


def test_double_install_rejected_and_layout_histogram_bounded(setup):
    handle = layout.install(setup.pipeline, setup.emit)
    with pytest.raises(layout.LayoutError, match="already installed"):
        layout.install(setup.pipeline, setup.emit)
    for batch in range(1, 19):
        setup.network.conv1(Tensor([0.0] * (batch * 12), (batch, 1, 3, 4)))
    handle.close()
    record = setup.records[-1]
    assert record["calls"] == 18
    assert len(record["input_layouts"]) == record["layout_limit"] == 16
    assert record["unlisted_input_calls"] == 2
    assert_restored(setup)


def test_emission_failure_on_close_does_not_leave_layout_or_hook_installed(setup):
    def emit(phase, started, **values):
        if values.get("checkpoint") == "after_inference":
            raise OSError("stderr unavailable")
        setup.emit(phase, started, **values)

    handle = layout.install(setup.pipeline, emit)
    with pytest.raises(OSError, match="stderr unavailable"):
        handle.close()
    assert_restored(setup)
    handle.close()


def test_unused_factor_reports_no_fabricated_input_timing(setup):
    handle = layout.install(setup.pipeline, setup.emit)
    handle.close()
    record = setup.records[-1]
    assert record["status"] == "not_observed"
    assert record["calls"] == 0
    assert record["elapsed_seconds"] is None and record["cpu_seconds"] is None


@pytest.mark.parametrize("name, parameters", [
    ("named_modules", ("self", "memo", "prefix", "remove_duplicate")),
    ("named_parameters", ("self", "prefix", "recurse", "remove_duplicate")),
    ("named_buffers", ("self", "prefix", "recurse", "remove_duplicate")),
])
def test_named_api_matches_installed_torch_and_rejects_plural(name, parameters):
    assert tuple(inspect.signature(getattr(Module, name)).parameters) == parameters
    method = getattr(Module(), name)
    with pytest.raises(TypeError, match="remove_duplicates"):
        list(method(remove_duplicates=False))
    list(method(remove_duplicate=False))
