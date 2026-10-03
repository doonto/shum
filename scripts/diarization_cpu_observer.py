"""Opt-in, reversible CPU timing for the installed Community-1 WeSpeaker path.

No inference libraries are imported here. Install after loading the pipeline,
call it normally, and close the observer in a finally block. Use one observer
and a synchronous pipeline call per process.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import inspect
import time
from typing import Any


class ObserverError(RuntimeError):
    """Detail profiling cannot be installed or safely restored."""


_MISSING = object()
_CROP_OBSERVER: ContextVar[Any] = ContextVar("diarization_crop_observer", default=None)
_SHAPE_LIMIT = 16
_SCOPES = {
    "detail_audio_crop": None,
    "detail_audio_decoder_init": "detail_audio_crop",
    "detail_audio_range_read": "detail_audio_crop",
    "detail_embedding_forward": None,
    "detail_embedding_fbank": "detail_embedding_forward",
    "detail_embedding_network": "detail_embedding_forward",
}


def _is_kind(value: Any, module: str, name: str) -> bool:
    return any(
        cls.__module__ == module and cls.__name__ == name
        for cls in type(value).__mro__
    )


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if isinstance(shape, (tuple, list)) and all(type(size) is int for size in shape):
        return tuple(shape)
    return None


def _argument(args: tuple, kwargs: dict, index: int, name: str) -> Any:
    return args[index] if len(args) > index else kwargs.get(name)


def _inputs(phase: str, args: tuple, kwargs: dict) -> dict:
    if phase == "detail_audio_crop":
        audio = _argument(args, kwargs, 0, "file")
        return {
            "input_kind": "waveform" if isinstance(audio, Mapping) and "waveform" in audio
            else "file",
        }
    if phase in ("detail_audio_decoder_init", "detail_audio_range_read"):
        return {}
    name = "fbank" if phase == "detail_embedding_network" else "waveforms"
    values = {"input_shape": _shape(_argument(args, kwargs, 0, name))}
    if phase != "detail_embedding_fbank":
        values["weights_shape"] = _shape(_argument(args, kwargs, 1, "weights"))
    return values


def _outputs(phase: str, result: Any) -> dict:
    if phase == "detail_audio_decoder_init":
        return {}
    if phase == "detail_audio_crop":
        return {"output_shape": _shape(result[0])}
    if phase == "detail_audio_range_read":
        return {"output_shape": _shape(result.data)}
    if phase == "detail_embedding_network":
        return {"output_shapes": tuple(_shape(value) for value in result)}
    return {"output_shape": _shape(result)}


@dataclass
class _Measurement:
    calls: int = 0
    completed_calls: int = 0
    failed_calls: int = 0
    nested_calls: int = 0
    standalone_calls: int = 0
    elapsed_seconds: float = 0.0
    cpu_seconds: float = 0.0
    input_items: int = 0
    shaped_input_calls: int = 0
    shapes: dict[tuple, int] = field(default_factory=dict)
    unlisted_shape_calls: int = 0
    metadata_errors: int = 0
    metadata_error_types: set[str] = field(default_factory=set)


@dataclass
class _Patch:
    owner: Any
    name: str
    original: Any
    wrapper: Callable

    def restore(self) -> None:
        if vars(self.owner).get(self.name, _MISSING) is not self.wrapper:
            raise ObserverError(f"Observed {self.name} changed while profiling; not overwriting it")
        if self.original is _MISSING:
            delattr(self.owner, self.name)
        else:
            setattr(self.owner, self.name, self.original)


class Observer:
    """Collect inclusive call intervals, then restore wrappers and emit sums."""

    def __init__(self, emit: Callable[..., Any]):
        self._emit = emit
        self._started = time.monotonic()
        self._measurements = {phase: _Measurement() for phase in _SCOPES}
        self._patches: list[_Patch] = []
        self._active_calls = 0
        self._closed = False
        self._scopes: ContextVar[tuple[str, ...]] = ContextVar(
            "diarization_detail_scopes", default=(),
        )

    def _metadata(self, phase: str, build: Callable, *args: Any) -> dict:
        try:
            return build(phase, *args)
        except Exception as exc:
            # Metadata must never replace a model result or its original exception.
            measurement = self._measurements[phase]
            measurement.metadata_errors += 1
            measurement.metadata_error_types.add(type(exc).__name__)
            return {}

    def _call(self, phase: str, original: Callable, args: tuple, kwargs: dict) -> Any:
        measurement = self._measurements[phase]
        metadata = self._metadata(phase, _inputs, args, kwargs)
        started = time.monotonic()
        cpu_started = time.process_time()
        parents = self._scopes.get()
        nested = _SCOPES[phase] in parents
        token = self._scopes.set((*parents, phase))
        self._active_calls += 1
        status = "failed"
        try:
            result = original(*args, **kwargs)
            status = "completed"
        finally:
            elapsed = time.monotonic() - started
            cpu = time.process_time() - cpu_started
            self._scopes.reset(token)
            self._active_calls -= 1
            measurement.calls += 1
            if nested:
                measurement.nested_calls += 1
            else:
                measurement.standalone_calls += 1
            measurement.elapsed_seconds += elapsed
            measurement.cpu_seconds += cpu
            if status == "completed":
                measurement.completed_calls += 1
            else:
                measurement.failed_calls += 1
                self._record_shapes(measurement, {**metadata, "status": status})
        metadata.update(self._metadata(phase, _outputs, result))
        self._record_shapes(measurement, {**metadata, "status": status})
        return result

    @staticmethod
    def _record_shapes(measurement: _Measurement, metadata: dict) -> None:
        shape = metadata.get("input_shape")
        if shape:
            measurement.shaped_input_calls += 1
            measurement.input_items += shape[0]
        key = tuple(sorted(metadata.items()))
        if key in measurement.shapes or len(measurement.shapes) < _SHAPE_LIMIT:
            measurement.shapes[key] = measurement.shapes.get(key, 0) + 1
        else:
            measurement.unlisted_shape_calls += 1

    def _patch(self, owner: Any, name: str, wrapper: Callable) -> None:
        original = vars(owner).get(name, _MISSING)
        wrapper._shum_cpu_observer = True
        setattr(owner, name, wrapper)
        self._patches.append(_Patch(owner, name, original, wrapper))

    def _wrap_instance(self, owner: Any, name: str, phase: str) -> None:
        original = getattr(owner, name)

        @wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if phase != "detail_audio_crop":
                return self._call(phase, original, args, kwargs)
            token = _CROP_OBSERVER.set(self)
            try:
                return self._call(phase, original, args, kwargs)
            finally:
                _CROP_OBSERVER.reset(token)

        self._patch(owner, name, wrapped)

    def _wrap_decoder(self, decoder: type, name: str, phase: str) -> None:
        original = getattr(decoder, name)

        @wraps(original)
        def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if _CROP_OBSERVER.get() is not self:
                return original(instance, *args, **kwargs)
            return self._call(
                phase, lambda *a, **kw: original(instance, *a, **kw), args, kwargs,
            )

        self._patch(decoder, name, wrapped)

    def _restore(self) -> None:
        failures = []
        unresolved = []
        for patch in reversed(self._patches):
            try:
                patch.restore()
            except Exception as exc:
                failures.append(exc)
                unresolved.append(patch)
        self._patches = list(reversed(unresolved))
        if failures:
            raise ObserverError(
                "Could not restore observer wrappers: " + "; ".join(str(exc) for exc in failures)
            ) from failures[0]

    def close(self) -> None:
        """Restore all wrappers before emitting; idempotent after restoration."""
        if self._closed:
            return
        if self._active_calls:
            raise ObserverError("Cannot close the observer during an observed call")
        self._restore()
        self._closed = True
        observed_elapsed = time.monotonic() - self._started
        for phase, measurement in self._measurements.items():
            if not measurement.calls:
                status = "not_observed"
            elif not measurement.failed_calls:
                status = "completed"
            elif measurement.completed_calls:
                status = "partial_failure"
            else:
                status = "failed"
            self._emit(
                phase, self._started,
                detail_version=1,
                status=status,
                measurement="sum_of_calls",
                scope="inclusive",
                nested_in=_SCOPES[phase],
                cpu_scope="process_all_threads",
                elapsed_seconds=measurement.elapsed_seconds if measurement.calls else None,
                cpu_seconds=measurement.cpu_seconds if measurement.calls else None,
                observer_elapsed_seconds=observed_elapsed,
                calls=measurement.calls,
                completed_calls=measurement.completed_calls,
                failed_calls=measurement.failed_calls,
                nested_calls=measurement.nested_calls,
                standalone_calls=measurement.standalone_calls,
                input_items=measurement.input_items,
                shaped_input_calls=measurement.shaped_input_calls,
                shapes=[{**dict(key), "calls": calls}
                        for key, calls in measurement.shapes.items()],
                shape_limit=_SHAPE_LIMIT,
                unlisted_shape_calls=measurement.unlisted_shape_calls,
                metadata_errors=measurement.metadata_errors,
                metadata_error_types=sorted(measurement.metadata_error_types),
            )


def install(pipeline: Any, emit: Callable[..., Any]) -> Observer:
    """Install detail profiling, or raise ObserverError without leaving hooks."""
    if not callable(emit):
        raise ObserverError("Detail profiling requires a callable emit(phase, started, **values)")
    embedding = getattr(pipeline, "_embedding", None)
    if not _is_kind(
        embedding, "pyannote.audio.pipelines.speaker_verification",
        "PyannoteAudioPretrainedSpeakerEmbedding",
    ):
        raise ObserverError("Detail profiling requires the PyTorch pyannote embedding backend")
    model = getattr(embedding, "model_", None)
    if not _is_kind(
        model, "pyannote.audio.models.embedding.wespeaker", "BaseWeSpeakerResNet",
    ):
        raise ObserverError("Detail profiling requires a PyTorch WeSpeaker ResNet model")
    if getattr(getattr(embedding, "device", None), "type", None) != "cpu":
        raise ObserverError("Detail profiling supports only synchronous CPU execution")
    if getattr(getattr(model, "device", embedding.device), "type", None) != "cpu":
        raise ObserverError("Detail profiling requires the embedding model on CPU")
    audio = getattr(pipeline, "_audio", None)
    if not _is_kind(audio, "pyannote.audio.core.io", "Audio"):
        raise ObserverError("Detail profiling requires pyannote Audio.crop")
    network = getattr(model, "resnet", None)
    if not _is_kind(
        network, "pyannote.audio.models.embedding.wespeaker.resnet", "ResNet",
    ):
        raise ObserverError("Detail profiling requires the WeSpeaker ResNet.forward boundary")
    targets = [
        (audio, "crop", "detail_audio_crop"),
        (model, "forward", "detail_embedding_forward"),
        (model, "compute_fbank", "detail_embedding_fbank"),
        (network, "forward", "detail_embedding_network"),
    ]
    for owner, name, _phase in targets:
        method = getattr(owner, name, None)
        if not callable(method) or getattr(method, "_shum_cpu_observer", False):
            raise ObserverError(f"Detail boundary {name} is missing or already observed")
        if (
            getattr(owner, "_compiled_call_impl", None) is not None
            or hasattr(method, "_torchdynamo_orig_callable")
        ):
            raise ObserverError(f"Detail boundary {name} is compiled and unsupported")
    crop = inspect.unwrap(audio.crop)
    crop_function = getattr(crop, "__func__", crop)
    decoder = getattr(crop_function, "__globals__", {}).get("AudioDecoder")
    if not isinstance(decoder, type) or not any(
        cls.__module__ == "torchcodec.decoders._audio_decoder" and cls.__name__ == "AudioDecoder"
        for cls in decoder.__mro__
    ):
        raise ObserverError("Detail profiling requires the loaded TorchCodec AudioDecoder")
    decoder_targets = [
        ("__init__", "detail_audio_decoder_init"),
        ("get_samples_played_in_range", "detail_audio_range_read"),
    ]
    for name, _phase in decoder_targets:
        method = getattr(decoder, name, None)
        if not inspect.isfunction(method) or getattr(method, "_shum_cpu_observer", False):
            raise ObserverError(f"TorchCodec boundary {name} is unsupported or already observed")
    observer = Observer(emit)
    try:
        for owner, name, phase in targets:
            observer._wrap_instance(owner, name, phase)
        for name, phase in decoder_targets:
            observer._wrap_decoder(decoder, name, phase)
    except BaseException:
        observer._restore()
        raise
    return observer
