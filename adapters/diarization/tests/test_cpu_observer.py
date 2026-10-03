"""Observer tests with Python-only stand-ins; never load inference libraries."""
from __future__ import annotations

from functools import wraps
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "diarization_cpu_observer.py"
SPEC = importlib.util.spec_from_file_location("diarization_cpu_observer", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
observer_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = observer_module
SPEC.loader.exec_module(observer_module)


class Clock:
    wall = 100.0
    cpu = 20.0

    def advance(self, wall, cpu):
        self.wall += wall
        self.cpu += cpu


class Tensor:
    def __init__(self, *shape):
        self.shape = shape


class AudioDecoder:
    clock = None
    failure = None
    calls = []
    samples = None

    def __init__(self, source):
        self.calls.append(("init", source))
        self.clock.advance(2.0, 1.0)
        if self.failure and self.failure[0] == "init":
            raise self.failure[1]

    def get_samples_played_in_range(self, start_seconds=0.0, stop_seconds=None):
        self.calls.append(("range", start_seconds, stop_seconds))
        self.clock.advance(3.0, 2.0)
        if self.failure and self.failure[0] == "range":
            raise self.failure[1]
        return self.samples


AudioDecoder.__module__ = "torchcodec.decoders._audio_decoder"


class Audio:
    def __init__(self, clock, waveform):
        self.clock = clock
        self.waveform = waveform
        self.calls = []
        self.failure = None

    def crop(self, file, segment, mode="raise"):
        self.calls.append((file, segment, mode))
        self.clock.advance(1.0, 0.2)
        if self.failure:
            raise self.failure
        if isinstance(file, dict) and "waveform" in file:
            result = file["waveform"]
        else:
            decoder = AudioDecoder(file)
            result = decoder.get_samples_played_in_range(segment.start, segment.end).data
        self.clock.advance(1.0, 0.4)
        return result, 16000


Audio.__module__ = "pyannote.audio.core.io"


class ResNet:
    def __init__(self, clock):
        self.clock = clock
        self.output = (Tensor(), Tensor(2, 256))
        self.calls = []
        self.failure = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, fbank, weights=None):
        self.calls.append((fbank, weights))
        self.clock.advance(6.0, 5.0)
        if self.failure:
            raise self.failure
        return self.output


ResNet.__module__ = "pyannote.audio.models.embedding.wespeaker.resnet"


class BaseWeSpeakerResNet:
    def __init__(self, clock):
        self.clock = clock
        self.resnet = ResNet(clock)
        self.features = Tensor(2, 98, 80)
        self.fbank_calls = []
        self.failure = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def compute_fbank(self, waveforms):
        self.fbank_calls.append(waveforms)
        self.clock.advance(4.0, 3.0)
        if self.failure:
            raise self.failure
        return self.features

    def forward(self, waveforms, weights=None):
        self.clock.advance(0.5, 0.1)
        fbank = self.compute_fbank(waveforms)
        result = self.resnet(fbank, weights=weights)[1]
        self.clock.advance(0.5, 0.2)
        return result


BaseWeSpeakerResNet.__module__ = "pyannote.audio.models.embedding.wespeaker"


class PyannoteAudioPretrainedSpeakerEmbedding:
    def __init__(self, clock):
        self.model_ = BaseWeSpeakerResNet(clock)
        self.device = SimpleNamespace(type="cpu")


PyannoteAudioPretrainedSpeakerEmbedding.__module__ = (
    "pyannote.audio.pipelines.speaker_verification"
)


@pytest.fixture
def setup(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(observer_module.time, "monotonic", lambda: clock.wall)
    monkeypatch.setattr(observer_module.time, "process_time", lambda: clock.cpu)
    monkeypatch.setattr(AudioDecoder, "clock", clock)
    monkeypatch.setattr(AudioDecoder, "calls", [])
    monkeypatch.setattr(AudioDecoder, "failure", None)
    waveform = Tensor(1, 16000)
    samples = SimpleNamespace(data=waveform)
    monkeypatch.setattr(AudioDecoder, "samples", samples)
    pipeline = SimpleNamespace(
        _audio=Audio(clock, waveform),
        _embedding=PyannoteAudioPretrainedSpeakerEmbedding(clock),
    )
    records = []

    def emit(phase, started, **values):
        # Matches cpu_profile_worker.event: explicit values override its timer.
        records.append({
            "phase": phase, "elapsed_seconds": clock.wall - started, **values,
        })

    return SimpleNamespace(
        clock=clock, pipeline=pipeline, records=records, emit=emit,
        waveform=waveform, samples=samples,
    )


def by_phase(setup):
    return {record["phase"]: record for record in setup.records}


def assert_restored(pipeline, decoder_init, decoder_range):
    assert "crop" not in vars(pipeline._audio)
    assert "forward" not in vars(pipeline._embedding.model_)
    assert "compute_fbank" not in vars(pipeline._embedding.model_)
    assert "forward" not in vars(pipeline._embedding.model_.resnet)
    assert AudioDecoder.__init__ is decoder_init
    assert AudioDecoder.get_samples_played_in_range is decoder_range


def test_exact_nested_measurements_preserve_arguments_and_results(setup):
    pipeline = setup.pipeline
    model = pipeline._embedding.model_
    initial_init = AudioDecoder.__init__
    initial_range = AudioDecoder.get_samples_played_in_range
    observer = observer_module.install(pipeline, setup.emit)
    segment = SimpleNamespace(start=1.0, end=2.0)
    audio = object()
    cropped = pipeline._audio.crop(audio, segment, mode="pad")
    assert cropped[0] is setup.waveform
    assert cropped[1] == 16000
    assert pipeline._audio.calls == [(audio, segment, "pad")]
    assert AudioDecoder.calls == [("init", audio), ("range", 1.0, 2.0)]
    waveforms = Tensor(2, 1, 16000)
    masks = Tensor(2, 50)
    result = model(waveforms, weights=masks)
    assert result is model.resnet.output[1]
    assert model.fbank_calls == [waveforms]
    assert model.resnet.calls == [(model.features, masks)]
    assert setup.records == []
    observer.close()
    assert_restored(pipeline, initial_init, initial_range)
    records = by_phase(setup)
    expected = {
        "detail_audio_crop": (7.0, 3.6, None),
        "detail_audio_decoder_init": (2.0, 1.0, "detail_audio_crop"),
        "detail_audio_range_read": (3.0, 2.0, "detail_audio_crop"),
        "detail_embedding_forward": (11.0, 8.3, None),
        "detail_embedding_fbank": (4.0, 3.0, "detail_embedding_forward"),
        "detail_embedding_network": (6.0, 5.0, "detail_embedding_forward"),
    }
    assert set(records) == set(expected)
    for phase, (wall, cpu, nested_in) in expected.items():
        record = records[phase]
        assert record["elapsed_seconds"] == pytest.approx(wall)
        assert record["cpu_seconds"] == pytest.approx(cpu)
        assert record["nested_in"] == nested_in
        assert record["scope"] == "inclusive"
        assert record["measurement"] == "sum_of_calls"
        assert record["cpu_scope"] == "process_all_threads"
        assert record["status"] == "completed"
        assert record["calls"] == record["completed_calls"] == 1
        assert record["failed_calls"] == 0
        assert record["nested_calls"] == (1 if nested_in else 0)
        assert record["standalone_calls"] == (0 if nested_in else 1)
        assert record["observer_elapsed_seconds"] == 18.0
        assert record["metadata_errors"] == 0
    assert records["detail_embedding_forward"]["input_items"] == 2
    assert records["detail_embedding_forward"]["shapes"] == [{
        "input_shape": (2, 1, 16000), "weights_shape": (2, 50),
        "output_shape": (2, 256), "status": "completed", "calls": 1,
    }]
    assert records["detail_embedding_network"]["shapes"][0]["output_shapes"] == (
        (), (2, 256),
    )
    assert records["detail_audio_range_read"]["shapes"][0]["output_shape"] == (1, 16000)
    observer.close()
    assert len(setup.records) == 6


def test_preloaded_crops_skip_decoder_without_fabricated_timings(setup):
    observer = observer_module.install(setup.pipeline, setup.emit)
    file = {"waveform": setup.waveform, "sample_rate": 16000}
    segment = SimpleNamespace(start=0.0, end=1.0)
    result = setup.pipeline._audio.crop(file=file, segment=segment, mode="pad")
    assert result[0] is setup.waveform
    assert AudioDecoder.calls == []
    observer.close()
    records = by_phase(setup)
    crop = records["detail_audio_crop"]
    assert crop["elapsed_seconds"] == 2.0
    assert crop["shapes"][0]["input_kind"] == "waveform"
    for phase in records.keys() - {"detail_audio_crop"}:
        assert records[phase]["status"] == "not_observed"
        assert records[phase]["calls"] == 0
        assert records[phase]["elapsed_seconds"] is None
        assert records[phase]["cpu_seconds"] is None


def test_unrelated_instances_and_decoder_calls_are_not_recorded(setup):
    observer = observer_module.install(setup.pipeline, setup.emit)
    decoder = AudioDecoder("unrelated")
    assert decoder.get_samples_played_in_range(0.0, 1.0) is setup.samples
    other = Audio(setup.clock, setup.waveform)
    other.crop("other", SimpleNamespace(start=0.0, end=1.0))
    BaseWeSpeakerResNet(setup.clock)(Tensor(2, 1, 16000))
    observer.close()
    assert all(record["calls"] == 0 for record in setup.records)
    assert observer_module._CROP_OBSERVER.get() is None


class OriginalFailure(BaseException):
    pass


@pytest.mark.parametrize("boundary, failed_phases", [
    ("crop", {"detail_audio_crop"}),
    ("init", {"detail_audio_crop", "detail_audio_decoder_init"}),
    ("range", {"detail_audio_crop", "detail_audio_range_read"}),
    ("fbank", {"detail_embedding_forward", "detail_embedding_fbank"}),
    ("network", {"detail_embedding_forward", "detail_embedding_network"}),
])
def test_original_exception_identity_and_partial_measurements(setup, boundary, failed_phases):
    error = OriginalFailure("same object")
    model = setup.pipeline._embedding.model_
    if boundary == "crop":
        setup.pipeline._audio.failure = error
    elif boundary in ("init", "range"):
        AudioDecoder.failure = (boundary, error)
    elif boundary == "fbank":
        model.failure = error
    else:
        model.resnet.failure = error
    initial_init = AudioDecoder.__init__
    initial_range = AudioDecoder.get_samples_played_in_range
    observer = observer_module.install(setup.pipeline, setup.emit)
    try:
        with pytest.raises(OriginalFailure) as caught:
            if boundary in ("crop", "init", "range"):
                setup.pipeline._audio.crop("file", SimpleNamespace(start=0.0, end=1.0))
            else:
                model(Tensor(2, 1, 16000), weights=Tensor(2, 50))
        assert caught.value is error
        assert observer_module._CROP_OBSERVER.get() is None
    finally:
        observer.close()
    assert_restored(setup.pipeline, initial_init, initial_range)
    records = by_phase(setup)
    assert {phase for phase, record in records.items() if record["failed_calls"]} == failed_phases
    for phase in failed_phases:
        assert records[phase]["status"] == "failed"
        assert records[phase]["failed_calls"] == records[phase]["calls"] == 1
        assert records[phase]["elapsed_seconds"] > 0


def test_recovered_model_probes_are_distinct_from_pipeline_failure(setup):
    model = setup.pipeline._embedding.model_
    observer = observer_module.install(setup.pipeline, setup.emit)
    model.failure = ValueError("probe too short")
    with pytest.raises(ValueError):
        model(Tensor(1, 1, 3))
    model.failure = None
    model(Tensor(2, 1, 16000))
    observer.close()
    record = by_phase(setup)["detail_embedding_forward"]
    assert record["status"] == "partial_failure"
    assert record["completed_calls"] == record["failed_calls"] == 1
    assert record["calls"] == 2
    assert record["input_items"] == 3


def test_existing_instance_wrappers_are_restored_exactly(setup):
    audio = setup.pipeline._audio
    original_crop = audio.crop
    seen = []

    @wraps(original_crop)
    def existing(*args, **kwargs):
        seen.append((args, kwargs))
        return original_crop(*args, **kwargs)

    audio.crop = existing
    observer = observer_module.install(setup.pipeline, setup.emit)
    audio.crop("file", SimpleNamespace(start=0.0, end=1.0))
    observer.close()
    assert audio.crop is existing
    assert len(seen) == 1


@pytest.mark.parametrize("unsupported", [
    "embedding", "model", "device", "model_device", "audio", "network",
    "fbank", "decoder", "compiled_module", "compiled_method",
])
def test_unsupported_backend_fails_before_any_changes(setup, monkeypatch, unsupported):
    pipeline = setup.pipeline
    if unsupported == "embedding":
        pipeline._embedding = SimpleNamespace(session_=object())
    elif unsupported == "model":
        pipeline._embedding.model_ = object()
    elif unsupported == "device":
        pipeline._embedding.device.type = "cuda"
    elif unsupported == "model_device":
        pipeline._embedding.model_.device = SimpleNamespace(type="cuda")
    elif unsupported == "audio":
        pipeline._audio = object()
    elif unsupported == "network":
        pipeline._embedding.model_.resnet = object()
    elif unsupported == "fbank":
        pipeline._embedding.model_.compute_fbank = None
    elif unsupported == "compiled_module":
        pipeline._embedding.model_.resnet._compiled_call_impl = object()
    elif unsupported == "compiled_method":
        original = pipeline._embedding.model_.forward

        @wraps(original)
        def compiled(*args, **kwargs):
            return original(*args, **kwargs)

        compiled._torchdynamo_orig_callable = original
        pipeline._embedding.model_.forward = compiled
    else:
        monkeypatch.setitem(Audio.crop.__globals__, "AudioDecoder", None)
    initial = {
        id(owner): dict(vars(owner))
        for owner in (pipeline._audio, pipeline._embedding, pipeline._embedding.model_)
        if hasattr(owner, "__dict__")
    } if unsupported != "embedding" else {}
    with pytest.raises(observer_module.ObserverError, match="Detail|TorchCodec"):
        observer_module.install(pipeline, setup.emit)
    for owner in (pipeline._audio, pipeline._embedding, getattr(pipeline._embedding, "model_", None)):
        if id(owner) in initial:
            assert vars(owner) == initial[id(owner)]
    assert setup.records == []


def test_double_install_is_rejected_without_disturbing_first_observer(setup):
    first = observer_module.install(setup.pipeline, setup.emit)
    try:
        with pytest.raises(observer_module.ObserverError, match="already observed"):
            observer_module.install(setup.pipeline, setup.emit)
        other_pipeline = SimpleNamespace(
            _audio=Audio(setup.clock, setup.waveform),
            _embedding=PyannoteAudioPretrainedSpeakerEmbedding(setup.clock),
        )
        with pytest.raises(observer_module.ObserverError, match="already observed"):
            observer_module.install(other_pipeline, setup.emit)
        assert "crop" not in vars(other_pipeline._audio)
    finally:
        first.close()


def test_installation_failure_rolls_back_all_prior_patches(setup, monkeypatch):
    initial_init = AudioDecoder.__init__
    initial_range = AudioDecoder.get_samples_played_in_range
    original = observer_module.Observer._patch

    def fail_final_patch(self, owner, name, wrapper):
        if name == "get_samples_played_in_range":
            raise RuntimeError("cannot install")
        original(self, owner, name, wrapper)

    monkeypatch.setattr(observer_module.Observer, "_patch", fail_final_patch)
    with pytest.raises(RuntimeError, match="cannot install"):
        observer_module.install(setup.pipeline, setup.emit)
    assert_restored(setup.pipeline, initial_init, initial_range)
    assert setup.records == []


def test_close_restores_before_emitting_even_when_emitter_raises(setup):
    initial_init = AudioDecoder.__init__
    initial_range = AudioDecoder.get_samples_played_in_range

    def emit(*args, **kwargs):
        assert_restored(setup.pipeline, initial_init, initial_range)
        raise OSError("broken stderr")

    observer = observer_module.install(setup.pipeline, emit)
    with pytest.raises(OSError, match="broken stderr"):
        observer.close()
    observer.close()


def test_close_rejects_mid_call_and_does_not_overwrite_foreign_changes(setup):
    model = setup.pipeline._embedding.model_
    original_forward = model.forward
    observer = None

    @wraps(original_forward)
    def forward(*args, **kwargs):
        with pytest.raises(observer_module.ObserverError, match="during an observed call"):
            observer.close()
        return original_forward(*args, **kwargs)

    model.forward = forward
    observer = observer_module.install(setup.pipeline, setup.emit)
    model(Tensor(2, 1, 16000))
    installed = model.forward
    foreign = lambda *args, **kwargs: object()
    model.forward = foreign
    with pytest.raises(observer_module.ObserverError, match="changed while profiling"):
        observer.close()
    assert model.forward is foreign
    assert setup.records == []
    with pytest.raises(observer_module.ObserverError, match="changed while profiling"):
        observer.close()
    model.forward = installed
    observer.close()
    assert model.forward is forward


def test_metadata_errors_are_reported_without_changing_results(setup):
    class UnknownShape:
        @property
        def shape(self):
            raise ValueError("metadata unavailable")

    observer = observer_module.install(setup.pipeline, setup.emit)
    model = setup.pipeline._embedding.model_
    assert model(UnknownShape()) is model.resnet.output[1]
    observer.close()
    records = by_phase(setup)
    for phase in ("detail_embedding_forward", "detail_embedding_fbank"):
        assert records[phase]["metadata_errors"] == 1
        assert records[phase]["metadata_error_types"] == ["ValueError"]
        assert records[phase]["completed_calls"] == 1


def test_shape_histograms_are_bounded_but_all_calls_and_items_are_counted(setup):
    observer = observer_module.install(setup.pipeline, setup.emit)
    model = setup.pipeline._embedding.model_
    for batch in range(1, 21):
        model.compute_fbank(Tensor(batch, 1, 16000))
    observer.close()
    record = by_phase(setup)["detail_embedding_fbank"]
    assert record["calls"] == record["shaped_input_calls"] == 20
    assert record["input_items"] == sum(range(1, 21))
    assert record["nested_calls"] == 0
    assert record["standalone_calls"] == 20
    assert len(record["shapes"]) == record["shape_limit"] == 16
    assert record["unlisted_shape_calls"] == 4
    assert record["elapsed_seconds"] == 80.0


def test_emit_must_be_callable(setup):
    with pytest.raises(observer_module.ObserverError, match="callable emit"):
        observer_module.install(setup.pipeline, None)
