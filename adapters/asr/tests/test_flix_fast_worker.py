"""Unit tests for flix_fast_worker.py.

Model loading/inference is injected via ``worker_factory`` (returning a
fake object with a ``transcribe()`` method), so these tests never need
``faster_whisper``/``ctranslate2`` installed -- same pattern as
``tests/test_main.py``'s ``FakeWhisperPipeline``.
"""
from __future__ import annotations

import io
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

import flix_fast_common as common
import flix_fast_worker as worker


def write_wav(path: Path, *, channels: int = 1, rate: int = 16000, sampwidth: int = 2, n_frames: int = 16000) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(rate)
        wav_file.writeframes(b"\x00" * n_frames * sampwidth * channels)


def make_ready_model_and_ct2_dir(tmp_path: Path, *, revision: str = "rev123", quantization: str = "int8") -> tuple[Path, Path]:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]})
    )
    (model_dir / "generation_config.json").write_text(json.dumps({"language": "de", "task": "transcribe"}))
    (model_dir / common.MANIFEST_FILENAME).write_text(json.dumps({"repo": "Flix-AI/flix-swissgerman-full", "revision": revision}))

    ct2_dir = tmp_path / "ct2"
    ct2_dir.mkdir()
    (ct2_dir / common.PROVENANCE_FILENAME).write_text(
        json.dumps(
            {
                "source_identity": revision,
                "quantization": quantization,
                "generation_defaults": {"language": "de", "task": "transcribe"},
            }
        )
    )
    return model_dir, ct2_dir


class FakeSegment(SimpleNamespace):
    pass


class FakeWorkerModel:
    def __init__(self, output_segments, *, language="de"):
        self._segments = output_segments
        self._language = language
        self.calls: list[dict] = []

    def transcribe(self, input_path, **kwargs):
        self.calls.append({"input_path": input_path, **kwargs})
        info = SimpleNamespace(language=self._language)
        return iter(self._segments), info


def make_factory(model: FakeWorkerModel):
    def factory(ct2_dir, threads, *, compute_type):
        factory.calls.append({"ct2_dir": ct2_dir, "threads": threads, "compute_type": compute_type})
        return model

    factory.calls = []
    return factory


# ---------------------------------------------------------------------------
# build_transcribe_fn: provenance / cache resolution
# ---------------------------------------------------------------------------

def test_build_transcribe_fn_errors_when_ct2_model_missing(tmp_path):
    model_dir, _ = make_ready_model_and_ct2_dir(tmp_path)
    with pytest.raises(common.AdapterError, match="No converted CTranslate2 model found"):
        worker.build_transcribe_fn(
            model_dir, 4, ct2_model=None, cache_root=str(tmp_path / "empty-cache"), quantization="int8",
            language_override=None, task_override=None, beam_size=1, best_of=1, vad_filter=False,
        )


def test_build_transcribe_fn_errors_on_provenance_identity_mismatch(tmp_path):
    model_dir, ct2_dir = make_ready_model_and_ct2_dir(tmp_path, revision="rev123")
    # Simulate a cache entry converted from a *different* model directory.
    (ct2_dir / common.PROVENANCE_FILENAME).write_text(
        json.dumps({"source_identity": "different-revision", "quantization": "int8"})
    )
    with pytest.raises(common.AdapterError, match="provenance mismatch"):
        worker.build_transcribe_fn(
            model_dir, 4, ct2_model=str(ct2_dir), cache_root=None, quantization="int8",
            language_override=None, task_override=None, beam_size=1, best_of=1, vad_filter=False,
        )


def test_build_transcribe_fn_errors_on_quantization_mismatch(tmp_path):
    model_dir, ct2_dir = make_ready_model_and_ct2_dir(tmp_path, quantization="int8")
    with pytest.raises(common.AdapterError, match="does not match the quantization"):
        worker.build_transcribe_fn(
            model_dir, 4, ct2_model=str(ct2_dir), cache_root=None, quantization="float32",
            language_override=None, task_override=None, beam_size=1, best_of=1, vad_filter=False,
        )


def test_build_transcribe_fn_loads_via_factory_with_defaults_from_generation_config(tmp_path):
    model_dir, ct2_dir = make_ready_model_and_ct2_dir(tmp_path)
    fake_model = FakeWorkerModel([FakeSegment(start=0.0, end=1.0, text=" hallo")])
    factory = make_factory(fake_model)

    transcribe_fn = worker.build_transcribe_fn(
        model_dir, 6, ct2_model=str(ct2_dir), cache_root=None, quantization="int8",
        language_override=None, task_override=None, beam_size=1, best_of=1, vad_filter=False,
        worker_factory=factory,
    )
    result = transcribe_fn("/abs/clip.wav", 1.0)

    assert factory.calls == [{"ct2_dir": str(ct2_dir), "threads": 6, "compute_type": "int8"}]
    assert fake_model.calls[0]["language"] == "de"
    assert fake_model.calls[0]["task"] == "transcribe"
    assert fake_model.calls[0]["beam_size"] == 1
    assert result["text"] == "hallo"
    assert result["language"] == "de"
    assert result["segments"] == [{"start": 0.0, "end": 1.0, "text": " hallo"}]


def test_build_transcribe_fn_honors_language_override(tmp_path):
    model_dir, ct2_dir = make_ready_model_and_ct2_dir(tmp_path)
    fake_model = FakeWorkerModel([FakeSegment(start=0.0, end=1.0, text="hi")], language="en")
    factory = make_factory(fake_model)

    transcribe_fn = worker.build_transcribe_fn(
        model_dir, 4, ct2_model=str(ct2_dir), cache_root=None, quantization="int8",
        language_override="en", task_override=None, beam_size=1, best_of=1, vad_filter=False,
        worker_factory=factory,
    )
    transcribe_fn("/abs/clip.wav", 1.0)
    assert fake_model.calls[0]["language"] == "en"


# ---------------------------------------------------------------------------
# transcribe(): segment/text assembly
# ---------------------------------------------------------------------------

def test_transcribe_joins_segment_text_and_uses_duration_for_missing_end():
    model = FakeWorkerModel([FakeSegment(start=0.0, end=None, text=" hallo welt")], language="de")
    result = worker.transcribe(model, "/abs/clip.wav", 2.5, language="de", task="transcribe", beam_size=1, best_of=1, vad_filter=False)
    assert result == {"text": "hallo welt", "language": "de", "segments": [{"start": 0.0, "end": 2.5, "text": " hallo welt"}]}

def test_fast_decode_disables_expensive_fallback_without_changing_default():
    model = FakeWorkerModel([FakeSegment(start=0.0, end=1.0, text=" hallo")])
    worker.transcribe(model, "/abs/clip.wav", 1.0, language="de", task="transcribe",
                      beam_size=1, best_of=1, vad_filter=False)
    assert "temperature" not in model.calls[-1]
    worker.transcribe(model, "/abs/clip.wav", 1.0, language="de", task="transcribe",
                      beam_size=1, best_of=1, vad_filter=False, fast_decode=True)
    assert model.calls[-1]["temperature"] == 0.0
    assert model.calls[-1]["compression_ratio_threshold"] is None
    assert model.calls[-1]["log_prob_threshold"] is None


def test_transcribe_wraps_engine_failures():
    class BrokenModel:
        def transcribe(self, *args, **kwargs):
            raise RuntimeError("boom")

    with pytest.raises(common.AdapterError, match="transcription failed"):
        worker.transcribe(BrokenModel(), "/abs/clip.wav", 1.0, language="de", task="transcribe", beam_size=1, best_of=1, vad_filter=False)


# ---------------------------------------------------------------------------
# serve(): wire protocol parity with main.py
# ---------------------------------------------------------------------------

def test_serve_writes_one_json_line_per_request(tmp_path):
    wav_path = tmp_path / "clip.wav"
    write_wav(wav_path, n_frames=16000)

    def transcribe_fn(input_path, duration):
        return {"text": "hallo", "language": "de", "segments": [{"start": 0.0, "end": duration, "text": "hallo"}]}

    in_stream = io.StringIO(json.dumps({"input": str(wav_path)}) + "\n")
    out_stream = io.StringIO()

    exit_code = worker.serve(transcribe_fn, in_stream=in_stream, out_stream=out_stream)

    assert exit_code == 0
    response = json.loads(out_stream.getvalue().strip())
    assert response == {"text": "hallo", "language": "de", "segments": [{"start": 0.0, "end": 1.0, "text": "hallo"}]}


def test_serve_reports_error_and_stops_on_bad_request():
    def transcribe_fn(input_path, duration):  # pragma: no cover - should not be called
        raise AssertionError("must not be called for a malformed request")

    in_stream = io.StringIO("not json\n")
    out_stream = io.StringIO()

    exit_code = worker.serve(transcribe_fn, in_stream=in_stream, out_stream=out_stream)

    assert exit_code == 1
    assert json.loads(out_stream.getvalue().strip())["error"]


# ---------------------------------------------------------------------------
# run(): argument parsing / single-clip mode end-to-end
# ---------------------------------------------------------------------------

def test_run_serve_requires_no_input_output(tmp_path, capsys):
    model_dir, _ = make_ready_model_and_ct2_dir(tmp_path)
    with pytest.raises(SystemExit):
        worker.run(["--model", str(model_dir), "--threads", "2", "--serve", "--input", "/abs/x.wav"])


def test_run_single_clip_end_to_end(tmp_path):
    model_dir, ct2_dir = make_ready_model_and_ct2_dir(tmp_path)
    wav_path = tmp_path / "clip.wav"
    write_wav(wav_path, n_frames=16000)
    output_path = tmp_path / "result.json"
    fake_model = FakeWorkerModel([FakeSegment(start=0.0, end=1.0, text=" hallo")])
    factory = make_factory(fake_model)

    exit_code = worker.run(
        [
            "--model", str(model_dir),
            "--ct2-model", str(ct2_dir),
            "--input", str(wav_path),
            "--output", str(output_path),
            "--threads", "4",
        ],
        worker_factory=factory,
    )

    assert exit_code == 0
    result = json.loads(output_path.read_text())
    assert result == {"text": "hallo", "language": "de", "segments": [{"start": 0.0, "end": 1.0, "text": " hallo"}]}
