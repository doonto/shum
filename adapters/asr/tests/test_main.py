"""Unit tests for the ASR adapter CLI.

These tests do not require torch/nemo/transformers to be installed: model
loading and inference are injected via the ``*_factory``/pipeline objects
passed into ``main.run()``, and WAV validation is exercised against small
real WAV files built with the standard-library ``wave`` module. A couple of
tests intentionally exercise the *real* ``default_*_factory`` functions to
confirm they fail with a clear ``AdapterError`` (not an unhandled
``ImportError``/traceback) when the corresponding heavy dependency is not
installed in the current environment.
"""
from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import pytest

import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_wav(path: Path, *, channels: int = 1, rate: int = 16000, sampwidth: int = 2, n_frames: int = 1600) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(rate)
        wav_file.writeframes(b"\x00" * n_frames * sampwidth * channels)


def make_nemo_checkpoint_dir(tmp_path: Path, name: str = "model", filename: str = "checkpoint.nemo") -> Path:
    """Build a nemo model *directory* containing a single '.nemo' file.

    The Rust model manager passes a per-model directory for every engine
    (consistent with whisper/qwen), so the adapter resolves the actual
    '.nemo' checkpoint file from inside that directory rather than
    expecting '--model' to point at the file directly.
    """
    model_dir = tmp_path / name
    model_dir.mkdir()
    checkpoint = model_dir / filename
    checkpoint.write_bytes(b"fake-nemo-archive")
    return model_dir


def make_transformers_checkpoint_dir(tmp_path: Path, name: str = "model") -> Path:
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}\n")
    return model_dir


class FakeHypothesis:
    def __init__(self, text: str, timestamp: dict | None = None):
        self.text = text
        self.timestamp = timestamp


class FakeNemoModel:
    """Stands in for a real ``nemo_asr.models.ASRModel`` instance."""

    def __init__(self, hypotheses, *, accepts_timestamps: bool = True):
        self._hypotheses = hypotheses
        self._accepts_timestamps = accepts_timestamps
        self.transcribe_calls: list[dict] = []

    def transcribe(self, paths, batch_size=1, timestamps=None, **kwargs):
        if timestamps is not None and not self._accepts_timestamps:
            raise TypeError("transcribe() got an unexpected keyword argument 'timestamps'")
        self.transcribe_calls.append({"paths": paths, "batch_size": batch_size, "timestamps": timestamps})
        return self._hypotheses


class FakeWhisperPipeline:
    def __init__(self, output):
        self._output = output
        self.calls: list[dict] = []

    def __call__(self, input_path, **kwargs):
        self.calls.append({"input_path": input_path, **kwargs})
        return self._output


class FakeBatchFeature(dict):
    def to(self, device, dtype):
        return self


class FakeQwenProcessor:
    def __init__(self, parsed):
        self._parsed = parsed
        self.decode_args = None

    def apply_transcription_request(self, audio):
        return FakeBatchFeature({"input_ids": FakeTensor([[1, 2, 3]])})

    def decode(self, generated_ids, return_format="raw"):
        self.decode_args = (generated_ids, return_format)
        return self._parsed


class FakeTensor:
    """Minimal stand-in for a torch tensor slice, only what the adapter needs."""

    def __init__(self, rows):
        self._rows = rows

    @property
    def shape(self):
        return (len(self._rows), len(self._rows[0]) if self._rows else 0)

    def __getitem__(self, item):
        # Only used as inputs["input_ids"].shape[1] in the adapter; not sliced further here.
        return self


class FakeQwenModel:
    def __init__(self, generated_ids):
        self.device = "cpu"
        self.dtype = "float32"
        self._generated_ids = generated_ids

    def generate(self, **kwargs):
        return self._generated_ids


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def test_parse_args_requires_all_flags():
    with pytest.raises(SystemExit):
        main.parse_args(["--engine", "nemo", "--model", "/x"])


def test_parse_args_rejects_unknown_engine():
    with pytest.raises(SystemExit):
        main.parse_args([
            "--engine", "bogus", "--model", "/m", "--input", "/i.wav",
            "--output", "/o.json", "--threads", "1",
        ])


def test_parse_args_ok():
    args = main.parse_args([
        "--engine", "whisper", "--model", "/m", "--input", "/i.wav",
        "--output", "/o.json", "--threads", "4",
    ])
    assert args.engine == "whisper"
    assert args.model == "/m"
    assert args.input == "/i.wav"
    assert args.output == "/o.json"
    assert args.threads == 4
    assert args.serve is False


def test_parse_args_serve_ok_without_input_output():
    args = main.parse_args(["--engine", "nemo", "--model", "/m", "--threads", "2", "--serve"])
    assert args.serve is True
    assert args.input is None
    assert args.output is None


def test_parse_args_serve_rejects_input():
    with pytest.raises(SystemExit):
        main.parse_args([
            "--engine", "nemo", "--model", "/m", "--threads", "2", "--serve",
            "--input", "/i.wav",
        ])


def test_parse_args_serve_rejects_output():
    with pytest.raises(SystemExit):
        main.parse_args([
            "--engine", "nemo", "--model", "/m", "--threads", "2", "--serve",
            "--output", "/o.json",
        ])


# ---------------------------------------------------------------------------
# validate_threads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("threads", [0, -1])
def test_validate_threads_rejects_non_positive(threads):
    with pytest.raises(main.AdapterError, match="positive"):
        main.validate_threads(threads)


def test_validate_threads_ok():
    assert main.validate_threads(4) == 4


# ---------------------------------------------------------------------------
# validate_model_path
# ---------------------------------------------------------------------------

def test_validate_model_path_rejects_relative_path():
    with pytest.raises(main.AdapterError, match="absolute"):
        main.validate_model_path("nemo", "relative/model.nemo")


def test_validate_model_path_nemo_missing_dir(tmp_path):
    missing = tmp_path / "missing-model"
    with pytest.raises(main.AdapterError, match="does not exist"):
        main.validate_model_path("nemo", str(missing))


def test_validate_model_path_nemo_dir_with_no_nemo_file(tmp_path):
    empty_dir = tmp_path / "model"
    empty_dir.mkdir()
    (empty_dir / "readme.txt").write_text("not a checkpoint\n")
    with pytest.raises(main.AdapterError, match="no '\\.nemo' checkpoint file"):
        main.validate_model_path("nemo", str(empty_dir))


def test_validate_model_path_nemo_dir_with_multiple_nemo_files(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "a.nemo").write_bytes(b"x")
    (model_dir / "b.nemo").write_bytes(b"y")
    with pytest.raises(main.AdapterError, match="multiple '\\.nemo' checkpoint files"):
        main.validate_model_path("nemo", str(model_dir))


def test_validate_model_path_nemo_ok(tmp_path):
    model_dir = make_nemo_checkpoint_dir(tmp_path)
    resolved = main.validate_model_path("nemo", str(model_dir))
    assert resolved == model_dir / "checkpoint.nemo"


@pytest.mark.parametrize("engine", ["whisper", "qwen"])
def test_validate_model_path_transformers_missing_dir(tmp_path, engine):
    missing = tmp_path / "missing-model"
    with pytest.raises(main.AdapterError, match="does not exist"):
        main.validate_model_path(engine, str(missing))


@pytest.mark.parametrize("engine", ["whisper", "qwen"])
def test_validate_model_path_transformers_missing_config(tmp_path, engine):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(main.AdapterError, match="config.json"):
        main.validate_model_path(engine, str(model_dir))


@pytest.mark.parametrize("engine", ["whisper", "qwen"])
def test_validate_model_path_transformers_ok(tmp_path, engine):
    model_dir = make_transformers_checkpoint_dir(tmp_path)
    assert main.validate_model_path(engine, str(model_dir)) == model_dir


# ---------------------------------------------------------------------------
# validate_input_wav
# ---------------------------------------------------------------------------

def test_validate_input_wav_rejects_relative_path():
    with pytest.raises(main.AdapterError, match="absolute"):
        main.validate_input_wav("relative/audio.wav")


def test_validate_input_wav_missing_file(tmp_path):
    missing = tmp_path / "missing.wav"
    with pytest.raises(main.AdapterError, match="does not exist"):
        main.validate_input_wav(str(missing))


def test_validate_input_wav_rejects_stereo(tmp_path):
    wav_path = tmp_path / "stereo.wav"
    write_wav(wav_path, channels=2)
    with pytest.raises(main.AdapterError, match="channel"):
        main.validate_input_wav(str(wav_path))


def test_validate_input_wav_rejects_wrong_rate(tmp_path):
    wav_path = tmp_path / "wrong_rate.wav"
    write_wav(wav_path, rate=44100)
    with pytest.raises(main.AdapterError, match="16000"):
        main.validate_input_wav(str(wav_path))


def test_validate_input_wav_rejects_non_wav(tmp_path):
    not_wav = tmp_path / "audio.wav"
    not_wav.write_bytes(b"not a real wav file")
    with pytest.raises(main.AdapterError, match="not a readable WAV"):
        main.validate_input_wav(str(not_wav))


def test_validate_input_wav_ok_returns_duration(tmp_path):
    wav_path = tmp_path / "ok.wav"
    write_wav(wav_path, rate=16000, n_frames=16000)  # exactly 1 second
    path, duration = main.validate_input_wav(str(wav_path))
    assert path == wav_path
    assert duration == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# validate_output_path
# ---------------------------------------------------------------------------

def test_validate_output_path_rejects_relative_path():
    with pytest.raises(main.AdapterError, match="absolute"):
        main.validate_output_path("relative/out.json")


def test_validate_output_path_rejects_missing_parent_dir(tmp_path):
    missing_parent = tmp_path / "no-such-dir" / "out.json"
    with pytest.raises(main.AdapterError, match="does not exist"):
        main.validate_output_path(str(missing_parent))


def test_validate_output_path_ok(tmp_path):
    out_path = tmp_path / "out.json"
    assert main.validate_output_path(str(out_path)) == out_path


# ---------------------------------------------------------------------------
# build_result / write_result
# ---------------------------------------------------------------------------

def test_build_result_normalizes_types():
    result = main.build_result("hello", "german", [{"start": 0, "end": 1, "text": "hi"}])
    assert result == {
        "text": "hello",
        "language": "german",
        "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
    }


def test_build_result_allows_null_language_and_empty_segments():
    result = main.build_result("", None, [])
    assert result == {"text": "", "language": None, "segments": []}


def test_build_result_rejects_end_before_start():
    with pytest.raises(main.AdapterError, match="before start"):
        main.build_result("x", None, [{"start": 2.0, "end": 1.0, "text": "x"}])


def test_build_result_rejects_non_string_language():
    with pytest.raises(main.AdapterError, match="language"):
        main.build_result("x", 123, [])


def test_write_result_is_atomic_and_readable(tmp_path):
    out_path = tmp_path / "out.json"
    result = {"text": "hi", "language": None, "segments": []}
    main.write_result(result, out_path)
    assert json.loads(out_path.read_text(encoding="utf-8")) == result
    # No leftover temp file.
    assert list(tmp_path.glob(".asr-*")) == []


def test_write_result_cleans_up_tmp_file_on_failure(tmp_path, monkeypatch):
    out_path = tmp_path / "out.json"

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(main.os, "replace", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        main.write_result({"text": "x", "language": None, "segments": []}, out_path)
    assert not out_path.exists()
    assert list(tmp_path.glob(".asr-*")) == []


# ---------------------------------------------------------------------------
# NeMo transcription logic
# ---------------------------------------------------------------------------

def test_nemo_transcribe_uses_segment_timestamps():
    hyp = FakeHypothesis(
        text="hello world",
        timestamp={"segment": [{"start": 0.1, "end": 0.5, "segment": "hello"}, {"start": 0.6, "end": 1.0, "segment": "world"}]},
    )
    model = FakeNemoModel([hyp])
    result = main.nemo_transcribe(model, "/audio.wav", duration=2.0)
    assert result["text"] == "hello world"
    assert result["language"] is None
    assert result["segments"] == [
        {"start": 0.1, "end": 0.5, "text": "hello"},
        {"start": 0.6, "end": 1.0, "text": "world"},
    ]
    assert model.transcribe_calls[0]["timestamps"] is True


def test_nemo_transcribe_falls_back_to_single_segment_without_timestamps():
    hyp = FakeHypothesis(text="hello world", timestamp=None)
    model = FakeNemoModel([hyp])
    result = main.nemo_transcribe(model, "/audio.wav", duration=3.5)
    assert result["segments"] == [{"start": 0.0, "end": 3.5, "text": "hello world"}]


def test_nemo_transcribe_falls_back_when_timestamps_kwarg_unsupported():
    hyp = FakeHypothesis(text="legacy model output", timestamp=None)
    model = FakeNemoModel([hyp], accepts_timestamps=False)
    result = main.nemo_transcribe(model, "/audio.wav", duration=1.25)
    assert result["text"] == "legacy model output"
    assert result["segments"] == [{"start": 0.0, "end": 1.25, "text": "legacy model output"}]


def test_nemo_transcribe_handles_tuple_hypothesis():
    hyp = FakeHypothesis(text="best hypothesis", timestamp=None)
    model = FakeNemoModel([(hyp, ["alt1", "alt2"])])
    result = main.nemo_transcribe(model, "/audio.wav", duration=1.0)
    assert result["text"] == "best hypothesis"


def test_nemo_transcribe_raises_on_empty_hypotheses():
    model = FakeNemoModel([])
    with pytest.raises(main.AdapterError, match="no hypotheses"):
        main.nemo_transcribe(model, "/audio.wav", duration=1.0)


def test_nemo_transcribe_raises_adapter_error_on_failure():
    class BoomModel:
        def transcribe(self, *args, **kwargs):
            raise RuntimeError("cuda out of memory")

    with pytest.raises(main.AdapterError, match="transcription failed"):
        main.nemo_transcribe(BoomModel(), "/audio.wav", duration=1.0)


# ---------------------------------------------------------------------------
# Whisper transcription logic
# ---------------------------------------------------------------------------

def test_whisper_transcribe_builds_segments_with_language():
    pipeline = FakeWhisperPipeline({
        "text": "hallo welt",
        "chunks": [
            {"text": "hallo", "timestamp": (0.0, 0.4), "language": "german"},
            {"text": " welt", "timestamp": (0.4, 0.9), "language": "german"},
        ],
    })
    result = main.whisper_transcribe(pipeline, "/audio.wav", duration=1.0)
    assert result["text"] == "hallo welt"
    assert result["language"] == "german"
    assert result["segments"] == [
        {"start": 0.0, "end": 0.4, "text": "hallo"},
        {"start": 0.4, "end": 0.9, "text": " welt"},
    ]
    assert pipeline.calls[0]["return_timestamps"] is True
    assert pipeline.calls[0]["return_language"] is True


def test_whisper_transcribe_fills_unresolved_final_timestamp_with_duration():
    pipeline = FakeWhisperPipeline({
        "text": "cut off",
        "chunks": [{"text": "cut off", "timestamp": (0.0, None), "language": None}],
    })
    result = main.whisper_transcribe(pipeline, "/audio.wav", duration=5.0)
    assert result["segments"] == [{"start": 0.0, "end": 5.0, "text": "cut off"}]
    assert result["language"] is None


def test_whisper_transcribe_handles_no_chunks():
    pipeline = FakeWhisperPipeline({"text": "", "chunks": []})
    result = main.whisper_transcribe(pipeline, "/audio.wav", duration=0.0)
    assert result == {"text": "", "language": None, "segments": []}


def test_whisper_transcribe_raises_adapter_error_on_failure():
    def boom(*args, **kwargs):
        raise RuntimeError("model forward failed")

    with pytest.raises(main.AdapterError, match="transcription failed"):
        main.whisper_transcribe(boom, "/audio.wav", duration=1.0)


# ---------------------------------------------------------------------------
# Qwen3-ASR transcription logic
# ---------------------------------------------------------------------------

def test_qwen_transcribe_reports_single_full_length_segment():
    model = FakeQwenModel(generated_ids=FakeTensor([[10, 11, 12]]))
    processor = FakeQwenProcessor(parsed=[{"language": "English", "transcription": "hello there"}])
    result = main.qwen_transcribe(model, processor, "/audio.wav", duration=4.2)
    assert result["text"] == "hello there"
    assert result["language"] == "English"
    assert result["segments"] == [{"start": 0.0, "end": 4.2, "text": "hello there"}]


def test_qwen_transcribe_handles_empty_transcription():
    model = FakeQwenModel(generated_ids=FakeTensor([[1]]))
    processor = FakeQwenProcessor(parsed=[{"language": None, "transcription": ""}])
    result = main.qwen_transcribe(model, processor, "/audio.wav", duration=1.0)
    assert result == {"text": "", "language": None, "segments": []}


def test_qwen_transcribe_raises_on_empty_parsed_output():
    model = FakeQwenModel(generated_ids=FakeTensor([[1]]))
    processor = FakeQwenProcessor(parsed=[])
    with pytest.raises(main.AdapterError, match="no output"):
        main.qwen_transcribe(model, processor, "/audio.wav", duration=1.0)


# ---------------------------------------------------------------------------
# End-to-end run() with injected factories
# ---------------------------------------------------------------------------

def test_run_end_to_end_nemo(tmp_path):
    model_dir = make_nemo_checkpoint_dir(tmp_path)
    wav_path = tmp_path / "audio.wav"
    write_wav(wav_path, rate=16000, n_frames=16000)
    out_path = tmp_path / "out.json"

    hyp = FakeHypothesis(text="hallo", timestamp=None)
    model = FakeNemoModel([hyp])

    exit_code = main.run(
        ["--engine", "nemo", "--model", str(model_dir), "--input", str(wav_path),
         "--output", str(out_path), "--threads", "2"],
        nemo_model_factory=lambda path, threads: model,
    )
    assert exit_code == 0
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == {"text": "hallo", "language": None, "segments": [{"start": 0.0, "end": 1.0, "text": "hallo"}]}


def test_run_end_to_end_whisper(tmp_path):
    model_dir = make_transformers_checkpoint_dir(tmp_path)
    wav_path = tmp_path / "audio.wav"
    write_wav(wav_path, rate=16000, n_frames=16000)
    out_path = tmp_path / "out.json"

    pipeline = FakeWhisperPipeline({
        "text": "hallo",
        "chunks": [{"text": "hallo", "timestamp": (0.0, 1.0), "language": "german"}],
    })

    exit_code = main.run(
        ["--engine", "whisper", "--model", str(model_dir), "--input", str(wav_path),
         "--output", str(out_path), "--threads", "2"],
        whisper_pipeline_factory=lambda path, threads: pipeline,
    )
    assert exit_code == 0
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == {"text": "hallo", "language": "german", "segments": [{"start": 0.0, "end": 1.0, "text": "hallo"}]}


def test_run_end_to_end_qwen(tmp_path):
    model_dir = make_transformers_checkpoint_dir(tmp_path)
    wav_path = tmp_path / "audio.wav"
    write_wav(wav_path, rate=16000, n_frames=16000)
    out_path = tmp_path / "out.json"

    model = FakeQwenModel(generated_ids=FakeTensor([[1, 2]]))
    processor = FakeQwenProcessor(parsed=[{"language": "English", "transcription": "hello"}])

    exit_code = main.run(
        ["--engine", "qwen", "--model", str(model_dir), "--input", str(wav_path),
         "--output", str(out_path), "--threads", "2"],
        qwen_model_factory=lambda path, threads: (model, processor),
    )
    assert exit_code == 0
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == {"text": "hello", "language": "English", "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}


def test_run_returns_nonzero_and_writes_no_output_on_validation_failure(tmp_path, capsys):
    wav_path = tmp_path / "audio.wav"
    write_wav(wav_path, channels=2)  # invalid: stereo
    out_path = tmp_path / "out.json"

    exit_code = main.run([
        "--engine", "whisper", "--model", str(tmp_path), "--input", str(wav_path),
        "--output", str(out_path), "--threads", "1",
    ])
    assert exit_code == 1
    assert not out_path.exists()
    captured = capsys.readouterr()
    assert "error:" in captured.err


def test_run_returns_nonzero_when_engine_transcription_fails(tmp_path, capsys):
    model_dir = make_nemo_checkpoint_dir(tmp_path)
    wav_path = tmp_path / "audio.wav"
    write_wav(wav_path, rate=16000, n_frames=16000)
    out_path = tmp_path / "out.json"

    class BoomModel:
        def transcribe(self, *args, **kwargs):
            raise RuntimeError("boom")

    exit_code = main.run(
        ["--engine", "nemo", "--model", str(model_dir), "--input", str(wav_path),
         "--output", str(out_path), "--threads", "1"],
        nemo_model_factory=lambda path, threads: BoomModel(),
    )
    assert exit_code == 1
    assert not out_path.exists()
    captured = capsys.readouterr()
    assert "error:" in captured.err


# ---------------------------------------------------------------------------
# Real default_*_factory functions: must fail with a clear AdapterError (not
# an unhandled ImportError) when the heavy dependency isn't installed.
# ---------------------------------------------------------------------------

def test_default_nemo_model_factory_missing_dependency_message():
    with pytest.raises(main.AdapterError, match="nemo_toolkit|'nemo' extra"):
        main.default_nemo_model_factory("/does/not/matter.nemo", 1)


def test_default_whisper_pipeline_factory_missing_dependency_message():
    with pytest.raises(main.AdapterError, match="'whisper' extra"):
        main.default_whisper_pipeline_factory("/does/not/matter", 1)


def test_default_qwen_model_factory_missing_dependency_message():
    with pytest.raises(main.AdapterError, match="'qwen' extra"):
        main.default_qwen_model_factory("/does/not/matter", 1)


# ---------------------------------------------------------------------------
# --serve mode: persistent worker, model loaded exactly once
# ---------------------------------------------------------------------------

def make_counting_factory(return_value):
    """Wrap a fixed return value in a factory that records call count/args."""
    calls: list[tuple] = []

    def factory(path, threads):
        calls.append((path, threads))
        return return_value

    factory.calls = calls
    return factory


def test_build_transcribe_fn_nemo_loads_model_exactly_once(tmp_path):
    hyp = FakeHypothesis(text="hallo welt", timestamp=None)
    model = FakeNemoModel([hyp])
    factory = make_counting_factory(model)

    transcribe_fn = main.build_transcribe_fn(
        "nemo", tmp_path / "model.nemo", 2, nemo_model_factory=factory,
    )
    assert len(factory.calls) == 1

    transcribe_fn("/a.wav", 1.0)
    transcribe_fn("/b.wav", 0.5)
    assert len(factory.calls) == 1  # still one load, for two transcribe calls


def test_serve_processes_two_requests_with_model_loaded_only_once(tmp_path):
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    write_wav(wav_a, rate=16000, n_frames=16000)  # 1.0s
    write_wav(wav_b, rate=16000, n_frames=8000)  # 0.5s

    hyp = FakeHypothesis(text="hallo welt", timestamp=None)
    model = FakeNemoModel([hyp])
    factory = make_counting_factory(model)

    transcribe_fn = main.build_transcribe_fn(
        "nemo", tmp_path / "model.nemo", 2, nemo_model_factory=factory,
    )
    assert len(factory.calls) == 1

    in_stream = io.StringIO(
        json.dumps({"input": str(wav_a)}) + "\n" + json.dumps({"input": str(wav_b)}) + "\n"
    )
    out_stream = io.StringIO()

    exit_code = main.serve(transcribe_fn, in_stream=in_stream, out_stream=out_stream)

    assert exit_code == 0
    assert len(factory.calls) == 1  # model loaded once, reused for both requests
    lines = [line for line in out_stream.getvalue().splitlines() if line]
    assert len(lines) == 2
    assert json.loads(lines[0]) == {
        "text": "hallo welt", "language": None,
        "segments": [{"start": 0.0, "end": 1.0, "text": "hallo welt"}],
    }
    assert json.loads(lines[1]) == {
        "text": "hallo welt", "language": None,
        "segments": [{"start": 0.0, "end": 0.5, "text": "hallo welt"}],
    }


def test_serve_skips_blank_lines_between_requests(tmp_path):
    wav_a = tmp_path / "a.wav"
    write_wav(wav_a, rate=16000, n_frames=16000)

    hyp = FakeHypothesis(text="hi", timestamp=None)
    model = FakeNemoModel([hyp])
    transcribe_fn = main.build_transcribe_fn(
        "nemo", tmp_path / "model.nemo", 1,
        nemo_model_factory=make_counting_factory(model),
    )

    in_stream = io.StringIO("\n   \n" + json.dumps({"input": str(wav_a)}) + "\n\n")
    out_stream = io.StringIO()

    exit_code = main.serve(transcribe_fn, in_stream=in_stream, out_stream=out_stream)

    assert exit_code == 0
    lines = [line for line in out_stream.getvalue().splitlines() if line]
    assert len(lines) == 1
    assert json.loads(lines[0])["text"] == "hi"


def test_serve_reports_structured_error_and_exits_nonzero_on_invalid_json(tmp_path):
    model = FakeNemoModel([FakeHypothesis(text="hi", timestamp=None)])
    transcribe_fn = main.build_transcribe_fn(
        "nemo", tmp_path / "model.nemo", 1,
        nemo_model_factory=make_counting_factory(model),
    )

    in_stream = io.StringIO("not json\n")
    out_stream = io.StringIO()

    exit_code = main.serve(transcribe_fn, in_stream=in_stream, out_stream=out_stream)

    assert exit_code == 1
    lines = [line for line in out_stream.getvalue().splitlines() if line]
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert "error" in parsed and isinstance(parsed["error"], str)


def test_serve_reports_structured_error_on_missing_input_key(tmp_path):
    model = FakeNemoModel([FakeHypothesis(text="hi", timestamp=None)])
    transcribe_fn = main.build_transcribe_fn(
        "nemo", tmp_path / "model.nemo", 1,
        nemo_model_factory=make_counting_factory(model),
    )

    in_stream = io.StringIO(json.dumps({"not_input": "/x.wav"}) + "\n")
    out_stream = io.StringIO()

    exit_code = main.serve(transcribe_fn, in_stream=in_stream, out_stream=out_stream)

    assert exit_code == 1
    parsed = json.loads(out_stream.getvalue().strip())
    assert "error" in parsed


def test_serve_reports_structured_error_when_transcription_fails_and_stops(tmp_path):
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    write_wav(wav_a, rate=16000, n_frames=16000)
    write_wav(wav_b, rate=16000, n_frames=16000)

    class BoomModel:
        def transcribe(self, *args, **kwargs):
            raise RuntimeError("boom")

    factory = make_counting_factory(BoomModel())
    transcribe_fn = main.build_transcribe_fn(
        "nemo", tmp_path / "model.nemo", 1, nemo_model_factory=factory,
    )

    in_stream = io.StringIO(
        json.dumps({"input": str(wav_a)}) + "\n" + json.dumps({"input": str(wav_b)}) + "\n"
    )
    out_stream = io.StringIO()

    exit_code = main.serve(transcribe_fn, in_stream=in_stream, out_stream=out_stream)

    assert exit_code == 1
    assert len(factory.calls) == 1  # model still loaded only once
    lines = [line for line in out_stream.getvalue().splitlines() if line]
    assert len(lines) == 1  # stopped after the first failing request
    parsed = json.loads(lines[0])
    assert "error" in parsed and "boom" in parsed["error"]


def test_serve_reports_structured_error_for_nonexistent_input_file(tmp_path):
    model = FakeNemoModel([FakeHypothesis(text="hi", timestamp=None)])
    transcribe_fn = main.build_transcribe_fn(
        "nemo", tmp_path / "model.nemo", 1,
        nemo_model_factory=make_counting_factory(model),
    )

    missing = tmp_path / "missing.wav"
    in_stream = io.StringIO(json.dumps({"input": str(missing)}) + "\n")
    out_stream = io.StringIO()

    exit_code = main.serve(transcribe_fn, in_stream=in_stream, out_stream=out_stream)

    assert exit_code == 1
    parsed = json.loads(out_stream.getvalue().strip())
    assert "does not exist" in parsed["error"]


def test_run_serve_end_to_end_loads_model_once_for_two_requests(tmp_path):
    model_dir = make_nemo_checkpoint_dir(tmp_path)
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    write_wav(wav_a, rate=16000, n_frames=16000)
    write_wav(wav_b, rate=16000, n_frames=16000)

    hyp = FakeHypothesis(text="serve works", timestamp=None)
    model = FakeNemoModel([hyp])
    factory = make_counting_factory(model)

    in_stream = io.StringIO(
        json.dumps({"input": str(wav_a)}) + "\n" + json.dumps({"input": str(wav_b)}) + "\n"
    )
    out_stream = io.StringIO()

    exit_code = main.run(
        ["--engine", "nemo", "--model", str(model_dir), "--threads", "2", "--serve"],
        nemo_model_factory=factory,
        in_stream=in_stream,
        out_stream=out_stream,
    )

    assert exit_code == 0
    assert len(factory.calls) == 1
    lines = [line for line in out_stream.getvalue().splitlines() if line]
    assert len(lines) == 2
    for line in lines:
        assert json.loads(line)["text"] == "serve works"


def test_run_serve_writes_json_only_to_stdout(tmp_path):
    """The --serve loop must never mix non-JSON text (like the single-clip
    'wrote <path>' message) into stdout: every line must parse as JSON."""
    model_dir = make_nemo_checkpoint_dir(tmp_path)
    wav_a = tmp_path / "a.wav"
    write_wav(wav_a, rate=16000, n_frames=16000)

    hyp = FakeHypothesis(text="clean stdout", timestamp=None)
    model = FakeNemoModel([hyp])

    in_stream = io.StringIO(json.dumps({"input": str(wav_a)}) + "\n")
    out_stream = io.StringIO()

    exit_code = main.run(
        ["--engine", "nemo", "--model", str(model_dir), "--threads", "1", "--serve"],
        nemo_model_factory=make_counting_factory(model),
        in_stream=in_stream,
        out_stream=out_stream,
    )

    assert exit_code == 0
    for line in out_stream.getvalue().splitlines():
        if line.strip():
            json.loads(line)  # raises if any non-JSON text leaked onto stdout


def test_run_serve_configures_offline_and_thread_env_exactly_once(tmp_path, monkeypatch):
    model_dir = make_nemo_checkpoint_dir(tmp_path)
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    write_wav(wav_a, rate=16000, n_frames=16000)
    write_wav(wav_b, rate=16000, n_frames=16000)

    offline_calls = []
    thread_calls = []
    monkeypatch.setattr(main, "configure_offline_environment", lambda: offline_calls.append(1))
    monkeypatch.setattr(main, "configure_thread_environment", lambda threads: thread_calls.append(threads))

    hyp = FakeHypothesis(text="env once", timestamp=None)
    model = FakeNemoModel([hyp])

    in_stream = io.StringIO(
        json.dumps({"input": str(wav_a)}) + "\n" + json.dumps({"input": str(wav_b)}) + "\n"
    )
    out_stream = io.StringIO()

    exit_code = main.run(
        ["--engine", "nemo", "--model", str(model_dir), "--threads", "3", "--serve"],
        nemo_model_factory=make_counting_factory(model),
        in_stream=in_stream,
        out_stream=out_stream,
    )

    assert exit_code == 0
    assert offline_calls == [1]  # exactly once, not once per request
    assert thread_calls == [3]  # exactly once, with the CLI --threads value


def test_run_serve_rejects_missing_model_directory_before_reading_stdin(tmp_path):
    in_stream = io.StringIO(json.dumps({"input": "/whatever.wav"}) + "\n")
    out_stream = io.StringIO()

    exit_code = main.run(
        ["--engine", "nemo", "--model", str(tmp_path / "does-not-exist"), "--threads", "1", "--serve"],
        in_stream=in_stream,
        out_stream=out_stream,
    )

    assert exit_code == 1
    assert out_stream.getvalue() == ""  # nothing written; failed before entering the serve loop
