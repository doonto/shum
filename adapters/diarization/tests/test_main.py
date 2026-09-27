"""Lightweight unit tests for the diarization adapter CLI.

These tests do not require torch/pyannote.audio to be installed: pipeline
loading is injected via ``pipeline_factory``, and WAV validation is
exercised against small real WAV files built with the standard-library
``wave`` module.
"""
from __future__ import annotations

import builtins
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_wav(path: Path, *, channels: int = 1, rate: int = 16000, sampwidth: int = 2, n_frames: int = 160) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(rate)
        wav_file.writeframes(b"\x00" * n_frames * sampwidth * channels)


class FakeSegment:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class FakeAnnotation:
    def __init__(self, tracks):
        self._tracks = tracks

    def itertracks(self, yield_label: bool = False):
        assert yield_label is True
        yield from self._tracks


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def test_parse_args_requires_all_four_flags():
    with pytest.raises(SystemExit):
        main.parse_args(["--model", "/x"])


def test_parse_args_ok():
    args = main.parse_args([
        "--model", "/m", "--input", "/i.wav", "--output", "/o.json", "--threads", "4",
    ])
    assert args.model == "/m"
    assert args.input == "/i.wav"
    assert args.output == "/o.json"
    assert args.threads == 4


# ---------------------------------------------------------------------------
# validate_model_dir
# ---------------------------------------------------------------------------

def test_validate_model_dir_rejects_relative_path():
    with pytest.raises(main.AdapterError, match="absolute"):
        main.validate_model_dir("relative/dir")


def test_validate_model_dir_missing(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(main.AdapterError, match="does not exist"):
        main.validate_model_dir(str(missing))


def test_validate_model_dir_missing_config(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(main.AdapterError, match="config.yaml"):
        main.validate_model_dir(str(model_dir))


def test_validate_model_dir_ok(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("pipeline: {}\n")
    result = main.validate_model_dir(str(model_dir))
    assert result == model_dir


# ---------------------------------------------------------------------------
# validate_input_wav
# ---------------------------------------------------------------------------

def test_validate_input_wav_rejects_relative_path():
    with pytest.raises(main.AdapterError, match="absolute"):
        main.validate_input_wav("relative.wav")


def test_validate_input_wav_missing(tmp_path):
    with pytest.raises(main.AdapterError, match="does not exist"):
        main.validate_input_wav(str(tmp_path / "missing.wav"))


def test_validate_input_wav_not_a_wav_file(tmp_path):
    bogus = tmp_path / "bogus.wav"
    bogus.write_text("not actually a wav file")
    with pytest.raises(main.AdapterError, match="not a readable WAV file"):
        main.validate_input_wav(str(bogus))


def test_validate_input_wav_rejects_stereo(tmp_path):
    wav_path = tmp_path / "stereo.wav"
    write_wav(wav_path, channels=2, rate=16000)
    with pytest.raises(main.AdapterError, match="channel"):
        main.validate_input_wav(str(wav_path))


def test_validate_input_wav_rejects_wrong_rate(tmp_path):
    wav_path = tmp_path / "44100.wav"
    write_wav(wav_path, channels=1, rate=44100)
    with pytest.raises(main.AdapterError, match="44100"):
        main.validate_input_wav(str(wav_path))


def test_validate_input_wav_accepts_16k_mono(tmp_path):
    wav_path = tmp_path / "ok.wav"
    write_wav(wav_path, channels=1, rate=16000)
    result = main.validate_input_wav(str(wav_path))
    assert result == wav_path


# ---------------------------------------------------------------------------
# validate_output_path / validate_threads
# ---------------------------------------------------------------------------

def test_validate_output_path_rejects_relative_path():
    with pytest.raises(main.AdapterError, match="absolute"):
        main.validate_output_path("relative/output.json")


def test_validate_output_path_rejects_missing_parent(tmp_path):
    with pytest.raises(main.AdapterError, match="does not exist"):
        main.validate_output_path(str(tmp_path / "missing-dir" / "out.json"))


def test_validate_output_path_ok(tmp_path):
    out = tmp_path / "out.json"
    assert main.validate_output_path(str(out)) == out


@pytest.mark.parametrize("threads", [0, -1])
def test_validate_threads_rejects_non_positive(threads):
    with pytest.raises(main.AdapterError, match="positive integer"):
        main.validate_threads(threads)


def test_validate_threads_ok():
    assert main.validate_threads(4) == 4


# ---------------------------------------------------------------------------
# Offline / thread environment configuration
# ---------------------------------------------------------------------------

def test_configure_offline_environment_sets_expected_vars(monkeypatch):
    for var in main.OFFLINE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    main.configure_offline_environment()
    for var, value in main.OFFLINE_ENV_VARS.items():
        assert __import__("os").environ[var] == value


def test_configure_thread_environment_sets_expected_vars(monkeypatch):
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(var, raising=False)
    main.configure_thread_environment(7)
    import os
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert os.environ[var] == "7"


# ---------------------------------------------------------------------------
# default_pipeline_factory (import-injection only; no real torch/pyannote needed)
# ---------------------------------------------------------------------------

def test_default_pipeline_factory_reports_missing_dependency(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("torch", "pyannote.audio"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(main.AdapterError, match="Missing dependency"):
        main.default_pipeline_factory("/some/model/dir", 1)


# ---------------------------------------------------------------------------
# annotation_to_turns / diarization_output_to_result
# ---------------------------------------------------------------------------

def test_annotation_to_turns_sorts_and_rounds():
    annotation = FakeAnnotation([
        (FakeSegment(2.0001, 3.0), None, "SPEAKER_01"),
        (FakeSegment(0.12345, 1.0), None, "SPEAKER_00"),
    ])
    turns = main.annotation_to_turns(annotation)
    assert turns == [
        {"start": 0.123, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_01"},
    ]


def test_diarization_output_to_result_requires_speaker_diarization():
    output = SimpleNamespace()
    with pytest.raises(main.AdapterError, match="speaker_diarization"):
        main.diarization_output_to_result(output)


def test_diarization_output_to_result_requires_exclusive_speaker_diarization():
    output = SimpleNamespace(speaker_diarization=FakeAnnotation([]))
    with pytest.raises(main.AdapterError, match="exclusive_speaker_diarization"):
        main.diarization_output_to_result(output)


def test_diarization_output_to_result_ok():
    output = SimpleNamespace(
        speaker_diarization=FakeAnnotation([(FakeSegment(0.0, 1.0), None, "A")]),
        exclusive_speaker_diarization=FakeAnnotation([(FakeSegment(0.0, 0.5), None, "A")]),
    )
    result = main.diarization_output_to_result(output)
    assert result == {
        "turns": [{"start": 0.0, "end": 1.0, "speaker": "A"}],
        "exclusive_turns": [{"start": 0.0, "end": 0.5, "speaker": "A"}],
    }


# ---------------------------------------------------------------------------
# write_result
# ---------------------------------------------------------------------------

def test_write_result_writes_json(tmp_path):
    out = tmp_path / "out.json"
    main.write_result({"turns": [], "exclusive_turns": []}, out)
    assert json.loads(out.read_text()) == {"turns": [], "exclusive_turns": []}


def test_write_result_leaves_no_partial_file_on_failure(tmp_path):
    out = tmp_path / "out.json"

    class Unserializable:
        pass

    with pytest.raises(TypeError):
        main.write_result({"bad": Unserializable()}, out)
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# run() end-to-end (fake pipeline_factory, no real torch/pyannote needed)
# ---------------------------------------------------------------------------

def _make_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("pipeline: {}\n")
    return model_dir


def test_run_success_writes_expected_json(tmp_path):
    model_dir = _make_model_dir(tmp_path)
    wav_path = tmp_path / "audio.wav"
    write_wav(wav_path, channels=1, rate=16000)
    output_path = tmp_path / "result.json"

    fake_output = SimpleNamespace(
        speaker_diarization=FakeAnnotation([(FakeSegment(0.0, 2.0), None, "SPEAKER_00")]),
        exclusive_speaker_diarization=FakeAnnotation([(FakeSegment(0.0, 1.5), None, "SPEAKER_00")]),
    )

    def fake_factory(model_dir_arg, threads_arg):
        assert model_dir_arg == str(model_dir)
        assert threads_arg == 2
        return lambda input_path: fake_output

    exit_code = main.run(
        [
            "--model", str(model_dir),
            "--input", str(wav_path),
            "--output", str(output_path),
            "--threads", "2",
        ],
        pipeline_factory=fake_factory,
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text()) == {
        "turns": [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}],
        "exclusive_turns": [{"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"}],
    }


def test_run_reports_explicit_error_for_missing_model(tmp_path, capsys):
    wav_path = tmp_path / "audio.wav"
    write_wav(wav_path, channels=1, rate=16000)
    output_path = tmp_path / "result.json"

    exit_code = main.run([
        "--model", str(tmp_path / "missing-model"),
        "--input", str(wav_path),
        "--output", str(output_path),
        "--threads", "1",
    ])

    assert exit_code == 1
    assert not output_path.exists()
    captured = capsys.readouterr()
    assert "error:" in captured.err


def test_run_wraps_pipeline_failure_without_writing_output(tmp_path, capsys):
    model_dir = _make_model_dir(tmp_path)
    wav_path = tmp_path / "audio.wav"
    write_wav(wav_path, channels=1, rate=16000)
    output_path = tmp_path / "result.json"

    def failing_factory(model_dir_arg, threads_arg):
        def failing_pipeline(input_path):
            raise RuntimeError("boom")
        return failing_pipeline

    exit_code = main.run(
        [
            "--model", str(model_dir),
            "--input", str(wav_path),
            "--output", str(output_path),
            "--threads", "1",
        ],
        pipeline_factory=failing_factory,
    )

    assert exit_code == 1
    assert not output_path.exists()
    captured = capsys.readouterr()
    assert "Diarization failed" in captured.err
