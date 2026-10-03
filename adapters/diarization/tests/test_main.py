"""Lightweight unit tests for the diarization adapter CLI.

These tests do not require torch/pyannote.audio to be installed: pipeline
loading is injected via ``pipeline_factory``, and WAV validation is
exercised against small real WAV files built with the standard-library
``wave`` module.
"""
from __future__ import annotations

import builtins
import json
import sys
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
    assert args.segmentation_batch_size == 1
    assert args.embedding_batch_size == 1
    assert not args.profile
    assert not args.preload_audio


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
        return lambda input_path, *, hook: fake_output

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


def test_run_configures_batches_and_profiles_stages(tmp_path, capsys):
    model_dir = _make_model_dir(tmp_path)
    wav_path = tmp_path / "audio.wav"
    write_wav(wav_path, channels=1, rate=16000)
    output_path = tmp_path / "result.json"
    fake_output = SimpleNamespace(
        speaker_diarization=FakeAnnotation([]),
        exclusive_speaker_diarization=FakeAnnotation([]),
    )

    class FakePipeline:
        segmentation_batch_size = 1
        embedding_batch_size = 1

        def __call__(self, input_path, *, hook):
            assert input_path == str(wav_path)
            assert self.segmentation_batch_size == 16
            assert self.embedding_batch_size == 32
            hook("segmentation", None, total=10, completed=1)
            hook("segmentation", None, total=10, completed=1)
            hook("segmentation", None, total=10, completed=2)
            hook("segmentation", None)
            hook("embeddings", None, total=4, completed=0)
            hook("embeddings", None, total=4, completed=1)
            hook("embeddings", None, total=4, completed=4)
            hook("embeddings", None)
            return fake_output

    assert main.run(
        [
            "--model", str(model_dir), "--input", str(wav_path),
            "--output", str(output_path), "--threads", "4",
            "--segmentation-batch-size", "16", "--embedding-batch-size", "32",
            "--profile",
        ],
        pipeline_factory=lambda _model, _threads: FakePipeline(),
    ) == 0
    assert json.loads(output_path.read_text()) == {"turns": [], "exclusive_turns": []}
    captured = capsys.readouterr()
    assert captured.err.count("diarization segmentation:") == 1
    assert captured.err.count("diarization segmentation progress 10%:") == 1
    assert captured.err.count("diarization segmentation progress 20%:") == 1
    assert "diarization embeddings:" in captured.err
    assert "diarization total:" in captured.err
    measurements = [
        json.loads(line)
        for line in captured.err.splitlines()
        if line.startswith('{"event":"shum.progress"')
    ]
    assert measurements == [
        {"event": "shum.progress", "version": 1, "phase": "segmentation", "completed": 1, "total": 10},
        {"event": "shum.progress", "version": 1, "phase": "segmentation", "completed": 2, "total": 10},
        {"event": "shum.progress", "version": 1, "phase": "embeddings", "completed": 0, "total": 4},
        {"event": "shum.progress", "version": 1, "phase": "embeddings", "completed": 1, "total": 4},
        {"event": "shum.progress", "version": 1, "phase": "embeddings", "completed": 4, "total": 4},
    ]


def test_unprofiled_embedded_progress_uses_real_hook_counts(capsys):
    class FakePipeline:
        def __call__(self, _input, *, hook):
            hook("embeddings", None, total=7, completed=0)
            hook("embeddings", None, total=7, completed=2)
            hook("embeddings", None, total=7, completed=2)
            hook("embeddings", None, total=7, completed=7)
            hook("embeddings", None)
            return "done"

    assert main.run_profiled_pipeline(FakePipeline(), "audio", profile=False) == "done"
    assert capsys.readouterr().err.splitlines() == [
        '{"event":"shum.progress","version":1,"phase":"embeddings","completed":0,"total":7}',
        '{"event":"shum.progress","version":1,"phase":"embeddings","completed":2,"total":7}',
        '{"event":"shum.progress","version":1,"phase":"embeddings","completed":7,"total":7}',
    ]


def test_run_preloads_once_and_keeps_global_speaker_timeline(tmp_path, capsys, monkeypatch):
    model_dir = _make_model_dir(tmp_path)
    wav_path = tmp_path / "audio.wav"
    write_wav(wav_path, channels=1, rate=16000)
    output_path = tmp_path / "result.json"
    waveform = object()
    calls = []

    def fake_loader(path):
        calls.append(path)
        return {"waveform": waveform, "sample_rate": 16000}

    monkeypatch.setattr(main, "load_waveform", fake_loader)
    fake_output = SimpleNamespace(
        speaker_diarization=FakeAnnotation([(FakeSegment(0.0, 2.0), None, "SPEAKER_00")]),
        exclusive_speaker_diarization=FakeAnnotation(
            [(FakeSegment(0.0, 2.0), None, "SPEAKER_00")]
        ),
    )

    def fake_pipeline(audio_input, *, hook):
        assert audio_input == {"waveform": waveform, "sample_rate": 16000}
        hook("segmentation", None)
        return fake_output

    assert main.run(
        [
            "--model", str(model_dir), "--input", str(wav_path),
            "--output", str(output_path), "--threads", "4",
            "--preload-audio", "--profile",
        ],
        pipeline_factory=lambda *_: fake_pipeline,
    ) == 0
    assert calls == [str(wav_path)]
    assert json.loads(output_path.read_text())["exclusive_turns"][0]["speaker"] == "SPEAKER_00"
    assert "diarization waveform load:" in capsys.readouterr().err


def test_preloaded_waveform_must_still_be_mono_and_16khz(monkeypatch):
    audio = SimpleNamespace(shape=(1, 160))
    monkeypatch.setitem(
        sys.modules,
        "torchaudio",
        SimpleNamespace(load=lambda _path: (audio, 8000)),
    )
    with pytest.raises(main.AdapterError, match="mono at 16 kHz"):
        main.load_waveform("/tmp/audio.wav")


def test_invalid_batch_size_fails_before_loading_model(tmp_path, capsys):
    model_dir = _make_model_dir(tmp_path)
    wav_path = tmp_path / "audio.wav"
    write_wav(wav_path, channels=1, rate=16000)
    output_path = tmp_path / "result.json"
    assert main.run(
        [
            "--model", str(model_dir), "--input", str(wav_path),
            "--output", str(output_path), "--threads", "4",
            "--embedding-batch-size", "0",
        ],
        pipeline_factory=lambda *_: pytest.fail("must not load the model"),
    ) == 1
    assert not output_path.exists()
    assert "--embedding-batch-size must be positive" in capsys.readouterr().err


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


def profile_records(stderr):
    return [
        json.loads(line) for line in stderr.splitlines()
        if line.startswith('{"event":"shum.profile"')
    ]


def test_profile_measures_hook_intervals_and_tail(monkeypatch, capsys):
    clock = [10.0]
    monkeypatch.setattr(main.time, "perf_counter", lambda: clock[0])

    def pipeline(audio, *, hook, num_speakers):
        assert audio == "unchanged"
        assert num_speakers == 3
        clock[0] = 12.0
        hook("segmentation", None, completed=0, total=2)
        clock[0] = 15.0
        hook("segmentation", None, completed=2, total=2)
        clock[0] = 16.0
        hook("segmentation", None)
        clock[0] = 17.0
        hook("speaker_count", None)
        clock[0] = 18.0
        hook("embeddings", None, completed=0, total=4)
        hook("embeddings", None, completed=0, total=4)
        clock[0] = 22.0
        hook("embeddings", None, completed=4, total=4)
        clock[0] = 23.0
        hook("embeddings", None)
        clock[0] = 28.0
        return "unchanged result"

    assert main.run_profiled_pipeline(
        pipeline, "unchanged", num_speakers=3,
    ) == "unchanged result"
    records = profile_records(capsys.readouterr().err)
    assert all(r["event"] == "shum.profile" and r["version"] == 1 for r in records)
    assert [r for r in records if r["status"] == "completed"] == [
        {"event": "shum.profile", "version": 1, "phase": "segmentation",
         "status": "completed", "elapsed_seconds": 6.0, "since": "pipeline_start",
         "completed": 2, "total": 2},
        {"event": "shum.profile", "version": 1, "phase": "embeddings",
         "status": "completed", "elapsed_seconds": 6.0, "since": "speaker_count",
         "completed": 4, "total": 4},
        {"event": "shum.profile", "version": 1, "phase": "clustering_assembly",
         "status": "completed", "elapsed_seconds": 5.0, "since": "embeddings"},
        {"event": "shum.profile", "version": 1, "phase": "pipeline_total",
         "status": "completed", "elapsed_seconds": 18.0},
    ]
    progress = [r for r in records if r["status"] == "progress"]
    assert [(r["phase"], r["completed"], r["total"]) for r in progress] == [
        ("segmentation", 0, 2), ("segmentation", 2, 2),
        ("embeddings", 0, 4), ("embeddings", 4, 4),
    ]
    assert all("elapsed_seconds" not in r for r in progress)


@pytest.mark.parametrize("after_embeddings", [False, True])
def test_profile_failure_retains_partial_counts(monkeypatch, capsys, after_embeddings):
    clock = [0.0]
    monkeypatch.setattr(main.time, "perf_counter", lambda: clock[0])

    def pipeline(_audio, *, hook):
        clock[0] = 2.0
        hook("segmentation", None)
        hook("embeddings", None, completed=1, total=3)
        if after_embeddings:
            clock[0] = 5.0
            hook("embeddings", None)
        clock[0] = 8.0
        raise RuntimeError("boom")

    with pytest.raises(main.AdapterError, match="boom"):
        main.run_profiled_pipeline(pipeline, "audio")
    records = profile_records(capsys.readouterr().err)
    failed = [r for r in records if r["status"] == "failed"]
    assert failed[-1]["phase"] == "pipeline_total"
    assert failed[-1]["elapsed_seconds"] == 8.0
    assert failed[0]["phase"] == ("clustering_assembly" if after_embeddings else "embeddings")
    assert failed[0]["elapsed_seconds"] == (3.0 if after_embeddings else 6.0)
    if not after_embeddings:
        assert failed[0]["completed"] == 1
        assert failed[0]["total"] == 3
    assert not any(r["phase"] == failed[0]["phase"] and r["status"] == "completed"
                   for r in records)


def test_profile_does_not_invent_missing_hook_timings(capsys):
    assert main.run_profiled_pipeline(lambda _audio, **_: "ok", "audio") == "ok"
    records = profile_records(capsys.readouterr().err)
    assert [r["phase"] for r in records] == ["pipeline_total"]


def test_profile_reports_effective_defaults_and_measured_lifecycle(tmp_path, monkeypatch, capsys):
    model = _make_model_dir(tmp_path)
    wav = tmp_path / "input.wav"
    write_wav(wav)
    out = tmp_path / "out.json"
    clock = [0.0]
    monkeypatch.setattr(main.time, "perf_counter", lambda: clock[0])
    waveform = {"waveform": object(), "sample_rate": 16000}

    class Pipeline:
        segmentation_batch_size = 8
        embedding_batch_size = 32

        def __call__(self, audio, *, hook):
            assert audio is waveform
            clock[0] += 5.0
            return SimpleNamespace(
                speaker_diarization=FakeAnnotation([]),
                exclusive_speaker_diarization=FakeAnnotation([]),
            )

    def factory(_model, threads):
        assert threads == 4
        clock[0] += 2.0
        return Pipeline()

    def preload(_path):
        clock[0] += 3.0
        return waveform

    original_write = main.write_result

    def write(result, path):
        clock[0] += 7.0
        original_write(result, path)

    monkeypatch.setattr(main, "load_waveform", preload)
    monkeypatch.setattr(main, "write_result", write)
    assert main.run([
        "--model", str(model), "--input", str(wav), "--output", str(out),
        "--threads", "4", "--profile", "--preload-audio",
    ], pipeline_factory=factory) == 0
    captured = capsys.readouterr()
    assert captured.out == f"wrote {out}\n"
    assert json.loads(out.read_text()) == {"turns": [], "exclusive_turns": []}
    records = profile_records(captured.err)
    configured = next(r for r in records if r["status"] == "configured")
    assert configured["parameters"] == {
        "threads": 4, "requested_segmentation_batch_size": 1,
        "requested_embedding_batch_size": 1, "preload_audio": True,
        "segmentation_batch_size": 8, "embedding_batch_size": 32,
    }
    assert "elapsed_seconds" not in configured
    durations = {r["phase"]: r["elapsed_seconds"] for r in records
                 if r["status"] == "completed"}
    assert durations == {
        "model_loading": 2.0, "batch_configuration": 0.0, "waveform_loading": 3.0,
        "pipeline_total": 5.0, "result_conversion": 0.0, "result_writing": 7.0,
    }


@pytest.mark.parametrize("phase", [
    "model_loading", "batch_configuration", "waveform_loading",
    "result_conversion", "result_writing",
])
def test_profile_lifecycle_failure_is_not_success(tmp_path, monkeypatch, capsys, phase):
    model = _make_model_dir(tmp_path)
    wav = tmp_path / "input.wav"
    write_wav(wav)
    out = tmp_path / "out.json"

    def fail(*_args):
        raise main.AdapterError("injected failure")

    output = SimpleNamespace(
        speaker_diarization=FakeAnnotation([]),
        exclusive_speaker_diarization=FakeAnnotation([]),
    )
    factory = lambda *_: lambda _audio, **_: output
    if phase == "model_loading":
        factory = fail
    else:
        target = {
            "batch_configuration": "configure_batch_sizes",
            "waveform_loading": "load_waveform",
            "result_conversion": "diarization_output_to_result",
            "result_writing": "write_result",
        }[phase]
        monkeypatch.setattr(main, target, fail)
    args = [
        "--model", str(model), "--input", str(wav), "--output", str(out),
        "--threads", "1", "--profile",
    ]
    if phase == "waveform_loading":
        args.append("--preload-audio")
    assert main.run(args, pipeline_factory=factory) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error: injected failure" in captured.err
    assert not out.exists()
    records = profile_records(captured.err)
    failed = [r for r in records if r["phase"] == phase]
    assert len(failed) == 1
    assert failed[0]["status"] == "failed"
    assert failed[0]["elapsed_seconds"] >= 0


def test_profile_reraises_unexpected_write_error(tmp_path, capsys):
    with pytest.raises(TypeError):
        with main.measured_phase("result_writing", True):
            main.write_result({"bad": object()}, tmp_path / "out.json")
    assert list(tmp_path.iterdir()) == []
    assert profile_records(capsys.readouterr().err)[0]["status"] == "failed"


def test_default_factory_keeps_local_cpu_and_thread_configuration(monkeypatch):
    calls = []
    pipeline = SimpleNamespace(to=lambda device: calls.append(("device", device)))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=lambda n: calls.append(("threads", n)),
        device=lambda name: name,
    ))
    monkeypatch.setitem(sys.modules, "pyannote.audio", SimpleNamespace(
        Pipeline=SimpleNamespace(from_pretrained=lambda path: (
            calls.append(("model", path)) or pipeline
        )),
    ))
    assert main.default_pipeline_factory("/local/model", 3) is pipeline
    assert calls == [("threads", 3), ("model", "/local/model"), ("device", "cpu")]


def test_preloading_preserves_decoder_waveform_and_sample_rate(monkeypatch):
    waveform = SimpleNamespace(shape=(1, 160))
    calls = []
    monkeypatch.setitem(sys.modules, "torchaudio", SimpleNamespace(
        load=lambda path: (calls.append(path) or waveform, 16000),
    ))
    result = main.load_waveform("/local/audio.wav")
    assert result["waveform"] is waveform
    assert result["sample_rate"] == 16000
    assert calls == ["/local/audio.wav"]
