"""Mocked protocol, provenance and timestamp tests for the Turbo CT2 worker."""
from __future__ import annotations

import io
import hashlib
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

import turbo_worker as worker


def ready_dirs(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    source, ct2 = tmp_path / "source", tmp_path / "ct2"
    source.mkdir()
    ct2.mkdir()
    (source / "config.json").write_text('{"model_type":"whisper"}')
    (source / "model.safetensors").write_bytes(b"source fixture")
    (source / ".shum-manifest.json").write_text(json.dumps({
        "repo": worker.REPO, "revision": worker.REVISION,
        "files": {"model.safetensors": worker.SOURCE_SHA256},
    }))
    for name in worker.CT2_ASSETS:
        (ct2 / name).write_bytes(b"fixture")
    (ct2 / worker.CT2_MANIFEST).write_text(json.dumps({
        "repo": worker.REPO, "revision": worker.REVISION,
        "files": {
            name: worker.CT2_SHA256 if name == "model.bin" else hashlib.sha256(b"fixture").hexdigest()
            for name in worker.CT2_ASSETS
        },
    }))
    original_hash = worker._sha256
    monkeypatch.setattr(worker, "_sha256", lambda path: (
        worker.SOURCE_SHA256 if path == source / "model.safetensors" else
        worker.CT2_SHA256 if path == ct2 / "model.bin" else original_hash(path)
    ))
    return source, ct2


def wav(path: Path, *, frames: int = 16000) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * frames)


class FakeModel:
    def __init__(self, segments=()):
        self.segments = segments
        self.calls = []

    def transcribe(self, path, **options):
        self.calls.append((path, options))
        return iter(self.segments), SimpleNamespace(language="de")


def segment(start, end, text=" grüezi"):
    return SimpleNamespace(start=start, end=end, text=text)


def test_cli_contract(tmp_path):
    parsed = worker.parse_args(["--engine", "whisper", "--model", str(tmp_path),
                                "--threads", "6", "--serve"])
    assert parsed.ct2_model is None
    for args in (["--model", str(tmp_path), "--threads", "6"],
                 ["--model", str(tmp_path), "--threads", "6", "--serve", "--input", "x"],
                 ["--model", str(tmp_path), "--threads", "0", "--serve"],
                 ["--engine", "qwen", "--model", str(tmp_path), "--threads", "6", "--serve"]):
        with pytest.raises(SystemExit):
            worker.parse_args(args)


def test_provenance_requires_pinned_identity_and_hashed_ct2(tmp_path, monkeypatch):
    source, ct2 = ready_dirs(tmp_path, monkeypatch)
    assert worker.validate_models(str(source), str(ct2)) == ct2
    manifest = source / ".shum-manifest.json"
    old = json.loads(manifest.read_text())
    manifest.write_text(json.dumps({**old, "revision": "untrusted"}))
    with pytest.raises(worker.AdapterError, match="pinned source repo/revision"):
        worker.validate_models(str(source), str(ct2))
    manifest.write_text(json.dumps(old))
    monkeypatch.setattr(worker, "_sha256", lambda _: "0" * 64)
    with pytest.raises(worker.AdapterError, match="Source model.safetensors SHA-256 mismatch"):
        worker.validate_models(str(source), str(ct2))


def test_ct2_interpretation_assets_are_hashed(tmp_path, monkeypatch):
    source, ct2 = ready_dirs(tmp_path, monkeypatch)
    (ct2 / "tokenizer.json").write_bytes(b"altered")
    with pytest.raises(worker.AdapterError, match="CT2 tokenizer.json SHA-256 mismatch"):
        worker.validate_models(str(source), str(ct2))
    (ct2 / "tokenizer.json").write_bytes(b"fixture")
    provenance = ct2 / worker.CT2_MANIFEST
    value = json.loads(provenance.read_text())
    del value["files"]["vocabulary.json"]
    provenance.write_text(json.dumps(value))
    with pytest.raises(worker.AdapterError, match="missing required publisher assets"):
        worker.validate_models(str(source), str(ct2))


def test_missing_ct2_never_downloads_or_converts(tmp_path, monkeypatch):
    source, _ = ready_dirs(tmp_path, monkeypatch)
    with pytest.raises(worker.AdapterError, match="Published CT2 directory does not exist"):
        worker.validate_models(str(source), str(tmp_path / "missing"))


def test_default_decode_and_model_reuse(tmp_path, monkeypatch):
    source, ct2 = ready_dirs(tmp_path, monkeypatch)
    model = FakeModel([segment(0.1, 0.9)])
    loads = []

    def factory(path, threads):
        loads.append((path, threads))
        return model

    transcribe = worker.build_transcribe_fn(str(source), str(ct2), 6, factory=factory)
    for _ in range(2):
        result = transcribe("/abs/clip.wav", 1.0)
        assert result == {"text": "grüezi", "language": "de",
                          "segments": [{"start": 0.1, "end": 0.9, "text": " grüezi"}]}
    assert loads == [(str(ct2), 6)]
    options = model.calls[0][1]
    assert options == {
        "language": "de", "task": "transcribe", "beam_size": 3, "best_of": 1,
        "vad_filter": True, "word_timestamps": False,
        "condition_on_previous_text": False,
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "compression_ratio_threshold": 2.4, "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
    }


def test_overshoot_clips_only_end_and_emits_auditable_raw_bounds():
    errors = io.StringIO()
    text, segments = worker.normalize_segments(
        [segment(26.12, 28.34)], 28.0, request_index=8, error_stream=errors,
    )
    assert text == "grüezi"
    assert segments == [{"start": 26.12, "end": 28.0, "text": " grüezi"}]
    assert json.loads(errors.getvalue()) == {
        "event": "asr_timestamp_adjustment", "request_index": 8, "segment_index": 0,
        "raw_start": 26.12, "raw_end": 28.34, "wav_duration": 28.0,
        "start": 26.12, "end": 28.0,
    }


def test_short_final_clip_retains_onset_and_logs_padded_decoder_end():
    errors = io.StringIO()
    text, segments = worker.normalize_segments(
        [segment(0.14, 30.12)], 2.4, request_index=3, error_stream=errors,
    )
    assert text == "grüezi"
    assert segments == [{"start": 0.14, "end": 2.4, "text": " grüezi"}]
    event = json.loads(errors.getvalue())
    assert event["raw_end"] == 30.12
    assert event["end"] == event["wav_duration"] == 2.4
    assert event["request_index"] == 3


@pytest.mark.parametrize("start,end,reason", [
    (28.01, 28.34, "entirely outside"),
    (-0.05, 0.4, "adjustment policy"),
    (0.2, 58.51, "adjustment policy"),
    (1.0, 0.4, "invalid raw"),
    (float("nan"), 0.4, "invalid raw"),
    (0, float("inf"), "invalid raw"),
])
def test_invalid_bounds_fail_not_discard(start, end, reason):
    with pytest.raises(worker.AdapterError, match=reason):
        worker.normalize_segments([segment(start, end)], 28, request_index=1,
                                  error_stream=io.StringIO())


def test_serve_indexed_error_and_success(tmp_path):
    clip = tmp_path / "clip.wav"
    wav(clip)
    calls = []

    def transcribe(path, duration, index):
        calls.append((path, duration, index))
        return {"text": "ok", "language": "de", "segments": []}

    output = io.StringIO()
    assert worker.serve(transcribe, in_stream=io.StringIO(
        json.dumps({"index": 3, "input": str(clip)}) + "\n"
    ), out_stream=output) == 0
    assert json.loads(output.getvalue())["index"] == 3
    assert calls == [(str(clip), 1.0, 3)]
    output = io.StringIO()
    assert worker.serve(transcribe, in_stream=io.StringIO(
        json.dumps({"index": 4, "input": str(tmp_path / "bad.wav")}) + "\n"
        + json.dumps({"index": 5, "input": str(clip)}) + "\n"
    ), out_stream=output, error_stream=io.StringIO()) == 1
    assert json.loads(output.getvalue())["index"] == 4
    assert len(output.getvalue().splitlines()) == 1
    assert len(calls) == 1


def test_serve_unexpected_model_failure_retains_index(tmp_path):
    clip = tmp_path / "clip.wav"
    wav(clip)

    def broken(*_):
        raise RuntimeError("decode failed")

    output = io.StringIO()
    assert worker.serve(broken, in_stream=io.StringIO(
        json.dumps({"index": 11, "input": str(clip)}) + "\n"
    ), out_stream=output, error_stream=io.StringIO()) == 1
    assert json.loads(output.getvalue()) == {"index": 11, "error": "decode failed"}


def test_single_clip_and_serve_run(tmp_path, monkeypatch):
    source, ct2 = ready_dirs(tmp_path, monkeypatch)
    clip = tmp_path / "clip.wav"
    wav(clip)
    model = FakeModel([segment(0.25, 1.02)])
    options = ["--engine", "whisper", "--model", str(source), "--ct2-model", str(ct2),
               "--threads", "6"]
    output = tmp_path / "result.json"
    assert worker.run(options + ["--input", str(clip), "--output", str(output)],
                      factory=lambda *_: model, error_stream=io.StringIO()) == 0
    assert json.loads(output.read_text())["segments"] == [
        {"start": 0.25, "end": 1.0, "text": " grüezi"}
    ]
    replies, errors = io.StringIO(), io.StringIO()
    assert worker.run(options + ["--serve"], factory=lambda *_: model,
                      in_stream=io.StringIO(json.dumps({"index": 7, "input": str(clip)}) + "\n"),
                      out_stream=replies, error_stream=errors) == 0
    assert json.loads(replies.getvalue())["index"] == 7
    assert json.loads(errors.getvalue())["request_index"] == 7
