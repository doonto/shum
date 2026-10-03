from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "asr_engine_study.py"
SPEC = importlib.util.spec_from_file_location("asr_engine_study", SCRIPT)
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    root = tmp_path_factory.mktemp("frozen")
    audio = root / "audio.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        for _ in range(1800):
            wav.writeframes(b"\0" * 32000)
    clips = []
    for index in range(67):
        path = root / f"{index:05d}.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(b"\0" * 32000)
        clips.append({
            "index": index, "path": str(path), "start": float(index), "end": float(index + 1),
            "sha256": digest(path), "frames": 16000,
            "pcm_sha256": hashlib.sha256(b"\0" * 32000).hexdigest(),
        })
    manifest = root / "corpus.json"
    manifest.write_text(json.dumps({
        "schema": 1, "audio_path": str(audio), "audio_sha256": digest(audio),
        "duration_seconds": 1800.0, "clips": clips,
        "source_model": {"path": "/missing/flix"},  # Must not be inspected.
    }))
    return root, manifest


@pytest.fixture
def setup_run(tmp_path, frozen, monkeypatch):
    root, manifest = frozen
    worker = tmp_path / "dummy.py"
    worker.write_text(
        "import json,sys,time,subprocess\n"
        "mode=sys.argv[1]\n"
        "if mode=='exit': sys.exit(7)\n"
        "if mode=='spawn': subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])\n"
        "batch=[]\n"
        "for line in sys.stdin:\n"
        " r=json.loads(line); batch.append(r)\n"
        " if mode=='timeout': time.sleep(2); continue\n"
        " if mode=='error': print(json.dumps({'index':r['index'],'error':'failure'}),flush=True); continue\n"
        " if mode=='misindex': print(json.dumps({'index':999,'text':'bad','segments':[]}),flush=True); continue\n"
        " if mode=='partial' and r['index']==1: sys.exit(9)\n"
        " if mode=='late' and r['index']==1:\n"
        "  print(json.dumps({'index':0,'text':'duplicate','segments':[]}),flush=True); continue\n"
        " if mode=='outoforder' and len(batch)%2: continue\n"
        " for item in (reversed(batch[-2:]) if mode=='outoforder' else [r]):\n"
        "  print(json.dumps({'index':item['index'],'text':'ok','segments':[]}),flush=True)\n"
    )
    source_file = tmp_path / "weights.bin"
    source_file.write_bytes(b"weights")
    engine_file = tmp_path / "engine.bin"
    engine_file.write_bytes(b"engine")
    runtime_file = tmp_path / "runtime.bin"
    runtime_file.write_bytes(b"runtime")
    source = tmp_path / "source.json"
    source.write_text(json.dumps({
        "engine": "dummy", "engine_sha256": digest(engine_file),
        "checkpoint": "local-test", "checkpoint_sha256": digest(source_file),
        "runtime": "python-test", "runtime_sha256": digest(runtime_file),
        "files": {
            str(source_file): digest(source_file),
            str(engine_file): digest(engine_file),
            str(runtime_file): digest(runtime_file),
        },
    }))
    initial = {
        "available_kib": 16 * 1024 * 1024, "rss_kib": 100,
        "own_swap_kib": 0, "swap_in_pages": 0, "swap_out_pages": 0,
        "psi_memory_full_avg10_percent": 0.0,
    }
    monkeypatch.setattr(study.study, "memory_sample", lambda pid=None: {
        **initial, "rss_kib": 321 if pid else 100,
    })

    def make(mode="normal", indices="0,1", timeout=3, slots=1, full=False):
        return SimpleNamespace(
            manifest=manifest, source_description=source,
            output=tmp_path / f"run-{len(list(tmp_path.glob('run-*')))}",
            indices=None if full else indices, full_corpus=full,
            timeout_seconds=timeout, estimated_peak_mib=4096, max_in_flight=slots,
            worker_command=[sys.executable, str(worker), mode],
        )

    return make, source_file, source


def run(args):
    return json.loads(study.run_diagnostic(args).read_text())


def test_corpus_is_generic_and_checks_audio_pcm_bounds(frozen):
    _, manifest = frozen
    assert len(study.verify_corpus(manifest)["clips"]) == 67
    original = manifest.read_text()
    data = json.loads(original)
    data["clips"][0]["start"] = -1
    try:
        manifest.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="bounds"):
            study.verify_corpus(manifest)
    finally:
        manifest.write_text(original)


def test_out_of_order_bounded_dispatch_and_runtime_telemetry(setup_run):
    make, _, _ = setup_run
    args = make("outoforder", "4,8,4,3", slots=2)
    report = run(args)
    assert report["status"] == "complete"
    assert report["completed_indices"] == [4, 8, 4, 3]
    assert report["missing_indices"] == []
    assert report["peak_rss_kib"] == 321
    assert report["quality_status"] == "unverified" and report["wer"] is None
    requests = [json.loads(line) for line in (args.output / "requests.jsonl").read_text().splitlines()]
    assert [r["index"] for r in requests] == [0, 1, 2, 3]
    assert [r["clip_index"] for r in requests] == [4, 8, 4, 3]
    assert [json.loads(line)["request_number"] for line in (args.output / "events.jsonl").read_text().splitlines()] == [1, 0, 3, 2]


@pytest.mark.parametrize("mode,status,completed", [
    ("exit", "error", []),
    ("error", "error", []),
    ("misindex", "error", []),
    ("partial", "error", [0]),
    ("late", "error", [0]),
    ("timeout", "timeout", []),
])
def test_worker_failures_have_non_success_reports(setup_run, mode, status, completed):
    make, _, _ = setup_run
    args = make(mode, timeout=0.2 if mode == "timeout" else 3)
    report = run(args)
    assert report["status"] == status
    assert report["reason"]
    assert report["completed_indices"] == completed
    assert report["full_speed_gate"] is False
    assert (args.output / "worker.stderr").exists()
    assert report["worker_exit_code"] is not None
    if mode in ("error", "misindex", "late"):
        assert (args.output / "responses.jsonl").read_text().strip()


def test_full_corpus_gate_requires_all_67_including_loading(setup_run):
    make, _, _ = setup_run
    args = make(full=True)
    report = run(args)
    assert report["status"] == "complete"
    assert report["completed_indices"] == list(range(67))
    assert report["full_speed_gate"] is True
    assert report["elapsed_seconds"] < 600


def test_process_group_cleanup_kills_worker_children(setup_run):
    make, _, _ = setup_run
    report = run(make("spawn", indices="0"))
    assert report["status"] == "complete"
    assert not study.group_alive(report["worker_pid"])


def test_admission_and_resource_guard_stop(setup_run, monkeypatch):
    make, _, _ = setup_run
    initial = study.study.memory_sample()
    monkeypatch.setattr(study.study, "memory_sample", lambda pid=None: {
        **initial, "available_kib": 4096 * 1024 if pid else 7000 * 1024,
    })
    args = make()
    report = run(args)
    assert report["status"] == "blocked"
    assert report["reason"] == "insufficient_memory_for_estimated_peak_plus_reserve"
    assert not (args.output / "worker.stderr").exists()
    monkeypatch.setattr(study.study, "memory_sample", lambda pid=None: {
        **initial, "available_kib": (4096 if pid else 12000) * 1024 - (1 if pid else 0),
    })
    args = make("spawn")
    report = run(args)
    assert report["status"] == "resource_abort"
    assert report["reason"] == "available_memory_below_4_gib_reserve"
    assert report["peak_rss_kib"] == initial["rss_kib"]
    assert not study.group_alive(report.get("worker_pid", -1))


def test_monitor_error_is_explicit_resource_stop(setup_run, monkeypatch):
    make, _, _ = setup_run
    initial = study.study.memory_sample()

    def sample(pid=None):
        if pid:
            raise OSError("monitor unavailable")
        return initial

    monkeypatch.setattr(study.study, "memory_sample", sample)
    report = run(make("spawn"))
    assert report["status"] == "resource_abort"
    assert "monitor unavailable" in report["reason"]
    assert report["full_speed_gate"] is False
    assert not study.group_alive(report["worker_pid"])


def test_runtime_source_and_manifest_hash_changes_reject_completion(setup_run, monkeypatch):
    make, source_file, source = setup_run
    sample = study.study.memory_sample
    changed = False

    def mutate(pid=None):
        nonlocal changed
        if pid and not changed:
            source_file.write_bytes(b"changed")
            changed = True
        return sample(pid)

    monkeypatch.setattr(study.study, "memory_sample", mutate)
    report = run(make())
    assert report["status"] == "error"
    assert "Artifact changed after run" in report["reason"]
    source.write_text("{}")
    report = run(make())
    assert report["status"] == "error"


def test_source_description_identity_hash_must_have_artifact(setup_run):
    make, _, source = setup_run
    data = json.loads(source.read_text())
    data["runtime_sha256"] = "a" * 64
    source.write_text(json.dumps(data))
    report = run(make())
    assert report["status"] == "error"
    assert "runtime_sha256" in report["reason"]


def test_prepared_runtime_artifacts_map_is_accepted_and_rechecked(setup_run):
    make, _, source = setup_run
    data = json.loads(source.read_text())
    data["runtime_artifacts_sha256"] = data.pop("files")
    data.update(repo="local/asr", revision="fixture", quality_status="unverified",
                comparison_description="Speed only; human review pending")
    source.write_text(json.dumps(data))
    report = run(make(indices="0"))
    assert report["status"] == "complete"
    assert report["source_description"]["repo"] == "local/asr"
    assert report["source_artifacts_sha256"] == data["runtime_artifacts_sha256"]


def test_conflicting_source_artifact_maps_are_rejected(setup_run):
    make, _, source = setup_run
    data = json.loads(source.read_text())
    data["runtime_artifacts_sha256"] = {}
    source.write_text(json.dumps(data))
    report = run(make())
    assert report["status"] == "error"
    assert "maps disagree" in report["reason"]


def test_cli_rejects_invalid_timeout_and_uses_nonzero_exit(setup_run):
    make, _, _ = setup_run
    args = make(timeout=601)
    with pytest.raises(ValueError, match="Timeout"):
        study.run_diagnostic(args)
    args = make("error", indices="0")
    command = [
        sys.executable, str(SCRIPT), "--manifest", str(args.manifest),
        "--output", str(args.output), "--indices", args.indices,
        "--source-description", str(args.source_description),
        "--worker-command", *args.worker_command,
    ]
    # The actual host must have enough headroom for 4 GiB + 4 GiB to launch.
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert json.loads((args.output / "summary.json").read_text())["status"] != "complete"


@pytest.mark.parametrize("response", [
    {"index": True, "text": "", "segments": []},
    {"index": 0, "text": "", "segments": [{"start": math.nan, "end": 1, "text": ""}]},
    {"index": 0, "text": "", "segments": "bad"},
])
def test_strict_response_validation(response):
    with pytest.raises(ValueError):
        study.validate_response(response, {0: (0, 0)})
