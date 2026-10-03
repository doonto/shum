from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "flix_study", Path(__file__).resolve().parents[3] / "scripts" / "flix_study.py",
)
assert SPEC is not None and SPEC.loader is not None
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_indices_preserve_order_and_repeated_requests():
    assert study.selected_indices("23,8,23") == [23, 8, 23]
    with pytest.raises(ValueError):
        study.selected_indices("67")


def test_resource_guard_preserves_reserve_and_requires_evidence_of_pressure():
    initial = {
        "available_kib": 12 * 1024 * 1024, "swap_in_pages": 10,
        "swap_out_pages": 20, "own_swap_kib": 0, "psi_memory_full_avg10_percent": 0.0,
    }
    assert study.resource_reason(initial, initial) is None
    assert study.resource_reason({**initial, "available_kib": 4096 * 1024}, initial) is None
    assert study.resource_reason({**initial, "available_kib": 4096 * 1024 - 1}, initial) == "available_memory_below_4_gib_reserve"
    assert study.resource_reason({**initial, "swap_out_pages": 21}, initial) is None
    assert study.resource_reason({**initial, "swap_out_pages": 21, "psi_memory_full_avg10_percent": 0.99}, initial) is None
    assert study.resource_reason({**initial, "psi_memory_full_avg10_percent": 1.0}, initial) is None
    assert study.resource_reason({**initial, "swap_out_pages": 21, "psi_memory_full_avg10_percent": 1.0}, initial) == "memory_psi_full_avg10_at_least_1_percent_with_new_system_swap_out"
    assert study.resource_reason({**initial, "own_swap_kib": 1}, initial) == "worker_process_tree_vmswap_increased"
    assert study.resource_reason({**initial, "swap_out_pages": 19}, initial) == "system_pswpout_counter_reset"
    assert study.resource_reason({**initial, "swap_in_pages": 11}, initial) is None


@pytest.mark.parametrize("response", [
    {"index": 4, "text": "", "segments": []},
    {"index": True, "text": "", "segments": []},
    {"text": "", "segments": []},
    {"index": 23, "error": "boom"},
])
def test_response_rejects_wrong_duplicate_or_error_identity(response):
    with pytest.raises(ValueError):
        study.validate_response(response, 23)


def test_response_accepts_indexed_contract():
    study.validate_response({"index": 23, "text": "text", "segments": []}, 23)


def test_runtime_provenance_keeps_venv_config_and_hashes_resolved_executable(tmp_path):
    base = tmp_path / "base-python"
    base.write_bytes(b"interpreter")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    executable = venv / "bin" / "python"
    executable.symlink_to(base)
    config = venv / "pyvenv.cfg"
    config.write_text("home = /base")
    worker = tmp_path / "worker.py"
    worker.write_text("print('test')")
    common = tmp_path / "flix_fast_common.py"
    common.write_text("value = 1")
    hashes = study.command_artifacts([str(executable), str(worker), "--serve"])
    assert hashes[str(base)] == study.sha256(base)
    assert hashes[str(config)] == study.sha256(config)
    assert hashes[str(common)] == study.sha256(common)


def test_startup_error_is_not_hidden_by_missing_request_index():
    with pytest.raises(ValueError, match="missing dependency"):
        study.validate_response({"error": "missing dependency"}, 0)


def test_repetition_metric_is_a_flag_not_a_quality_score():
    assert study.repetition_fraction("one two three") == 0.0
    assert study.repetition_fraction("one two three four " * 10) > 0.5


def test_atomic_report_is_persistent(tmp_path):
    path = tmp_path / "summary.json"
    study.write_json(path, {"status": "timeout", "missing_indices": [23]})
    assert json.loads(path.read_text())["status"] == "timeout"
    assert not path.with_suffix(".json.tmp").exists()


def test_report_rejects_nonfinite_measurements(tmp_path):
    with pytest.raises(ValueError):
        study.write_json(tmp_path / "summary.json", {"elapsed": float("nan")})
    assert list(tmp_path.iterdir()) == []


def mock_run_args(tmp_path, monkeypatch, code, *, timeout=5.0, indices="23,8"):
    manifest = tmp_path / "corpus.json"
    manifest.write_text("{}")
    source, converted = tmp_path / "source", tmp_path / "converted"
    monkeypatch.setattr(study, "verify_corpus", lambda *_: {
        "source_model": {"path": str(source)}, "ct2_model": {"path": str(converted)},
        "clips": [{"path": f"/clip-{index}.wav"} for index in range(67)],
    })
    monkeypatch.setattr(study, "memory_sample", lambda *_: {
        "available_kib": 12 * 1024 * 1024, "rss_kib": 123,
        "swap_in_pages": 0, "swap_out_pages": 0,
        "own_swap_kib": 0, "psi_memory_full_avg10_percent": 0.0,
    })
    return SimpleNamespace(
        manifest=manifest, output=tmp_path / "run", indices=indices,
        timeout_seconds=timeout, estimated_peak_mib=4096,
        worker_command=[
            "/usr/bin/python3", "-u", "-c", code, "--model", str(source),
            "--ct2-model", str(converted), "--serve",
        ],
    )


def test_run_records_request_and_window_indices_separately(tmp_path, monkeypatch):
    args = mock_run_args(tmp_path, monkeypatch, (
        "import sys,json\n"
        "for line in sys.stdin:\n"
        " r=json.loads(line); print(json.dumps({'index':r['index'],'text':'ok','segments':[]}),flush=True)\n"
    ), indices="23,8,23")
    summary = study.read_json(study.run_diagnostic(args))
    assert summary["status"] == "complete"
    assert summary["completed_indices"] == [23, 8, 23]
    assert summary["missing_indices"] == []
    assert summary["full_speed_gate"] is False
    assert summary["memory_full_avg10_threshold_percent"] == 1.0
    assert "not worker attribution" in summary["resource_guard_semantics"]
    assert summary["resource_samples"][0]["own_swap_kib"] == 0
    assert summary["resource_samples"][-1]["psi_memory_full_avg10_percent"] == 0.0
    events = [json.loads(line) for line in (args.output / "events.jsonl").read_text().splitlines()]
    assert [event["index"] for event in events] == [23, 8, 23]
    assert [event["request_number"] for event in events] == [0, 1, 2]


def test_full_corpus_speed_gate_requires_all_67_windows_without_quality_claim(tmp_path, monkeypatch):
    args = mock_run_args(tmp_path, monkeypatch, (
        "import sys,json\n"
        "for line in sys.stdin:\n"
        " r=json.loads(line); print(json.dumps({'index':r['index'],'text':'ok','segments':[]}),flush=True)\n"
    ))
    args.full_corpus = True
    args.indices = None
    report = study.read_json(study.run_diagnostic(args))
    assert report["completed_indices"] == list(range(67))
    assert report["full_speed_gate"] is True
    assert report["quality_status"] == "unverified"


def test_timeout_persists_partial_report_and_rejects_late_result(tmp_path, monkeypatch):
    args = mock_run_args(tmp_path, monkeypatch, (
        "import sys,json,time\n"
        "r=json.loads(sys.stdin.readline()); time.sleep(0.4)\n"
        "print(json.dumps({'index':r['index'],'text':'late','segments':[]}),flush=True)\n"
    ), timeout=0.1)
    summary = study.read_json(study.run_diagnostic(args))
    assert summary["status"] == "timeout"
    assert summary["completed_indices"] == []
    assert summary["missing_indices"] == [23, 8]
    assert not list(args.output.glob("result-*.json"))


def test_worker_failure_persists_report(tmp_path, monkeypatch):
    args = mock_run_args(tmp_path, monkeypatch, "import sys; sys.exit(7)")
    summary = study.read_json(study.run_diagnostic(args))
    assert summary["status"] == "error"
    assert summary["worker_exit_code"] == 7
    assert summary["missing_indices"] == [23, 8]


def test_review_keeps_quality_unverified_and_preserves_variants(tmp_path, monkeypatch):
    args = mock_run_args(tmp_path, monkeypatch, (
        "import sys,json\n"
        "for line in sys.stdin:\n"
        " r=json.loads(line); print(json.dumps({'index':r['index'],'text':'one two three four '*10,'segments':[]}),flush=True)\n"
    ))
    study.run_diagnostic(args)
    args.manifest.write_text(json.dumps({
        "clips": [{"path": f"/clip-{i}.wav", "start": i, "end": i + 1, "frames": 16000} for i in range(67)],
    }))
    summary_path = args.output / "summary.json"
    summary = study.read_json(summary_path)
    summary["manifest_sha256"] = study.sha256(args.manifest)
    study.write_json(summary_path, summary)
    report = study.read_json(study.review_report(SimpleNamespace(
        manifest=args.manifest, study_dir=tmp_path, output=tmp_path / "review.json",
    )))
    assert report["quality_status"] == "unverified"
    assert report["wer"] is None and report["cer"] is None
    assert report["runs"][0]["windows"][0]["flags"] == ["repeated_fourgrams"]
    assert report["review_windows"][0]["reference_text"] is None


def test_review_discovers_nested_matrix_profiles_without_label_collisions(tmp_path, monkeypatch):
    args = mock_run_args(tmp_path, monkeypatch, (
        "import sys,json\n"
        "for line in sys.stdin:\n"
        " r=json.loads(line); print(json.dumps({'index':r['index'],'text':'ok','segments':[]}),flush=True)\n"
    ))
    args.output = tmp_path / "matrix-a" / "profile"
    study.run_diagnostic(args)
    args.manifest.write_text(json.dumps({
        "clips": [{"path": f"/clip-{i}.wav", "start": i, "end": i + 1, "frames": 16000} for i in range(67)],
    }))
    summary_path = args.output / "summary.json"
    summary = study.read_json(summary_path)
    summary["manifest_sha256"] = study.sha256(args.manifest)
    study.write_json(summary_path, summary)
    report = study.read_json(study.review_report(SimpleNamespace(
        manifest=args.manifest, study_dir=tmp_path, output=tmp_path / "review.json",
    )))
    assert report["runs"][0]["run"] == "matrix-a/profile"
    assert report["review_windows"][0]["variants"][0]["run"] == "matrix-a/profile"


def test_insufficient_memory_never_launches_worker(tmp_path, monkeypatch):
    args = mock_run_args(tmp_path, monkeypatch, "raise AssertionError('must not run')")
    monkeypatch.setattr(study, "memory_sample", lambda *_: {
        "available_kib": 7000 * 1024, "rss_kib": 0,
        "swap_in_pages": 0, "swap_out_pages": 0,
        "own_swap_kib": 0, "psi_memory_full_avg10_percent": 0.0,
    })
    summary = study.read_json(study.run_diagnostic(args))
    assert summary["status"] == "blocked"
    assert summary["missing_indices"] == [23, 8]
    assert not (args.output / "worker.stderr").exists()


def test_prelaunch_monitor_error_never_launches_worker(tmp_path, monkeypatch):
    args = mock_run_args(tmp_path, monkeypatch, "raise AssertionError('must not run')")
    monkeypatch.setattr(study, "memory_sample", lambda *_: (_ for _ in ()).throw(
        study.common.AdapterError("cgroup read failed")))
    summary = study.read_json(study.run_diagnostic(args))
    assert summary["status"] == "blocked"
    assert "resource_monitor_error_before_launch" in summary["reason"]
    assert not (args.output / "worker.stderr").exists()


def test_running_monitor_error_is_resource_abort(tmp_path, monkeypatch):
    args = mock_run_args(tmp_path, monkeypatch, "import time; time.sleep(2)")
    initial = {
        "available_kib": 12 * 1024 * 1024, "rss_kib": 0,
        "swap_in_pages": 0, "swap_out_pages": 0,
        "own_swap_kib": 0, "psi_memory_full_avg10_percent": 0.0,
    }
    def sample(pid=None):
        if pid is not None:
            raise OSError("PSI read failed")
        return initial
    monkeypatch.setattr(study, "memory_sample", sample)
    summary = study.read_json(study.run_diagnostic(args))
    assert summary["status"] == "resource_abort"
    assert "PSI read failed" in summary["reason"]
