"""Metadata/guard tests only: no ONNX sessions or model inference."""
from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import flix_onnx_pilot as pilot


def put_json(root: Path, name: str, data: dict) -> None:
    (root / name).write_text(json.dumps(data), encoding="utf-8")


def fixture_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    source, graphs = tmp_path / "source", tmp_path / "graphs"
    source.mkdir()
    graphs.mkdir()
    put_json(source, "config.json", {
        "model_type": "whisper", "num_mel_bins": 128, "dtype": "bfloat16",
    })
    put_json(graphs, "config.json", {
        "model_type": "whisper", "num_mel_bins": 128, "dtype": "float32",
    })
    generation = {
        "language": "de", "task": "transcribe", "suppress_tokens": [1, 2],
        "begin_suppress_tokens": [220, 50257],
    }
    put_json(source, "generation_config.json", generation)
    put_json(graphs, "generation_config.json", {
        **generation, "transformers_version": "different-export-version",
    })
    put_json(source, "processor_config.json", {
        "feature_extractor": {"feature_size": 128, "sampling_rate": 16000},
    })
    put_json(graphs, "preprocessor_config.json", {
        "feature_size": 128, "sampling_rate": 16000,
    })
    put_json(source, "tokenizer_config.json", {
        "language": "de", "task": "transcribe",
        "extra_special_tokens": ["<|de|>", "<|transcribe|>"],
    })
    put_json(source, "tokenizer.json", {"added_tokens": [
        {"content": "<|de|>", "id": 1, "special": True},
        {"content": "<|transcribe|>", "id": 2, "special": True},
    ]})
    (source / "model.safetensors").write_bytes(b"fake-test-only")
    files = {p.name: pilot.digest(p) for p in source.iterdir()}
    put_json(source, ".shum-manifest.json", {
        "repo": "Flix-AI/flix-swissgerman-full",
        "revision": pilot.REVISION, "files": files,
    })
    entries = {}
    for name in pilot.GRAPH_NAMES:
        for suffix in (".onnx", ".onnx_data"):
            (graphs / (name + suffix)).write_bytes((name + suffix).encode())
        entries[name] = {
            "initializer_count": 1, "exact_initializers": 1,
            "onnx_sha256": pilot.digest(graphs / (name + ".onnx")),
            "external_data_sha256": pilot.digest(graphs / (name + ".onnx_data")),
        }
    audit = tmp_path / "audit.json"
    put_json(tmp_path, "audit.json", {
        "exact_identity": True, "source_parameters_total": 1259,
        "unique_source_parameters_matched": 1259,
        "source": {"sha256": files["model.safetensors"]},
        "graphs": entries,
    })
    return source, graphs, audit


def test_identity_guards_every_graph_and_source_hash(tmp_path):
    source, graphs, audit = fixture_paths(tmp_path)
    result = pilot.verify_identity(source, graphs, audit)
    assert result["revision"] == pilot.REVISION
    assert result["suppressed_tokens"] == [1, 2]
    (graphs / "decoder_model.onnx_data").write_bytes(b"changed")
    with pytest.raises(pilot.PilotError, match="checksum mismatch"):
        pilot.verify_identity(source, graphs, audit)
    (graphs / "decoder_model.onnx_data").write_bytes(b"decoder_model.onnx_data")
    (source / "tokenizer.json").write_text('{"other": true}')
    with pytest.raises(pilot.PilotError, match="Source checksum mismatch: tokenizer.json"):
        pilot.verify_identity(source, graphs, audit)


def test_identity_rejects_generation_config_mismatch_and_link(tmp_path):
    source, graphs, audit = fixture_paths(tmp_path)
    put_json(graphs, "generation_config.json", {"language": "en", "task": "translate"})
    with pytest.raises(pilot.PilotError, match="Generation defaults"):
        pilot.verify_identity(source, graphs, audit)
    (graphs / "generation_config.json").unlink()
    (graphs / "generation_config.json").symlink_to(source / "generation_config.json")
    # The generation file is JSON metadata; graph binaries and source hashes are
    # independently protected against linked paths.
    (graphs / "encoder_model.onnx").unlink()
    (graphs / "encoder_model.onnx").symlink_to(source / "model.safetensors")
    with pytest.raises(pilot.PilotError, match="Missing or linked"):
        pilot.verify_identity(source, graphs, audit)


def test_nonempty_special_tokens_are_mapped_losslessly_and_every_id_checked(tmp_path):
    source, _, _ = fixture_paths(tmp_path)
    mapping, entries = pilot.source_token_mapping(source)
    assert mapping == {
        "flix_extra_special_000": "<|de|>",
        "flix_extra_special_001": "<|transcribe|>",
    }

    class Tokenizer:
        all_special_tokens = ["<|de|>", "<|transcribe|>"]

        def __len__(self):
            return 3

        def convert_tokens_to_ids(self, token):
            return {"<|de|>": 1, "<|transcribe|>": 2}[token]

    pilot.validate_fast_token_ids(Tokenizer(), entries, mapping, 3)
    with pytest.raises(pilot.PilotError, match="vocabulary size"):
        pilot.validate_fast_token_ids(Tokenizer(), entries, mapping, 4)
    entries[0]["id"] = 10
    with pytest.raises(pilot.PilotError, match="changed an added-token ID"):
        pilot.validate_fast_token_ids(Tokenizer(), entries, mapping, 3)
    put_json(source, "tokenizer.json", {"added_tokens": [
        {"content": "<|de|>", "id": 1, "special": False},
        {"content": "<|transcribe|>", "id": 2, "special": True},
    ]})
    with pytest.raises(pilot.PilotError, match="missing or not special"):
        pilot.source_token_mapping(source)


def test_scope_must_have_kernel_ram_limit_and_zero_swap(tmp_path):
    scope = tmp_path / "scope"
    scope.mkdir()
    (scope / "memory.max").write_text("1073741824")
    (scope / "memory.swap.max").write_text("0")
    (scope / "memory.current").write_text("1024")
    assert pilot.scope_memory(scope, pilot.GIB)["memory_max_bytes"] == pilot.GIB
    (scope / "memory.swap.max").write_text("max")
    with pytest.raises(pilot.PilotError, match="Kernel cgroup"):
        pilot.scope_memory(scope, pilot.GIB)
    (scope / "memory.swap.max").write_text("0")
    with pytest.raises(pilot.PilotError, match="Kernel cgroup"):
        pilot.scope_memory(scope, pilot.GIB - 1)


def test_budget_and_pressure_enforce_reserve_swap_and_psi():
    assert pilot.budget(14 * pilot.GIB) == 9 * pilot.GIB
    assert pilot.budget(12 * pilot.GIB) == 8 * pilot.GIB
    with pytest.raises(pilot.PilotError):
        pilot.budget(4 * pilot.GIB)
    initial = {
        "available_bytes": 10 * pilot.GIB, "own_vmswap_kib": 0,
        "system_pswpout_pages": 20, "psi_memory_full_avg10_percent": 0.0,
    }
    assert pilot.pressure_reason(initial, initial) is None
    assert pilot.pressure_reason({**initial, "available_bytes": pilot.RESERVE-1}, initial) == "4_gib_reserve_lost"
    assert pilot.pressure_reason({**initial, "own_vmswap_kib": 1}, initial) == "own_swap_detected"
    assert pilot.pressure_reason({**initial, "cgroup_swap_current_bytes": 1}, initial) == "own_swap_detected"
    assert pilot.pressure_reason({
        **initial, "system_pswpout_pages": 21, "psi_memory_full_avg10_percent": 1.0,
    }, initial) == "swap_out_with_full_memory_psi"


def test_128_new_token_cap_reports_eos_and_possible_truncation():
    flags = pilot.termination_flags([50258] + [42] * 128, 50257)
    assert flags == {
        "eos_observed": False, "artificial_max_new_tokens": 128,
        "cap_reached_possible": True, "possible_truncation": True,
        "termination": "possible_128_token_cap_truncation",
    }
    ended = pilot.termination_flags([50258, 42, 50257], 50257)
    assert ended["eos_observed"] and not ended["possible_truncation"]
    assert ended["termination"] == "eos_observed_not_audio_completeness_proof"
    assert pilot.termination_flags([50258, 42], [50257, 50256])["possible_truncation"]


def test_metadata_dry_run_never_starts_runtime(tmp_path, monkeypatch):
    source, graphs, audit = fixture_paths(tmp_path)
    report = tmp_path / "report.json"
    monkeypatch.setattr(pilot, "mem_available", lambda: 14 * pilot.GIB)
    monkeypatch.setattr(pilot, "run", lambda *_: pytest.fail("Dry-run started runtime"))
    assert pilot.main([
        "--source", str(source), "--graphs", str(graphs), "--audit", str(audit),
        "--report", str(report),
    ]) == 0
    written = json.loads(report.read_text())
    assert written["status"] == "metadata_ready_runtime_unverified"
    assert written["budget_bytes"] == 9 * pilot.GIB
    assert written["profile"]["cache"] is False
    assert written["profile"]["artificial_cap_128"] is True
    assert written["profile"]["normal_benchmark_comparable"] is False
    assert written["requirements"]["runtime_api"].startswith("optimum.")
    assert "text" not in written


def test_run_requires_window_and_output_protected(tmp_path, monkeypatch):
    source, graphs, audit = fixture_paths(tmp_path)
    monkeypatch.setattr(pilot, "mem_available", lambda: 14 * pilot.GIB)
    assert pilot.main([
        "--source", str(source), "--graphs", str(graphs), "--audit", str(audit),
        "--report", str(tmp_path / "blocked.json"), "--run",
    ]) == 2
    assert "exactly one --input" in json.loads((tmp_path / "blocked.json").read_text())["error"]
    assert pilot.main([
        "--source", str(source), "--graphs", str(graphs), "--audit", str(audit),
        "--report", str(source / "forbidden.json"),
    ]) == 1
    assert not (source / "forbidden.json").exists()


def test_run_uses_systemd_hard_cap_and_official_no_cache_profile(monkeypatch):
    seen = {}
    scope = "/sys/fs/cgroup/user.slice/flix-test.scope"

    class FakeProcess:
        stdout = io.StringIO("\n".join(json.dumps(item) for item in (
            {"phase": "scope_ready", "pid": 999, "cgroup": scope, "memory": {}},
            {"phase": "loaded", "startup_seconds": 0.5, "memory": {}},
            {"phase": "decoded", "text": "diagnostic", "tokens": 3,
             "decode_seconds": 0.2, "memory": {},
             **pilot.termination_flags([50258, 42, 50257], 50257)},
        )) + "\n")
        stderr = io.StringIO("")
        returncode = 0

        def poll(self):
            return 0

    def fake_popen(command, **kwargs):
        seen["command"] = command
        seen["environment"] = kwargs["env"]
        return FakeProcess()

    monkeypatch.setattr(pilot.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(pilot, "scope_memory", lambda *_: {})
    monkeypatch.setattr(pilot, "scoped_activity", lambda *_: {
        "scoped_pids": [], "pid_mismatch": False,
    })
    monkeypatch.setattr(pilot, "sample", lambda *_: {
        "available_bytes": 12 * pilot.GIB, "own_vmswap_kib": 0,
        "system_pswpout_pages": 0, "psi_memory_full_avg10_percent": 0.0,
    })
    report = {
        "budget_bytes": 8 * pilot.GIB, "identity": {
            "suppressed_tokens": [1, 2], "begin_suppress_tokens": [220],
        },
    }
    args = SimpleNamespace(
        source=Path("/source"), graphs=Path("/graphs"),
        input=Path("/one-window.wav"), report=Path("/diagnostics/report.json"),
        threads=2, seed=123,
    )
    pilot.run(args, report)
    command = seen["command"]
    assert command[:4] == ["systemd-run", "--user", "--scope", "--quiet"]
    assert f"--property=MemoryMax={8 * pilot.GIB}" in command
    assert "--property=MemorySwapMax=0" in command
    assert "--scope-unit" in command
    assert command[command.index("--scope-unit") + 1] == report["scope_unit"]
    assert seen["environment"]["HF_HUB_OFFLINE"] == "1"
    assert report["scope_verified"] and report["status"] == "runtime_compatibility_observed_text_unverified"
    assert report["text"] == "diagnostic"
    assert report["termination"]["eos_observed"]
    assert report["cleanup"]["status"] == "verified_stopped"


def test_controller_preserves_verified_decoded_sample_after_scope_disappears(monkeypatch):
    last = {
        "available_bytes": 12 * pilot.GIB, "own_vmswap_kib": 0,
        "system_pswpout_pages": 0, "psi_memory_full_avg10_percent": 0.0,
        "cgroup_memory_peak_bytes": 12345,
    }

    class Finished:
        stdout = io.StringIO("\n".join(json.dumps(item) for item in (
            {"phase": "scope_ready", "pid": 999, "cgroup": "/sys/fs/cgroup/owned.scope", "memory": last},
            {"phase": "decoded", "text": "possibly partial", "tokens": 129,
             "decode_seconds": 1.0, "memory": last,
             **pilot.termination_flags([50258] + [42] * 128, 50257)},
        )) + "\n")
        stderr = io.StringIO("")
        returncode = 0

        def poll(self):
            return 0

    monkeypatch.setattr(pilot.subprocess, "Popen", lambda *_args, **_kwargs: Finished())
    monkeypatch.setattr(pilot, "scope_memory", lambda *_: {})
    monkeypatch.setattr(pilot, "scoped_activity", lambda *_: {
        "scoped_pids": [], "pid_mismatch": False,
    })
    scope_reads = [0]

    def disappearing_sample(scope=None, pid=None):
        if scope:
            scope_reads[0] += 1
            if scope_reads[0] > 1:
                raise FileNotFoundError("scope removed by systemd")
        return last

    monkeypatch.setattr(pilot, "sample", disappearing_sample)
    report = {"budget_bytes": 8 * pilot.GIB, "identity": {
        "suppressed_tokens": [1], "begin_suppress_tokens": [220],
    }}
    args = SimpleNamespace(source=Path("/source"), graphs=Path("/graphs"),
                           input=Path("/one.wav"), report=Path("/report.json"), threads=2, seed=0)
    pilot.run(args, report)
    assert report["status"] == "runtime_compatibility_observed_text_unverified"
    assert report["cgroup_removed_after_completion"]
    assert report["last_memory"] == last
    assert report["peak_cgroup_bytes"] == 12345
    assert report["termination"]["possible_truncation"]
    assert report["cleanup"]["status"] == "verified_stopped"


def test_scope_disappearance_while_worker_live_is_error(monkeypatch):
    class Live:
        stdout = io.StringIO(json.dumps({
            "phase": "scope_ready", "pid": 999, "cgroup": "/sys/fs/cgroup/owned.scope",
            "memory": {},
        }) + "\n")
        stderr = io.StringIO("")
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, *, timeout):
            self.returncode = 0

    monkeypatch.setattr(pilot.subprocess, "Popen", lambda *_args, **_kwargs: Live())
    monkeypatch.setattr(pilot.subprocess, "run", lambda *_args, **_kwargs:
                        SimpleNamespace(returncode=0, stderr=""))
    monkeypatch.setattr(pilot, "scope_memory", lambda *_: {})
    monkeypatch.setattr(pilot, "scoped_activity", lambda *_: {
        "scoped_pids": [], "pid_mismatch": False,
    })

    def sample(scope=None, pid=None):
        if scope:
            raise FileNotFoundError("scope vanished")
        return {"available_bytes": 12 * pilot.GIB, "own_vmswap_kib": 0,
                "system_pswpout_pages": 0, "psi_memory_full_avg10_percent": 0.0}

    monkeypatch.setattr(pilot, "sample", sample)
    report = {"budget_bytes": 8 * pilot.GIB, "identity": {
        "suppressed_tokens": [1], "begin_suppress_tokens": [220],
    }}
    args = SimpleNamespace(source=Path("/source"), graphs=Path("/graphs"),
                           input=Path("/one.wav"), report=Path("/report.json"), threads=2, seed=0)
    with pytest.raises(pilot.PilotError, match="scope vanished while worker is live"):
        pilot.run(args, report)
    assert "scope vanished while worker is live" in report["operation_error"]
    assert report["cleanup"]["status"] == "verified_stopped"


def test_failed_worker_traceback_wins_over_transient_vmrss_race(monkeypatch):
    class Failed:
        stdout = io.StringIO(json.dumps({
            "phase": "scope_ready", "pid": 999, "cgroup": "/sys/fs/cgroup/owned.scope",
            "memory": {"available_bytes": 12 * pilot.GIB, "own_vmswap_kib": 0,
                       "system_pswpout_pages": 0, "psi_memory_full_avg10_percent": 0.0},
        }) + "\n")
        stderr = io.StringIO("AttributeError: 'list' object has no attribute 'keys'\n")
        returncode = 1

        def poll(self):
            return 1

    monkeypatch.setattr(pilot.subprocess, "Popen", lambda *_args, **_kwargs: Failed())
    monkeypatch.setattr(pilot, "scope_memory", lambda *_: {})
    monkeypatch.setattr(pilot, "scoped_activity", lambda *_: {
        "scoped_pids": [], "pid_mismatch": False,
    })

    def racing_sample(scope=None, pid=None):
        if scope:
            raise pilot.common.AdapterError("Invalid VmRSS/VmSwap for worker 999")
        return {"available_bytes": 12 * pilot.GIB, "own_vmswap_kib": 0,
                "system_pswpout_pages": 0, "psi_memory_full_avg10_percent": 0.0}

    monkeypatch.setattr(pilot, "sample", racing_sample)
    report = {"budget_bytes": 8 * pilot.GIB, "identity": {
        "suppressed_tokens": [1], "begin_suppress_tokens": [220],
    }}
    args = SimpleNamespace(source=Path("/source"), graphs=Path("/graphs"),
                           input=Path("/one.wav"), report=Path("/report.json"), threads=2, seed=0)
    with pytest.raises(pilot.PilotError, match="Worker exited 1.*AttributeError"):
        pilot.run(args, report)
    assert "AttributeError" in report["operation_error"]
    assert report["cleanup"]["status"] == "verified_stopped"


def test_transient_vmrss_retries_are_bounded_and_persistent_live_error_fails(monkeypatch):
    calls = []

    def transient(scope=None, pid=None):
        calls.append(1)
        if len(calls) < 3:
            raise pilot.common.AdapterError("Invalid VmRSS/VmSwap for worker 999")
        return {"own_vmswap_kib": 0}

    monkeypatch.setattr(pilot, "sample", transient)
    monkeypatch.setattr(pilot.time, "sleep", lambda _: None)
    live = SimpleNamespace(poll=lambda: None)
    assert pilot.monitored_sample(None, 999, live) == {"own_vmswap_kib": 0}
    assert len(calls) == 3

    def persistent(scope=None, pid=None):
        raise pilot.common.AdapterError("Invalid VmRSS/VmSwap for worker 999")

    monkeypatch.setattr(pilot, "sample", persistent)
    with pytest.raises(pilot.common.AdapterError, match="Invalid VmRSS"):
        pilot.monitored_sample(None, 999, live)


def test_cleanup_failure_records_stop_error_and_unverified_scoped_worker(monkeypatch):
    class Stuck:
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, *, timeout):
            if self.returncode is None:
                raise pilot.subprocess.TimeoutExpired("systemd-run", timeout)
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(pilot, "scoped_activity", lambda *_: {
        "scoped_pids": [12345], "pid_mismatch": False,
    })
    monkeypatch.setattr(pilot.subprocess, "run", lambda *_args, **_kwargs:
                        SimpleNamespace(returncode=1, stderr="permission denied"))
    monkeypatch.setattr(pilot.time, "sleep", lambda _: None)
    signals = []
    monkeypatch.setattr(pilot.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    report = {}
    with pytest.raises(pilot.PilotError, match="Owned scope cleanup failed"):
        pilot.stop_owned_job(Stuck(), "owned-unit", Path("/sys/fs/cgroup/owned.scope"), 12345, report)
    assert report["cleanup"]["systemctl_exit_code"] == 1
    assert report["cleanup"]["systemctl_error"] == "permission denied"
    assert report["cleanup"]["status"] == "unverified_cleanup_failure"
    assert "termination could not be verified" in report["cleanup"]["error"]
    assert [sig for _, sig in signals] == [pilot.signal.SIGTERM, pilot.signal.SIGKILL]


def test_launch_rechecks_fresh_headroom_and_reduces_kernel_cap(tmp_path, monkeypatch):
    source, graphs, audit = fixture_paths(tmp_path)
    values = iter((14 * pilot.GIB, 11 * pilot.GIB))
    monkeypatch.setattr(pilot, "mem_available", lambda: next(values))
    monkeypatch.setattr(pilot, "validate_wav", lambda _: 1.0)
    monkeypatch.setattr(pilot, "runtime_requirements", lambda: {"available": True})
    captured = {}

    def fake_run(_, report):
        captured["cap"] = report["budget_bytes"]
        report["status"] = "runtime_compatibility_observed_text_unverified"

    monkeypatch.setattr(pilot, "run", fake_run)
    report = tmp_path / "report.json"
    assert pilot.main([
        "--source", str(source), "--graphs", str(graphs), "--audit", str(audit),
        "--report", str(report), "--run", "--input", str(tmp_path / "synthetic.wav"),
    ]) == 0
    assert captured["cap"] == 7 * pilot.GIB
    assert json.loads(report.read_text())["available_at_launch_bytes"] == 11 * pilot.GIB
