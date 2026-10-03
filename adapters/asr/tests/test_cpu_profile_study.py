import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("cpu_profile_study", SCRIPTS / "cpu_profile_study.py")
meter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(meter)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    worker.write_text("""import json,sys,time,subprocess
mode=sys.argv[1]
if mode=='child': subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
for line in sys.stdin:
 r=json.loads(line)
 if mode in ('timeout','child'): time.sleep(5)
 response={'index':r['index'],'text':'speech','segments':[]}
 print(json.dumps(response),flush=True)
 if mode=='duplicate': print(json.dumps(response),flush=True)
""")
    monkeypatch.setattr(meter, "foreign_model_processes", lambda excluded: [])
    monkeypatch.setattr(meter, "cpu_limits", lambda: {"effective_cpu_capacity": 24})
    monkeypatch.setattr(meter.BackgroundCPU, "initial", lambda self: 0.0)
    monkeypatch.setattr(meter.BackgroundCPU, "observe", lambda self, roots: 0.0)
    original = meter.sample

    def available(roots):
        value = original(roots)
        value.update(available_kib=32 * 1024 * 1024, own_swap_kib=0,
                     swap_out_pages=0, psi_memory_full_avg10_percent=0)
        return value

    monkeypatch.setattr(meter, "sample", available)
    spec = {"component": "turbo", "processes": 2, "threads": 2,
            "estimated_peak_mib": 4096, "limit_seconds": 2,
            "command": [sys.executable, str(worker), "normal"],
            "inputs": [{"index": i, "path": "/unused.wav", "duration": 1} for i in range(4)]}
    passport = {"file_stats": {}, "session_deadline_epoch": time.time() + 60}
    return spec, passport, tmp_path / "output"


def test_fresh_parallel_workers_count_unique_results_and_cleanup(setup):
    spec, passport, output = setup
    result = meter.run(spec, output, passport)
    assert result["status"] == "complete"
    assert result["completed_indices"] == [0, 1, 2, 3]
    assert result["cleanup_verified"]
    assert result["cpu_seconds"] > 0
    assert not result["full_speed_gate"]
    assert all(code == 0 for code in result["exit_codes"])


@pytest.mark.parametrize("mode", ["timeout", "child"])
def test_deadline_stops_only_owned_groups(setup, mode):
    spec, passport, output = setup
    spec.update(limit_seconds=0.4)
    spec["command"][-1] = mode
    result = meter.run(spec, output, passport)
    assert result["status"] == "timeout"
    assert result["cleanup_verified"]
    assert not result["full_speed_gate"]


def test_duplicate_is_not_success(setup):
    spec, passport, output = setup
    spec["command"][-1] = "duplicate"
    result = meter.run(spec, output, passport)
    assert result["status"] == "error"
    assert result["cleanup_verified"]


def test_swap_guard_remains_resource_failure(setup, monkeypatch):
    spec, passport, output = setup
    original = meter.sample

    def swapping(roots):
        result = original(roots)
        if roots:
            result["own_swap_kib"] = 1
        return result

    monkeypatch.setattr(meter, "sample", swapping)
    result = meter.run(spec, output, passport)
    assert result["status"] == "resource_abort"
    assert result["cleanup_verified"]
    assert not result["full_speed_gate"]


def test_corrupt_diarization_and_overlapping_coverage():
    meter.validate_diarization({"turns": [], "exclusive_turns": []}, 1800)
    with pytest.raises(ValueError):
        meter.validate_diarization({"turns": [{"start": 0, "end": float("nan"),
                                               "speaker": "A"}], "exclusive_turns": []}, 1800)
    assert meter.union_duration([{"start": 0, "end": 2}, {"start": 1, "end": 3}]) == 3


def test_global_swap_out_alone_is_telemetry():
    initial = {"available_kib": 10 * 1024 * 1024, "own_swap_kib": 0,
               "swap_out_pages": 0, "psi_memory_full_avg10_percent": 0}
    current = {**initial, "swap_out_pages": 5}
    assert meter.study.resource_reason(current, initial) is None
    current["psi_memory_full_avg10_percent"] = 1
    assert meter.study.resource_reason(current, initial) is not None


def test_diarization_stdout_receipt_is_not_asr_protocol(setup):
    spec, passport, output = setup
    worker = Path(spec["command"][1])
    worker.write_text(
        "import json,pathlib,sys\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({'turns':[], 'exclusive_turns':[]}))\n"
        "print(json.dumps({'output':sys.argv[1]}),flush=True)\n"
    )
    spec.update(component="diarization", processes=1, duration_seconds=120,
                command=[sys.executable, str(worker), "{output}"])
    result = meter.run(spec, output, passport)
    assert result["status"] == "complete"
    assert result["cleanup_verified"]
    assert result["completed_indices"] == [0]
    assert (output / "worker-0.stdout").is_file()


def test_observation_preserves_unlimited_vad_option(capsys):
    spec = importlib.util.spec_from_file_location("cpu_profile_worker", SCRIPTS / "cpu_profile_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    worker.event("inference", time.monotonic(), vad_options={"max_speech_duration_s": float("inf")})
    value = json.loads(capsys.readouterr().err)
    assert value["vad_options"]["max_speech_duration_s"] == "+inf"
    assert value["nonfinite_fields"] == [{"path": "vad_options.max_speech_duration_s", "value": "+inf"}]


def test_primeline_unindexed_single_request_protocol(setup):
    spec, passport, output = setup
    worker = Path(spec["command"][1])
    worker.write_text(
        "import json,sys\n"
        "for line in sys.stdin:\n"
        " json.loads(line)\n"
        " print(json.dumps({'text':'speech','segments':[]}),flush=True)\n"
    )
    spec.update(component="primeline", processes=1)
    result = meter.run(spec, output, passport)
    assert result["status"] == "complete"
    assert result["completed_indices"] == [0, 1, 2, 3]
    assert result["cleanup_verified"]


@pytest.mark.parametrize("component", ["primeline", "turbo"])
def test_missing_or_duplicate_protocol_response_is_rejected(setup, component):
    spec, passport, output = setup
    worker = Path(spec["command"][1])
    worker.write_text(
        "import json,sys\n"
        "for line in sys.stdin:\n"
        " r=json.loads(line)\n"
        " response={'text':'speech','segments':[]}\n"
        " print(json.dumps(response),flush=True)\n"
        " print(json.dumps(response),flush=True)\n"
    )
    spec.update(component=component, processes=1)
    result = meter.run(spec, output, passport)
    assert result["status"] == "error"
    assert result["cleanup_verified"]
    assert not result["full_speed_gate"]
    assert (output / "worker-0.stdout").is_file()


@pytest.mark.parametrize("flag", ["--detail-profile", "--embedding-channels-last"])
def test_detail_profile_is_diarization_only(monkeypatch, flag):
    spec = importlib.util.spec_from_file_location("profile_worker", SCRIPTS / "cpu_profile_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    monkeypatch.setattr(sys, "argv", ["profile_worker", "turbo", flag])
    with pytest.raises(ValueError, match="diarization only"):
        worker.main()


def test_detail_observer_restores_hooks_after_pipeline_failure(monkeypatch):
    spec = importlib.util.spec_from_file_location("profile_worker", SCRIPTS / "cpu_profile_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    pipeline = object()
    closed = []
    observer = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_interop_threads=lambda value: None, get_num_threads=lambda: 8,
        get_num_interop_threads=lambda: 1))
    monkeypatch.setitem(sys.modules, "threadpoolctl", SimpleNamespace(threadpool_info=lambda: []))
    monkeypatch.setitem(sys.modules, "diarization_cpu_observer", SimpleNamespace(
        install=lambda value, emit: observer if value is pipeline else None))

    def run(arguments, *, pipeline_factory):
        assert "--detail-profile" not in arguments
        assert pipeline_factory("/unused", 8) is pipeline
        raise OSError("pipeline failure")

    monkeypatch.setitem(sys.modules, "main", SimpleNamespace(
        configure_thread_environment=lambda threads: None,
        default_pipeline_factory=lambda directory, threads: pipeline, run=run))
    monkeypatch.setattr(sys, "argv", ["profile_worker", "diarization", "--detail-profile"])
    with pytest.raises(OSError, match="pipeline failure"):
        worker.main()
    assert closed == [True]


def test_layout_restored_even_when_detail_cleanup_fails(monkeypatch):
    spec = importlib.util.spec_from_file_location("profile_worker", SCRIPTS / "cpu_profile_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    closed = []

    def observer_close():
        closed.append("observer")
        raise OSError("observer cleanup failure")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_interop_threads=lambda value: None, get_num_threads=lambda: 8,
        get_num_interop_threads=lambda: 1))
    monkeypatch.setitem(sys.modules, "threadpoolctl", SimpleNamespace(threadpool_info=lambda: []))
    monkeypatch.setitem(sys.modules, "diarization_cpu_observer", SimpleNamespace(
        install=lambda pipeline, emit: SimpleNamespace(close=observer_close)))
    monkeypatch.setitem(sys.modules, "diarization_cpu_layout", SimpleNamespace(
        install=lambda pipeline, emit: SimpleNamespace(close=lambda: closed.append("layout"))))

    def run(arguments, *, pipeline_factory):
        assert "--embedding-channels-last" not in arguments
        pipeline_factory("/unused", 8)
        return 0

    monkeypatch.setitem(sys.modules, "main", SimpleNamespace(
        configure_thread_environment=lambda threads: None,
        default_pipeline_factory=lambda directory, threads: object(), run=run))
    monkeypatch.setattr(sys, "argv", [
        "profile_worker", "diarization", "--detail-profile", "--embedding-channels-last"])
    with pytest.raises(OSError, match="observer cleanup failure"):
        worker.main()
    assert closed == ["observer", "layout"]
