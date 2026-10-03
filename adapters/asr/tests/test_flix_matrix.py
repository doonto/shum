from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("flix_matrix", SCRIPTS / "flix_matrix.py")
assert SPEC is not None and SPEC.loader is not None
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


def test_matrix_changes_only_fast_decode_between_paired_profiles(tmp_path):
    corpus = {
        "source_model": {"path": "/model"},
        "ct2_model": {"path": "/ct2", "provenance": {"quantization": "int8"}},
    }
    fast = matrix.build_worker_command(corpus, tmp_path / "python", tmp_path / "worker", "fast-repeat", 6)
    fallback = matrix.build_worker_command(corpus, tmp_path / "python", tmp_path / "worker", "fallback-repeat", 6)
    assert [argument for argument in fast if argument != "--fast-decode"] == fallback
    assert matrix.PROFILES["fast-repeat"][0] == matrix.PROFILES["fallback-repeat"][0]
    assert "--reset-seed-per-request" in fast


def test_matrix_preserves_virtualenv_interpreter_symlink(tmp_path):
    binary = tmp_path / "base-python"
    binary.touch()
    interpreter = tmp_path / "venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(binary)
    corpus = {
        "source_model": {"path": "/model"},
        "ct2_model": {"path": "/ct2", "provenance": {"quantization": "int8"}},
    }
    command = matrix.build_worker_command(corpus, interpreter, tmp_path / "worker", "fast-repeat", 6)
    assert command[0] == str(interpreter)
    assert command[0] != str(interpreter.resolve())


@pytest.mark.parametrize("status", ["complete", "resource_abort", "timeout"])
def test_matrix_is_sequential_and_persists_stop_reason(tmp_path, monkeypatch, status):
    python, worker = tmp_path / "python", tmp_path / "worker"
    python.touch()
    worker.touch()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr(matrix.study, "verify_corpus", lambda *_: {
        "source_model": {"path": "/model"},
        "ct2_model": {"path": "/ct2", "provenance": {"quantization": "int8"}},
    })
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        out = Path(command[command.index("--output") + 1])
        out.mkdir()
        matrix.study.write_json(out / "summary.json", {"status": status, "reason": None if status == "complete" else "pressure"})
        return SimpleNamespace(returncode=0 if status == "complete" else 1)

    monkeypatch.setattr(matrix.subprocess, "run", execute)
    args = SimpleNamespace(
        manifest=manifest, output=tmp_path / "matrix", python=python, worker=worker,
        threads=6, profiles=["fast-repeat", "fallback-repeat"],
    )
    report = matrix.study.read_json(matrix.run_matrix(args))
    assert len(calls) == (1 if status == "resource_abort" else 2)
    expected = {"complete": "finished", "resource_abort": "stopped", "timeout": "finished_with_failures"}
    assert report["status"] == expected[status]
    assert report["quality_status"] == "unverified"
    assert report["full_speed_gate"] is False
