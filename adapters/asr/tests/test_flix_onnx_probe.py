"""Read-only Flix probe tests; no ONNX, torch, export or inference imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import flix_onnx_probe as probe


def make_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    payloads = {
        "config.json": {"model_type": "whisper", "num_mel_bins": 128, "dtype": "bfloat16", "transformers_version": "5.0.0"},
        "processor_config.json": {"feature_extractor": {"feature_size": 128, "sampling_rate": 16000}},
        "generation_config.json": {"language": "de", "task": "transcribe", "suppress_tokens": [1, 2]},
        "tokenizer.json": {"model": {}},
        "tokenizer_config.json": {"added_tokens_decoder": {"1": {"content": "<|de|>"}}},
    }
    for name, content in payloads.items():
        (source / name).write_text(json.dumps(content))
    (source / "model.safetensors").write_bytes(b"bf16-stub")
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    (source / ".shum-manifest.json").write_text(
        json.dumps({"repo": probe.REPO, "revision": probe.REVISION, "files": files})
    )
    return source


def make_graphs(tmp_path: Path, source: Path) -> Path:
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    (graphs / "config.json").write_text(json.dumps({"model_type": "whisper", "num_mel_bins": 128, "dtype": "float32"}))
    generation = json.loads((source / "generation_config.json").read_text())
    generation["transformers_version"] = "4.57.6"
    (graphs / "generation_config.json").write_text(json.dumps(generation))
    (graphs / "preprocessor_config.json").write_text(json.dumps({"feature_size": 128, "sampling_rate": 16000}))
    for name in probe.NO_CACHE_FILES:
        (graphs / name).write_bytes(b"graph-placeholder")
    return graphs


def test_source_metadata_uses_manifest_not_weight_contents(tmp_path, monkeypatch):
    source = make_source(tmp_path)
    original = Path.read_bytes

    def metadata_only(path):
        if path.name == "model.safetensors":
            raise AssertionError("weight data was read")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", metadata_only)
    metadata = probe.source_metadata(source)
    assert metadata["repo"] == probe.REPO
    assert metadata["revision"] == probe.REVISION
    assert metadata["source_weight_bytes"] == len(b"bf16-stub")
    assert metadata["processor"] == {"feature_size": 128, "sampling_rate": 16000}
    assert metadata["generation"]["suppress_tokens"] == [1, 2]
    assert metadata["integrity"] == "manifest_declared_only_weights_not_rehashed"


@pytest.mark.parametrize("key,value", [("repo", "other"), ("revision", "main")])
def test_wrong_source_rejected(tmp_path, key, value):
    source = make_source(tmp_path)
    path = source / ".shum-manifest.json"
    manifest = json.loads(path.read_text())
    manifest[key] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(probe.ProbeError, match="manifest"):
        probe.source_metadata(source)


def test_wrong_processor_and_dtype_rejected(tmp_path):
    source = make_source(tmp_path)
    (source / "processor_config.json").write_text(json.dumps({"feature_extractor": {"feature_size": 80, "sampling_rate": 16000}}))
    with pytest.raises(probe.ProbeError, match="processor"):
        probe.source_metadata(source)
    (source / "processor_config.json").write_text(json.dumps({"feature_extractor": {"feature_size": 128, "sampling_rate": 16000}}))
    (source / "config.json").write_text(json.dumps({"model_type": "whisper", "num_mel_bins": 128, "dtype": "float16"}))
    with pytest.raises(probe.ProbeError, match="BF16"):
        probe.source_metadata(source)


def test_exact_export_memory_boundary_and_cgroup(tmp_path):
    source = make_source(tmp_path)
    threshold = 28 * probe.GIB
    assert probe.probe(source, threshold - 1)["status"] == "blocked"
    ready = probe.probe(source, threshold)
    assert ready["status"] == "ready_for_export_review"
    assert ready["export_gate"]["required_available_bytes"] == threshold
    assert probe.gate(32 * probe.GIB, 8 * 4 * probe.GIB)["required_available_bytes"] == 36 * probe.GIB
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.max").write_text(str(30 * probe.GIB))
    (cgroup / "memory.current").write_text(str(3 * probe.GIB))
    assert probe.effective_available("MemAvailable: 40000000 kB", cgroup, membership="0::/") == 27 * probe.GIB
    (cgroup / "memory.current").write_text("invalid")
    with pytest.raises(probe.ProbeError, match="cgroup"):
        probe.effective_available("MemAvailable: 40000000 kB", cgroup, membership="0::/")


def test_existing_graphs_unproven_and_load_gate(tmp_path):
    source = make_source(tmp_path)
    graphs = make_graphs(tmp_path, source)
    report = probe.probe(source, 19 * probe.GIB, graphs)
    assert report["status"] == "blocked"
    assert report["existing_graphs"]["config_matches_source_except_version"]
    assert report["existing_graphs"]["provenance"] == "missing"
    assert report["existing_graphs"]["load_gate_no_cache"]["required_available_bytes"] == 20 * probe.GIB
    assert "existing_graph_load_memory_gate_failed" in report["reasons"]
    assert "existing_graph_source_provenance_missing" in report["reasons"]
    assert not report["existing_graphs"]["graph_weights_and_tensor_values_inspected"]


def test_matching_provenance_still_not_proof_of_compatibility(tmp_path):
    source = make_source(tmp_path)
    graphs = make_graphs(tmp_path, source)
    metadata = probe.source_metadata(source)
    declared = {
        "source_repo": probe.REPO, "source_revision": probe.REVISION,
        "source_manifest_sha256": metadata["manifest_sha256"],
        "precision": "fp32", "cache": False, "status": "verified_export",
    }
    path = graphs / probe.PROVENANCE_FILENAME
    path.write_text(json.dumps(declared))
    report = probe.probe(source, 40 * probe.GIB, graphs)
    assert report["existing_graphs"]["provenance"] == "matching_declaration"
    assert report["status"] == "blocked"
    assert "existing_graph_numerical_and_generation_parity_unverified" in report["reasons"]
    declared["source_revision"] = "wrong"
    path.write_text(json.dumps(declared))
    assert probe.probe(source, 40 * probe.GIB, graphs)["existing_graphs"]["provenance"] == "mismatch"


def test_graph_generation_mismatch_reported(tmp_path):
    source = make_source(tmp_path)
    graphs = make_graphs(tmp_path, source)
    (graphs / "generation_config.json").write_text(json.dumps({"language": "en", "task": "translate"}))
    report = probe.probe(source, 40 * probe.GIB, graphs)
    assert "existing_graph_config_mismatch" in report["reasons"]


def test_report_persistence_and_cli_exit_contract(tmp_path, monkeypatch, capsys):
    source = make_source(tmp_path)
    graphs = make_graphs(tmp_path, source)
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(probe, "effective_available", lambda _: 12 * probe.GIB)
    argv = ["--model", str(source), "--graphs", str(graphs), "--report", str(report_path)]
    assert probe.main(argv) == 2
    stored = json.loads(report_path.read_text())
    assert stored["status"] == "blocked"
    assert stored["export_gate"]["required_available_bytes"] == 28 * probe.GIB
    assert capsys.readouterr().out.startswith("blocked:")
    with pytest.raises(probe.ProbeError, match="outside"):
        probe.write_report(stored, source / "report.json", (source, graphs))
    assert not (source / "report.json").exists()
    monkeypatch.setattr(probe, "effective_available", lambda _: 40 * probe.GIB)
    assert probe.main(["--model", str(source), "--report", str(report_path)]) == 0
    assert json.loads(report_path.read_text())["status"] == "ready_for_export_review"
    assert probe.main(["--model", str(source), "--report", str(source / "bad.json")]) == 1
