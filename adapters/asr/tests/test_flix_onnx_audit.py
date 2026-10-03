"""Tiny external-data identity tests; never access the pinned multi-GB graphs."""
from __future__ import annotations

import hashlib
import json
import os
import resource
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest

import flix_onnx_audit as audit


def unit_audit(monkeypatch, source: Path, graphs: Path) -> dict:
    # The full pytest process may already exceed 512 MiB of address space.
    # Only the numerical algorithm runs in-process; the real cap is tested
    # independently through the CLI in a fresh interpreter below.
    with monkeypatch.context() as patched:
        patched.setattr(audit, "memory_guard", lambda: resource.getrlimit(resource.RLIMIT_AS))
        return audit.audit(source, graphs)


def run_cli(source: Path, graphs: Path, report: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(Path(audit.__file__).absolute()),
         "--source", str(source), "--graphs", str(graphs), "--report", str(report)],
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=30, check=False,
    )


def bf16(values: list[float]) -> bytes:
    return (np.array(values, dtype="<f4").view("<u4") >> 16).astype("<u2").tobytes()


def make_source(root: Path, weights: dict[str, tuple[list[int], list[float]]]) -> Path:
    source = root / "source"
    source.mkdir()
    header = {"__metadata__": {"format": "pt"}}
    payload = bytearray()
    for name, (shape, values) in weights.items():
        start = len(payload)
        payload += bf16(values)
        header[name] = {"dtype": "BF16", "shape": shape,
                        "data_offsets": [start, len(payload)]}
    encoded = json.dumps(header).encode()
    (source / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    (source / "config.json").write_text(json.dumps({
        "model_type": "whisper", "num_mel_bins": 128, "dtype": "bfloat16",
    }))
    (source / ".shum-manifest.json").write_text(json.dumps({
        "repo": audit.REPO, "revision": audit.REVISION,
        "files": {"model.safetensors": hashlib.sha256(
            (source / "model.safetensors").read_bytes(),
        ).hexdigest()},
    }))
    return source


def make_graph(root: Path, name: str,
               initializers: list[tuple[str, list[int], list[float]]],
               *, location: str | None = None) -> None:
    data = bytearray()
    tensors = []
    for tensor_name, shape, values in initializers:
        tensor = onnx.TensorProto()
        tensor.name = tensor_name
        tensor.dims.extend(shape)
        tensor.data_type = onnx.TensorProto.FLOAT
        tensor.data_location = onnx.TensorProto.EXTERNAL
        external = {"location": location or name + ".onnx_data",
                    "offset": str(len(data)), "length": str(4 * len(values))}
        for key, value in external.items():
            field = tensor.external_data.add()
            field.key, field.value = key, value
        tensors.append(tensor)
        data.extend(np.array(values, dtype="<f4").tobytes())
    graph = onnx.helper.make_graph([], name, [], [], initializer=tensors)
    onnx.save_model(onnx.helper.make_model(graph), root / (name + ".onnx"))
    (root / (name + ".onnx_data")).write_bytes(data)


def fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = make_source(tmp_path, {
        "model.encoder.direct.weight": ([2, 3], [1, 2, 3, 4, 5, 6]),
        "model.decoder.proj.weight": ([2, 3], [-1, -2, -3, 7, 8, 9]),
    })
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    make_graph(graphs, "encoder_model", [("direct.weight", [2, 3], [1, 2, 3, 4, 5, 6])])
    transpose = [-1, 7, -2, 8, -3, 9]
    make_graph(graphs, "decoder_model", [("onnx::MatMul_3", [3, 2], transpose)])
    make_graph(graphs, "decoder_with_past_model", [("onnx::MatMul_4", [3, 2], transpose)])
    return source, graphs


def test_exact_direct_transpose_and_compatible_partial_schema(tmp_path, monkeypatch):
    source, graphs = fixture(tmp_path)
    report = unit_audit(monkeypatch, source, graphs)
    assert not report["exact_identity"]  # two tensors cannot attest to the pinned 1,259
    assert report["source_parameters_total"] == 2
    assert report["unique_source_parameters_matched"] == 2
    assert report["unmapped_source_parameters"] == []
    assert report["source"]["manifest_hash_matches"]
    assert report["graphs"]["encoder_model"]["exact_initializers"] == 1
    decoder = report["graphs"]["decoder_model"]
    assert decoder["initializers"][0]["transform"] == "transpose"
    assert decoder["exact_initializers"] == decoder["initializer_count"] == 1
    assert decoder["external_data_sha256"] == hashlib.sha256(
        (graphs / "decoder_model.onnx_data").read_bytes()
    ).hexdigest()
    assert report["graphs"]["decoder_with_past_model"]["exact_initializers"] == 1
    assert report["method"]["load_external_data"] is False


def test_named_initializer_reports_bitwise_mismatch(tmp_path, monkeypatch):
    source, graphs = fixture(tmp_path)
    make_graph(graphs, "encoder_model", [
        ("direct.weight", [2, 3], [1, 2, 3, 4, 5, 10]),
    ])
    report = unit_audit(monkeypatch, source, graphs)
    bad = report["graphs"]["encoder_model"]["mismatches"]
    assert len(bad) == 1
    assert bad[0]["source"] == "model.encoder.direct.weight"
    assert bad[0]["first_mismatched_flat_indices"] == [5]
    assert bad[0]["mismatched_elements"] == 1
    assert not report["exact_identity"]


def test_duplicate_values_remain_ambiguous_without_name_evidence(tmp_path, monkeypatch):
    values = [1, 2, 3, 4, 5, 6]
    source = make_source(tmp_path, {
        "model.encoder.a.weight": ([2, 3], values),
        "model.encoder.b.weight": ([2, 3], values),
    })
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    for name in audit.GRAPHS:
        make_graph(graphs, name, [("onnx::MatMul_0", [3, 2], [1, 4, 2, 5, 3, 6])])
    report = unit_audit(monkeypatch, source, graphs)
    record = report["graphs"]["encoder_model"]["initializers"][0]
    assert record["status"] == "ambiguous"
    assert record["candidate_sources"] == [
        "model.encoder.a.weight", "model.encoder.b.weight",
    ]
    assert report["unique_source_parameters_matched"] == 0
    assert not report["exact_identity"]


def test_unrepresented_parameter_never_claims_complete_identity(tmp_path, monkeypatch):
    source, graphs = fixture(tmp_path)
    weights = {
        "model.encoder.direct.weight": ([2, 3], [1, 2, 3, 4, 5, 6]),
        "model.decoder.proj.weight": ([2, 3], [-1, -2, -3, 7, 8, 9]),
    }
    weights.update({
        f"model.encoder.unused.{i}.weight": ([1], [float(i)])
        for i in range(1257)
    })
    source = make_source_replacing(source, weights)
    report = unit_audit(monkeypatch, source, graphs)
    assert report["source_parameters_total"] == 1259
    assert report["unique_source_parameters_matched"] == 2
    assert len(report["unmapped_source_parameters"]) == 1257
    assert not report["exact_identity"]


def make_source_replacing(source: Path, weights: dict[str, tuple[list[int], list[float]]]) -> Path:
    # Regenerate the tiny fixture under the same test root.
    for path in source.iterdir():
        path.unlink()
    source.rmdir()
    return make_source(source.parent, weights)


def test_rejects_external_path_traversal_and_protected_report(tmp_path, monkeypatch):
    source, graphs = fixture(tmp_path)
    make_graph(graphs, "decoder_model", [
        ("onnx::MatMul_3", [3, 2], [-1, 7, -2, 8, -3, 9]),
    ], location="../other.onnx_data")
    with pytest.raises(audit.AuditError, match="path/offset/length invalid"):
        unit_audit(monkeypatch, source, graphs)
    report_path = tmp_path / "audit.json"
    cli = run_cli(source, graphs, report_path)
    assert cli.returncode == 2, cli.stderr
    assert "path/offset/length invalid" in json.loads(report_path.read_text())["error"]
    with pytest.raises(SystemExit):
        audit.main(["--source", str(source), "--graphs", str(graphs),
                    "--report", str(source / "forbidden.json")])
    assert not (source / "forbidden.json").exists()


def test_real_cli_enforces_cap_in_fresh_process(tmp_path):
    source, graphs = fixture(tmp_path)
    report_path = tmp_path / "bounded.json"
    completed = run_cli(source, graphs, report_path)
    assert completed.returncode == 2, completed.stderr
    report = json.loads(report_path.read_text())
    assert report["graphs"]["encoder_model"]["exact_initializers"] == 1
    assert report["graphs"]["decoder_model"]["exact_initializers"] == 1
    assert report["method"]["memory_limit_mib"] == 512
    assert report["peak_self_rss_mib"] <= 512
    assert not report["exact_identity"]  # toy source, not the pinned checkpoint


def test_real_cli_restores_soft_limit_after_partial_result(tmp_path):
    source, graphs = fixture(tmp_path)
    report_path = tmp_path / "restored.json"
    script = (
        "import resource,sys\n"
        "from flix_onnx_audit import main\n"
        "before=resource.getrlimit(resource.RLIMIT_AS)\n"
        "result=main(sys.argv[1:])\n"
        "assert resource.getrlimit(resource.RLIMIT_AS)==before\n"
        "print('soft_limit_restored')\n"
        "sys.exit(result)\n"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, "--source", str(source),
         "--graphs", str(graphs), "--report", str(report_path)],
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
             "PYTHONPATH": str(Path(audit.__file__).parent)},
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 2, result.stderr
    assert "soft_limit_restored" in result.stdout
    assert json.loads(report_path.read_text())["peak_self_rss_mib"] <= 512
