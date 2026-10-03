#!/usr/bin/env python3
"""Metadata-only Flix ONNX feasibility probe (no export, model load or inference)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import flix_fast_common as common

REPO = "Flix-AI/flix-swissgerman-full"
REVISION = "a9c347a904af5d31b28c537ac732e0618aed6136"
GIB = 1024**3
RESERVE_BYTES = 4 * GIB
EXPORT_MIN_PEAK_BYTES = 24 * GIB
LOAD_MIN_PEAK_BYTES = 16 * GIB
PROVENANCE_FILENAME = "flix-onnx-provenance.json"
NO_CACHE_FILES = (
    "encoder_model.onnx",
    "encoder_model.onnx_data",
    "decoder_model.onnx",
    "decoder_model.onnx_data",
)


class ProbeError(RuntimeError):
    """An invalid or unsafe probe input, not a compatibility result."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProbeError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProbeError(f"Expected JSON object: {path}")
    return value


def directory(value: str, flag: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_dir():
        raise ProbeError(f"{flag} must be an existing absolute directory: {value}")
    return path.resolve()


def source_metadata(source: Path) -> dict[str, Any]:
    manifest_path = source / ".shum-manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("repo") != REPO or manifest.get("revision") != REVISION:
        raise ProbeError(f"Source manifest must declare {REPO}@{REVISION}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ProbeError("Source manifest has no file hash inventory")
    config = read_json(source / "config.json")
    if config.get("model_type") != "whisper" or config.get("num_mel_bins") != 128 or config.get("dtype") != "bfloat16":
        raise ProbeError("Source config is not a BF16 128-mel Whisper checkpoint")
    processor_path = source / "processor_config.json"
    if not processor_path.is_file():
        processor_path = source / "preprocessor_config.json"
    processor = read_json(processor_path)
    feature = processor.get("feature_extractor", processor)
    if not isinstance(feature, dict) or feature.get("feature_size") != 128 or feature.get("sampling_rate") != 16000:
        raise ProbeError("Source processor must specify 128 mel bins and 16000 Hz")
    generation = read_json(source / "generation_config.json")
    if generation.get("language") != "de" or generation.get("task") != "transcribe":
        raise ProbeError("Source generation defaults must specify de/transcribe")
    for name in ("config.json", processor_path.name, "generation_config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
        if not isinstance(files.get(name), str) or len(files[name]) != 64:
            raise ProbeError(f"Source manifest lacks a hash for {name}")
        path = source / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(source):
            raise ProbeError(f"Source file missing or linked outside snapshot: {path}")
    # Deliberately stat only: this is a metadata probe, not a full source integrity check.
    return {
        "repo": REPO,
        "revision": REVISION,
        "source_path": str(source),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_weight_bytes": (source / "model.safetensors").stat().st_size,
        "source_dtype": config.get("dtype"),
        "source_transformers_version": config.get("transformers_version"),
        "processor": {"feature_size": feature["feature_size"], "sampling_rate": feature["sampling_rate"]},
        "generation": generation,
        "integrity": "manifest_declared_only_weights_not_rehashed",
    }


def effective_available(
    meminfo: str, cgroup: Path = Path("/sys/fs/cgroup"), *, membership: str | None = None,
) -> int:
    available = None
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) == 3 and fields[2] == "kB":
                try:
                    available = int(fields[1]) * 1024
                except ValueError as exc:
                    raise ProbeError("Invalid MemAvailable in /proc/meminfo") from exc
            break
    if available is None or available < 0:
        raise ProbeError("Cannot establish MemAvailable from /proc/meminfo")
    try:
        headroom = common.cgroup_headroom_bytes(cgroup, membership=membership)
        if headroom is not None:
            available = min(available, headroom)
    except common.AdapterError as exc:
        raise ProbeError(str(exc)) from exc
    return available


def gate(available: int, peak: int) -> dict[str, int | bool]:
    if available < 0 or peak < 0:
        raise ProbeError("Memory figures cannot be negative")
    return {
        "available_bytes": available,
        "conservative_peak_estimate_bytes": peak,
        "reserve_bytes": RESERVE_BYTES,
        "required_available_bytes": peak + RESERVE_BYTES,
        "admitted": available >= peak + RESERVE_BYTES,
    }


def graph_metadata(graphs: Path, source: dict[str, Any], available: int) -> dict[str, Any]:
    config = read_json(graphs / "config.json")
    processor = read_json(graphs / "preprocessor_config.json")
    generation = read_json(graphs / "generation_config.json")
    source_generation = source["generation"].copy()
    source_generation.pop("transformers_version", None)
    graph_generation = generation.copy()
    graph_generation.pop("transformers_version", None)
    matching = (
        config.get("model_type") == "whisper"
        and config.get("num_mel_bins") == 128
        and processor.get("feature_size") == 128
        and processor.get("sampling_rate") == 16000
        and graph_generation == source_generation
    )
    sizes = {}
    for name in NO_CACHE_FILES:
        path = graphs / name
        if path.is_symlink() or not path.is_file():
            raise ProbeError(f"Existing no-cache graph file missing or linked: {path}")
        sizes[name] = path.stat().st_size
    data_bytes = sizes["encoder_model.onnx_data"] + sizes["decoder_model.onnx_data"]
    provenance_path = graphs / PROVENANCE_FILENAME
    provenance = None
    if provenance_path.is_file():
        declared = read_json(provenance_path)
        provenance = (
            declared.get("source_repo") == REPO
            and declared.get("source_revision") == REVISION
            and declared.get("source_manifest_sha256") == source["manifest_sha256"]
            and declared.get("precision") == "fp32"
            and declared.get("cache") is False
            and declared.get("status") == "verified_export"
        )
    return {
        "path": str(graphs),
        "files_bytes": sizes,
        "config_dtype_claim": config.get("dtype"),
        "config_matches_source_except_version": matching,
        "provenance": "matching_declaration" if provenance else ("mismatch" if provenance is False else "missing"),
        "provenance_file": str(provenance_path),
        "load_gate_no_cache": gate(available, max(LOAD_MIN_PEAK_BYTES, 2 * data_bytes)),
        "graph_weights_and_tensor_values_inspected": False,
    }


def probe(source: Path, available: int, graphs: Path | None = None) -> dict[str, Any]:
    metadata = source_metadata(source)
    weight_bytes = metadata["source_weight_bytes"]
    if weight_bytes <= 0:
        raise ProbeError("Source checkpoint weight file is empty")
    export_gate = gate(available, max(EXPORT_MIN_PEAK_BYTES, 8 * weight_bytes))
    reasons: list[str] = []
    if not export_gate["admitted"]:
        reasons.append("export_memory_gate_failed")
    report = {
        "status": "blocked",
        "source": metadata,
        "export_gate": export_gate,
        "precision_review": "BF16-to-FP32 numerical preservation not verified; no FP16 intermediate permitted",
        "cache": False,
        "existing_graphs": None,
        "reasons": reasons,
        "scope": "metadata_only_no_export_or_inference",
    }
    if graphs is not None:
        existing = graph_metadata(graphs, metadata, available)
        report["existing_graphs"] = existing
        if not existing["config_matches_source_except_version"]:
            reasons.append("existing_graph_config_mismatch")
        if existing["provenance"] != "matching_declaration":
            reasons.append("existing_graph_source_provenance_" + existing["provenance"])
        if not existing["load_gate_no_cache"]["admitted"]:
            reasons.append("existing_graph_load_memory_gate_failed")
        # Even declared provenance cannot verify precision or generation parity.
        reasons.append("existing_graph_numerical_and_generation_parity_unverified")
    elif export_gate["admitted"]:
        report["status"] = "ready_for_export_review"
    return report


def write_report(report: dict[str, Any], destination: Path, protected: tuple[Path, ...]) -> None:
    if not destination.is_absolute() or not destination.parent.is_dir():
        raise ProbeError("--report must name a file in an existing absolute directory")
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise ProbeError("--report must not be a symlink or directory")
    resolved = destination.resolve()
    if any(resolved == root or root in resolved.parents for root in protected):
        raise ProbeError("--report must be outside source and graph directories")
    fd, temporary = tempfile.mkstemp(prefix=".flix-onnx-probe-", suffix=".json", dir=str(destination.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Absolute path to managed pinned source")
    parser.add_argument("--graphs", help="Absolute path to existing ONNX graphs (optional)")
    parser.add_argument("--report", required=True, help="Absolute JSON report path outside model and graphs")
    args = parser.parse_args(argv)
    try:
        source = directory(args.model, "--model")
        graphs = directory(args.graphs, "--graphs") if args.graphs else None
        available = effective_available(Path("/proc/meminfo").read_text(encoding="ascii"))
        report = probe(source, available, graphs)
        write_report(report, Path(args.report), (source, graphs) if graphs else (source,))
    except (OSError, ProbeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{report['status']}: {args.report}")
    return 2 if report["status"] == "blocked" else 0


if __name__ == "__main__":
    sys.exit(main())
