#!/usr/bin/env python3
"""Bounded BF16-to-FP32 initializer identity audit for pinned Flix ONNX graphs.

Reads the safetensors header, ONNX protobufs without external data, and bounded
external-data slices. Does not load a model, run inference, or change inputs.
Whole-checkpoint identity requires all 1,259 source tensors and all three graphs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import resource
import struct
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
MIB = 1024**2
CAP = 512 * MIB
REVISION = "a9c347a904af5d31b28c537ac732e0618aed6136"
REPO = "Flix-AI/flix-swissgerman-full"
PINNED_MODEL_SHA256 = "cd5bf4fdfd7fbcabed5bc52c44954f1ce28ff339e53f1f595dc06bf7f6131de6"
GRAPHS = ("encoder_model", "decoder_model", "decoder_with_past_model")


class AuditError(RuntimeError):
    pass


def checked_file(root: Path, name: str) -> Path:
    if Path(name).name != name or name in (".", ".."):
        raise AuditError("External-data location must be a filename inside the graph directory")
    path = root / name
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise AuditError(f"File is missing, linked, or outside its directory: {path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"Expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * MIB):
            digest.update(chunk)
    return digest.hexdigest()


def memory_guard() -> tuple[int, int]:
    previous = resource.getrlimit(resource.RLIMIT_AS)
    limit = min(CAP, previous[1]) if previous[1] != resource.RLIM_INFINITY else CAP
    vm_size = next(int(line.split()[1]) * 1024 for line in
                   Path("/proc/self/status").read_text(encoding="ascii").splitlines()
                   if line.startswith("VmSize:"))
    if vm_size >= limit:
        raise AuditError("Current virtual memory already exceeds the 512 MiB audit cap")
    resource.setrlimit(resource.RLIMIT_AS, (limit, previous[1]))
    return previous


def source_header(source: Path) -> tuple[int, dict[str, Any]]:
    with source.open("rb", buffering=0) as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise AuditError("Safetensors header length is missing")
        header_length = struct.unpack("<Q", length_bytes)[0]
        if not 0 < header_length <= 16 * MIB:
            raise AuditError("Safetensors header exceeds the bounded metadata limit")
        header = json.loads(stream.read(header_length))
    if not isinstance(header, dict):
        raise AuditError("Invalid safetensors header")
    tensors = {name: info for name, info in header.items() if name != "__metadata__"}
    for name, info in tensors.items():
        if not isinstance(info, dict) or not isinstance(info.get("data_offsets"), list) or (
            len(info["data_offsets"]) != 2
        ):
            raise AuditError(f"Invalid source tensor metadata: {name}")
    end = 0
    for name, info in sorted(tensors.items(), key=lambda item: item[1]["data_offsets"][0]):
        if not isinstance(name, str) or not isinstance(info, dict) or info.get("dtype") != "BF16":
            raise AuditError(f"Non-BF16 source tensor: {name}")
        shape, (start, stop) = info["shape"], info["data_offsets"]
        if not shape or any(not isinstance(x, int) or x <= 0 for x in shape):
            raise AuditError(f"Invalid source dimensions: {name}")
        size = 1
        for dim in shape:
            size *= dim
        if start != end or stop - start != size * 2:
            raise AuditError(f"Noncontiguous or wrong-length source tensor: {name}")
        end = stop
    if 8 + header_length + end != source.stat().st_size:
        raise AuditError("Safetensors payload has missing or extra bytes")
    return 8 + header_length, tensors


def samples(count: int) -> list[int]:
    return sorted({0, 1, 2, count // 11, count // 7, count // 3, count // 2,
                   2 * count // 3, 5 * count // 7, count - 3, count - 2, count - 1}
                  & set(range(count)))


def fingerprint_index(fd: int, base: int, tensors: dict[str, Any]
                      ) -> dict[tuple[tuple[int, ...], tuple[int, ...]], list[tuple[str, str]]]:
    index: dict[tuple[tuple[int, ...], tuple[int, ...]], list[tuple[str, str]]] = defaultdict(list)
    for name, info in tensors.items():
        dims = tuple(info["shape"])
        count = 1
        for dim in dims:
            count *= dim
        start = base + info["data_offsets"][0]
        for transpose in ((False, True) if len(dims) == 2 else (False,)):
            positions = (
                (i % dims[0]) * dims[1] + i // dims[0] if transpose else i
                for i in samples(count)
            )
            fingerprint = tuple(struct.unpack("<H", os.pread(fd, 2, start + 2 * p))[0] << 16
                                for p in positions)
            key = (dims[::-1] if transpose else dims, fingerprint)
            index[key].append((name, "transpose" if transpose else "direct"))
    return index


def compare(fd: int, base: int, info: dict[str, Any], graph_fd: int,
            offset: int, transform: str) -> tuple[int, int, list[int]]:
    import numpy as np

    dims = info["shape"]
    count = 1
    for dim in dims:
        count *= dim
    source_offset = base + info["data_offsets"][0]
    mismatches, first = 0, []
    if transform == "direct":
        for i in range(0, count, 524288):
            length = min(524288, count - i)
            source = np.frombuffer(os.pread(fd, 2 * length, source_offset + 2 * i), dtype="<u2")
            graph = np.frombuffer(os.pread(graph_fd, 4 * length, offset + 4 * i), dtype="<u4")
            if len(source) != length or len(graph) != length:
                raise AuditError("Short initializer read")
            different = (source.astype("<u4") << 16) != graph
            mismatches += int(np.count_nonzero(different))
            if np.any(different) and len(first) < 3:
                first.append(i + int(np.flatnonzero(different)[0]))
    else:
        rows, cols = dims
        page = mmap.ALLOCATIONGRANULARITY
        aligned = source_offset // page * page
        with mmap.mmap(fd, source_offset - aligned + 2 * count,
                       offset=aligned, access=mmap.ACCESS_READ) as mapped:
            source = np.ndarray((rows, cols), dtype="<u2", buffer=mapped,
                                offset=source_offset - aligned)
            step = max(1, min(cols, 1048576 // rows))
            for row in range(0, cols, step):
                length = min(cols - row, step)
                expected = source[:, row:row + length].T.astype("<u4") << 16
                graph = np.frombuffer(os.pread(graph_fd, rows * length * 4,
                                               offset + row * rows * 4), dtype="<u4")
                if len(graph) != rows * length:
                    raise AuditError("Short initializer read")
                different = expected.ravel() != graph
                mismatches += int(np.count_nonzero(different))
                if np.any(different) and len(first) < 3:
                    first.append(row * rows + int(np.flatnonzero(different)[0]))
                del expected, graph, different
            del source
    return count, mismatches, first


def verify_source(root: Path) -> tuple[Path, int, dict[str, Any], dict[str, Any]]:
    manifest = read_json(checked_file(root, ".shum-manifest.json"))
    if (manifest.get("repo"), manifest.get("revision")) != (REPO, REVISION):
        raise AuditError("Source manifest is not the pinned Flix revision")
    config = read_json(checked_file(root, "config.json"))
    if (config.get("model_type"), config.get("num_mel_bins"), config.get("dtype")) != (
        "whisper", 128, "bfloat16"
    ):
        raise AuditError("Source must be BF16 128-mel Whisper")
    source = checked_file(root, "model.safetensors")
    base, tensors = source_header(source)
    expected = manifest.get("files", {}).get("model.safetensors")
    if not isinstance(expected, str) or len(expected) != 64:
        raise AuditError("Pinned manifest lacks a model.safetensors digest")
    actual = sha256(source)
    return source, base, tensors, {
        "path": str(source), "byte_size": source.stat().st_size,
        "header_bytes": base - 8, "tensor_count": len(tensors), "dtype": "BF16",
        "revision": REVISION, "sha256": actual, "expected_manifest_sha256": expected,
        "manifest_hash_matches": actual == expected,
        "pinned_hash_matches": actual == PINNED_MODEL_SHA256,
    }


def graph_audit(graphs: Path, name: str, source_fd: int, base: int, tensors: dict[str, Any],
                index: dict[tuple[tuple[int, ...], tuple[int, ...]], list[tuple[str, str]]]
                ) -> tuple[dict[str, Any], set[str]]:
    import onnx

    graph_path = checked_file(graphs, name + ".onnx")
    data_path = checked_file(graphs, name + ".onnx_data")
    model = onnx.load(str(graph_path), load_external_data=False)
    records: list[dict[str, Any]] = []
    mapped: set[str] = set()
    transforms: Counter[str] = Counter()
    previous, seen = 0, set()
    graph_fd = os.open(data_path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        for initializer in model.graph.initializer:
            if initializer.name in seen:
                raise AuditError(f"Duplicate graph initializer name: {initializer.name}")
            seen.add(initializer.name)
            if initializer.data_type != onnx.TensorProto.FLOAT or (
                initializer.data_location != onnx.TensorProto.EXTERNAL
            ):
                raise AuditError(f"Initializer is not external FP32: {initializer.name}")
            fields = {field.key: field.value for field in initializer.external_data}
            dims = tuple(initializer.dims)
            count = 1
            for dim in dims:
                count *= dim
            if not dims or any(dim <= 0 for dim in dims):
                raise AuditError(f"Invalid initializer dimensions: {initializer.name}")
            offset, length = int(fields["offset"]), int(fields["length"])
            if (fields["location"] != data_path.name or offset != previous or
                    length != 4 * count or offset + length > data_path.stat().st_size):
                raise AuditError(f"External-data path/offset/length invalid: {initializer.name}")
            previous = offset + length
            fingerprint = tuple(struct.unpack("<I", os.pread(graph_fd, 4, offset + 4 * p))[0]
                                for p in samples(count))
            candidates = index.get((dims, fingerprint), [])
            named = (initializer.name if initializer.name in tensors else
                     "model.encoder." + initializer.name if name == "encoder_model" else None)
            if named in tensors and tuple(tensors[named]["shape"]) == dims:
                candidates = [(named, "direct")]
            matches = []
            checked = []
            for candidate, transform in candidates:
                size, errors, first = compare(source_fd, base, tensors[candidate],
                                              graph_fd, offset, transform)
                checked.append((candidate, transform, size, errors, first))
                if errors == 0:
                    matches.append((candidate, transform))
            record: dict[str, Any] = {
                "initializer": initializer.name, "shape": list(dims),
                "offset": offset, "length": length,
            }
            if len(matches) == 1 and len(checked) == 1:
                candidate, transform, size, _, _ = checked[0]
                record.update(source=candidate, transform=transform, compared_elements=size,
                              mismatched_elements=0, status="exact")
                mapped.add(candidate)
                transforms[transform] += 1
            elif len(matches) == 1 and len(checked) > 1:
                candidate, transform = matches[0]
                record.update(source=candidate, transform=transform,
                              compared_elements=count, mismatched_elements=0, status="exact")
                mapped.add(candidate)
                transforms[transform] += 1
            elif len(matches) > 1:
                record.update(status="ambiguous", candidate_sources=sorted({x[0] for x in matches}))
            elif len(checked) == 1:
                candidate, transform, size, errors, first = checked[0]
                record.update(source=candidate, transform=transform, compared_elements=size,
                              mismatched_elements=errors, first_mismatched_flat_indices=first,
                              status="mismatch")
                mapped.add(candidate)
            else:
                record.update(status="unmapped", candidate_count=len(checked))
            records.append(record)
        if previous != data_path.stat().st_size:
            raise AuditError(f"Unaccounted bytes at end of graph external data: {name}")
    finally:
        os.close(graph_fd)
    statuses = Counter(row["status"] for row in records)
    return {
        "onnx_path": str(graph_path), "onnx_sha256": sha256(graph_path),
        "external_data_path": str(data_path), "external_data_sha256": sha256(data_path),
        "external_data_bytes": data_path.stat().st_size,
        "initializer_count": len(records), "mapped_initializers": statuses["exact"] + statuses["mismatch"],
        "exact_initializers": statuses["exact"], "mismatched_initializers": statuses["mismatch"],
        "ambiguous_initializers": statuses["ambiguous"], "unmapped_initializers_count": statuses["unmapped"],
        "transforms": dict(transforms), "unique_source_parameters": len(mapped),
        "source_parameters_not_in_graph": sorted(set(tensors) - mapped),
        "unmapped_graph_initializers": [row for row in records if row["status"] == "unmapped"],
        "mismatches": [row for row in records if row["status"] == "mismatch"],
        "total_compared_elements": sum(row.get("compared_elements", 0) for row in records),
        "initializers": records,
    }, mapped


def audit(source: Path, graphs: Path) -> dict[str, Any]:
    if not source.is_absolute() or not source.is_dir() or not graphs.is_absolute() or not graphs.is_dir():
        raise AuditError("--source and --graphs must be existing absolute directories")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    previous = memory_guard()
    started = time.monotonic()
    try:
        import onnx
        import numpy

        model_file, base, tensors, source_report = verify_source(source)
        maximum_source_tensor = max(
            (item["data_offsets"][1] - item["data_offsets"][0] for item in tensors.values()),
            default=0,
        )
        result: dict[str, Any] = {
            "schema_version": 1, "purpose": "Initializer BF16-to-FP32 bitwise identity; not runtime parity",
            "source": source_report,
            "tool_versions": {"onnx": onnx.__version__, "numpy": numpy.__version__},
            "method": {
                "load_external_data": False, "memory_limit_mib": 512,
                "enforcement": "RLIMIT_AS <= 512 MiB; process virtual size caps its RSS",
                "max_graph_read_bytes": 4 * MIB, "max_source_read_bytes": 4 * MIB,
                "max_mapped_source_tensor_bytes": maximum_source_tensor,
                "source_mapping": "12-value direct/transpose fingerprint, followed by full bitwise comparison",
            },
            "graphs": {}, "mismatches": [],
        }
        source_fd = os.open(model_file, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            index = fingerprint_index(source_fd, base, tensors)
            mapped = set()
            exact_sources = set()
            for name in GRAPHS:
                graph_report, names = graph_audit(graphs, name, source_fd, base, tensors, index)
                result["graphs"][name] = graph_report
                mapped.update(names)
                exact_sources.update(row["source"] for row in graph_report["initializers"]
                                     if row["status"] == "exact")
                result["mismatches"].extend(dict(item, graph=name) for item in graph_report["mismatches"])
        finally:
            os.close(source_fd)
        result.update(
            source_parameters_total=len(tensors),
            unique_source_parameters_matched=len(mapped),
            unique_source_parameters_exact=len(exact_sources),
            unmapped_source_parameters=sorted(set(tensors) - mapped),
            peak_self_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            elapsed_seconds=round(time.monotonic() - started, 1),
        )
        result["exact_identity"] = (
            len(tensors) == 1259 and not result["unmapped_source_parameters"] and
            source_report["manifest_hash_matches"] and source_report["pinned_hash_matches"] and
            all(graph["initializer_count"] > 0 and
                graph["initializer_count"] == graph["exact_initializers"]
                for graph in result["graphs"].values()) and len(result["graphs"]) == len(GRAPHS)
        )
        result["scope_note"] = ("Initializer identity does not verify graph operators, "
                                "runtime arithmetic, decoding, or generation parity.")
        return result
    finally:
        resource.setrlimit(resource.RLIMIT_AS, previous)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Absolute pinned source directory")
    parser.add_argument("--graphs", type=Path, required=True, help="Absolute read-only ONNX graph directory")
    parser.add_argument("--report", type=Path, required=True, help="Absolute JSON destination outside input directories")
    args = parser.parse_args(argv)
    if not args.report.is_absolute() or not args.report.parent.is_dir() or args.report.is_symlink():
        parser.error("--report must be an absolute non-symlink path in an existing directory")
    if any(args.report.resolve().is_relative_to(path.resolve()) for path in (args.source, args.graphs)):
        parser.error("--report must be outside source and graphs")
    previous = None
    try:
        previous = memory_guard()
        try:
            result = audit(args.source, args.graphs)
        except (AuditError, OSError, ValueError, KeyError, MemoryError) as exc:
            result = {"schema_version": 1, "exact_identity": False,
                      "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        result["peak_self_rss_mib"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1,
        )
        with args.report.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except (AuditError, OSError, ValueError, MemoryError) as exc:
        if previous is not None:
            raise
        result = {"schema_version": 1, "exact_identity": False, "status": "error",
                  "error": f"{type(exc).__name__}: {exc}"}
        with args.report.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    finally:
        if previous is not None:
            resource.setrlimit(resource.RLIMIT_AS, previous)
    print(f"{'exact' if result['exact_identity'] else 'not_verified'}: {args.report}")
    return 0 if result["exact_identity"] else 2


if __name__ == "__main__":
    sys.exit(main())
