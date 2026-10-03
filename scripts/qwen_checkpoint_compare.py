#!/usr/bin/env python3
"""Bounded bitwise parameter audit: original Qwen3 ASR versus HF packaging.

This checks source weights, not quantized graph weights or runtime equivalence.
No model loading, inference, network calls or application test suite.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def header(path):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < size <= 16 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header size: {path}")
        value = json.loads(stream.read(size))
    return 8 + size, {key: data for key, data in value.items() if key != "__metadata__"}


def hf_name(name):
    for old, new in (
        ("thinker.audio_tower.proj1.", "model.multi_modal_projector.linear_1."),
        ("thinker.audio_tower.proj2.", "model.multi_modal_projector.linear_2."),
        ("thinker.audio_tower.", "model.audio_tower."),
        ("thinker.model.", "model.language_model."),
    ):
        if name.startswith(old):
            return new + name[len(old):]
    if name == "thinker.lm_head.weight":
        return "model.language_model.embed_tokens.weight"
    raise ValueError(f"unmapped original tensor: {name}")


def tensor_hash(path, base, data):
    start, end = data["data_offsets"]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(base + start)
        remaining = end - start
        while remaining:
            chunk = stream.read(min(remaining, 2 * 1024 * 1024))
            if not chunk:
                raise ValueError(f"truncated tensor: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite existing audit")
    originals = {}
    for path in sorted(args.original.glob("*.safetensors")):
        base, tensors = header(path)
        for name, data in tensors.items():
            if name in originals:
                raise ValueError(f"duplicate original tensor: {name}")
            originals[name] = (path, base, data)
    hf_path = args.hf / "model.safetensors"
    base, tensors = header(hf_path)
    matches, mismatches, used = [], [], set()
    for name, (path, original_base, data) in sorted(originals.items()):
        counterpart = hf_name(name)
        other = tensors.get(counterpart)
        if other is None or data["dtype"] != other["dtype"] or data["shape"] != other["shape"]:
            mismatches.append({"original": name, "hf": counterpart, "reason": "shape/dtype/missing"})
            continue
        a = tensor_hash(path, original_base, data)
        b = tensor_hash(hf_path, base, other)
        used.add(counterpart)
        item = {"original": name, "hf": counterpart, "dtype": data["dtype"],
                "shape": data["shape"], "original_sha256": a, "hf_sha256": b}
        (matches if a == b else mismatches).append(item)
    missing = sorted(set(tensors) - used)
    passed = bool(originals) and not mismatches and not missing
    report = {
        "status": "bitwise_parameters_match" if passed else "mismatch",
        "original": str(args.original), "hf": str(args.hf),
        "original_tensor_count": len(originals), "hf_tensor_count": len(tensors),
        "matched_original_tensors": len(matches), "mismatches": mismatches,
        "unmapped_hf_tensors": missing, "parameters": matches,
        "caveat": "Source packaging audit only; does not audit graph quantization or runtime/preprocessing equivalence.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "parameters"}))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
