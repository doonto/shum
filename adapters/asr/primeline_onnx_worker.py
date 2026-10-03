#!/usr/bin/env python3
"""Offline primeLine FP32 ONNX worker using the public onnx-asr TDT decoder."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
import wave
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

import main as common

MANIFEST = ".shum-primeline-onnx.json"
SOURCE_MANIFEST = ".shum-manifest.json"
SOURCE_REPO = "primeline/parakeet-primeline"
SOURCE_REVISION = "3f1a9bcb611dfeda53fe74fe5f1a3d5701e8023e"
CONFIG = {
    "model_type": "nemo-conformer-tdt",
    "features_size": 128,
    "subsampling_factor": 8,
    "max_tokens_per_step": 10,
}
REQUIRED = {"encoder-model.onnx", "decoder_joint-model.onnx", "vocab.txt", "config.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_file(root: Path, name: str) -> Path:
    if (not isinstance(name, str) or not name or Path(name).name != name
            or name in (".", "..")):
        raise common.AdapterError(f"unsafe artifact filename: {name!r}")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise common.AdapterError(f"missing or symlinked artifact: {path}")
    return path


def source_checkpoint(model: str) -> Path:
    directory = Path(model)
    if not directory.is_absolute() or directory.is_symlink() or not directory.is_dir():
        raise common.AdapterError("--model must be an existing absolute checkpoint directory")
    return common.find_nemo_checkpoint_file(directory)


def source_provenance(source: Path) -> dict[str, str]:
    root = source.parent
    try:
        path = regular_file(root, SOURCE_MANIFEST)
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise common.AdapterError(f"invalid source model manifest: {exc}") from exc
    files = manifest.get("files")
    if (manifest.get("repo") != SOURCE_REPO or manifest.get("revision") != SOURCE_REVISION
            or not isinstance(files, dict) or source.name not in files
            or not all(isinstance(name, str) and name and
                       all(part not in ("", ".", "..") for part in name.split("/")) and
                       "\\" not in name and not name.startswith("/")
                       and isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
                       for name, digest in files.items())):
        raise common.AdapterError("source model manifest has wrong repo/revision/files schema")
    actual = {}
    for current, directories, names in os.walk(root, followlinks=False):
        for name in directories + names:
            item = Path(current) / name
            if item.is_symlink():
                raise common.AdapterError(f"symlink in source model: {item}")
        for name in names:
            item = Path(current) / name
            if item.name == SOURCE_MANIFEST and item.parent == root:
                continue
            relative = item.relative_to(root).as_posix()
            actual[relative] = sha256(item)
    if actual != files:
        raise common.AdapterError("source model files do not match Rust model manifest")
    return {
        "repo": manifest["repo"],
        "revision": manifest["revision"],
        "checkpoint_sha256": actual[source.name],
        "manifest_sha256": sha256(path),
    }


def artifact_dir(source: Path, explicit: str | None) -> Path:
    if explicit is not None:
        directory = Path(explicit)
    else:
        pointer = source.parent.parent / MANIFEST
        if pointer.is_symlink() or not pointer.is_file():
            raise common.AdapterError(f"--onnx-model required unless {pointer} is present")
        try:
            value = json.loads(pointer.read_text(encoding="utf-8"))["artifact_dir"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise common.AdapterError(f"invalid model-relative ONNX pointer: {exc}") from exc
        if (not isinstance(value, str) or not value or
                any(part in (".", "..") for part in Path(value).parts)):
            raise common.AdapterError("model-relative artifact_dir must not traverse directories")
        directory = Path(value) if Path(value).is_absolute() else pointer.parent / value
    if not directory.is_absolute() or directory.is_symlink() or not directory.is_dir():
        raise common.AdapterError("--onnx-model must resolve to an existing absolute artifact directory")
    return directory


def export_audit(directory: Path) -> tuple[dict[str, Any], dict[str, int]]:
    try:
        audit = json.loads(regular_file(directory, "export-check.json").read_text(encoding="utf-8"))
        config = json.loads(regular_file(directory, "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise common.AdapterError(f"invalid ONNX export audit/config: {exc}") from exc
    if config != CONFIG:
        raise common.AdapterError("ONNX export is not the measured FP32 primeLine TDT configuration")
    if audit.get("checkpoint_float_dtypes") != {"torch.float32": 701}:
        raise common.AdapterError("export source is not the audited FP32 checkpoint")
    dtypes = audit.get("onnx_initializer_dtypes")
    if (not isinstance(dtypes, dict) or set(dtypes) != REQUIRED.intersection({
            "encoder-model.onnx", "decoder_joint-model.onnx"})
            or any(not isinstance(counts, dict) or not counts
                   or any(key != "FLOAT" for key in counts) for counts in dtypes.values())):
        raise common.AdapterError("ONNX graphs are not the audited FP32 export")
    inventory = audit.get("files")
    if (not isinstance(inventory, dict) or not REQUIRED <= set(inventory)
            or any(type(size) is not int or size <= 0 for size in inventory.values())):
        raise common.AdapterError("incomplete ONNX export inventory")
    for name, size in inventory.items():
        if regular_file(directory, name).stat().st_size != size:
            raise common.AdapterError(f"ONNX export size mismatch: {name}")
    return audit, inventory


def create_manifest(source: Path, directory: Path) -> dict[str, Any]:
    audit, inventory = export_audit(directory)
    provenance = source_provenance(source)
    if provenance["checkpoint_sha256"] != audit.get("source_sha256"):
        raise common.AdapterError("ONNX export/source checkpoint mismatch")
    return {
        "version": 1,
        "source": provenance,
        "export_audit_sha256": sha256(regular_file(directory, "export-check.json")),
        "artifact_sha256": {name: sha256(regular_file(directory, name))
                            for name in sorted(inventory)},
    }


def verify_artifacts(source: Path, directory: Path) -> None:
    try:
        manifest = json.loads(regular_file(directory, MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise common.AdapterError(f"missing or invalid ONNX provenance: {exc}") from exc
    audit, inventory = export_audit(directory)
    if (manifest.get("version") != 1
            or not isinstance(manifest.get("source"), dict)
            or manifest["source"].get("checkpoint_sha256") != audit.get("source_sha256")
            or manifest.get("export_audit_sha256") != sha256(regular_file(directory, "export-check.json"))
            or not isinstance(manifest.get("artifact_sha256"), dict)
            or set(manifest["artifact_sha256"]) != set(inventory)):
        raise common.AdapterError("ONNX export manifest does not match source/audit/inventory")
    if source_provenance(source) != manifest["source"]:
        raise common.AdapterError("ONNX export/source manifest mismatch")
    for name, expected in manifest["artifact_sha256"].items():
        if not isinstance(expected, str) or sha256(regular_file(directory, name)) != expected:
            raise common.AdapterError(f"ONNX artifact hash mismatch: {name}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=("nemo",))
    parser.add_argument("--model", required=True)
    parser.add_argument("--onnx-model", help="Absolute directory of audited FP32 ONNX graphs")
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--print-manifest", action="store_true",
                        help="Print a provenance manifest for an audited export; do not load the model")
    args = parser.parse_args(argv)
    if args.print_manifest:
        if args.serve or args.input or args.output or not args.onnx_model:
            parser.error("--print-manifest requires --onnx-model and no clip/serve arguments")
    elif args.serve:
        if args.input or args.output:
            parser.error("--serve cannot be combined with --input/--output")
    elif not args.input or not args.output:
        parser.error("single-clip mode requires --input and --output")
    return args


def load_model(directory: Path, threads: int) -> Any:
    try:
        import onnx_asr
        import onnxruntime as ort
    except ImportError as exc:
        raise common.AdapterError("Install onnx-asr and onnxruntime in an isolated CPU environment") from exc
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    try:
        model = onnx_asr.load_model(
            "nemo-conformer-tdt", path=directory, quantization=None,
            providers=["CPUExecutionProvider"], sess_options=options,
        ).with_timestamps()
        if (model.asr._encoder.get_providers() != ["CPUExecutionProvider"]
                or model.asr._decoder_joint.get_providers() != ["CPUExecutionProvider"]):
            raise common.AdapterError("ONNX graphs must run on CPUExecutionProvider only")
        return model
    except common.AdapterError:
        raise
    except Exception as exc:
        raise common.AdapterError(f"ONNX model load failed: {exc}") from exc


def transcribe(model: Any, input_path: str, duration: float) -> dict[str, Any]:
    try:
        result = model.recognize(Path(input_path))
        text = result.text
        if not isinstance(text, str):
            raise ValueError("decoder returned no text")
        return common.build_result(text, None, [
            {"start": 0.0, "end": duration, "text": text}
        ] if text else [])
    except Exception as exc:
        raise common.AdapterError(f"ONNX transcription failed for {input_path!r}: {exc}") from exc


def validate_clip(path: str) -> tuple[Path, float]:
    input_path, duration = common.validate_input_wav(path)
    try:
        with wave.open(str(input_path), "rb") as wav:
            if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
                raise common.AdapterError("input must be uncompressed mono PCM16 WAV")
            if not wav.getnframes():
                raise common.AdapterError("input WAV is empty")
    except wave.Error as exc:
        raise common.AdapterError(f"invalid input WAV: {exc}") from exc
    return input_path, duration


def serve(model: Any, stdin: TextIO, stdout: TextIO) -> int:
    for index, raw in enumerate(stdin):
        if not raw.strip():
            continue
        try:
            request = json.loads(raw)
            if not isinstance(request, dict) or not isinstance(request.get("input"), str):
                raise common.AdapterError("request must be a JSON object with string 'input'")
            path, duration = validate_clip(request["input"])
            with contextlib.redirect_stdout(sys.stderr):
                response = transcribe(model, str(path), duration)
        except Exception as exc:
            error = f"request[{index}]: {exc}"
            print(error, file=sys.stderr)
            stdout.write(json.dumps({"error": error}, ensure_ascii=False) + "\n")
            stdout.flush()
            return 1
        stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        stdout.flush()
    return 0


def run(argv: Sequence[str] | None = None, *,
        factory: Callable[[Path, int], Any] = load_model,
        stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    args = parse_args(argv)
    try:
        threads = common.validate_threads(args.threads)
        source = source_checkpoint(args.model)
        directory = artifact_dir(source, args.onnx_model)
        if args.print_manifest:
            (stdout or sys.stdout).write(json.dumps(create_manifest(source, directory), indent=2) + "\n")
            return 0
        common.configure_offline_environment()
        common.configure_thread_environment(threads)
        verify_artifacts(source, directory)
        if not args.serve:
            path, duration = validate_clip(args.input)
            output = common.validate_output_path(args.output)
        with contextlib.redirect_stdout(sys.stderr):
            model = factory(directory, threads)
        if args.serve:
            return serve(model, stdin or sys.stdin, stdout or sys.stdout)
        with contextlib.redirect_stdout(sys.stderr):
            result = transcribe(model, str(path), duration)
        common.write_result(result, output)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        if args.serve:
            stream = stdout or sys.stdout
            stream.write(json.dumps({"error": str(exc)}, ensure_ascii=False) + "\n")
            stream.flush()
        return 1


if __name__ == "__main__":
    sys.exit(run())
