#!/usr/bin/env python3
"""Offline, opt-in split FP32 ONNX export of the pinned Swiss German Turbo source.

Dry-run validates the source without importing model packages. --run exports
the encoder, decoder and cached decoder in separate bounded processes.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = "gcoli/whisper-large-v3-turbo-swiss-german-mit"
REVISION = "540fb2e7a9135048963c5de9822579c96f8962e5"
REQUIRED = ("config.json", "generation_config.json", "preprocessor_config.json",
            "tokenizer.json", "tokenizer_config.json", "model.safetensors", "README.md")
GIB = 1024 ** 3
STAGE_BUDGET = 9 * GIB
RESERVE = 4 * GIB
SCOPE_LIMIT = 24 * GIB
STAGES = ("encoder", "decoder", "decoder-past")
STAGE_NAMES = {"encoder": "encoder_model", "decoder": "decoder_model",
               "decoder-past": "decoder_with_past_model"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(source: Path) -> dict:
    if not source.is_absolute() or not source.is_dir():
        raise ValueError("--source must be an existing absolute directory")
    api = json.loads((source / ".hf-api.json").read_text())
    if api.get("sha") != REVISION or api.get("private") or api.get("gated"):
        raise ValueError("Hub metadata does not identify the public pinned revision")
    artifacts = {entry["rfilename"]: entry for entry in api["siblings"]}
    if any(name not in artifacts for name in REQUIRED):
        raise ValueError("Pinned revision is missing a required source artifact")
    hashes = {}
    for name, entry in artifacts.items():
        if name.startswith("ct2/"):
            continue
        path = source / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(source):
            raise ValueError(f"Missing or linked source artifact: {name}")
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"Incorrect size for source artifact: {name}")
        digest = sha256(path)
        if entry.get("lfs") and digest != entry["lfs"]["sha256"]:
            raise ValueError(f"LFS SHA-256 mismatch: {name}")
        if not entry.get("lfs") and git_blob_sha1(path) != entry["blobId"]:
            raise ValueError(f"Git blob SHA-1 mismatch: {name}")
        hashes[name] = digest
    config = json.loads((source / "config.json").read_text())
    generation = json.loads((source / "generation_config.json").read_text())
    feature = json.loads((source / "preprocessor_config.json").read_text())
    if (config.get("model_type"), config.get("num_mel_bins"), config.get("dtype")) != (
            "whisper", 128, "float16"):
        raise ValueError("Unexpected Turbo model architecture/precision")
    if (generation.get("language"), generation.get("task")) != ("german", "transcribe"):
        raise ValueError("Unexpected Turbo generation policy")
    if (feature.get("feature_size"), feature.get("sampling_rate")) != (128, 16000):
        raise ValueError("Unexpected Turbo feature extractor")
    return {"repo": REPO, "revision": REVISION, "source_sha256": hashes,
            "source_path": str(source)}


def available_bytes(*, exclude_own_cgroup: bool = False) -> int:
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
                if ":" in line)
    available = int(info["MemAvailable"].strip().split()[0]) * 1024
    membership = next((line.split("::", 1)[1] for line in
                       Path("/proc/self/cgroup").read_text().splitlines()
                       if line.startswith("0::")), None)
    if membership is None:
        raise RuntimeError("Cannot inspect cgroup v2 headroom")
    path = Path("/sys/fs/cgroup") / membership.lstrip("/")
    root = Path("/sys/fs/cgroup")
    own_path = path
    while path.is_relative_to(root):
        if path == root:
            break
        if path != own_path or not exclude_own_cgroup:
            limit = (path / "memory.max").read_text().strip()
            if limit != "max":
                available = min(available, int(limit) - int((path / "memory.current").read_text()))
        path = path.parent
    return available


def own_cgroup() -> Path:
    membership = next((line.split("::", 1)[1] for line in
                       Path("/proc/self/cgroup").read_text().splitlines()
                       if line.startswith("0::")), None)
    if membership is None:
        raise RuntimeError("cgroup v2 is required for a bounded export")
    path = Path("/sys/fs/cgroup") / membership.lstrip("/")
    try:
        limit = (path / "memory.max").read_text().strip()
        swap = int((path / "memory.swap.max").read_text().strip())
    except (OSError, ValueError) as exc:
        raise RuntimeError("Cannot inspect the kernel cgroup limits") from exc
    if limit == "max" or int(limit) > SCOPE_LIMIT or swap != 0:
        raise RuntimeError("Run in a cgroup with MemoryMax <=24 GiB and MemorySwapMax=0")
    return path


def pressure_sample(group: Path) -> dict:
    vmstat = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    psi = Path("/proc/pressure/memory").read_text().splitlines()
    full = next(line for line in psi if line.startswith("full "))
    events = dict(line.split() for line in (group / "memory.events").read_text().splitlines())
    return {"available": available_bytes(), "pswpout": int(vmstat["pswpout"]),
            "psi_full": float(next(part.split("=")[1] for part in full.split()
                                   if part.startswith("avg10="))),
            "swap": int((group / "memory.swap.current").read_text()),
            "scope_current": int((group / "memory.current").read_text()),
            "oom_kill": int(events["oom_kill"])}


def guarded_stage(source: Path, output: Path, stage: str, group: Path,
                  env: dict, report_path: Path,
                  child_command: list[str] | None = None, deadline_seconds: float = 900) -> None:
    if available_bytes() < STAGE_BUDGET + RESERVE:
        raise RuntimeError(f"{stage}: 9 GiB stage budget plus 4 GiB reserve unavailable")
    start = pressure_sample(group)
    began = time.monotonic()
    command = child_command or [sys.executable, str(Path(__file__).resolve()), "--source", str(source),
                                "--output", str(output), "--execute-stage", stage]
    process = subprocess.Popen(command, env=env)
    report = {"stage": stage, "status": "blocked", "source": str(source), "output": str(output),
              "command": command, "pid": process.pid,
              "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "initial": start, "minimum_available_bytes": start["available"],
              "maximum_scope_growth_bytes": 0, "maximum_psi_full_avg10_percent": start["psi_full"]}
    try:
        while process.poll() is None:
            time.sleep(0.5)
            if time.monotonic() - began > deadline_seconds:
                raise RuntimeError(f"{stage}: supervised stage deadline ({deadline_seconds:.1f}s)")
            current = pressure_sample(group)
            report["minimum_available_bytes"] = min(report["minimum_available_bytes"], current["available"])
            report["maximum_scope_growth_bytes"] = max(
                report["maximum_scope_growth_bytes"], current["scope_current"] - start["scope_current"])
            report["maximum_psi_full_avg10_percent"] = max(
                report["maximum_psi_full_avg10_percent"], current["psi_full"])
            report["last"] = current
            if (current["available"] < RESERVE or current["swap"]
                    or current["scope_current"] - start["scope_current"] > STAGE_BUDGET
                    or current["oom_kill"] > start["oom_kill"]
                    or (current["pswpout"] > start["pswpout"] and current["psi_full"] >= 1.0)):
                raise RuntimeError(f"{stage}: memory reserve/swap/PSI/OOM stop: {current}")
        if process.returncode:
            raise RuntimeError(f"{stage}: stage exited {process.returncode}")
        report["status"] = "export_process_complete_unverified"
    except BaseException as exc:
        report["stop_reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        report["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report["elapsed_seconds"] = round(time.monotonic() - began, 3)
        report["exit_code"] = process.returncode
        report["child_exited"] = process.poll() is not None
        report_path.write_text(json.dumps(report, indent=2) + "\n")


def stage_files(output: Path, stage: str) -> dict[str, str]:
    name = STAGE_NAMES[stage]
    folder = output / stage
    graph = folder / (name + ".onnx")
    if not graph.is_file() or graph.is_symlink() or graph.stat().st_size < 1024:
        raise RuntimeError(f"Missing or invalid {stage} graph")
    files = [file for file in folder.iterdir() if file.is_file()]
    if any(file.is_symlink() for file in files):
        raise RuntimeError("Linked graph data is not allowed")
    return {str(file.relative_to(output)): sha256(file) for file in files}


def prepared_report(output: Path, manifest: dict) -> dict:
    report = json.loads((output / ".shum-turbo-onnx-prepared.json").read_text())
    if report.get("source") != manifest or report.get("export_dtype") != "fp32":
        raise RuntimeError("Prepared source identity differs from the current source")
    return report


def finish_stage(output: Path, stage: str) -> None:
    marker = output / f".stage-{stage}.json"
    if marker.exists():
        raise RuntimeError(f"{stage} stage already finished")
    marker.write_text(json.dumps(stage_files(output, stage), indent=2) + "\n")


def finalize(source: Path, output: Path, manifest: dict) -> None:
    report = prepared_report(output, manifest)
    for stage in STAGES:
        expected = json.loads((output / f".stage-{stage}.json").read_text())
        if stage_files(output, stage) != expected:
            raise RuntimeError(f"{stage} graph differs from its completed stage")
    config = json.loads((source / "config.json").read_text())
    config["dtype"] = "float32"
    config.pop("torch_dtype", None)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in ("generation_config.json", "tokenizer.json", "tokenizer_config.json",
                 "preprocessor_config.json", "special_tokens_map.json",
                 "added_tokens.json", "vocab.json", "merges.txt", "normalizer.json"):
        shutil.copyfile(source / name, output / name)
    report["graphs_sha256"] = {
        str(file.relative_to(output)): sha256(file)
        for file in sorted(output.rglob("*")) if file.is_file()
    }
    report["complete"] = True
    (output / ".shum-turbo-onnx.json").write_text(json.dumps(report, indent=2) + "\n")


def export_stage(source: Path, output: Path, stage: str) -> None:
    import torch
    import optimum.exporters.onnx.model_configs  # noqa: F401 - registers Whisper's ONNX task
    from safetensors import safe_open
    from transformers import WhisperConfig, WhisperForConditionalGeneration
    from optimum.exporters.tasks import TasksManager
    from optimum.exporters.utils import DisableCompileContextManager, get_encoder_decoder_models_for_export
    from optimum.exporters.utils import check_dummy_inputs_are_allowed

    torch.set_num_threads(1)
    task = "automatic-speech-recognition-with-past"
    model_config = WhisperConfig.from_pretrained(str(source), local_files_only=True)
    with torch.device("meta"):
        model = WhisperForConditionalGeneration(model_config)
    prefix = "model.encoder." if stage == "encoder" else "model.decoder."
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as weights:
        keys = [name for name in weights.keys() if name.startswith(prefix)]
        expected = {name for name, _ in model.named_parameters() if name.startswith(prefix)}
        if set(keys) != expected:
            raise RuntimeError(f"{stage}: source/model parameter keys differ")
        for name in keys:
            owner_name, param_name = name.rsplit(".", 1)
            owner = model.get_submodule(owner_name)
            value = weights.get_tensor(name).to(dtype=torch.float32)
            if tuple(value.shape) != tuple(owner._parameters[param_name].shape):
                raise RuntimeError(f"{stage}: mismatched tensor shape: {name}")
            owner._parameters[param_name] = torch.nn.Parameter(value, requires_grad=False)
    if stage != "encoder":
        model.proj_out.weight = model.model.decoder.embed_tokens.weight
        if model.proj_out.weight.device.type != "cpu":
            raise RuntimeError("Decoder projection was not materialized")
    if any(param.device.type != "cpu" for name, param in model.named_parameters()
           if name.startswith(prefix)):
        raise RuntimeError(f"{stage}: a required parameter remains on meta")
    model.config.use_cache = True
    constructor = TasksManager.get_exporter_config_constructor(
        model=model, exporter="onnx", task=task, library_name="transformers")
    config = constructor(model.config, float_dtype="fp32")
    submodels = get_encoder_decoder_models_for_export(model, config)
    key = STAGE_NAMES[stage]
    if set(submodels) != set(STAGE_NAMES.values()):
        raise RuntimeError(f"Unexpected Optimum cached submodels: {list(submodels)}")
    submodel, onnx_config = submodels[key]
    submodel.eval()
    submodel.config.return_dict = True
    for name, value in (onnx_config.values_override or {}).items():
        setattr(submodel.config, name, value)
    dummy = onnx_config.rename_ambiguous_inputs(
        onnx_config.generate_dummy_inputs(framework="pt"))
    folder = output / stage
    folder.mkdir(mode=0o700)
    destination = folder / (key + ".onnx")
    with torch.no_grad(), DisableCompileContextManager(), onnx_config.patch_model_for_export(submodel):
        check_dummy_inputs_are_allowed(submodel, dummy)
        axes = dict(list(onnx_config.ordered_inputs(submodel).items())
                    + list(onnx_config.outputs.items()))
        torch.onnx.export(
            submodel, (dummy,), f=str(destination),
            input_names=list(onnx_config.ordered_inputs(submodel)),
            output_names=list(onnx_config.outputs),
            dynamic_axes=axes, do_constant_folding=False,
            opset_version=onnx_config.DEFAULT_ONNX_OPSET, dynamo=False,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="Actually export; default only validates source")
    parser.add_argument("--prepare", action="store_true", help="Create a verified empty export directory")
    parser.add_argument("--stage", choices=STAGES, help="Run one monitored export stage")
    parser.add_argument("--finalize", action="store_true", help="Verify both stages and write provenance")
    parser.add_argument("--report", type=Path, help="Unused absolute JSON report path for --stage")
    parser.add_argument("--execute-stage", choices=STAGES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sum((args.run, args.prepare, bool(args.stage), args.finalize, bool(args.execute_stage))) > 1:
        parser.error("Select one of --run, --prepare, --stage, --finalize")
    if args.report and (not args.stage or not args.report.is_absolute()
                        or args.report.is_symlink() or args.report.exists()):
        parser.error("--report requires --stage and an unused absolute JSON path")
    source, output = args.source.resolve(), args.output
    if not output.is_absolute() or output.is_symlink():
        parser.error("--output must be an absolute non-symlink path")
    if (args.prepare or args.run or not any((args.stage, args.finalize, args.execute_stage))) and output.exists():
        parser.error("--output must be an unused absolute path")
    if output.resolve().is_relative_to(source) or source.is_relative_to(output.resolve()):
        parser.error("Output and source must be separate directories")
    manifest = source_manifest(source)
    # Header-derived FP32 weights: encoder 2.373 GiB, decoder 0.640 GiB.
    # Selective loading avoids materializing the other branch in each stage;
    # Encoder weights, traced graph and serialized weights are about 2.373
    # GiB each; 9 GiB leaves about 1.88 GiB for tracing and process overhead.
    needed = STAGE_BUDGET + RESERVE
    report = {"source": manifest, "export_dtype": "fp32", "cache": False,
              "export_strategy": "selective_fp32_encoder_decoder_decoder_past_no_folding",
              "estimated_needed_bytes": needed, "available_bytes": available_bytes()}
    if not any((args.run, args.prepare, args.stage, args.finalize, args.execute_stage)):
        print(json.dumps(report, indent=2), flush=True)
        return 0
    if args.prepare or args.run:
        for package in ("optimum-onnx", "transformers", "torch", "onnx", "onnxruntime"):
            report.setdefault("versions", {})[package] = importlib.metadata.version(package)
        output.mkdir(mode=0o700, parents=True)
        (output / ".shum-turbo-onnx-prepared.json").write_text(json.dumps(report, indent=2) + "\n")
        if args.prepare:
            return 0
    else:
        prepared_report(output, manifest)
    if args.finalize:
        finalize(source, output, manifest)
        return 0
    group = own_cgroup()
    if report["available_bytes"] < needed:
        raise RuntimeError("Insufficient headroom for split FP32 export plus 4 GiB reserve")
    if args.execute_stage:
        export_stage(source, output, args.execute_stage)
        return 0
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1", "OMP_NUM_THREADS": "1"})
    try:
        stages = (args.stage,) if args.stage else STAGES
        for stage in stages:
            if (output / f".stage-{stage}.json").exists() or (output / stage).exists():
                raise RuntimeError(f"{stage} graph already exists; refusing to overwrite")
            report_path = args.report if args.stage else output / f".attempt-{stage}.json"
            guarded_stage(source, output, stage, group, env, report_path)
            finish_stage(output, stage)
            stage_report = json.loads(report_path.read_text())
            stage_report["status"] = "complete"
            stage_report["graph_sha256"] = stage_files(output, stage)
            report_path.write_text(json.dumps(stage_report, indent=2) + "\n")
        if args.run:
            finalize(source, output, manifest)
    except BaseException:
        print(f"Export incomplete; do not use {output}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Turbo ONNX export blocked: {exc}", file=sys.stderr)
        sys.exit(2)
