#!/usr/bin/env python3
"""Export pinned Turbo as cached FP32 ONNX graphs with streamed external weights.

Public Torch/Optimum ONNX configs generate parameterless graphs. Separate
processes attach source-mapped FP32 tensors to those graphs as external data.
No graph is accepted without exact name, shape, dtype and source checks.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import turbo_onnx_export as base

STRATEGY = "streamed_external_fp32_cached_no_constant_folding"


def structure(source: Path, output: Path, stage: str) -> None:
    import torch
    import onnx
    import optimum.exporters.onnx.model_configs  # noqa: F401 - registers Whisper
    from safetensors import safe_open
    from transformers import WhisperConfig, WhisperForConditionalGeneration
    from optimum.exporters.tasks import TasksManager
    from optimum.exporters.utils import (DisableCompileContextManager,
                                         check_dummy_inputs_are_allowed,
                                         get_encoder_decoder_models_for_export)

    torch.set_num_threads(1)
    config = WhisperConfig.from_pretrained(str(source), local_files_only=True)
    with torch.device("meta"):
        model = WhisperForConditionalGeneration(config)
    prefix = "model.encoder." if stage == "encoder" else "model.decoder."
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as weights:
        keys = [key for key in weights.keys() if key.startswith(prefix)]
        expected = {key for key, _ in model.named_parameters() if key.startswith(prefix)}
        if set(keys) != expected:
            raise ValueError(f"{stage}: source/model parameter inventory differs")
        for key in keys:
            owner_name, name = key.rsplit(".", 1)
            owner = model.get_submodule(owner_name)
            value = weights.get_tensor(key).to(torch.float32)
            if tuple(value.shape) != tuple(owner._parameters[name].shape):
                raise ValueError(f"{stage}: incompatible weight shape: {key}")
            owner._parameters[name] = torch.nn.Parameter(value, requires_grad=False)
    if stage != "encoder":
        model.proj_out.weight = model.model.decoder.embed_tokens.weight
        if model.proj_out.weight is not model.model.decoder.embed_tokens.weight:
            raise ValueError("Decoder LM head is not tied to source embedding")
    task = "automatic-speech-recognition-with-past"
    model.config.use_cache = True
    constructor = TasksManager.get_exporter_config_constructor(
        model=model, exporter="onnx", task=task, library_name="transformers")
    export_config = constructor(model.config, float_dtype="fp32")
    submodels = get_encoder_decoder_models_for_export(model, export_config)
    if set(submodels) != set(base.STAGE_NAMES.values()):
        raise ValueError("Unexpected cached Whisper submodel inventory")
    name = base.STAGE_NAMES[stage]
    submodel, onnx_config = submodels[name]
    submodel.eval()
    submodel.config.return_dict = True
    for key, value in (onnx_config.values_override or {}).items():
        setattr(submodel.config, key, value)
    dummy = onnx_config.rename_ambiguous_inputs(
        onnx_config.generate_dummy_inputs(framework="pt"))
    runtime_names = list(onnx_config.ordered_inputs(submodel))
    folder = output / stage
    folder.mkdir(mode=0o700)
    path = folder / (name + ".structure.onnx")
    with torch.no_grad(), DisableCompileContextManager(), onnx_config.patch_model_for_export(submodel):
        check_dummy_inputs_are_allowed(submodel, dummy)
        torch.onnx.export(
            submodel, (dummy,), str(path),
            input_names=runtime_names, output_names=list(onnx_config.outputs),
            dynamic_axes=dict(list(onnx_config.ordered_inputs(submodel).items())
                              + list(onnx_config.outputs.items())),
            do_constant_folding=False, export_params=False,
            keep_initializers_as_inputs=True,
            opset_version=onnx_config.DEFAULT_ONNX_OPSET, dynamo=False,
        )
    graph = onnx.load(str(path), load_external_data=False).graph
    if graph.initializer:
        raise ValueError("Structure-only export unexpectedly contains initializers")
    params = dict(submodel.named_parameters())
    runtime = {item.name for item in graph.input if item.name not in params}
    missing = set(runtime_names) - runtime
    permitted_missing = {"encoder_hidden_states"} if stage == "decoder-past" else set()
    if not runtime.issubset(runtime_names) or missing != permitted_missing:
        raise ValueError(f"Unmapped runtime input names: {sorted(runtime ^ set(runtime_names))}")
    source_map = {}
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as weights:
        for item in graph.input:
            if item.name in runtime:
                continue
            src = (prefix + item.name if stage == "encoder" else item.name)
            if item.name == "proj_out.weight" and stage != "encoder":
                src = "model.decoder.embed_tokens.weight"
            if item.name not in params or src not in weights.keys():
                raise ValueError(f"Unknown graph weight input: {item.name}")
            shape = list(weights.get_slice(src).get_shape())
            tensor = item.type.tensor_type
            graph_shape = [dim.dim_value for dim in tensor.shape.dim]
            if (weights.get_slice(src).get_dtype() != "F16"
                    or tensor.elem_type != onnx.TensorProto.FLOAT
                    or shape != graph_shape or shape != list(params[item.name].shape)):
                raise ValueError(f"Source/graph shape or dtype mismatch: {item.name}")
            source_map[item.name] = {"source": src, "shape": shape, "dtype": "float32"}
    if not source_map:
        raise ValueError("Parameterless graph has no mapped weights")
    (folder / ".mapping.json").write_text(
        json.dumps({"runtime_inputs": [name for name in runtime_names if name in runtime],
                    "pruned_optimum_inputs": sorted(missing), "weights": source_map,
                    "structure_sha256": base.sha256(path)}, indent=2) + "\n")


def attach(source: Path, output: Path, stage: str) -> None:
    import numpy as np
    import onnx
    import torch
    from onnx.external_data_helper import set_external_data
    from safetensors import safe_open

    folder = output / stage
    name = base.STAGE_NAMES[stage]
    original = folder / (name + ".structure.onnx")
    mapping = json.loads((folder / ".mapping.json").read_text())
    if base.sha256(original) != mapping["structure_sha256"]:
        raise ValueError("Structure graph changed after mapping")
    graph = onnx.load(str(original), load_external_data=False)
    runtime = set(mapping["runtime_inputs"])
    weights = mapping["weights"]
    graph_weights = {item.name for item in graph.graph.input if item.name not in runtime}
    if graph_weights != set(weights) or graph.graph.initializer:
        raise ValueError("Graph inputs differ from validated weight inventory")
    if {item.name for item in graph.graph.input if item.name in runtime} != runtime:
        raise ValueError("Graph runtime input inventory differs from Optimum config")
    data = folder / (name + ".onnx.data")
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as source_weights, \
            data.open("xb") as stream:
        for item in list(graph.graph.input):
            if item.name in runtime:
                continue
            entry = weights[item.name]
            source_slice = source_weights.get_slice(entry["source"])
            if (source_slice.get_dtype() != "F16"
                    or source_slice.get_shape() != entry["shape"]
                    or [dim.dim_value for dim in item.type.tensor_type.shape.dim] != entry["shape"]
                    or item.type.tensor_type.elem_type != onnx.TensorProto.FLOAT):
                raise ValueError(f"Source/graph weight changed: {item.name}")
            value = source_weights.get_tensor(entry["source"]).to(dtype=torch.float32)
            raw = np.ascontiguousarray(value.numpy().astype("<f4", copy=False))
            offset = stream.tell()
            stream.write(memoryview(raw).cast("B"))
            stream.flush()
            os.fdatasync(stream.fileno())
            tensor = onnx.TensorProto()
            tensor.name = item.name
            tensor.data_type = onnx.TensorProto.FLOAT
            tensor.dims.extend(entry["shape"])
            tensor.raw_data = b""
            set_external_data(tensor, data.name, offset=offset, length=raw.nbytes)
            tensor.ClearField("raw_data")
            graph.graph.initializer.append(tensor)
        if len(graph.graph.initializer) != len(weights):
            raise ValueError("Not every source weight became an external initializer")
        indices = [index for index, item in enumerate(graph.graph.input) if item.name in weights]
        for index in reversed(indices):
            del graph.graph.input[index]
    if {item.name for item in graph.graph.input} != runtime:
        raise ValueError("Runtime inputs changed while attaching weights")
    weighted = folder / (name + ".onnx")
    onnx.save_model(graph, str(weighted))
    metadata = {"graph": weighted.name, "graph_sha256": base.sha256(weighted),
                "external_data": data.name, "external_data_sha256": base.sha256(data),
                "source_weights_count": len(weights), "external_bytes": data.stat().st_size}
    (folder / ".attachment.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--stage", choices=base.STAGES)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--execute-phase", choices=("structure", "attach"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sum((args.prepare, bool(args.stage), args.finalize)) != 1:
        parser.error("Select exactly one of --prepare, --stage or --finalize")
    if args.execute_phase and not args.stage:
        parser.error("--execute-phase requires --stage")
    source, output = args.source.resolve(), args.output
    if not output.is_absolute() or output.is_symlink() or not args.report_dir.is_absolute():
        parser.error("Source, output and report directory must be absolute")
    if output.resolve().is_relative_to(source) or source.is_relative_to(output.resolve()):
        parser.error("Output and source must be separate")
    if not args.report_dir.is_dir():
        parser.error("--report-dir must exist")
    manifest = base.source_manifest(source)
    if args.prepare:
        if output.exists():
            parser.error("--output must be unused")
        output.mkdir(mode=0o700, parents=True)
        report = {"source": manifest, "export_dtype": "fp32",
                  "export_strategy": STRATEGY, "cache": True,
                  "estimated_needed_bytes": base.STAGE_BUDGET + base.RESERVE}
        (output / ".shum-turbo-onnx-prepared.json").write_text(json.dumps(report, indent=2) + "\n")
        return
    prepared = base.prepared_report(output, manifest)
    if prepared.get("export_strategy") != STRATEGY:
        raise ValueError("Wrong stream-export strategy")
    if args.finalize:
        base.finalize(source, output, manifest)
        return
    if args.execute_phase:
        base.own_cgroup()
        if base.available_bytes() < base.STAGE_BUDGET + base.RESERVE:
            raise RuntimeError("9 GiB stage budget plus 4 GiB effective reserve unavailable")
        (structure if args.execute_phase == "structure" else attach)(source, output, args.stage)
        return
    group = base.own_cgroup()
    if base.available_bytes() < base.STAGE_BUDGET + base.RESERVE:
        raise RuntimeError("9 GiB stage budget plus 4 GiB effective reserve unavailable")
    if (output / args.stage).exists() or (output / f".stage-{args.stage}.json").exists():
        raise ValueError("Refusing to overwrite a prior stage attempt")
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1", "OMP_NUM_THREADS": "1"})
    began = time.monotonic()
    for phase in ("structure", "attach"):
        report = args.report_dir / f"turbo-onnx-stream-{args.stage}-{phase}.json"
        if report.exists():
            raise ValueError(f"Refusing to replace existing report: {report}")
        command = [sys.executable, str(Path(__file__).resolve()), "--source", str(source),
                   "--output", str(output), "--stage", args.stage, "--execute-phase", phase,
                   "--report-dir", str(args.report_dir)]
        base.guarded_stage(source, output, args.stage, group, env, report, child_command=command,
                           deadline_seconds=max(1, 900 - (time.monotonic() - began)))
    base.finish_stage(output, args.stage)
    report = args.report_dir / f"turbo-onnx-stream-{args.stage}-complete.json"
    report.write_text(json.dumps({"stage": args.stage, "status": "complete",
                                  "graph_sha256": base.stage_files(output, args.stage),
                                  "elapsed_seconds": round(time.monotonic() - began, 3),
                                  "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat()},
                                 indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Turbo ONNX stream export blocked: {exc}", file=sys.stderr)
        sys.exit(2)
