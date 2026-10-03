#!/usr/bin/env python3
"""Guarded decoder-only ORT dynamic MatMul INT8 pilot for pinned Turbo graphs."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import turbo_onnx_export as base

GRAPH_NAMES = {"decoder": "decoder_model", "decoder-past": "decoder_with_past_model"}
PHASES = tuple(f"{stage}-{phase}" for stage in GRAPH_NAMES
               for phase in ("preprocess", "quantize"))


def read_fp32(graphs: Path) -> dict:
    if not graphs.is_absolute() or graphs.is_symlink():
        raise ValueError("--graphs must be an absolute, non-symlink FP32 graph directory")
    manifest = json.loads((graphs / ".shum-turbo-onnx.json").read_text())
    if (manifest.get("complete") is not True or manifest.get("export_dtype") != "fp32"
            or manifest.get("source", {}).get("revision") != base.REVISION):
        raise ValueError("FP32 graph manifest is incomplete or not the pinned model")
    return manifest


def checked_hash(graphs: Path, manifest: dict, name: str) -> str:
    expected = manifest["graphs_sha256"].get(name)
    path = graphs / name
    if (not expected or path.is_symlink() or not path.is_file()
            or not path.resolve().is_relative_to(graphs) or base.sha256(path) != expected):
        raise ValueError(f"FP32 source graph or external data changed: {name}")
    return expected


def paths(graphs: Path, output: Path, stage: str) -> tuple[Path, Path, Path]:
    stem = GRAPH_NAMES[stage]
    return (graphs / stage / f"{stem}.onnx",
            output / stage / f"{stem}.pre.onnx",
            output / stage / f"{stem}.onnx")


def constant_matmuls(graph) -> int:
    names = {item.name for item in graph.initializer}
    return sum(node.op_type == "MatMul" and len(node.input) > 1
               and node.input[1] in names for node in graph.node)


def execute(graphs: Path, output: Path, phase: str) -> None:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quant_pre_process, quantize_dynamic

    base.own_cgroup()
    if os.environ.get("SHUM_TURBO_QUANT_GUARDED") != "1":
        raise RuntimeError("A monitored 9 GiB quantization stage is required")
    stage, step = phase.rsplit("-", 1)
    manifest = read_fp32(graphs)
    original, preprocessed, quantized = paths(graphs, output, stage)
    folder = output / stage
    record = folder / f".{step}.json"
    if record.exists():
        raise ValueError(f"Phase already recorded: {record}")
    if step == "preprocess":
        original_name = str(original.relative_to(graphs))
        external_name = original_name + ".data"
        input_hashes = {
            original_name: checked_hash(graphs, manifest, original_name),
            external_name: checked_hash(graphs, manifest, external_name),
        }
        optimized = folder / f"{GRAPH_NAMES[stage]}.ort-basic.onnx"
        if preprocessed.exists() or optimized.exists():
            raise ValueError(f"Preprocessed graph already exists: {preprocessed}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 16
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.optimized_model_filepath = str(optimized)
        options.add_session_config_entry(
            "session.optimized_model_external_initializers_file_name", optimized.name + ".data")
        options.add_session_config_entry("session.disable_prepacking", "1")
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        session = ort.InferenceSession(str(original), options, providers=["CPUExecutionProvider"])
        del session
        if not optimized.is_file():
            raise ValueError("ORT basic optimization did not save a graph")
        optimized_data = folder / (optimized.name + ".data")
        if not optimized_data.is_file():
            raise ValueError("ORT basic optimization did not save its external initializers")
        quant_pre_process(input_model=str(optimized), output_model_path=str(preprocessed),
                          skip_optimization=True, skip_symbolic_shape=True,
                          skip_onnx_shape=False, save_as_external_data=False)
        graph = onnx.load(str(preprocessed), load_external_data=False).graph
        direct = constant_matmuls(graph)
        if direct <= 0:
            raise ValueError("ORT preprocessing left zero directly quantizable MatMul weights")
        result = {"phase": phase, "source_sha256": manifest["source"]["source_sha256"],
                  "fp32_input_sha256": input_hashes, "direct_constant_matmul": direct,
                  "ort_optimized_sha256": base.sha256(optimized),
                  "ort_optimized_data_sha256": base.sha256(optimized_data),
                  "preprocessed_sha256": base.sha256(preprocessed)}
    else:
        previous = json.loads((folder / ".preprocess.json").read_text())
        if previous.get("phase") != f"{stage}-preprocess":
            raise ValueError("Missing validated preprocessing stage")
        if base.sha256(preprocessed) != previous["preprocessed_sha256"]:
            raise ValueError("Preprocessed model changed after validation")
        graph = onnx.load(str(preprocessed), load_external_data=False).graph
        if constant_matmuls(graph) != previous["direct_constant_matmul"]:
            raise ValueError("Preprocessed constant-weight MatMul inventory changed")
        if quantized.exists() or (folder / (quantized.name + ".data")).exists():
            raise ValueError("Quantized output already exists")
        quantize_dynamic(str(preprocessed), str(quantized),
                         op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8,
                         per_channel=False, use_external_data_format=True,
                         extra_options={"MatMulConstBOnly": True})
        qgraph = onnx.load(str(quantized), load_external_data=False).graph
        coverage = {
            "matmul_integer": sum(node.op_type == "MatMulInteger" for node in qgraph.node),
            "remaining_matmul": sum(node.op_type == "MatMul" for node in qgraph.node),
            "int8_initializers": sum(item.data_type == onnx.TensorProto.INT8
                                     for item in qgraph.initializer),
        }
        data = folder / (quantized.name + ".data")
        if (coverage["matmul_integer"] <= 0 or coverage["int8_initializers"] <= 0
                or not data.is_file() or data.stat().st_size == 0):
            raise ValueError(f"No proven external-weight dynamic MatMul INT8 coverage: {coverage}")
        result = {"phase": phase, "source_sha256": manifest["source"]["source_sha256"],
                  "fp32_input_sha256": previous["fp32_input_sha256"],
                  "preprocessed_sha256": previous["preprocessed_sha256"],
                  "coverage": coverage,
                  "quantized_sha256": {
                      str(quantized.relative_to(output)): base.sha256(quantized),
                      str(data.relative_to(output)): base.sha256(data),
                  }}
    record.write_text(json.dumps(result, indent=2) + "\n")


def finalize(graphs: Path, output: Path) -> None:
    manifest = read_fp32(graphs)
    if (output / ".shum-turbo-onnx-mixed.json").exists():
        raise ValueError("Mixed graph manifest already exists")
    quantized_hashes = {}
    coverage = {}
    phases = {}
    for stage in GRAPH_NAMES:
        original, preprocessed, quantized = paths(graphs, output, stage)
        before = json.loads((output / stage / ".preprocess.json").read_text())
        after = json.loads((output / stage / ".quantize.json").read_text())
        if (before.get("phase") != f"{stage}-preprocess"
                or after.get("phase") != f"{stage}-quantize"
                or before["preprocessed_sha256"] != after["preprocessed_sha256"]
                or before["fp32_input_sha256"] != after["fp32_input_sha256"]
                or before["source_sha256"] != manifest["source"]["source_sha256"]
                or after["source_sha256"] != manifest["source"]["source_sha256"]
                or base.sha256(preprocessed) != before["preprocessed_sha256"]):
            raise ValueError(f"Broken quantization provenance: {stage}")
        optimized = output / stage / f"{GRAPH_NAMES[stage]}.ort-basic.onnx"
        if (base.sha256(optimized) != before["ort_optimized_sha256"]
                or base.sha256(optimized.parent / (optimized.name + ".data"))
                != before["ort_optimized_data_sha256"]):
            raise ValueError(f"ORT optimized graph changed: {stage}")
        for name, expected in before["fp32_input_sha256"].items():
            checked_hash(graphs, manifest, name)
            if manifest["graphs_sha256"][name] != expected:
                raise ValueError(f"Source changed during decoder quantization: {name}")
        for name, expected in after["quantized_sha256"].items():
            if base.sha256(output / name) != expected:
                raise ValueError(f"Quantized graph changed: {name}")
        if set(after["quantized_sha256"]) != {
                str(quantized.relative_to(output)),
                str((quantized.parent / (quantized.name + ".data")).relative_to(output))}:
            raise ValueError(f"Incomplete quantized decoder inventory: {stage}")
        quantized_hashes.update(after["quantized_sha256"])
        coverage[stage] = after["coverage"]
        phases[stage] = {"preprocess": before, "quantize": after}
    result = {"complete": True, "profile": "fp32_encoder_dynamic_int8_decoders",
              "source": manifest["source"], "fp32_graphs": str(graphs),
              "fp32_graphs_sha256": manifest["graphs_sha256"],
              "quantized_sha256": quantized_hashes, "coverage": coverage,
              "phases": phases, "ort_version": importlib.metadata.version("onnxruntime"),
              "onnx_version": importlib.metadata.version("onnx"),
              "worker_policy": "FP32 encoder; dynamic QInt8 MatMul decoders; no CT2 VAD/fallback"}
    (output / ".shum-turbo-onnx-mixed.json").write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--stage", choices=PHASES)
    mode.add_argument("--execute-stage", choices=PHASES)
    mode.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if not all(path.is_absolute() for path in (args.graphs, args.output, args.report_dir)):
        parser.error("All paths must be absolute")
    graphs, output = args.graphs.resolve(), args.output
    if (output.resolve() == graphs or output.resolve().is_relative_to(graphs)
            or output.is_symlink()):
        raise ValueError("Mixed output must not replace or sit within frozen FP32 graphs")
    if args.prepare:
        read_fp32(graphs)
        if output.exists() or args.report_dir.exists():
            raise ValueError("Output and report directories must both be fresh")
        output.mkdir(mode=0o700)
        args.report_dir.mkdir(mode=0o700)
        for stage in GRAPH_NAMES:
            (output / stage).mkdir(mode=0o700)
    elif args.execute_stage:
        execute(graphs, output, args.execute_stage)
    elif args.stage:
        if not output.is_dir() or not args.report_dir.is_dir():
            raise ValueError("Run --prepare before supervised stages")
        group = base.own_cgroup()
        command = [sys.executable, str(Path(__file__).resolve()), "--graphs", str(graphs),
                   "--output", str(output), "--report-dir", str(args.report_dir),
                   "--execute-stage", args.stage]
        env = dict(os.environ, SHUM_TURBO_QUANT_GUARDED="1")
        report_path = args.report_dir / f"{args.stage}.guard.json"
        if report_path.exists():
            raise ValueError("Stage report already exists; do not retry unchanged")
        try:
            base.guarded_stage(graphs, output, args.stage, group, env, report_path,
                               child_command=command)
        except BaseException as exc:
            if not report_path.exists():
                report_path.write_text(json.dumps(
                    {"phase": args.stage, "status": "admission_blocked",
                     "reason": f"{type(exc).__name__}: {exc}"}, indent=2) + "\n")
            raise
    else:
        finalize(graphs, output)


if __name__ == "__main__":
    main()
