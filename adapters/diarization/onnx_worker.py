#!/usr/bin/env python3
"""Offline ONNX Community-1 diarization adapter with explicit local weights."""
from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import main as base

PipelineFactory = Callable[[str, str, str, int, float], Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Community-1 pipeline with PLDA artifacts.")
    parser.add_argument("--segmentation-model", required=True, help="Local segmentation ONNX file.")
    parser.add_argument("--segmentation-sha256", required=True)
    parser.add_argument("--embedding-model", required=True, help="Local speaker embedding ONNX file.")
    parser.add_argument("--embedding-sha256", required=True)
    parser.add_argument("--input", required=True, help="Absolute 16 kHz mono WAV file.")
    parser.add_argument("--output", required=True, help="Absolute output JSON path.")
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--clustering-threshold", type=float, default=0.7)
    return parser.parse_args(argv)


def verify_weight(path_arg: str, expected_sha256: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute() or not path.is_file():
        raise base.AdapterError(f"ONNX weight must be an existing absolute file: {path}")
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise base.AdapterError("ONNX weight SHA-256 must be a lowercase 64-character digest")
    with path.open("rb") as weight:
        actual = hashlib.file_digest(weight, "sha256").hexdigest()
    if actual != expected_sha256:
        raise base.AdapterError(f"ONNX weight SHA-256 mismatch: {path}")
    return path


def default_pipeline_factory(
    model_dir: str, segmentation: str, embedding: str, threads: int, threshold: float,
) -> Any:
    try:
        import onnxruntime as ort
        from pyannote_onnx_community import ONNXSpeakerDiarization
        from pyannote_onnx_community.config import SDConfig
    except ImportError as exc:
        raise base.AdapterError(
            "Missing ONNX diarization dependencies; install pyannote-onnx-community"
        ) from exc
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    try:
        seg_session = ort.InferenceSession(
            segmentation, sess_options=options, providers=providers,
        )
        emb_session = ort.InferenceSession(
            embedding, sess_options=options, providers=providers,
        )
        return ONNXSpeakerDiarization(
            seg_session=seg_session,
            emb_session=emb_session,
            config=SDConfig(plda_repo_id=model_dir, clustering_threshold=threshold),
            providers=providers,
        )
    except Exception as exc:
        raise base.AdapterError(f"Cannot initialize local ONNX diarization: {exc}") from exc


def run(
    argv: Sequence[str] | None = None, pipeline_factory: PipelineFactory | None = None,
) -> int:
    args = parse_args(argv)
    try:
        model_dir = base.validate_model_dir(args.model)
        for name in ("plda/plda.npz", "plda/xvec_transform.npz"):
            if not (model_dir / name).is_file():
                raise base.AdapterError(f"Community-1 model is missing {name}")
        input_path = base.validate_input_wav(args.input)
        output_path = base.validate_output_path(args.output)
        threads = base.validate_threads(args.threads)
        if args.num_speakers is not None and args.num_speakers < 1:
            raise base.AdapterError("--num-speakers must be positive")
        if not math.isfinite(args.clustering_threshold) or not 0 < args.clustering_threshold < 1:
            raise base.AdapterError("--clustering-threshold must be between 0 and 1")
        segmentation = verify_weight(args.segmentation_model, args.segmentation_sha256)
        embedding = verify_weight(args.embedding_model, args.embedding_sha256)
        base.configure_offline_environment()
        base.configure_thread_environment(threads)
        factory = pipeline_factory or default_pipeline_factory
        pipeline = factory(
            str(model_dir), str(segmentation), str(embedding), threads,
            args.clustering_threshold,
        )
        output = (
            base.run_profiled_pipeline(
                pipeline, str(input_path), num_speakers=args.num_speakers,
            )
            if args.profile else base.run_pipeline(
                pipeline, str(input_path), num_speakers=args.num_speakers,
            )
        )
        base.write_result(base.diarization_output_to_result(output), output_path)
    except base.AdapterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
