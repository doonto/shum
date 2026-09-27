#!/usr/bin/env python3
"""Offline, CPU-only speaker diarization adapter (pyannote Community-1).

Isolated CLI adapter: reads a 16 kHz mono WAV file, runs a locally installed
``pyannote/speaker-diarization-community-1`` pipeline entirely on CPU without
any network access or audio upload, and writes the result as JSON.

Usage:
    python main.py --model /absolute/local/model/directory \
                    --input /absolute/16khz/mono.wav \
                    --output /absolute/result.json \
                    --threads N

Output JSON shape:
    {
      "turns": [{"start": <seconds>, "end": <seconds>, "speaker": "<label>"}, ...],
      "exclusive_turns": [...]
    }

Design notes:
* Heavy imports (``torch``, ``pyannote.audio``) are performed lazily inside
  ``default_pipeline_factory`` so that argument parsing, validation, and
  result-formatting logic can be unit tested without those dependencies
  installed, and so that failures below fail fast with clear messages.
* All failures raise ``AdapterError`` and are reported as a single explicit
  error message on stderr with a non-zero exit code. There is no silent
  fallback: this adapter will never resample/downmix audio behind the
  user's back, retry over the network, or substitute another model.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Callable, Sequence


class AdapterError(RuntimeError):
    """Explicit, user-facing adapter failure (never a silent fallback)."""


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Offline CPU speaker diarization using a local pyannote "
            "Community-1 pipeline."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Absolute path to a local pyannote Community-1 model directory "
            "(a full local checkout of the pipeline repository, containing "
            "config.yaml)."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Absolute path to a 16 kHz mono WAV file to diarize.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Absolute path to write the resulting diarization JSON to.",
    )
    parser.add_argument(
        "--threads",
        required=True,
        type=int,
        help="Number of CPU threads to use for inference.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# Validation (fail fast, explicit errors, no silent fallback)
# ---------------------------------------------------------------------------

def validate_model_dir(model_path: str) -> Path:
    path = Path(model_path)
    if not path.is_absolute():
        raise AdapterError(f"--model must be an absolute path, got: {model_path!r}")
    if not path.is_dir():
        raise AdapterError(f"--model directory does not exist: {path}")
    config_path = path / "config.yaml"
    if not config_path.is_file():
        raise AdapterError(
            "--model directory is missing config.yaml (not a valid local "
            f"pyannote pipeline checkout): {path}. Clone the full pipeline "
            "repository, e.g.:\n"
            f"  git clone https://hf.co/pyannote/speaker-diarization-community-1 {path}"
        )
    return path


def validate_input_wav(input_path: str) -> Path:
    path = Path(input_path)
    if not path.is_absolute():
        raise AdapterError(f"--input must be an absolute path, got: {input_path!r}")
    if not path.is_file():
        raise AdapterError(f"--input file does not exist: {path}")
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as wav_file:
            channels = wav_file.getnchannels()
            rate = wav_file.getframerate()
    except wave.Error as exc:
        raise AdapterError(f"--input is not a readable WAV file: {path} ({exc})") from exc

    problems = []
    if channels != 1:
        problems.append(f"{channels} channel(s) (expected mono, 1 channel)")
    if rate != 16000:
        problems.append(f"{rate} Hz sample rate (expected 16000 Hz)")
    if problems:
        raise AdapterError(
            "--input does not meet the required 16 kHz mono WAV format: "
            + "; ".join(problems)
            + f". File: {path}. Convert it first (e.g. with ffmpeg/sox); "
              "this adapter will not silently resample or downmix audio."
        )
    return path


def validate_output_path(output_path: str) -> Path:
    path = Path(output_path)
    if not path.is_absolute():
        raise AdapterError(f"--output must be an absolute path, got: {output_path!r}")
    parent = path.parent
    if not parent.is_dir():
        raise AdapterError(f"--output directory does not exist: {parent}")
    return path


def validate_threads(threads: int) -> int:
    if threads < 1:
        raise AdapterError(f"--threads must be a positive integer, got: {threads}")
    return threads


# ---------------------------------------------------------------------------
# Offline / CPU environment setup
# ---------------------------------------------------------------------------

OFFLINE_ENV_VARS = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


def configure_offline_environment() -> None:
    """Force huggingface_hub/transformers into offline mode.

    This runs before ``torch``/``pyannote.audio`` are imported so any
    attempt to reach the network fails with an explicit error instead of
    silently downloading a model or falling back to a cloud pipeline
    (e.g. pyannote's hosted ``precision-2``). Audio is never uploaded:
    this adapter only ever loads a local pipeline directory and never
    passes a pyannoteAI API token.
    """
    os.environ.update(OFFLINE_ENV_VARS)


def configure_thread_environment(threads: int) -> None:
    """Pin CPU thread counts before numpy/torch are imported."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads)


# ---------------------------------------------------------------------------
# Pipeline loading & execution
# (heavy imports are lazy and injectable so this module stays testable
#  without torch/pyannote.audio installed)
# ---------------------------------------------------------------------------

PipelineFactory = Callable[[str, int], Any]


def default_pipeline_factory(model_dir: str, threads: int) -> Any:
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise AdapterError(
            "Missing dependency: install requirements.txt from "
            "adapters/diarization (pip install -r "
            "adapters/diarization/requirements.txt)."
        ) from exc

    torch.set_num_threads(threads)

    try:
        pipeline = Pipeline.from_pretrained(model_dir)
    except Exception as exc:  # exact type depends on huggingface_hub version
        raise AdapterError(
            f"Failed to load the local pyannote pipeline from {model_dir!r} "
            "in offline mode. No network fallback will be attempted. "
            "Verify the directory is a complete local checkout of the "
            f"pipeline repository (original error: {exc})."
        ) from exc

    if pipeline is None:
        raise AdapterError(
            f"pyannote Pipeline.from_pretrained returned no pipeline for {model_dir!r}."
        )

    pipeline.to(torch.device("cpu"))
    return pipeline


def run_pipeline(pipeline: Any, input_path: str) -> Any:
    try:
        return pipeline(input_path)
    except Exception as exc:
        raise AdapterError(f"Diarization failed for {input_path!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# Result conversion
# ---------------------------------------------------------------------------

def annotation_to_turns(annotation: Any) -> list[dict[str, Any]]:
    turns = [
        {
            "start": round(float(segment.start), 3),
            "end": round(float(segment.end), 3),
            "speaker": str(speaker),
        }
        for segment, _track, speaker in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda turn: (turn["start"], turn["end"], turn["speaker"]))
    return turns


def diarization_output_to_result(output: Any) -> dict[str, list[dict[str, Any]]]:
    if not hasattr(output, "speaker_diarization"):
        raise AdapterError(
            "Pipeline output has no 'speaker_diarization' attribute; the "
            "loaded model is not a pyannote Community-1 pipeline."
        )
    if not hasattr(output, "exclusive_speaker_diarization"):
        raise AdapterError(
            "Pipeline output has no 'exclusive_speaker_diarization' "
            "attribute; the loaded model does not support exclusive "
            "speaker diarization (requires pyannote Community-1)."
        )
    return {
        "turns": annotation_to_turns(output.speaker_diarization),
        "exclusive_turns": annotation_to_turns(output.exclusive_speaker_diarization),
    }


# ---------------------------------------------------------------------------
# Output writing (atomic: a failure never leaves a partial/corrupt file)
# ---------------------------------------------------------------------------

def write_result(result: dict, output_path: Path) -> None:
    fd, tmp_name = tempfile.mkstemp(
        prefix=".diarization-", suffix=".json.tmp", dir=str(output_path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(result, tmp_file, ensure_ascii=False, indent=2)
            tmp_file.write("\n")
        os.replace(tmp_name, output_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_name)
        raise


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(argv: Sequence[str] | None = None, pipeline_factory: PipelineFactory | None = None) -> int:
    args = parse_args(argv)
    try:
        threads = validate_threads(args.threads)
        model_path = validate_model_dir(args.model)
        input_path = validate_input_wav(args.input)
        output_path = validate_output_path(args.output)

        configure_offline_environment()
        configure_thread_environment(threads)

        factory = pipeline_factory or default_pipeline_factory
        pipeline = factory(str(model_path), threads)
        diarization_output = run_pipeline(pipeline, str(input_path))
        result = diarization_output_to_result(diarization_output)
        write_result(result, output_path)
    except AdapterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {output_path}")
    return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
