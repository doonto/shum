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
import time
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
    parser.add_argument(
        "--segmentation-batch-size", type=int, default=1,
        help="Segmentation inference batch size (default: 1).",
    )
    parser.add_argument(
        "--embedding-batch-size", type=int, default=1,
        help="Speaker embedding batch size (default: 1).",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Log elapsed time for diarization stages to stderr; never log audio or speech.",
    )
    parser.add_argument(
        "--preload-audio", action="store_true",
        help="Load the whole WAV into RAM once so embedding crops avoid repeated file decoding.",
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


def emit_profile(phase: str, status: str, **fields: Any) -> None:
    print(
        json.dumps(
            {"event": "shum.profile", "version": 1, "phase": phase,
             "status": status, **fields},
            separators=(",", ":"),
        ),
        file=sys.stderr,
        flush=True,
    )


@contextlib.contextmanager
def measured_phase(phase: str, enabled: bool):
    if not enabled:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    except BaseException:
        emit_profile(
            phase, "failed", elapsed_seconds=time.perf_counter() - started,
        )
        raise
    else:
        emit_profile(
            phase, "completed", elapsed_seconds=time.perf_counter() - started,
        )


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


def run_pipeline(pipeline: Any, audio_input: Any, **options: Any) -> Any:
    try:
        return pipeline(audio_input, **options)
    except Exception as exc:
        raise AdapterError(f"Diarization failed: {exc}") from exc


def load_waveform(input_path: str) -> dict[str, Any]:
    try:
        import torchaudio
    except ImportError as exc:
        raise AdapterError("Missing dependency: torchaudio is required for --preload-audio") from exc
    try:
        waveform, sample_rate = torchaudio.load(input_path)
    except Exception as exc:
        raise AdapterError(f"Cannot preload WAV: {exc}") from exc
    if sample_rate != 16000 or waveform.shape[0] != 1:
        raise AdapterError("Preloaded audio must be mono at 16 kHz")
    return {"waveform": waveform, "sample_rate": sample_rate}


def configure_batch_sizes(pipeline: Any, segmentation: int, embedding: int) -> None:
    for name, size in (
        ("segmentation_batch_size", segmentation),
        ("embedding_batch_size", embedding),
    ):
        if size < 1:
            raise AdapterError(f"--{name.replace('_', '-')} must be positive, got {size}")
        if size != 1:
            if not hasattr(pipeline, name):
                raise AdapterError(f"Loaded diarization pipeline does not support {name}")
            setattr(pipeline, name, size)


def run_profiled_pipeline(
    pipeline: Any, audio_input: Any, *, profile: bool = True, **options: Any
) -> Any:
    started = time.perf_counter()
    previous = started
    progress: dict[str, int] = {}
    measurements: dict[str, tuple[int, int]] = {}
    boundary = started
    boundary_name = "pipeline_start"
    active_phase: str | None = None
    clustering_started: float | None = None

    def hook(step: str, _artifact: Any, **kwargs: Any) -> None:
        nonlocal previous, boundary, boundary_name, active_phase, clustering_started
        if kwargs.get("total") is not None:
            total = kwargs["total"]
            completed = kwargs.get("completed", 0)
            if (
                step in ("segmentation", "embeddings")
                and isinstance(total, int)
                and isinstance(completed, int)
                and total > 0
                and 0 <= completed <= total
                and measurements.get(step) != (completed, total)
            ):
                measurements[step] = (completed, total)
                if profile:
                    active_phase = step
                    emit_profile(step, "progress", completed=completed, total=total)
                print(
                    json.dumps(
                        {
                            "event": "shum.progress",
                            "version": 1,
                            "phase": step,
                            "completed": completed,
                            "total": total,
                        },
                        separators=(",", ":"),
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            if profile and total > 0:
                tenths = min(10, completed * 10 // total)
                if tenths > progress.get(step, 0) and tenths < 10:
                    progress[step] = tenths
                    print(
                        f"diarization {step} progress {tenths * 10}%:"
                        f" {time.perf_counter() - started:.2f}s elapsed",
                        file=sys.stderr,
                    )
            return
        if not profile:
            return
        now = time.perf_counter()
        print(f"diarization {step}: {now - previous:.2f}s", file=sys.stderr)
        previous = now
        if step in ("segmentation", "embeddings"):
            emit_profile(
                step, "completed", elapsed_seconds=now - boundary,
                since=boundary_name,
                **({"completed": measurements[step][0], "total": measurements[step][1]}
                   if step in measurements else {}),
            )
            active_phase = None
            if step == "embeddings":
                clustering_started = now
                active_phase = "clustering_assembly"
        boundary = now
        boundary_name = step

    try:
        result = pipeline(audio_input, hook=hook, **options)
    except Exception as exc:
        if profile:
            now = time.perf_counter()
            if active_phase is not None:
                emit_profile(
                    active_phase, "failed",
                    elapsed_seconds=now - (
                        clustering_started if active_phase == "clustering_assembly"
                        and clustering_started is not None else boundary
                    ),
                    since="embeddings" if active_phase == "clustering_assembly" else boundary_name,
                    **({"completed": measurements[active_phase][0],
                        "total": measurements[active_phase][1]}
                       if active_phase in measurements else {}),
                )
            emit_profile("pipeline_total", "failed", elapsed_seconds=now - started)
        raise AdapterError(f"Diarization failed: {exc}") from exc
    if profile:
        now = time.perf_counter()
        if clustering_started is not None:
            emit_profile(
                "clustering_assembly", "completed",
                elapsed_seconds=now - clustering_started, since="embeddings",
            )
        emit_profile("pipeline_total", "completed", elapsed_seconds=now - started)
        print(f"diarization total: {now - started:.2f}s", file=sys.stderr)
    return result


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
        for name, size in (
            ("segmentation-batch-size", args.segmentation_batch_size),
            ("embedding-batch-size", args.embedding_batch_size),
        ):
            if size < 1:
                raise AdapterError(f"--{name} must be positive, got {size}")

        configure_offline_environment()
        configure_thread_environment(threads)

        factory = pipeline_factory or default_pipeline_factory
        parameters = {
            "threads": threads,
            "requested_segmentation_batch_size": args.segmentation_batch_size,
            "requested_embedding_batch_size": args.embedding_batch_size,
            "preload_audio": args.preload_audio,
        }
        if args.profile:
            emit_profile("parameters", "requested", parameters=parameters)
        with measured_phase("model_loading", args.profile):
            pipeline = factory(str(model_path), threads)
        with measured_phase("batch_configuration", args.profile):
            configure_batch_sizes(
                pipeline, args.segmentation_batch_size, args.embedding_batch_size,
            )
        if args.profile:
            emit_profile(
                "parameters", "configured",
                parameters={
                    **parameters,
                    "segmentation_batch_size": getattr(pipeline, "segmentation_batch_size", None),
                    "embedding_batch_size": getattr(pipeline, "embedding_batch_size", None),
                },
            )
        audio_input: Any = str(input_path)
        if args.preload_audio:
            loading_started = time.perf_counter()
            with measured_phase("waveform_loading", args.profile):
                audio_input = load_waveform(str(input_path))
            if args.profile:
                print(
                    f"diarization waveform load: {time.perf_counter() - loading_started:.2f}s",
                    file=sys.stderr,
                )
        diarization_output = run_profiled_pipeline(
            pipeline, audio_input, profile=args.profile
        )
        with measured_phase("result_conversion", args.profile):
            result = diarization_output_to_result(diarization_output)
        with measured_phase("result_writing", args.profile):
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
