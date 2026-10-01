#!/usr/bin/env python3
"""Offline, CPU-only ASR adapter for NeMo, Whisper (Transformers) and Qwen3-ASR.

Isolated CLI adapter: reads a 16 kHz mono WAV file, runs a locally installed
speech-to-text model entirely on CPU without any network access, and writes
the result as JSON. A single ``--engine`` flag selects which model family to
load; each engine's heavy Python dependencies (``nemo_toolkit``, ``torch`` +
``transformers``) are expected to live in their own isolated environment,
because they are large and can conflict with each other (see README.md).

Usage (single clip, loads the model once and exits):
    python main.py --engine {nemo,whisper,qwen} \
                    --model /absolute/local/model/path \
                    --input /absolute/16khz/mono.wav \
                    --output /absolute/result.json \
                    --threads N

Usage (persistent worker, loads the model exactly ONCE and then serves many
clips -- for callers that would otherwise pay the model-load cost once per
chunk, e.g. a pipeline processing hundreds of short chunks):
    python main.py --engine {nemo,whisper,qwen} \
                    --model /absolute/local/model/path \
                    --threads N \
                    --serve

    In ``--serve`` mode, ``--input``/``--output`` are not used. Instead, the
    adapter reads newline-delimited JSON requests from stdin, one per line:
        {"input": "/absolute/16khz/mono.wav"}
    and writes exactly one JSON-lines response to stdout per request, using
    the same ``{text, language, segments}`` schema as single-clip mode. On
    any request/transcription failure it writes a single
    ``{"error": "..."}`` line to stdout, prints diagnostics to stderr, and
    exits non-zero (it does not attempt to keep serving after an error).
    stdout carries JSON only; all other logging goes to stderr.

Model path per engine (``--model`` is always a directory, consistent
across engines with how a model manager typically lays out per-model
checkouts):
    nemo:    an absolute path to a directory containing exactly one original
             NVIDIA NeMo ``.nemo`` checkpoint file (e.g. a
             Parakeet/Conformer/Citrinet checkpoint).
    whisper: an absolute path to a local Hugging Face Transformers Whisper
             checkpoint directory (``config.json`` + weights + processor
             files), e.g. a Flix SwissGerman Whisper fine-tune.
    qwen:    an absolute path to a local Hugging Face Qwen3-ASR checkpoint
             directory (``config.json`` + weights + processor files).

Output JSON shape:
    {
      "text": "<full transcription>",
      "language": "<language name or code>" | null,
      "segments": [{"start": <seconds>, "end": <seconds>, "text": "<segment text>"}, ...]
    }

Design notes:
* Heavy imports (``torch``, ``nemo``, ``transformers``) are performed lazily
  inside the ``default_*_factory`` functions so that argument parsing,
  validation, and result-formatting logic can be unit tested without those
  dependencies installed, and so real model loading/inference can be
  injected in tests via the ``*_factory``/``*_transcribe`` parameters of
  ``run()``.
* All failures raise ``AdapterError`` and are reported as a single explicit
  error message on stderr with a non-zero exit code. There is no silent
  fallback: this adapter will never resample/downmix audio behind the
  user's back, retry over the network, or substitute another engine/model.
* Segment-level timestamps are only reported when the underlying model and
  its documented API actually expose them (NeMo Parakeet-family models,
  Whisper's chunked decoding). Qwen3-ASR alone does not produce timestamps
  (that requires a separate Qwen3-ForcedAligner checkpoint, out of scope
  for this single-model contract), so its adapter reports one segment
  spanning the whole input rather than inventing sub-segment timing.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

ENGINE_CHOICES = ("nemo", "whisper", "qwen")


class AdapterError(RuntimeError):
    """Explicit, user-facing adapter failure (never a silent fallback)."""


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Offline CPU speech-to-text transcription using a local NeMo, "
            "Whisper (Transformers), or Qwen3-ASR checkpoint."
        ),
    )
    parser.add_argument(
        "--engine",
        required=True,
        choices=ENGINE_CHOICES,
        help="ASR engine/model family to run.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Absolute path to the local model directory: for --engine nemo "
            "it must contain exactly one '.nemo' checkpoint file; for "
            "--engine whisper/qwen it must be a Transformers checkpoint "
            "directory."
        ),
    )
    parser.add_argument(
        "--input",
        required=False,
        default=None,
        help=(
            "Absolute path to a 16 kHz mono WAV file to transcribe. "
            "Required unless --serve is given (in --serve mode, one input "
            "path per request is instead read from stdin)."
        ),
    )
    parser.add_argument(
        "--output",
        required=False,
        default=None,
        help=(
            "Absolute path to write the resulting transcription JSON to. "
            "Required unless --serve is given (in --serve mode, results are "
            "instead written as JSON lines to stdout)."
        ),
    )
    parser.add_argument(
        "--threads",
        required=True,
        type=int,
        help="Number of CPU threads to use for inference.",
    )
    parser.add_argument(
        "--offload-dir",
        default=None,
        help=(
            "Opt in to Accelerate CPU/disk offload for whisper/qwen with a "
            "2 GiB model CPU budget. Absolute private cache root outside "
            "the model snapshot; each engine/revision gets its own directory."
        ),
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help=(
            "Run as a persistent worker instead of processing a single "
            "clip: load the model exactly once, then read newline-"
            "delimited JSON requests ({\"input\": \"/abs/clip.wav\"}) from "
            "stdin and write one JSON result (or {\"error\": ...}) per line "
            "to stdout until stdin is closed or the first error. Mutually "
            "exclusive with --input/--output."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.serve:
        if args.input is not None or args.output is not None:
            parser.error("--serve cannot be combined with --input/--output")
    else:
        missing = [
            name
            for name, value in (("--input", args.input), ("--output", args.output))
            if value is None
        ]
        if missing:
            parser.error(f"the following arguments are required: {', '.join(missing)}")
    return args


# ---------------------------------------------------------------------------
# Validation (fail fast, explicit errors, no silent fallback)
# ---------------------------------------------------------------------------

def validate_threads(threads: int) -> int:
    if threads < 1:
        raise AdapterError(f"--threads must be a positive integer, got: {threads}")
    return threads


def find_nemo_checkpoint_file(model_dir: Path) -> Path:
    """Resolve the single original '.nemo' checkpoint file inside a model directory.

    The model directory holds exactly one '.nemo' file, matching the same
    per-model-directory layout used for the whisper/qwen engines. There is
    no attempt to guess which file to load when zero or more than one
    '.nemo' file is present; both are explicit, reported errors.
    """
    nemo_files = sorted(model_dir.glob("*.nemo"))
    if not nemo_files:
        raise AdapterError(
            f"--model directory contains no '.nemo' checkpoint file: {model_dir}"
        )
    if len(nemo_files) > 1:
        names = ", ".join(f.name for f in nemo_files)
        raise AdapterError(
            "--model directory contains multiple '.nemo' checkpoint files "
            f"({names}); expected exactly one in {model_dir}"
        )
    return nemo_files[0]


def validate_model_path(engine: str, model_path: str) -> Path:
    path = Path(model_path)
    if not path.is_absolute():
        raise AdapterError(f"--model must be an absolute path, got: {model_path!r}")
    if engine == "nemo":
        if not path.is_dir():
            raise AdapterError(f"--model directory does not exist: {path}")
        return find_nemo_checkpoint_file(path)
    else:
        # whisper / qwen: local Transformers checkpoint directories.
        if not path.is_dir():
            raise AdapterError(f"--model directory does not exist: {path}")
        config_path = path / "config.json"
        if not config_path.is_file():
            raise AdapterError(
                "--model directory is missing config.json (not a valid "
                f"local Transformers checkpoint): {path}"
            )
    return path


def validate_offload_dir(engine: str, model_dir: Path, offload_dir: str | None) -> Path | None:
    if offload_dir is None:
        return None
    if engine == "nemo":
        raise AdapterError("--offload-dir is supported only for whisper and qwen")
    root = Path(offload_dir)
    if not root.is_absolute():
        raise AdapterError(f"--offload-dir must be an absolute path, got: {offload_dir!r}")
    if root.is_symlink() or root.resolve().is_relative_to(model_dir.resolve()):
        raise AdapterError("--offload-dir must not be a symlink or inside the model snapshot")
    revision = model_dir.name if re.fullmatch(r"[0-9a-fA-F]{40}", model_dir.name) else "local"
    identity = hashlib.sha256(os.fsencode(model_dir.resolve())).hexdigest()[:12]
    engine_dir = root / engine
    cache = engine_dir / f"{revision}-{identity}"
    for directory in (root, engine_dir, cache):
        if directory.is_symlink():
            raise AdapterError(f"--offload-dir cache must not contain symlinks: {directory}")
        try:
            directory.mkdir(mode=0o700, parents=False, exist_ok=True)
        except OSError as exc:
            raise AdapterError(f"Cannot create --offload-dir cache {directory}: {exc}") from exc
        if not directory.is_dir() or stat.S_IMODE(directory.stat().st_mode) & 0o077:
            raise AdapterError(f"--offload-dir cache must be a private directory (mode 0700): {directory}")
    return cache


def validate_input_wav(input_path: str) -> tuple[Path, float]:
    path = Path(input_path)
    if not path.is_absolute():
        raise AdapterError(f"--input must be an absolute path, got: {input_path!r}")
    if not path.is_file():
        raise AdapterError(f"--input file does not exist: {path}")
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as wav_file:
            channels = wav_file.getnchannels()
            rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
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
    duration = n_frames / float(rate) if rate else 0.0
    return path, duration


def validate_output_path(output_path: str) -> Path:
    path = Path(output_path)
    if not path.is_absolute():
        raise AdapterError(f"--output must be an absolute path, got: {output_path!r}")
    parent = path.parent
    if not parent.is_dir():
        raise AdapterError(f"--output directory does not exist: {parent}")
    return path


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

    This runs before ``torch``/``transformers``/``nemo`` are imported so any
    attempt to reach the network fails with an explicit error instead of
    silently downloading a model. Every engine below also passes
    ``local_files_only=True`` explicitly to its own loading calls as a
    second, independent guard.
    """
    os.environ.update(OFFLINE_ENV_VARS)


def configure_thread_environment(threads: int) -> None:
    """Pin CPU thread counts before numpy/torch are imported."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads)


# ---------------------------------------------------------------------------
# Result validation & atomic output writing
# ---------------------------------------------------------------------------

REQUIRED_SEGMENT_KEYS = ("start", "end", "text")


def build_result(text: str, language: str | None, segments: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_segments = []
    for i, segment in enumerate(segments):
        for key in REQUIRED_SEGMENT_KEYS:
            if key not in segment:
                raise AdapterError(f"internal error: segment[{i}] missing key {key!r}")
        start = float(segment["start"])
        end = float(segment["end"])
        if end < start:
            raise AdapterError(
                f"internal error: segment[{i}] end ({end}) is before start ({start})"
            )
        normalized_segments.append({"start": start, "end": end, "text": str(segment["text"])})

    if language is not None and not isinstance(language, str):
        raise AdapterError(f"internal error: language must be a string or null, got: {language!r}")

    return {"text": str(text), "language": language, "segments": normalized_segments}


def write_result(result: dict[str, Any], output_path: Path) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=".asr-", suffix=".json.tmp", dir=str(output_path.parent))
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
# NeMo engine
# ---------------------------------------------------------------------------

NemoModelFactory = Callable[[str, int], Any]


def default_nemo_model_factory(model_path: str, threads: int) -> Any:
    try:
        import torch
        import nemo.collections.asr as nemo_asr
    except ImportError as exc:
        raise AdapterError(
            "Missing dependency for --engine nemo: install the 'nemo' extra "
            "in its own isolated uv environment (see README.md), e.g.:\n"
            "  UV_PROJECT_ENVIRONMENT=.venv-nemo uv sync --extra nemo"
        ) from exc

    torch.set_num_threads(threads)

    try:
        model = nemo_asr.models.ASRModel.restore_from(restore_path=model_path, map_location="cpu")
    except Exception as exc:
        raise AdapterError(
            f"Failed to load the NeMo checkpoint from {model_path!r} "
            f"(original error: {exc})"
        ) from exc

    model.eval()
    if hasattr(model, "freeze"):
        model.freeze()
    return model


def nemo_transcribe(model: Any, input_path: str, duration: float) -> dict[str, Any]:
    """Transcribe with NeMo's documented ``transcribe(..., timestamps=True)`` API.

    Segment-level timestamps are currently only produced by NeMo for
    Parakeet-family models (CTC/RNNT/TDT decoders); other architectures
    raise a ``TypeError`` for the ``timestamps`` kwarg on older NeMo
    releases, or simply omit the ``.timestamp`` attribute on the returned
    hypothesis. In both cases we fall back to a single segment spanning
    the whole file rather than guessing sub-segment timing.
    """
    try:
        hypotheses = model.transcribe([input_path], batch_size=1, timestamps=True)
    except TypeError:
        hypotheses = model.transcribe([input_path], batch_size=1)
    except Exception as exc:
        raise AdapterError(f"NeMo transcription failed for {input_path!r}: {exc}") from exc

    if not hypotheses:
        raise AdapterError(f"NeMo model returned no hypotheses for {input_path!r}")

    hypothesis = hypotheses[0]
    if isinstance(hypothesis, (tuple, list)):
        hypothesis = hypothesis[0]

    text = hypothesis if isinstance(hypothesis, str) else getattr(hypothesis, "text", None)
    if text is None:
        raise AdapterError(
            "NeMo hypothesis has no usable text (neither a plain string nor "
            "a 'text' attribute)"
        )

    segments: list[dict[str, Any]] = []
    timestamp = getattr(hypothesis, "timestamp", None)
    if isinstance(timestamp, dict):
        for stamp in timestamp.get("segment") or []:
            segment_text = stamp.get("segment", stamp.get("word", stamp.get("char", "")))
            segments.append({"start": stamp["start"], "end": stamp["end"], "text": segment_text})

    if not segments:
        # No model-provided timestamps: report the whole file as one segment.
        segments = [{"start": 0.0, "end": duration, "text": text}]

    # NeMo ASR checkpoints are monolingual and expose no language detection.
    return {"text": text, "language": None, "segments": segments}


# ---------------------------------------------------------------------------
# Whisper (Transformers) engine
# ---------------------------------------------------------------------------

WhisperPipelineFactory = Callable[..., Any]


def default_whisper_pipeline_factory(
    model_dir: str, threads: int, *, offload_folder: str | None = None
) -> Any:
    try:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    except ImportError as exc:
        raise AdapterError(
            "Missing dependency for --engine whisper: install the "
            "'whisper' extra in its own isolated uv environment (see "
            "README.md), e.g.:\n"
            "  UV_PROJECT_ENVIRONMENT=.venv-whisper uv sync --extra whisper"
        ) from exc

    torch.set_num_threads(threads)

    try:
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        load_options: dict[str, Any] = {"dtype": torch.float32, "local_files_only": True}
        if offload_folder is not None:
            load_options.update(
                device_map="auto",
                max_memory={"cpu": "2GiB"},
                offload_folder=offload_folder,
                offload_state_dict=True,
            )
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_dir, **load_options)
        pipeline_options: dict[str, Any] = {
            "task": "automatic-speech-recognition",
            "model": model,
            "tokenizer": processor.tokenizer,
            "feature_extractor": processor.feature_extractor,
        }
        if offload_folder is None:
            pipeline_options["device"] = "cpu"
        asr_pipeline = pipeline(**pipeline_options)
    except Exception as exc:
        raise AdapterError(
            f"Failed to load the Whisper Transformers checkpoint from "
            f"{model_dir!r} (original error: {exc})"
        ) from exc
    return asr_pipeline


def whisper_transcribe(asr_pipeline: Any, input_path: str, duration: float) -> dict[str, Any]:
    """Transcribe with the documented ``automatic-speech-recognition`` pipeline.

    ``return_timestamps=True`` yields segment-level ``chunks`` (start/end +
    text); ``return_language=True`` additionally populates a per-chunk
    ``language`` field (Whisper-only, per the pipeline's own validation).
    """
    try:
        output = asr_pipeline(
            input_path,
            chunk_length_s=30,
            return_timestamps=True,
            return_language=True,
        )
    except Exception as exc:
        raise AdapterError(f"Whisper transcription failed for {input_path!r}: {exc}") from exc

    if not isinstance(output, dict):
        raise AdapterError(
            f"Unexpected Whisper pipeline output type: {type(output).__name__}"
        )

    text = output.get("text", "")
    chunks = output.get("chunks") or []

    segments: list[dict[str, Any]] = []
    language: str | None = None
    for chunk in chunks:
        start, end = chunk.get("timestamp", (None, None))
        if start is None:
            continue
        if end is None:
            # The final chunk's closing timestamp token is sometimes not
            # resolved; treat it as running to the end of the audio rather
            # than inventing a shorter time or dropping the text.
            end = duration
        segments.append({"start": start, "end": end, "text": chunk.get("text", "")})
        chunk_language = chunk.get("language")
        if chunk_language and language is None:
            language = chunk_language

    return {"text": text, "language": language, "segments": segments}


# ---------------------------------------------------------------------------
# Qwen3-ASR (Transformers) engine
# ---------------------------------------------------------------------------

QwenModelFactory = Callable[..., tuple[Any, Any]]


def default_qwen_model_factory(
    model_dir: str, threads: int, *, offload_folder: str | None = None
) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:
        raise AdapterError(
            "Missing dependency for --engine qwen: install the 'qwen' "
            "extra (requires transformers>=4.57.6 with Qwen3-ASR support) "
            "in its own isolated uv environment (see README.md), e.g.:\n"
            "  UV_PROJECT_ENVIRONMENT=.venv-qwen uv sync --extra qwen"
        ) from exc

    torch.set_num_threads(threads)

    try:
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        load_options: dict[str, Any] = {"dtype": torch.float32, "local_files_only": True}
        if offload_folder is not None:
            load_options.update(
                device_map="auto",
                max_memory={"cpu": "2GiB"},
                offload_folder=offload_folder,
                offload_state_dict=True,
            )
        model = AutoModelForMultimodalLM.from_pretrained(model_dir, **load_options)
    except Exception as exc:
        raise AdapterError(
            f"Failed to load the Qwen3-ASR checkpoint from {model_dir!r} "
            f"(original error: {exc})"
        ) from exc

    if offload_folder is None:
        model.to("cpu")
    model.eval()
    return model, processor


def qwen_transcribe(model: Any, processor: Any, input_path: str, duration: float) -> dict[str, Any]:
    """Transcribe with Qwen3-ASR's documented ``apply_transcription_request`` API.

    Qwen3-ASR alone does not emit segment-level timestamps (that requires a
    separate Qwen3-ForcedAligner checkpoint, out of scope for this
    single-model adapter contract), so one segment spanning the whole
    input is reported instead of fabricating sub-segment timing.

    ``model.generate`` is a ``@torch.no_grad()``-decorated method in
    Transformers' ``GenerationMixin``, so no separate ``torch.no_grad()``
    context is needed here (and none is imported) to keep this
    function testable without ``torch`` installed.
    """
    try:
        inputs = processor.apply_transcription_request(audio=input_path)
        inputs = inputs.to(model.device, model.dtype)
    except Exception as exc:
        raise AdapterError(
            f"Failed to prepare Qwen3-ASR inputs for {input_path!r}: {exc}"
        ) from exc

    generation_config = model.generation_config
    max_tokens = generation_config.max_new_tokens
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise AdapterError("Qwen3-ASR generation_config.max_new_tokens must be a positive integer")
    try:
        output_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    except Exception as exc:
        raise AdapterError(f"Qwen3-ASR generation failed for {input_path!r}: {exc}") from exc

    generated_count = output_ids.shape[1] - inputs["input_ids"].shape[1]
    if generated_count >= max_tokens:
        eos_ids = generation_config.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else eos_ids
        if not eos_ids or output_ids[0, -1].item() not in eos_ids:
            raise AdapterError(
                f"Qwen3-ASR hit the {max_tokens}-token generation limit without EOS for "
                f"{input_path!r}; the response may be incomplete"
            )

    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    try:
        parsed = processor.decode(generated_ids, return_format="parsed")
    except Exception as exc:
        raise AdapterError(f"Failed to decode Qwen3-ASR output: {exc}") from exc

    if not parsed:
        raise AdapterError("Qwen3-ASR produced no output")

    entry = parsed[0]
    text = entry.get("transcription", "") or ""
    language = entry.get("language")

    segments = [{"start": 0.0, "end": duration, "text": text}] if text else []
    return {"text": text, "language": language, "segments": segments}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

TranscribeFn = Callable[[str, float], dict[str, Any]]


def build_transcribe_fn(
    engine: str,
    model_path: Path,
    threads: int,
    *,
    offload_folder: Path | None = None,
    nemo_model_factory: NemoModelFactory | None = None,
    whisper_pipeline_factory: WhisperPipelineFactory | None = None,
    qwen_model_factory: QwenModelFactory | None = None,
) -> TranscribeFn:
    """Load the model/pipeline for ``engine`` exactly once.

    Returns a bound callable ``(input_path, duration) -> raw_result`` that
    reuses the already-loaded model for every call. This is the single
    choke point that both single-clip mode (one call) and ``--serve`` mode
    (one call per stdin request, sharing the same loaded model) go through,
    guaranteeing the heavy ``default_*_factory`` is invoked at most once per
    process regardless of how many clips are transcribed.
    """
    if engine == "nemo":
        factory = nemo_model_factory or default_nemo_model_factory
        model = factory(str(model_path), threads)
        return lambda input_path, duration: nemo_transcribe(model, input_path, duration)
    elif engine == "whisper":
        factory = whisper_pipeline_factory or default_whisper_pipeline_factory
        asr_pipeline = (
            factory(str(model_path), threads, offload_folder=str(offload_folder))
            if offload_folder is not None else factory(str(model_path), threads)
        )
        return lambda input_path, duration: whisper_transcribe(asr_pipeline, input_path, duration)
    elif engine == "qwen":
        factory = qwen_model_factory or default_qwen_model_factory
        model, processor = (
            factory(str(model_path), threads, offload_folder=str(offload_folder))
            if offload_folder is not None else factory(str(model_path), threads)
        )
        return lambda input_path, duration: qwen_transcribe(model, processor, input_path, duration)
    else:  # pragma: no cover - guarded by argparse choices
        raise AdapterError(f"Unknown engine: {engine}")


def serve(
    transcribe_fn: TranscribeFn,
    *,
    in_stream: TextIO,
    out_stream: TextIO,
) -> int:
    """Persistent worker loop: one JSON-lines request in, one response out.

    The model is already loaded (``transcribe_fn`` is bound to it) before
    this loop starts, so no model loading happens per request. Reads
    newline-delimited ``{"input": "/abs/clip.wav"}`` requests from
    ``in_stream`` until EOF (returns 0) or the first failure: on failure,
    writes a single ``{"error": "..."}`` JSON line to ``out_stream``, logs
    diagnostics to stderr, and returns 1 immediately without attempting to
    process further requests. ``out_stream`` only ever receives JSON lines,
    one per request, flushed immediately so a caller reading it as a pipe
    sees each response as soon as it is ready.
    """
    for raw_line in in_stream:
        line = raw_line.strip()
        if not line:
            continue

        error_message: str | None = None
        result: dict[str, Any] | None = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or not isinstance(request.get("input"), str):
                raise AdapterError(
                    "request must be a JSON object with a string 'input' "
                    f"key, got: {line!r}"
                )
            input_path, duration = validate_input_wav(request["input"])
            with contextlib.redirect_stdout(sys.stderr):
                raw_result = transcribe_fn(str(input_path), duration)
            result = build_result(raw_result["text"], raw_result["language"], raw_result["segments"])
        except AdapterError as exc:
            error_message = str(exc)
        except json.JSONDecodeError as exc:
            error_message = f"invalid JSON request line: {exc}"
        except Exception as exc:  # pragma: no cover - defensive: unexpected engine failure
            error_message = f"unexpected failure while transcribing request: {exc}"

        if error_message is not None:
            print(f"error: {error_message}", file=sys.stderr)
            out_stream.write(json.dumps({"error": error_message}, ensure_ascii=False) + "\n")
            out_stream.flush()
            return 1

        assert result is not None
        out_stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_stream.flush()

    return 0


def run(
    argv: Sequence[str] | None = None,
    *,
    nemo_model_factory: NemoModelFactory | None = None,
    whisper_pipeline_factory: WhisperPipelineFactory | None = None,
    qwen_model_factory: QwenModelFactory | None = None,
    in_stream: TextIO | None = None,
    out_stream: TextIO | None = None,
) -> int:
    args = parse_args(argv)
    try:
        threads = validate_threads(args.threads)
        model_path = validate_model_path(args.engine, args.model)
        offload_folder = validate_offload_dir(args.engine, model_path, args.offload_dir)
    except AdapterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    configure_offline_environment()
    configure_thread_environment(threads)

    if args.serve:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                transcribe_fn = build_transcribe_fn(
                    args.engine,
                    model_path,
                    threads,
                    offload_folder=offload_folder,
                    nemo_model_factory=nemo_model_factory,
                    whisper_pipeline_factory=whisper_pipeline_factory,
                    qwen_model_factory=qwen_model_factory,
                )
        except AdapterError as exc:
            print(f"error: {exc}", file=sys.stderr)
            (out_stream or sys.stdout).write(json.dumps({"error": str(exc)}, ensure_ascii=False) + "\n")
            (out_stream or sys.stdout).flush()
            return 1
        return serve(
            transcribe_fn,
            in_stream=in_stream or sys.stdin,
            out_stream=out_stream or sys.stdout,
        )

    try:
        input_path, duration = validate_input_wav(args.input)
        output_path = validate_output_path(args.output)
        transcribe_fn = build_transcribe_fn(
            args.engine,
            model_path,
            threads,
            offload_folder=offload_folder,
            nemo_model_factory=nemo_model_factory,
            whisper_pipeline_factory=whisper_pipeline_factory,
            qwen_model_factory=qwen_model_factory,
        )
        raw_result = transcribe_fn(str(input_path), duration)
        result = build_result(raw_result["text"], raw_result["language"], raw_result["segments"])
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
