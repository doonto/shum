#!/usr/bin/env python3
"""Shared, dependency-light helpers for the ``flix_fast_*`` prototype.

This module backs both ``flix_fast_convert.py`` (original pinned Hugging
Face Whisper checkpoint -> CTranslate2 int8 model directory) and
``flix_fast_worker.py`` (CLI-compatible transcription worker built on top
of the converted directory, via ``faster-whisper``/CTranslate2).

Everything here is standard-library only (``json``/``hashlib``/``pathlib``)
so both callers can validate arguments and resolve paths without importing
``torch``/``transformers``/``ctranslate2``/``faster_whisper`` -- exactly the
same "lazy heavy imports" discipline ``main.py`` uses, so this module (and
its unit tests) never require those large dependencies to be installed.

This is a standalone prototype: it does not import, and is not imported
by, ``main.py``. See ``flix_fast_README.md`` for scope, status, and the
provenance/compatibility notes summarized below.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = ".shum-manifest.json"
PROVENANCE_FILENAME = "flix-fast-provenance.json"
PREPROCESSOR_CONFIG_FILENAME = "preprocessor_config.json"

# Keys accepted by faster-whisper's ``FeatureExtractor.__init__``. Only
# these are copied into the flattened preprocessor config written by the
# converter (see ``flatten_feature_extractor_config`` below).
FEATURE_EXTRACTOR_KEYS = ("feature_size", "sampling_rate", "hop_length", "chunk_length", "n_fft")

QUANTIZATION_CHOICES = (
    "int8",
    "int8_float32",
    "int8_float16",
    "int8_bfloat16",
    "int16",
    "float16",
    "bfloat16",
    "float32",
)

OFFLINE_ENV_VARS = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


class AdapterError(RuntimeError):
    """Explicit, user-facing failure (never a silent fallback or retry)."""


def configure_offline_environment() -> None:
    """Force huggingface_hub/transformers into offline mode.

    Must run before ``torch``/``transformers``/``ctranslate2`` are
    imported, mirroring ``main.py``'s ``configure_offline_environment``.
    """
    os.environ.update(OFFLINE_ENV_VARS)


def configure_thread_environment(threads: int) -> None:
    """Pin CPU thread counts before numpy/torch/ctranslate2 are imported."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads)


# ---------------------------------------------------------------------------
# Model directory validation & provenance
# ---------------------------------------------------------------------------

def require_absolute(value: str, flag: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise AdapterError(f"{flag} must be an absolute path, got: {value!r}")
    return path


def validate_whisper_model_dir(model_path: str) -> Path:
    """Validate ``--model`` is a local Transformers Whisper checkpoint dir.

    Only reads ``config.json`` as plain JSON (no ``transformers`` import),
    so this stays usable without the heavy conversion/inference
    dependencies installed, exactly like ``main.py``'s validation layer.
    """
    path = require_absolute(model_path, "--model")
    if not path.is_dir():
        raise AdapterError(f"--model directory does not exist: {path}")
    config_path = path / "config.json"
    if not config_path.is_file():
        raise AdapterError(
            f"--model directory is missing config.json (not a valid local "
            f"Transformers checkpoint): {path}"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"--model config.json is not readable/valid JSON: {config_path} ({exc})") from exc

    architectures = config.get("architectures") or []
    model_type = config.get("model_type")
    if model_type != "whisper" and "WhisperForConditionalGeneration" not in architectures:
        raise AdapterError(
            "--model does not look like a Whisper checkpoint (expected "
            "model_type == 'whisper' or architectures including "
            f"'WhisperForConditionalGeneration') in {config_path}"
        )
    return path


def read_manifest(model_dir: Path) -> dict[str, Any] | None:
    """Read the optional ``.shum-manifest.json`` (repo/revision) if present.

    This file is written by shum's model manager for pinned local
    checkouts (see e.g. ``models/flix/<revision>/.shum-manifest.json``). It
    is entirely optional: any local Transformers Whisper checkpoint
    directory works without one, falling back to a content hash of the
    directory path for cache-key/provenance purposes.
    """
    manifest_path = model_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"{manifest_path} is not readable/valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise AdapterError(f"{manifest_path} must contain a JSON object")
    return data


def model_identity(model_dir: Path) -> str:
    """A short, stable identifier for ``model_dir`` used as a cache key.

    Prefers the pinned revision from ``.shum-manifest.json`` (readable,
    matches the model directory's own name for shum-managed checkouts);
    falls back to a hash of the resolved absolute path for arbitrary local
    checkpoints so the prototype is not limited to shum-managed models.
    """
    manifest = read_manifest(model_dir)
    if manifest is not None:
        revision = manifest.get("revision")
        if isinstance(revision, str) and revision:
            return revision
    return hashlib.sha256(os.fsencode(model_dir.resolve())).hexdigest()[:16]


def read_generation_defaults(model_dir: Path) -> tuple[str | None, str]:
    """Read the checkpoint's default (language, task) from generation_config.json.

    Falls back to ``(None, "transcribe")`` (auto-detect language, transcribe
    task) when the file or the fields are absent, which matches
    ``transformers``' own ``GenerationConfig``/pipeline defaults.
    """
    config_path = model_dir / "generation_config.json"
    if not config_path.is_file():
        return None, "transcribe"
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"{config_path} is not readable/valid JSON: {exc}") from exc
    language = data.get("language")
    task = data.get("task") or "transcribe"
    return (language if isinstance(language, str) else None), task


def flatten_feature_extractor_config(model_dir: Path) -> dict[str, Any]:
    """Build a flat ``preprocessor_config.json`` payload for faster-whisper.

    faster-whisper's ``WhisperModel`` reads a *flat* ``preprocessor_config.json``
    (top-level ``feature_size``/``sampling_rate``/``hop_length``/
    ``chunk_length``/``n_fft`` keys) from the CTranslate2 output directory.
    Newer ``transformers`` releases (this checkpoint declares
    ``transformers_version: 5.0.0``) instead save a unified
    ``processor_config.json`` with those same fields *nested* under a
    ``"feature_extractor"`` object -- copying that file as-is would silently
    leave faster-whisper on its 80-mel-bin default and break every
    transcription with a shape mismatch against this checkpoint's 128 mel
    bins. This function reads whichever original processor/preprocessor
    config file is present and returns the flat dict faster-whisper expects,
    restricted to the keys its ``FeatureExtractor`` constructor accepts.
    """
    for filename in ("processor_config.json", "preprocessor_config.json"):
        config_path = model_dir / filename
        if config_path.is_file():
            break
    else:
        raise AdapterError(
            f"--model directory has neither processor_config.json nor "
            f"preprocessor_config.json: {model_dir}"
        )
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"{config_path} is not readable/valid JSON: {exc}") from exc

    nested = data.get("feature_extractor")
    source = nested if isinstance(nested, dict) else data
    flat = {key: source[key] for key in FEATURE_EXTRACTOR_KEYS if key in source}
    if "feature_size" not in flat:
        raise AdapterError(
            f"Could not determine feature_size (mel bin count) from {config_path}; "
            "refusing to guess a default that could silently corrupt audio features."
        )
    return flat


# ---------------------------------------------------------------------------
# CTranslate2 cache resolution (shared by convert.py and worker.py so both
# agree on the default cache layout without either importing the other).
# ---------------------------------------------------------------------------

def default_cache_root() -> Path:
    env = os.environ.get("SHUM_FLIX_FAST_CACHE")
    if env:
        return require_absolute(env, "SHUM_FLIX_FAST_CACHE")
    return Path.home() / ".cache" / "shum" / "flix-fast-ct2"


def resolve_ct2_output_dir(model_dir: Path, *, cache_root: str | None, quantization: str) -> Path:
    """Resolve the default CTranslate2 output directory for ``model_dir``.

    Deliberately lives *outside* the shum repository and outside the
    original model snapshot (default: ``~/.cache/shum/flix-fast-ct2/...``),
    so the converted copy never touches ``.shum-manifest.json``-tracked
    original files and is never accidentally committed.
    """
    root = require_absolute(cache_root, "--cache-root") if cache_root else default_cache_root()
    return root / model_identity(model_dir) / quantization


# ---------------------------------------------------------------------------
# Provenance file (written by flix_fast_convert.py, verified by flix_fast_worker.py)
# ---------------------------------------------------------------------------

def load_provenance(ct2_dir: Path) -> dict[str, Any]:
    provenance_path = ct2_dir / PROVENANCE_FILENAME
    if not provenance_path.is_file():
        raise AdapterError(
            f"{provenance_path} is missing: {ct2_dir} was not produced by "
            "flix_fast_convert.py (or is from an older/incompatible run). "
            "Re-run the converter before using this directory."
        )
    try:
        data = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"{provenance_path} is not readable/valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise AdapterError(f"{provenance_path} must contain a JSON object")
    return data


def verify_provenance_matches_model(provenance: dict[str, Any], model_dir: Path) -> None:
    """Refuse to serve a CTranslate2 directory converted from a different checkpoint.

    Compares the ``source_identity`` recorded at conversion time against
    the identity of the ``--model`` directory passed to the worker now.
    This is a safety net on top of the pinned-revision directory layout:
    it catches a ``--ct2-model`` pointed at the wrong/stale cache entry
    (e.g. explicitly overridden, or the cache moved) before any audio is
    transcribed with mismatched weights/vocabulary/mel-bin config.
    """
    expected = provenance.get("source_identity")
    actual = model_identity(model_dir)
    if expected != actual:
        raise AdapterError(
            "CTranslate2 model provenance mismatch: converted from "
            f"source_identity={expected!r} but --model resolves to "
            f"{actual!r} ({model_dir}). Re-run flix_fast_convert.py against "
            "this --model, or point --ct2-model at the matching cache entry."
        )


# ---------------------------------------------------------------------------
# WAV validation, output-path validation & atomic JSON writing (mirrors
# main.py's contract exactly so results are byte-for-byte compatible).
# ---------------------------------------------------------------------------

def validate_threads(threads: int) -> int:
    if threads < 1:
        raise AdapterError(f"--threads must be a positive integer, got: {threads}")
    return threads


def validate_input_wav(input_path: str) -> tuple[Path, float]:
    import contextlib
    import wave

    path = require_absolute(input_path, "--input")
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
              "this worker will not silently resample or downmix audio."
        )
    duration = n_frames / float(rate) if rate else 0.0
    return path, duration


def validate_output_path(output_path: str) -> Path:
    path = require_absolute(output_path, "--output")
    parent = path.parent
    if not parent.is_dir():
        raise AdapterError(f"--output directory does not exist: {parent}")
    return path


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
            raise AdapterError(f"internal error: segment[{i}] end ({end}) is before start ({start})")
        normalized_segments.append({"start": start, "end": end, "text": str(segment["text"])})

    if language is not None and not isinstance(language, str):
        raise AdapterError(f"internal error: language must be a string or null, got: {language!r}")

    return {"text": str(text), "language": language, "segments": normalized_segments}


def write_result(result: dict[str, Any], output_path: Path) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=".flix-fast-", suffix=".json.tmp", dir=str(output_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(result, tmp_file, ensure_ascii=False, indent=2)
            tmp_file.write("\n")
        os.replace(tmp_name, output_path)
    except BaseException:
        import contextlib as _contextlib

        with _contextlib.suppress(OSError):
            os.remove(tmp_name)
        raise
