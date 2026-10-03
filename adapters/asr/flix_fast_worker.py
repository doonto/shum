#!/usr/bin/env python3
"""CLI-compatible fast Flix Whisper worker (CTranslate2 / faster-whisper).

Prototype standalone adapter: same CLI/JSON contract as
``main.py --engine whisper`` (single-clip and ``--serve`` modes, identical
``{text, language, segments}`` output schema, identical 16 kHz mono WAV
input validation, identical offline/CPU-thread environment setup), but
backed by a CTranslate2 int8 model converted from the *original* pinned HF
checkpoint via ``flix_fast_convert.py`` instead of loading the full
Transformers model directly. It is entirely standalone: it does not
import, and is not imported by, ``main.py`` (some validation/env-setup
logic is intentionally duplicated rather than shared, matching that file's
own "no cross-adapter dependency" design).

Usage (single clip):
    python flix_fast_worker.py \
        --model /absolute/models/flix/<pinned-revision> \
        --input /absolute/16khz/mono.wav \
        --output /absolute/result.json \
        --threads N

Usage (persistent worker, model loaded exactly once):
    python flix_fast_worker.py \
        --model /absolute/models/flix/<pinned-revision> \
        --threads N --serve

``--model`` is always the *original* HF checkpoint directory (for
provenance/config: revision manifest, generation defaults, mel-bin count)
-- this worker never reads model weights from it. The actual CTranslate2
weights are resolved from (in order): an explicit ``--ct2-model``, or the
default cache layout under ``--cache-root``/``SHUM_FLIX_FAST_CACHE``
(``flix_fast_common.resolve_ct2_output_dir``). This worker deliberately
does **not** auto-convert on a cache miss: converting loads the full
checkpoint into RAM, which this "fast worker" should never do silently as
a side effect of starting up. Run ``flix_fast_convert.py`` first; a clear
``AdapterError`` explains this if the cache entry is missing.

See ``flix_fast_README.md`` for current status, the quality/compatibility
notes, and the feasibility analysis for the <10 minute / 30 minute audio
target this prototype was built to investigate.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

import flix_fast_common as common
from flix_fast_common import AdapterError

WorkerFactory = Callable[..., Any]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flix_fast_worker.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--engine",
        default=None,
        choices=("whisper",),
        help=(
            "Optional, for CLI-contract parity with main.py's --engine "
            "flag: the Rust pipeline always passes '--engine whisper' for "
            "a whisper-engine role, so this worker accepts and strictly "
            "validates that value (any other engine is rejected by "
            "argparse's choices check) while staying optional for manual/"
            "standalone runs that omit it entirely."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Absolute path to the original local HF Transformers Whisper checkpoint directory (for provenance/config only).",
    )
    parser.add_argument(
        "--ct2-model",
        default=None,
        help="Absolute path to an already-converted CTranslate2 model directory. Overrides cache resolution.",
    )
    parser.add_argument(
        "--cache-root",
        default=None,
        help="Absolute cache root used to resolve the converted model when --ct2-model is omitted.",
    )
    parser.add_argument(
        "--quantization",
        default="int8",
        choices=common.QUANTIZATION_CHOICES,
        help="Quantization used to resolve the cached converted model directory (default: int8). Must match how flix_fast_convert.py was run.",
    )
    parser.add_argument("--input", default=None, help="Absolute path to a 16 kHz mono WAV file to transcribe.")
    parser.add_argument("--output", default=None, help="Absolute path to write the resulting transcription JSON to.")
    parser.add_argument("--threads", required=True, type=int, help="Number of CPU threads to use for inference.")
    parser.add_argument(
        "--offload-dir",
        default=None,
        help=(
            "Not supported by this worker (rejected with a clear error in "
            "run()): main.py accepts --offload-dir for its Transformers-"
            "based whisper/qwen engines, but this worker serves an "
            "already-quantized CTranslate2 model that has no equivalent "
            "disk-offload mode. Accepted here only so a misconfigured "
            "--offload-dir is reported explicitly instead of failing as an "
            "unrecognized argument."
        ),
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run as a persistent worker (see module docstring). Mutually exclusive with --input/--output.",
    )
    parser.add_argument("--language", default=None, help="Force a language code; default: read from the checkpoint's generation_config.json.")
    parser.add_argument("--task", default=None, choices=("transcribe", "translate"), help="Default: read from the checkpoint's generation_config.json (normally 'transcribe').")
    parser.add_argument("--beam-size", type=int, default=1, help="Beam search width (default: 1 = greedy, matching main.py's whisper engine default).")
    parser.add_argument("--best-of", type=int, default=1, help="Candidates when sampling with temperature>0 (default: 1).")
    parser.add_argument("--seed", type=int, default=None, help="Optional CTranslate2 sampling seed (0..4294967295); default: engine-controlled.")
    parser.add_argument(
        "--reset-seed-per-request", action="store_true",
        help="Request a seed reset before each WAV; native worker sampling can remain stateful (experimental).",
    )
    parser.add_argument(
        "--fast-decode",
        action="store_true",
        help="Use one greedy decoding pass without fallback retries; experimental, verify accuracy against the default profile.",
    )
    parser.add_argument(
        "--vad-filter",
        action="store_true",
        help="Enable faster-whisper's Silero VAD filter (default: off, for parity with main.py's whisper engine which does not filter silence).",
    )
    parser.add_argument(
        "--diagnostics-dir",
        default=None,
        help="Write optional local load/decode diagnostics to a new empty absolute directory; normal ASR JSON is unchanged.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.seed is not None and not 0 <= args.seed <= 4294967295:
        parser.error("--seed must be between 0 and 4294967295")
    if args.reset_seed_per_request and args.seed is None:
        parser.error("--reset-seed-per-request requires --seed")
    if args.serve:
        if args.input is not None or args.output is not None:
            parser.error("--serve cannot be combined with --input/--output")
    else:
        missing = [name for name, value in (("--input", args.input), ("--output", args.output)) if value is None]
        if missing:
            parser.error(f"the following arguments are required: {', '.join(missing)}")
    return args


# ---------------------------------------------------------------------------
# Model loading & transcription
# ---------------------------------------------------------------------------

def default_worker_factory(ct2_dir: str, threads: int, *, compute_type: str) -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise AdapterError(
            "Missing dependency: install faster-whisper (+ ctranslate2) in "
            "an isolated environment, e.g.:\n"
            "  uv venv --python 3.11 .venv-flix-fast\n"
            "  uv pip install --python .venv-flix-fast/bin/python "
            "-r flix_fast_requirements.txt --extra-index-url "
            "https://download.pytorch.org/whl/cpu\n"
            "(see flix_fast_README.md)"
        ) from exc
    try:
        return WhisperModel(ct2_dir, device="cpu", compute_type=compute_type, cpu_threads=threads, local_files_only=True)
    except Exception as exc:
        raise AdapterError(f"Failed to load the CTranslate2 model from {ct2_dir!r} (original error: {exc})") from exc


def transcribe(
    model: Any,
    input_path: str,
    duration: float,
    *,
    language: str | None,
    task: str,
    beam_size: int,
    best_of: int,
    vad_filter: bool,
    fast_decode: bool = False,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        decode_options = (
            {"temperature": 0.0, "compression_ratio_threshold": None, "log_prob_threshold": None}
            if fast_decode else {}
        )
        segments, info = model.transcribe(
            input_path,
            language=language,
            task=task,
            beam_size=beam_size,
            best_of=best_of,
            vad_filter=vad_filter,
            word_timestamps=False,
            condition_on_previous_text=False,
            **decode_options,
        )
        segment_list = list(segments)
    except Exception as exc:
        raise AdapterError(f"Whisper transcription failed for {input_path!r}: {exc}") from exc

    result_segments: list[dict[str, Any]] = []
    for seg in segment_list:
        end = seg.end if seg.end is not None else duration
        result_segments.append({"start": seg.start, "end": end, "text": seg.text})

    text = "".join(seg.text for seg in segment_list).strip()
    resolved_language = getattr(info, "language", None) or language
    if diagnostics is not None:
        options = getattr(info, "transcription_options", None)
        vad_options = getattr(info, "vad_options", None)
        diagnostics.update({
            "input_path": input_path,
            "input_duration_seconds": duration,
            "language": resolved_language,
            "task": task,
            "transcription_options": dataclasses.asdict(options) if dataclasses.is_dataclass(options) else None,
            "vad_options": dataclasses.asdict(vad_options) if dataclasses.is_dataclass(vad_options) else None,
            "duration_after_vad_seconds": getattr(info, "duration_after_vad", None),
            "vad_retained_intervals": None,
            "vad_interval_status": "not_exposed_by_transcription_info",
            "decoder_token_limit": getattr(model, "max_length", None),
            "truncation_status": "not_exposed_by_engine",
            "fallback_attempts": None,
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "tokens": getattr(seg, "tokens", None),
                    "token_count": len(seg.tokens) if getattr(seg, "tokens", None) is not None else None,
                    "temperature": getattr(seg, "temperature", None),
                    "avg_logprob": getattr(seg, "avg_logprob", None),
                    "compression_ratio": getattr(seg, "compression_ratio", None),
                    "no_speech_prob": getattr(seg, "no_speech_prob", None),
                }
                for seg in segment_list
            ],
        })
    return {"text": text, "language": resolved_language, "segments": result_segments}


# ---------------------------------------------------------------------------
# Orchestration (mirrors main.py's build_transcribe_fn/serve/run shape)
# ---------------------------------------------------------------------------

TranscribeFn = Callable[[str, float], dict[str, Any]]


def default_seed_setter(seed: int) -> None:
    try:
        import ctranslate2
        ctranslate2.set_random_seed(seed)
    except (ImportError, RuntimeError, ValueError) as exc:
        raise AdapterError(f"Cannot set CTranslate2 sampling seed: {exc}") from exc


def build_transcribe_fn(
    model_dir: Path,
    threads: int,
    *,
    ct2_model: str | None,
    cache_root: str | None,
    quantization: str,
    language_override: str | None,
    task_override: str | None,
    beam_size: int,
    best_of: int,
    vad_filter: bool,
    worker_factory: WorkerFactory | None = None,
    fast_decode: bool = False,
    diagnostics_dir: Path | None = None,
    seed: int | None = None,
    reset_seed_per_request: bool = False,
    seed_setter: Callable[[int], None] | None = None,
) -> TranscribeFn:
    if seed is not None and (type(seed) is not int or not 0 <= seed <= 4294967295):
        raise AdapterError("sampling seed must be an integer between 0 and 4294967295")
    if reset_seed_per_request and seed is None:
        raise AdapterError("reset_seed_per_request requires a seed")
    if ct2_model is not None:
        ct2_dir = common.require_absolute(ct2_model, "--ct2-model")
        if not ct2_dir.is_dir():
            raise AdapterError(f"--ct2-model directory does not exist: {ct2_dir}")
    else:
        ct2_dir = common.resolve_ct2_output_dir(model_dir, cache_root=cache_root, quantization=quantization)
        if not ct2_dir.is_dir():
            raise AdapterError(
                f"No converted CTranslate2 model found at {ct2_dir} for --model "
                f"{model_dir} (quantization={quantization}). Run: python "
                f"flix_fast_convert.py --model {model_dir} --quantization {quantization} "
                "first, or pass --ct2-model explicitly."
            )

    provenance = common.load_provenance(ct2_dir)
    common.verify_provenance_matches_model(provenance, model_dir)
    if provenance.get("quantization") != quantization:
        raise AdapterError(
            f"--quantization {quantization!r} does not match the quantization "
            f"recorded in {ct2_dir}'s provenance ({provenance.get('quantization')!r})"
        )

    manifest_language, manifest_task = common.read_generation_defaults(model_dir)
    generation_defaults = provenance.get("generation_defaults") or {}
    default_language = manifest_language or generation_defaults.get("language")
    default_task = task_override or manifest_task or generation_defaults.get("task") or "transcribe"
    language = language_override or default_language

    factory = worker_factory or default_worker_factory
    set_seed = seed_setter or default_seed_setter
    if seed is not None:
        set_seed(seed)
    load_started = time.monotonic()
    model = factory(str(ct2_dir), threads, compute_type=quantization)
    if diagnostics_dir is not None:
        versions = {}
        for package in ("faster-whisper", "ctranslate2"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        backend = getattr(model, "model", None)
        write_diagnostics(diagnostics_dir / "ready.json", {
            "ready_monotonic_seconds": time.monotonic(),
            "model_load_seconds": time.monotonic() - load_started,
            "requested_compute_type": quantization,
            "effective_compute_type": getattr(backend, "compute_type", None),
            "versions": versions,
            "python_executable": sys.executable,
            "python_prefix": sys.prefix,
            "python_version": sys.version,
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "thread_environment": {
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
            },
            "source_identity": provenance.get("source_identity"),
            "language": language,
            "task": default_task,
            "threads": threads,
            "beam_size": beam_size,
            "best_of": best_of,
            "vad_filter": vad_filter,
            "fast_decode": fast_decode,
            "condition_on_previous_text": False,
            "seed": seed,
            "seed_policy": "uncontrolled" if seed is None else ("per_request_reset_requested" if reset_seed_per_request else "per_worker"),
            "seed_reproducibility": "not_guaranteed",
        })

    request_index = 0

    def transcribe_fn(input_path: str, duration: float) -> dict[str, Any]:
        nonlocal request_index
        started = time.monotonic()
        if reset_seed_per_request and seed is not None:
            set_seed(seed)
        diagnostic: dict[str, Any] | None = {} if diagnostics_dir is not None else None
        result = transcribe(
            model, input_path, duration, language=language, task=default_task,
            beam_size=beam_size, best_of=best_of, vad_filter=vad_filter,
            fast_decode=fast_decode, diagnostics=diagnostic,
        )
        if diagnostics_dir is not None and diagnostic is not None:
            diagnostic["decode_seconds"] = time.monotonic() - started
            diagnostic["completed_monotonic_seconds"] = time.monotonic()
            diagnostic["seed"] = seed
            diagnostic["reset_seed_per_request"] = reset_seed_per_request
            write_diagnostics(diagnostics_dir / f"request-{request_index:05d}.json", diagnostic)
            request_index += 1
        return result

    return transcribe_fn


def normalize_diagnostics(value: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    nonfinite_fields: list[dict[str, str]] = []

    def normalize(item: Any, key: str) -> Any:
        if isinstance(item, float) and not math.isfinite(item):
            encoded = "nan" if math.isnan(item) else ("+inf" if item > 0 else "-inf")
            nonfinite_fields.append({"path": key, "value": encoded})
            return encoded
        if isinstance(item, dict):
            return {name: normalize(child, f"{key}.{name}" if key else name) for name, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [normalize(child, f"{key}[{index}]") for index, child in enumerate(item)]
        return item

    normalized = normalize(value, "")
    normalized["nonfinite_fields"] = nonfinite_fields
    return normalized, nonfinite_fields


def write_diagnostics(path: Path, value: dict[str, Any]) -> None:
    normalized, nonfinite_fields = normalize_diagnostics(value)
    try:
        json.dumps(normalized, allow_nan=False)
        common.write_result(normalized, path)
    except (OSError, TypeError, ValueError) as exc:
        raise AdapterError(f"Cannot write diagnostics to {path}: {exc}") from exc
    unexpected = [
        field for field in nonfinite_fields
        if not (field["path"] == "vad_options.max_speech_duration_s" and field["value"] == "+inf")
    ]
    if unexpected:
        print(f"warning: engine diagnostic values are non-finite in {path}: {unexpected}", file=sys.stderr)


def serve(transcribe_fn: TranscribeFn, *, in_stream: TextIO, out_stream: TextIO) -> int:
    """Identical wire protocol to ``main.py``'s ``serve()``: one JSON-lines
    request in, one response out, in order, until EOF or the first error."""
    for raw_line in in_stream:
        line = raw_line.strip()
        if not line:
            continue

        error_message: str | None = None
        result: dict[str, Any] | None = None
        request_index: int | None = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise AdapterError(f"request must be a JSON object with a string 'input' key, got: {line!r}")
            if "index" in request:
                if type(request["index"]) is not int or request["index"] < 0:
                    raise AdapterError("request 'index' must be a nonnegative integer")
                request_index = request["index"]
            if not isinstance(request.get("input"), str):
                raise AdapterError("request must contain a string 'input' key")
            input_path, duration = common.validate_input_wav(request["input"])
            with contextlib.redirect_stdout(sys.stderr):
                raw_result = transcribe_fn(str(input_path), duration)
            result = common.build_result(raw_result["text"], raw_result["language"], raw_result["segments"])
        except AdapterError as exc:
            error_message = str(exc)
        except json.JSONDecodeError as exc:
            error_message = f"invalid JSON request line: {exc}"
        except Exception as exc:  # pragma: no cover - defensive: unexpected engine failure
            error_message = f"unexpected failure while transcribing request: {exc}"

        if error_message is not None:
            print(f"error: {error_message}", file=sys.stderr)
            error_result: dict[str, Any] = {"error": error_message}
            if request_index is not None:
                error_result["index"] = request_index
            out_stream.write(json.dumps(error_result, ensure_ascii=False) + "\n")
            out_stream.flush()
            return 1

        assert result is not None
        if request_index is not None:
            result["index"] = request_index
        out_stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_stream.flush()

    return 0


def run(
    argv: Sequence[str] | None = None,
    *,
    worker_factory: WorkerFactory | None = None,
    in_stream: TextIO | None = None,
    out_stream: TextIO | None = None,
) -> int:
    args = parse_args(argv)
    try:
        threads = common.validate_threads(args.threads)
        model_dir = common.validate_whisper_model_dir(args.model)
        diagnostics_dir = None
        if args.diagnostics_dir is not None:
            diagnostics_dir = common.require_absolute(args.diagnostics_dir, "--diagnostics-dir")
            try:
                diagnostics_dir.mkdir(mode=0o700, exist_ok=True)
                if any(diagnostics_dir.iterdir()):
                    raise AdapterError(f"--diagnostics-dir must be empty: {diagnostics_dir}")
            except OSError as exc:
                raise AdapterError(f"Cannot prepare diagnostics directory {diagnostics_dir}: {exc}") from exc
        if args.offload_dir is not None:
            raise AdapterError(
                "--offload-dir is not supported by flix_fast_worker.py: "
                "this worker serves an already-quantized CTranslate2 "
                "model, which needs no Transformers-style CPU/disk "
                "offload. Remove --offload-dir for any role configured to "
                "run this worker instead of main.py."
            )
    except AdapterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    common.configure_offline_environment()
    common.configure_thread_environment(threads)

    def make_transcribe_fn() -> TranscribeFn:
        return build_transcribe_fn(
            model_dir,
            threads,
            ct2_model=args.ct2_model,
            cache_root=args.cache_root,
            quantization=args.quantization,
            language_override=args.language,
            task_override=args.task,
            beam_size=args.beam_size,
            best_of=args.best_of,
            vad_filter=args.vad_filter,
            fast_decode=args.fast_decode,
            worker_factory=worker_factory,
            diagnostics_dir=diagnostics_dir,
            seed=args.seed,
            reset_seed_per_request=args.reset_seed_per_request,
        )

    if args.serve:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                transcribe_fn = make_transcribe_fn()
        except AdapterError as exc:
            print(f"error: {exc}", file=sys.stderr)
            (out_stream or sys.stdout).write(json.dumps({"error": str(exc)}, ensure_ascii=False) + "\n")
            (out_stream or sys.stdout).flush()
            return 1
        return serve(transcribe_fn, in_stream=in_stream or sys.stdin, out_stream=out_stream or sys.stdout)

    try:
        input_path, duration = common.validate_input_wav(args.input)
        output_path = common.validate_output_path(args.output)
        transcribe_fn = make_transcribe_fn()
        raw_result = transcribe_fn(str(input_path), duration)
        result = common.build_result(raw_result["text"], raw_result["language"], raw_result["segments"])
        common.write_result(result, output_path)
    except AdapterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {output_path}")
    return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
