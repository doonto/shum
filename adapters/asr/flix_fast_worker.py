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
import json
import sys
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
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
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
    return {"text": text, "language": resolved_language, "segments": result_segments}


# ---------------------------------------------------------------------------
# Orchestration (mirrors main.py's build_transcribe_fn/serve/run shape)
# ---------------------------------------------------------------------------

TranscribeFn = Callable[[str, float], dict[str, Any]]


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
) -> TranscribeFn:
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
    model = factory(str(ct2_dir), threads, compute_type=quantization)
    return lambda input_path, duration: transcribe(
        model,
        input_path,
        duration,
        language=language,
        task=default_task,
        beam_size=beam_size,
        best_of=best_of,
        vad_filter=vad_filter,
        fast_decode=fast_decode,
    )


def serve(transcribe_fn: TranscribeFn, *, in_stream: TextIO, out_stream: TextIO) -> int:
    """Identical wire protocol to ``main.py``'s ``serve()``: one JSON-lines
    request in, one response out, in order, until EOF or the first error."""
    for raw_line in in_stream:
        line = raw_line.strip()
        if not line:
            continue

        error_message: str | None = None
        result: dict[str, Any] | None = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or not isinstance(request.get("input"), str):
                raise AdapterError(f"request must be a JSON object with a string 'input' key, got: {line!r}")
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
    worker_factory: WorkerFactory | None = None,
    in_stream: TextIO | None = None,
    out_stream: TextIO | None = None,
) -> int:
    args = parse_args(argv)
    try:
        threads = common.validate_threads(args.threads)
        model_dir = common.validate_whisper_model_dir(args.model)
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
