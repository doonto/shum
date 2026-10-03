#!/usr/bin/env python3
"""Offline, one-model-per-process Swiss German Turbo CT2 ASR worker."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

import flix_fast_common as common

AdapterError = common.AdapterError
REPO = "gcoli/whisper-large-v3-turbo-swiss-german-mit"
REVISION = "540fb2e7a9135048963c5de9822579c96f8962e5"
SOURCE_SHA256 = "b37f10e1b905e7c6a0a7d3eb7462083b6e12caa71c06e26c83245d33337d246d"
CT2_SHA256 = "baecb116482ebb80a434d4a71ab50d6642808b258b5ce0f82cfde571aca3dfa2"
CT2_MANIFEST = ".shum-turbo-ct2.json"
CT2_ASSETS = frozenset(("model.bin", "config.json", "tokenizer.json",
                        "preprocessor_config.json", "vocabulary.json"))
TIMESTAMP_GRID_SECONDS = 0.02
# Whisper's last decoding window is padded to 30 seconds even for a short clip.
MAX_END_OVERSHOOT_SECONDS = 30.50
MAX_START_UNDERSHOOT_SECONDS = TIMESTAMP_GRID_SECONDS * 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("whisper",), default="whisper")
    parser.add_argument("--model", required=True, help="Absolute pinned local HF source directory")
    parser.add_argument("--ct2-model", help="Absolute directory of pinned publisher CT2 assets")
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--input", help="Absolute 16 kHz mono WAV path")
    parser.add_argument("--output", help="Absolute result JSON path")
    parser.add_argument("--serve", action="store_true", help="Read indexed JSON-line requests")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.serve:
        if args.input is not None or args.output is not None:
            parser.error("--serve cannot be combined with --input/--output")
    elif args.input is None or args.output is None:
        parser.error("--input and --output are required unless --serve is set")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AdapterError(f"Cannot read local manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdapterError(f"Local manifest must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise AdapterError(f"Cannot hash local model file {path}: {exc}") from exc
    return digest.hexdigest()


def validate_models(model_arg: str, ct2_arg: str | None) -> Path:
    model_dir = common.validate_whisper_model_dir(model_arg)
    source = _read_json(model_dir / common.MANIFEST_FILENAME)
    if (source.get("repo"), source.get("revision")) != (REPO, REVISION):
        raise AdapterError(f"--model must have pinned source repo/revision {REPO}@{REVISION}")
    hashes = source.get("files")
    if not isinstance(hashes, dict) or hashes.get("model.safetensors") != SOURCE_SHA256:
        raise AdapterError("--model manifest must record the pinned model.safetensors SHA-256")
    if _sha256(model_dir / "model.safetensors") != SOURCE_SHA256:
        raise AdapterError(f"Source model.safetensors SHA-256 mismatch: {model_dir}")
    # The source directory is provenance metadata, not a conversion input.
    if ct2_arg is None:
        ct2_dir = model_dir / "ct2"
    else:
        ct2_dir = common.require_absolute(ct2_arg, "--ct2-model")
    if not ct2_dir.is_dir():
        raise AdapterError(f"Published CT2 directory does not exist: {ct2_dir}; provide --ct2-model")
    manifest = _read_json(ct2_dir / CT2_MANIFEST)
    if (manifest.get("repo"), manifest.get("revision")) != (REPO, REVISION):
        raise AdapterError("CT2 manifest repo/revision does not match the pinned published snapshot")
    hashes = manifest.get("files")
    if not isinstance(hashes, dict) or hashes.get("model.bin") != CT2_SHA256:
        raise AdapterError("CT2 manifest does not record the pinned publisher model.bin SHA-256")
    if not CT2_ASSETS.issubset(hashes):
        raise AdapterError(f"CT2 manifest missing required publisher assets: {sorted(CT2_ASSETS - hashes.keys())}")
    for filename, expected in hashes.items():
        if (not isinstance(filename, str) or filename in ("", ".", "..")
                or Path(filename).name != filename or filename.startswith(".")
                or not isinstance(expected, str) or len(expected) != 64
                or any(char not in "0123456789abcdef" for char in expected)):
            raise AdapterError(f"CT2 manifest has an invalid asset entry: {filename!r}")
        path = ct2_dir / filename
        if not path.is_file() or path.is_symlink():
            raise AdapterError(f"Published CT2 asset is missing or is a symlink: {path}")
        if _sha256(path) != expected:
            raise AdapterError(f"CT2 {filename} SHA-256 mismatch: {path}")
    return ct2_dir


def default_factory(model_dir: str, threads: int) -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise AdapterError("faster-whisper is required in the offline ASR environment") from exc
    try:
        model = WhisperModel(model_dir, device="cpu", compute_type="int8",
                             cpu_threads=threads, num_workers=1, local_files_only=True)
        effective = model.model.compute_type
        if not effective.startswith("int8"):
            raise AdapterError(f"Effective CT2 compute type is {effective!r}, expected INT8")
        return model
    except AdapterError:
        raise
    except Exception as exc:
        raise AdapterError(f"Cannot load local CPU CT2 model: {exc}") from exc


def normalize_segments(segments: Any, duration: float, *, request_index: int | None,
                       error_stream: TextIO) -> tuple[str, list[dict[str, Any]]]:
    result = []
    texts = []
    if not math.isfinite(duration) or duration <= 0:
        raise AdapterError(f"Invalid WAV duration: {duration}")
    for position, segment in enumerate(segments):
        start, end = segment.start, segment.end
        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                or not math.isfinite(start) or not math.isfinite(end) or end <= start):
            raise AdapterError(f"segment[{position}] invalid raw timestamp interval: {start!r}..{end!r}")
        if end <= 0 or start >= duration:
            raise AdapterError(f"segment[{position}] entirely outside WAV bounds: {start}..{end} vs 0..{duration}")
        if start < -MAX_START_UNDERSHOOT_SECONDS or end > duration + MAX_END_OVERSHOOT_SECONDS:
            raise AdapterError(
                f"segment[{position}] raw bounds exceed timestamp adjustment policy: "
                f"{start}..{end} vs 0..{duration}"
            )
        text = segment.text
        if not isinstance(text, str):
            raise AdapterError(f"segment[{position}] has non-string text")
        adjusted_start, adjusted_end = max(0.0, start), min(duration, end)
        if adjusted_end <= adjusted_start:
            raise AdapterError(f"segment[{position}] has no interval within WAV bounds")
        if adjusted_start != start or adjusted_end != end:
            event = {
                "event": "asr_timestamp_adjustment", "request_index": request_index,
                "segment_index": position, "raw_start": start, "raw_end": end,
                "wav_duration": duration, "start": adjusted_start, "end": adjusted_end,
            }
            print(json.dumps(event, allow_nan=False), file=error_stream, flush=True)
        result.append({"start": adjusted_start, "end": adjusted_end, "text": text})
        texts.append(text)
    return "".join(texts).strip(), result


def build_transcribe_fn(model_dir: str, ct2_dir: str, threads: int,
                        *, factory: Callable[[str, int], Any] = default_factory,
                        error_stream: TextIO = sys.stderr) -> Callable[[str, float, int | None], dict[str, Any]]:
    ct2 = validate_models(model_dir, ct2_dir)
    model = factory(str(ct2), threads)

    def transcribe(input_path: str, duration: float, index: int | None = None) -> dict[str, Any]:
        try:
            segments, info = model.transcribe(
                input_path, language="de", task="transcribe", beam_size=3, best_of=1,
                vad_filter=True, word_timestamps=False, condition_on_previous_text=False,
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
            )
            text, normalized = normalize_segments(segments, duration, request_index=index,
                                                  error_stream=error_stream)
            return common.build_result(text, info.language, normalized)
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError(f"CT2 transcription failed: {exc}") from exc

    return transcribe


def serve(transcribe: Callable[[str, float, int | None], dict[str, Any]],
          *, in_stream: TextIO, out_stream: TextIO, error_stream: TextIO = sys.stderr) -> int:
    for line in in_stream:
        if not line.strip():
            continue
        index = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise AdapterError("request must be a JSON object")
            if "index" in request:
                index = request["index"]
                if type(index) is not int or index < 0:
                    index = None
                    raise AdapterError("request index must be a nonnegative integer")
            if not isinstance(request.get("input"), str):
                raise AdapterError("request must have a string input path")
            path, duration = common.validate_input_wav(request["input"])
            with contextlib.redirect_stdout(error_stream):
                result = transcribe(str(path), duration, index)
        except Exception as exc:
            message = str(exc)
            print(f"error: {message}", file=error_stream, flush=True)
            result = {"error": message}
            if index is not None:
                result["index"] = index
            out_stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_stream.flush()
            return 1
        if index is not None:
            result["index"] = index
        out_stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
        out_stream.flush()
    return 0


def run(argv: Sequence[str] | None = None, *, factory: Callable[[str, int], Any] = default_factory,
        in_stream: TextIO | None = None, out_stream: TextIO | None = None,
        error_stream: TextIO | None = None) -> int:
    args = parse_args(argv)
    errors = error_stream if error_stream is not None else sys.stderr
    output = out_stream if out_stream is not None else sys.stdout
    common.configure_offline_environment()
    common.configure_thread_environment(args.threads)
    try:
        if not args.serve:
            path, duration = common.validate_input_wav(args.input)
            result_path = common.validate_output_path(args.output)
        with contextlib.redirect_stdout(errors):
            transcribe = build_transcribe_fn(args.model, args.ct2_model, args.threads,
                                             factory=factory, error_stream=errors)
        if args.serve:
            return serve(transcribe, in_stream=in_stream if in_stream is not None else sys.stdin,
                         out_stream=output, error_stream=errors)
        with contextlib.redirect_stdout(errors):
            result = transcribe(str(path), duration)
        common.write_result(result, result_path)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=errors, flush=True)
        if args.serve:
            output.write(json.dumps({"error": str(exc)}, ensure_ascii=False) + "\n")
            output.flush()
        return 1


if __name__ == "__main__":
    sys.exit(run())
