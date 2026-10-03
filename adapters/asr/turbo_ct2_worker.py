#!/usr/bin/env python3
"""Indexed, offline CPU experiment for a pinned published Turbo CT2 model."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import dataclasses
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import queue
import sys
import threading
import time

import flix_fast_common as common
from flix_fast_worker import write_diagnostics


model = None
diagnostics_dir = None


def initialize(model_dir: str, threads: int, diagnostics: str | None) -> None:
    global model, diagnostics_dir
    common.configure_offline_environment()
    common.configure_thread_environment(threads)
    from faster_whisper import WhisperModel

    started = time.monotonic()
    model = WhisperModel(model_dir, device="cpu", compute_type="int8",
                         cpu_threads=threads, num_workers=1, local_files_only=True)
    compute = model.model.compute_type
    if not compute.startswith("int8"):
        raise ValueError(f"unexpected effective compute type: {compute}")
    diagnostics_dir = Path(diagnostics) if diagnostics else None
    if diagnostics_dir:
        write_diagnostics(diagnostics_dir / f"ready-{os.getpid()}.json", {
            "pid": os.getpid(), "load_seconds": time.monotonic() - started,
            "compute_type": compute, "threads": threads,
            "versions": {name: importlib.metadata.version(name)
                         for name in ("faster-whisper", "ctranslate2", "onnxruntime")},
        })
    print(f"Turbo CT2 ready: pid={os.getpid()}, compute={compute}, threads={threads}",
          file=sys.stderr, flush=True)


def transcribe(request: dict) -> dict:
    number = request["index"]
    path, duration = common.validate_input_wav(request["input"])
    started = time.monotonic()
    segments, info = model.transcribe(
        str(path), language="de", task="transcribe", beam_size=3, best_of=1,
        vad_filter=True, word_timestamps=False, condition_on_previous_text=False,
    )
    parts = list(segments)
    text = "".join(segment.text for segment in parts).strip()
    result = common.build_result(text, info.language, [
        {"start": 0.0, "end": duration, "text": text}
    ])
    if diagnostics_dir:
        options = info.transcription_options
        write_diagnostics(diagnostics_dir / f"request-{number:05d}.json", {
            "request_index": number, "input": str(path), "worker_pid": os.getpid(),
            "elapsed_seconds": time.monotonic() - started,
            "input_duration_seconds": duration,
            "duration_after_vad_seconds": info.duration_after_vad,
            "transcription_options": dataclasses.asdict(options),
            "vad_options": dataclasses.asdict(info.vad_options),
            "decoder_token_limit": model.max_length,
            "truncation_status": "not_exposed_by_engine",
            "output_timestamp_policy": "whole WAV window; raw decoder timestamps retained below",
            "raw_segment_bounds_outside_input": [
                i for i, s in enumerate(parts)
                if s.start < 0 or s.end < s.start or s.end > duration
            ],
            "segments": [
                {"start": s.start, "end": s.end, "tokens": s.tokens,
                 "temperature": s.temperature, "avg_logprob": s.avg_logprob,
                 "compression_ratio": s.compression_ratio, "no_speech_prob": s.no_speech_prob}
                for s in parts
            ],
        })
    return {"index": number, **result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()
    if not args.model.is_absolute() or not (args.model / "model.bin").is_file():
        parser.error("model must be an absolute existing CT2 directory")
    if args.threads < 1 or not 1 <= args.workers <= 4:
        parser.error("threads must be positive; workers must be between 1 and 4")
    if args.diagnostics:
        if not args.diagnostics.is_absolute():
            parser.error("diagnostics must be an absolute path")
        args.diagnostics.mkdir(mode=0o700, parents=True, exist_ok=True)
        if any(args.diagnostics.iterdir()):
            parser.error("diagnostics directory must be empty")
    common.configure_offline_environment()
    common.configure_thread_environment(args.threads)
    incoming = queue.Queue()

    def read_requests():
        try:
            for line in sys.stdin:
                incoming.put(line)
        except (OSError, UnicodeError) as exc:
            incoming.put(exc)
        finally:
            incoming.put(None)

    threading.Thread(target=read_requests, daemon=True).start()
    eof = False
    pending = {}
    seen = set()
    number = None
    try:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize,
            initargs=(str(args.model), args.threads,
                      str(args.diagnostics) if args.diagnostics else None),
        ) as executor:
            while not eof or pending:
                if not eof and len(pending) < args.workers:
                    try:
                        line = incoming.get(timeout=0.05)
                    except queue.Empty:
                        line = ""
                    if line is None:
                        eof = True
                    elif isinstance(line, Exception):
                        raise line
                    elif line:
                        request = json.loads(line)
                        number = request["index"]
                        if type(number) is not int or number < 0 or number in seen:
                            raise ValueError("request index must be a unique nonnegative integer")
                        seen.add(number)
                        pending[executor.submit(transcribe, request)] = number
                if pending:
                    completed, _ = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
                    for future in completed:
                        number = pending.pop(future)
                        print(json.dumps(future.result(), ensure_ascii=False, allow_nan=False),
                              flush=True)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, UnicodeError, common.AdapterError) as exc:
        print(f"Turbo CT2 failure: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        print(json.dumps({"index": number, "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
