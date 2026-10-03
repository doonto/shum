#!/usr/bin/env python3
"""Experimental indexed bridge to the one-clip original Qwen C backend."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import wave
from pathlib import Path

from qwen_gguf_worker import AdapterError, validate_input, verify_file


def transcribe(args: argparse.Namespace, path: Path) -> dict:
    command = [
        args.binary, "-d", args.model, "-i", str(path),
        "-t", str(args.threads), "-S", "0", "--silent",
    ]
    if args.language is not None:
        command += ["--language", args.language]
    result = subprocess.run(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=sys.stderr, text=True, encoding="utf-8", check=False,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
    )
    if result.returncode != 0:
        raise AdapterError(f"native Qwen exited with status {result.returncode}")
    if not result.stdout.endswith("\n"):
        raise AdapterError("native Qwen response lacks the CLI's final newline")
    text = result.stdout[:-1]
    with wave.open(str(path), "rb") as audio:
        duration = audio.getnframes() / audio.getframerate()
    return {
        "text": text, "language": None,
        "segments": [{"start": 0.0, "end": duration, "text": text}] if text.strip() else [],
    }


def serve(args: argparse.Namespace) -> None:
    if args.threads < 1:
        raise AdapterError("threads must be positive")
    verify_file(args.binary, args.binary_sha256)
    model = Path(args.model)
    if not model.is_absolute() or not model.is_dir():
        raise AdapterError("model must be an absolute local original Qwen directory")
    if json.loads((model / "config.json").read_text())["model_type"] != "qwen3_asr":
        raise AdapterError("model is not an original Qwen3-ASR checkpoint")
    for line in sys.stdin:
        value = json.loads(line)
        if not isinstance(value, dict) or type(value.get("index")) is not int or value["index"] < 0:
            raise AdapterError("request requires a nonnegative integer index")
        if not isinstance(value.get("input"), str):
            raise AdapterError("request requires an absolute WAV path")
        response = transcribe(args, validate_input(value["input"]))
        print(json.dumps({"index": value["index"], **response}, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--language", choices=("German", "Russian"))
    parser.add_argument("--serve", required=True, action="store_true")
    args = parser.parse_args()
    try:
        serve(args)
    except (AdapterError, OSError, ValueError, KeyError, UnicodeError) as exc:
        print(json.dumps({"error": str(exc)}), flush=True)
        print(f"native Qwen adapter: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
