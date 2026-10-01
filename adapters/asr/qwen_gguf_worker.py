#!/usr/bin/env python3
"""Experimental offline Qwen3-ASR adapter using one shared llama.cpp server."""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import ctypes
import hashlib
import http.client
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, TextIO


TEMPLATE = """{%- set ns = namespace(system_text="", audio_tokens="") -%}
{%- for m in messages -%}
  {%- if m.role == 'system' -%}
    {%- if m.content is string -%}
      {%- set ns.system_text = ns.system_text + m.content -%}
    {%- else -%}
      {%- for c in m.content -%}
        {%- if c.type == 'text' and c.text is defined -%}
          {%- set ns.system_text = ns.system_text + c.text -%}
        {%- endif -%}
      {%- endfor -%}
    {%- endif -%}
  {%- endif -%}
  {%- if m.role == 'user' and m.content is string -%}
    {%- set ns.audio_tokens = ns.audio_tokens + m.content -%}
  {%- endif -%}
  {%- if m.content is not string -%}
    {%- for c in m.content -%}
      {%- if c.type == 'audio' or ('audio' in c) or ('audio_url' in c) -%}
        {%- set ns.audio_tokens = ns.audio_tokens + "<|audio_start|><|audio_pad|><|audio_end|>" -%}
      {%- endif -%}
    {%- endfor -%}
  {%- endif -%}
{%- endfor -%}
{{- '<|im_start|>system\\n' + ns.system_text + '<|im_end|>\\n' -}}
{{- '<|im_start|>user\\n' + ns.audio_tokens + '<|im_end|>\\n' -}}
{%- if add_generation_prompt -%}
{{- '<|im_start|>assistant\\n' -}}
{%- if asr_language is defined and asr_language -%}
{{- 'language ' + asr_language + '<asr_text>' -}}
{%- endif -%}
{%- endif -%}
"""
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
PREFIX_RE = re.compile(r"\Alanguage ([A-Za-z]+|None)<asr_text>")


class AdapterError(RuntimeError):
    pass


class ServerLoading(AdapterError):
    pass


class UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float = 180) -> None:
        super().__init__("localhost", timeout=timeout)
        self.path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


def request(socket_path: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    connection = UnixConnection(socket_path)
    try:
        headers = {"Content-Type": "application/json"}
        connection.request(
            "POST" if body is not None else "GET",
            path,
            json.dumps(body).encode() if body is not None else None,
            headers,
        )
        response = connection.getresponse()
        data = response.read()
        if path == "/health" and response.status == 503:
            raise ServerLoading("local ASR server is still loading")
        if response.status != 200:
            raise AdapterError(f"local ASR server returned HTTP {response.status}: {data[:400]!r}")
        value = json.loads(data)
        if not isinstance(value, dict):
            raise AdapterError("local ASR server returned a non-object response")
        return value
    finally:
        connection.close()


def verify_file(path: str, expected: str) -> Path:
    file = Path(path)
    if not file.is_absolute() or not file.is_file() or not HASH_RE.fullmatch(expected):
        raise AdapterError(f"expected an absolute file path and SHA-256 for {path}")
    digest = hashlib.sha256()
    with file.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise AdapterError(f"SHA-256 mismatch for {file}")
    return file


def validate_input(path: str) -> Path:
    file = Path(path)
    if not file.is_absolute() or not file.is_file():
        raise AdapterError("ASR input must be an existing absolute WAV file")
    with wave.open(str(file), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getframerate() != 16000 or wav.getsampwidth() != 2:
            raise AdapterError("ASR input must be 16 kHz mono 16-bit PCM WAV")
        if wav.getnframes() == 0:
            raise AdapterError("ASR input WAV must contain samples")
    return file


def decode_result(result: dict[str, Any], duration: float) -> dict[str, Any]:
    choices = result.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise AdapterError("Qwen returned an invalid choice count")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise AdapterError(f"Qwen did not finish decoding: {choice.get('finish_reason')}")
    text = choice.get("message", {}).get("content")
    if not isinstance(text, str):
        raise AdapterError("Qwen returned no transcription text")
    match = PREFIX_RE.match(text)
    language = match.group(1) if match and match.group(1) != "None" else None
    if match:
        text = text[match.end():]
    words = text.lower().split()
    groups = [tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
    if groups and (len(groups) - len(set(groups))) / len(groups) > 0.5:
        raise AdapterError("Qwen repeated more than half its four-word sequences")
    return {
        "text": text.strip(),
        "language": language,
        "segments": [{"start": 0.0, "end": duration, "text": text.strip()}] if text.strip() else [],
    }


def transcribe(socket_path: str, input_path: Path, language: str | None) -> dict[str, Any]:
    with wave.open(str(input_path), "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()
    body: dict[str, Any] = {
        "messages": [{
            "role": "user",
            "content": [{
                "type": "input_audio",
                "input_audio": {"data": base64.b64encode(input_path.read_bytes()).decode("ascii")},
            }],
        }],
        "temperature": 0,
        "max_tokens": 512,
        "repeat_penalty": 1.15,
    }
    if language:
        body["chat_template_kwargs"] = {"asr_language": language}
    result = decode_result(request(socket_path, "/v1/chat/completions", body), duration)
    if language:
        result["language"] = language
    return result


def serve(
    socket_path: str, slots: int, language: str | None, source: TextIO, target: TextIO
) -> None:
    write_lock = threading.Lock()
    failed = threading.Event()
    pending: set[concurrent.futures.Future[dict[str, Any]]] = set()

    def send(index: int | None, future: concurrent.futures.Future[dict[str, Any]]) -> None:
        try:
            result = future.result()
        except Exception as exc:
            print(f"Qwen window {index}: {exc}", file=sys.stderr, flush=True)
            result = {"error": str(exc)}
            failed.set()
        if index is not None:
            result = {"index": index, **result}
        with write_lock:
            target.write(json.dumps(result, ensure_ascii=False) + "\n")
            target.flush()

    with concurrent.futures.ThreadPoolExecutor(max_workers=slots) as pool:
        for raw in source:
            if failed.is_set():
                break
            value = json.loads(raw)
            if not isinstance(value, dict) or not isinstance(value.get("input"), str):
                raise AdapterError("request requires an absolute WAV input")
            index = value.get("index")
            if index is not None and (type(index) is not int or index < 0):
                raise AdapterError("window index must be a nonnegative integer")
            file = validate_input(value["input"])
            future = pool.submit(transcribe, socket_path, file, language)
            future.add_done_callback(lambda completed, i=index: send(i, completed))
            pending.add(future)
            if len(pending) >= slots:
                concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                pending = {entry for entry in pending if not entry.done()}
        concurrent.futures.wait(pending)
    if failed.is_set():
        raise AdapterError("one or more Qwen windows failed; see indexed worker responses")


def parent_death_signal(parent_pid: int) -> None:
    if sys.platform != "linux":
        raise AdapterError("Qwen GGUF worker currently requires Linux parent-death supervision")
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(1, signal.SIGTERM) != 0:
        raise OSError(ctypes.get_errno(), "cannot supervise local ASR server")
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)


def run(args: argparse.Namespace) -> None:
    if args.engine != "qwen" or not (1 <= args.slots <= 4) or args.threads < 1:
        raise AdapterError("expected --engine qwen, positive threads and 1-4 slots")
    if args.offload_dir:
        raise AdapterError("llama.cpp GGUF cannot use Transformers disk offload")
    model = Path(args.model)
    if not model.is_absolute() or not model.is_dir():
        raise AdapterError("--model must be the pinned original Qwen3-ASR directory")
    if json.loads((model / "config.json").read_text()).get("model_type") != "qwen3_asr":
        raise AdapterError("--model is not an original Qwen3-ASR checkpoint")
    server = verify_file(args.server, args.server_sha256)
    gguf = verify_file(args.gguf, args.gguf_sha256)
    mmproj = verify_file(args.mmproj, args.mmproj_sha256)
    with tempfile.TemporaryDirectory(prefix="shum-qwen-") as private:
        socket_path = str(Path(private) / "server.sock")
        template = Path(private) / "template.jinja"
        template.write_text(TEMPLATE)
        command = [
            str(server), "-m", str(gguf), "--mmproj", str(mmproj),
            "--no-mmproj-offload", "--offline", "-c", str(2048 * args.slots),
            "-np", str(args.slots), "-t", str(args.threads), "-tb", str(args.threads),
            "-n", "512", "--host", socket_path, "--chat-template-file", str(template),
        ]
        parent_pid = os.getpid()
        child = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=sys.stderr, preexec_fn=lambda: parent_death_signal(parent_pid),
        )
        try:
            for _ in range(300):
                if child.poll() is not None:
                    raise AdapterError(f"local ASR server exited during startup: {child.returncode}")
                try:
                    if request(socket_path, "/health").get("status") == "ok":
                        break
                except (OSError, http.client.HTTPException, ServerLoading):
                    pass
                time.sleep(0.1)
            else:
                raise AdapterError("local ASR server did not start in 30 seconds")
            serve(socket_path, args.slots, args.language, sys.stdin, sys.stdout)
        finally:
            if child.poll() is None:
                child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--offload-dir")
    parser.add_argument("--server", required=True)
    parser.add_argument("--server-sha256", required=True)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--gguf-sha256", required=True)
    parser.add_argument("--mmproj", required=True)
    parser.add_argument("--mmproj-sha256", required=True)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--language", choices=("German", "Russian"))
    args = parser.parse_args()
    try:
        run(args)
    except (AdapterError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Qwen GGUF adapter: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
