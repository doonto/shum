#!/usr/bin/env python3
"""Synchronous, stdlib-only transport for ``shum api``.

This is not a durable job queue: each HTTP request owns a separate API process,
and disconnecting a client does not create a resumable HTTP job. Workflow
checkpoints and resume semantics belong to shum, not this transport.
"""
from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import math
import os
import re
import selectors
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

VERSION = 1
MAX_REQUEST = 64 * 1024
MAX_RESPONSE = 1024 * 1024
MAX_STDERR = 64 * 1024
READ_TIMEOUT = 10


class TransportError(Exception):
    def __init__(self, code: str, message: str, status: int = 502, stderr: bytes = b""):
        super().__init__(message)
        self.code, self.message, self.status, self.stderr = code, message, status, stderr


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _json(data: bytes) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"), object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON number")),
        )
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise TransportError("INVALID_JSON", "Expected one UTF-8 JSON document", 400) from exc


def _encode(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _envelope(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TransportError("INVALID_REQUEST", "Request must be a JSON object", 400)
    if type(value.get("version")) is not int or value["version"] != VERSION:
        raise TransportError("UNSUPPORTED_VERSION", "Expected request version 1", 400)
    if not isinstance(value.get("id"), str) or not value["id"]:
        raise TransportError("INVALID_REQUEST", "Request id must be a nonempty string", 400)
    if not isinstance(value.get("method"), str) or not value["method"]:
        raise TransportError("INVALID_REQUEST", "Request method must be a nonempty string", 400)
    if not isinstance(value.get("params"), dict):
        raise TransportError("INVALID_REQUEST", "Request params must be an object", 400)
    if set(value) != {"version", "id", "method", "params"}:
        raise TransportError("INVALID_REQUEST", "Unexpected request fields", 400)
    return value


def parse_request(data: bytes) -> dict[str, Any]:
    if len(data) > MAX_REQUEST:
        raise TransportError("REQUEST_TOO_LARGE", "Request exceeds 64 KiB", 413)
    return _envelope(_json(data))


def _terminate_tree(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        if proc.poll() is None:
            proc.kill()
    proc.wait()


def _read_child(proc: subprocess.Popen[bytes], payload: bytes, timeout: float | None) -> tuple[bytes, bytes]:
    output, errors = bytearray(), bytearray()
    start = time.monotonic()
    with selectors.DefaultSelector() as ready:
        for stream, name in ((proc.stdout, "stdout"), (proc.stderr, "stderr")):
            assert stream is not None
            os.set_blocking(stream.fileno(), False)
            ready.register(stream, selectors.EVENT_READ, name)
        assert proc.stdin is not None
        os.set_blocking(proc.stdin.fileno(), False)
        ready.register(proc.stdin, selectors.EVENT_WRITE, "stdin")
        sent = 0
        while ready.get_map():
            remaining = None if timeout is None else timeout - (time.monotonic() - start)
            if remaining is not None and remaining <= 0:
                _terminate_tree(proc)
                raise TransportError("TIMEOUT", "API process exceeded the configured timeout", 504, bytes(errors))
            for key, _ in ready.select(remaining):
                if key.data == "stdin":
                    try:
                        sent += os.write(key.fileobj.fileno(), payload[sent:])
                    except BrokenPipeError:
                        sent = len(payload)
                    if sent == len(payload):
                        ready.unregister(key.fileobj)
                        key.fileobj.close()
                    continue
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    ready.unregister(key.fileobj)
                    continue
                if key.data == "stdout":
                    output.extend(chunk)
                    if len(output) > MAX_RESPONSE:
                        _terminate_tree(proc)
                        raise TransportError("RESPONSE_TOO_LARGE", "API response exceeds 1 MiB", stderr=bytes(errors))
                else:
                    errors.extend(chunk[: max(0, MAX_STDERR - len(errors))])
        remaining = None if timeout is None else timeout - (time.monotonic() - start)
        try:
            proc.wait(timeout=max(0, remaining) if remaining is not None else None)
        except subprocess.TimeoutExpired as exc:
            _terminate_tree(proc)
            raise TransportError("TIMEOUT", "API process exceeded the configured timeout", 504, bytes(errors)) from exc
    return bytes(output), bytes(errors)


def call_api(
    request: dict[str, Any], shum: str, config: str | None = None,
    installation: str | None = None, timeout: float | None = None,
) -> tuple[dict[str, Any], bytes]:
    command = [shum]
    if config is not None:
        command.extend(("--config", config))
    if installation is not None:
        command.extend(("--installation", installation))
    command.append("api")
    try:
        proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, start_new_session=True,
        )
    except OSError as exc:
        raise TransportError("API_UNAVAILABLE", "Could not start shum api") from exc
    try:
        output, errors = _read_child(proc, _encode(request) + b"\n", timeout)
    except (BrokenPipeError, OSError) as exc:
        _terminate_tree(proc)
        raise TransportError("API_IO", "Communication with shum api failed") from exc
    except BaseException:
        _terminate_tree(proc)
        raise
    try:
        response = _json(output)
    except TransportError as exc:
        raise TransportError("INVALID_RESPONSE", "shum api returned invalid JSON", stderr=errors) from exc
    if (
        not isinstance(response, dict)
        or type(response.get("version")) is not int or response["version"] != VERSION
        or response.get("id") != request["id"]
        or (("result" in response) == ("error" in response))
        or set(response) - {"version", "id", "result", "error"}
        or ("result" in response and not isinstance(response["result"], dict))
        or ("error" in response and (
            not isinstance(response["error"], dict)
            or not isinstance(response["error"].get("code"), str)
            or not isinstance(response["error"].get("message"), str)
        ))
    ):
        raise TransportError("INVALID_RESPONSE", "shum api returned an invalid response envelope", stderr=errors)
    if proc.returncode != 0 and "result" in response:
        raise TransportError("API_EXIT", "shum api exited unsuccessfully", stderr=errors)
    return response, errors


def _error(exc: TransportError, identifier: str | None = None) -> dict[str, Any]:
    return {"version": VERSION, "id": identifier, "error": {"code": exc.code, "message": exc.message}}


def _stderr(data: bytes, token: str | None = None) -> None:
    if not data:
        return
    text = data.decode("utf-8", errors="replace")
    if token:
        text = text.replace(token, "[REDACTED]")
    sys.stderr.write(text)
    if len(data) >= MAX_STDERR:
        sys.stderr.write("\n[shum api stderr truncated at 64 KiB]\n")
    elif not text.endswith("\n"):
        sys.stderr.write("\n")


def _token(path: str) -> str:
    if not os.path.isabs(path):
        raise ValueError("--token-file must be an absolute path")
    parent = Path(path).parent
    info = parent.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Token parent directory must be owned by you and private (0700)")
    flags = os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | flags, 0o600)
    except FileExistsError:
        fd = os.open(path, os.O_RDONLY | flags)
        created = False
    else:
        created = True
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError("Token file must be a regular, owner-only 0600 file")
        if created:
            token = os.urandom(32).hex()
            os.write(fd, token.encode("ascii"))
        else:
            token = os.read(fd, 128).decode("ascii")
        if not re.fullmatch(r"[0-9a-f]{64}", token):
            raise ValueError("Token file must contain exactly 64 lowercase hex characters")
        return token
    finally:
        os.close(fd)


class HeadlessServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    address_family = socket.AF_INET

    def __init__(self, host: str, port: int, token: str, options: argparse.Namespace):
        self.token = token
        self.options = options
        super().__init__((host, port), HeadlessHandler)


class HeadlessHandler(BaseHTTPRequestHandler):
    server: HeadlessServer
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(READ_TIMEOUT)

    def log_message(self, format: str, *args: Any) -> None:
        # Paths and headers supplied by clients must not enter logs.
        pass

    def _reply(self, status: int, body: dict[str, Any]) -> None:
        data = _encode(body).replace(self.server.token.encode("ascii"), b"[REDACTED]")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            if self.command != "HEAD":
                self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            pass
        self.close_connection = True

    def _authorized(self) -> bool:
        values = self.headers.get_all("Authorization", [])
        if len(values) != 1:
            return False
        try:
            provided = values[0].encode("latin-1")
        except UnicodeEncodeError:
            return False
        return hmac.compare_digest(provided, ("Bearer " + self.server.token).encode("ascii"))

    def _handle(self, method: str) -> None:
        if not self._authorized():
            self._reply(401, _error(TransportError("UNAUTHORIZED", "Bearer authorization required", 401)))
            return
        if self.headers.get_all("Origin", []):
            self._reply(403, _error(TransportError("ORIGIN_FORBIDDEN", "Browser origins are not accepted", 403)))
            return
        if method == "GET" and self.path == "/health":
            self._reply(200, {"version": VERSION, "ready": True})
            return
        if method != "POST" or self.path != "/v1/requests":
            self._reply(404, _error(TransportError("NOT_FOUND", "Unknown endpoint", 404)))
            return
        try:
            if self.headers.get_all("Transfer-Encoding", []):
                raise TransportError("INVALID_REQUEST", "Transfer-Encoding is not supported", 400)
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or not re.fullmatch(r"[0-9]{1,6}", lengths[0]):
                raise TransportError("INVALID_REQUEST", "Exactly one valid Content-Length is required", 400)
            length = int(lengths[0])
            if length > MAX_REQUEST:
                raise TransportError("REQUEST_TOO_LARGE", "Request exceeds 64 KiB", 413)
            content_types = self.headers.get_all("Content-Type", [])
            if len(content_types) != 1 or content_types[0].lower().strip() not in (
                "application/json", "application/json; charset=utf-8"
            ):
                raise TransportError("INVALID_REQUEST", "Content-Type must be application/json", 400)
            data = self.rfile.read(length)
            if len(data) != length:
                raise TransportError("INVALID_REQUEST", "Incomplete request body", 400)
            request = parse_request(data)
            self.connection.settimeout(None)
            response, errors = call_api(
                request, self.server.options.shum, self.server.options.config,
                self.server.options.installation, self.server.options.timeout,
            )
            _stderr(errors, self.server.token)
            self._reply(200, response)
        except TransportError as exc:
            _stderr(exc.stderr, self.server.token)
            self._reply(exc.status, _error(exc, locals().get("request", {}).get("id")))
        except socket.timeout:
            self._reply(408, _error(TransportError("READ_TIMEOUT", "Request body timed out", 408)))

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def do_OPTIONS(self) -> None:
        self._handle("OPTIONS")

    def do_HEAD(self) -> None:
        self._handle("HEAD")

    def do_PUT(self) -> None:
        self._handle("PUT")

    def do_PATCH(self) -> None:
        self._handle("PATCH")

    def do_DELETE(self) -> None:
        self._handle("DELETE")


def _positive(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be positive") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("timeout must be a finite positive number")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shum", default=shutil.which("shum") or "shum", help="shum executable")
    parser.add_argument("--config", help="pass --config to shum")
    parser.add_argument("--installation", help="pass --installation to shum")
    parser.add_argument("--timeout", type=_positive, help="optional API process timeout, seconds")
    commands = parser.add_subparsers(dest="command", required=True)
    call = commands.add_parser("call", help="send one request and print its JSON response")
    source = call.add_mutually_exclusive_group()
    source.add_argument("--request", help="request JSON")
    source.add_argument("--request-file", help="read JSON request from this file (default: stdin)")
    serve = commands.add_parser("serve", help="start an authenticated loopback HTTP transport")
    serve.add_argument("--host", default="127.0.0.1", help="IPv4 loopback address")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--token-file", required=True, help="absolute path in a private directory")
    args = parser.parse_args(argv)
    if args.command == "call":
        identifier = None
        try:
            if args.request is not None:
                data = args.request.encode("utf-8")
            elif args.request_file is not None:
                with open(args.request_file, "rb") as stream:
                    data = stream.read(MAX_REQUEST + 1)
            else:
                data = sys.stdin.buffer.read(MAX_REQUEST + 1)
            request = parse_request(data)
            identifier = request["id"]
            response, errors = call_api(request, args.shum, args.config, args.installation, args.timeout)
            _stderr(errors)
            success = "error" not in response
        except TransportError as exc:
            _stderr(exc.stderr)
            response, success = _error(exc, identifier), False
        except OSError:
            response, success = _error(TransportError("INPUT_IO", "Could not read request"),  identifier), False
        print(_encode(response).decode("utf-8"))
        return 0 if success else 1
    try:
        if not ipaddress.IPv4Address(args.host).is_loopback:
            raise ValueError("--host must be an IPv4 loopback address")
        token = _token(args.token_file)
        with HeadlessServer(args.host, args.port, token, args) as server:
            server.serve_forever()
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
