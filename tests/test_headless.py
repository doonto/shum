"""Offline transport tests; no shum models, providers, or persistent servers."""
from __future__ import annotations

from http.client import HTTPConnection
import json
import shutil
import stat
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import shum_headless as headless


@pytest.fixture
def private_fixture():
    root = Path(__file__).parent / ".headless-fixtures"
    root.mkdir(mode=0o700, exist_ok=True)
    directory = root / uuid.uuid4().hex
    directory.mkdir(mode=0o700)
    try:
        binary = directory / "shum-fake"
        binary.write_text(
            f"""#!{sys.executable}
import json, os, sys, time
import subprocess
request = json.loads(sys.stdin.readline())
mode = request["method"]
if mode == "bad-json":
    print("diagnostic noise")
elif mode == "bad-id":
    print(json.dumps({{"version": 1, "id": "other", "result": {{}}}}))
elif mode == "bad-version":
    print(json.dumps({{"version": 2, "id": request["id"], "result": {{}}}}))
elif mode == "exit":
    sys.stderr.write("failure detail")
    sys.exit(7)
elif mode == "error":
    print(json.dumps({{"version": 1, "id": request["id"], "error": {{"code": "UNKNOWN_METHOD", "message": "Unknown method"}}}}))
elif mode == "error-exit":
    print(json.dumps({{"version": 1, "id": request["id"], "error": {{"code": "INVALID_METHOD", "message": "Rejected"}}}}))
    sys.exit(1)
elif mode == "oversize":
    print("x" * (1024 * 1024 + 1))
elif mode == "slow":
    time.sleep(1)
    print(json.dumps({{"version": 1, "id": request["id"], "result": {{}}}}))
elif mode == "tree":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    with open(request["params"]["pid_file"], "w") as stream:
        stream.write(str(child.pid))
    time.sleep(30)
elif mode == "stderr":
    sys.stderr.write(request["params"]["secret"] + "!" * (70 * 1024))
    print(json.dumps({{"version": 1, "id": request["id"], "result": {{}}}}))
elif mode == "workflow.start":
    print(json.dumps({{"version": 1, "id": request["id"], "result": {{"params": request["params"]}}}}))
else:
    print(json.dumps({{"version": 1, "id": request["id"], "result": {{"method": mode, "args": sys.argv[1:]}}}}))
""",
            encoding="utf-8",
        )
        binary.chmod(0o700)
        yield directory, binary
    finally:
        shutil.rmtree(directory)
        if not any(root.iterdir()):
            root.rmdir()


def request(method="installation.status", identifier="case"):
    return {"version": 1, "id": identifier, "method": method, "params": {}}


def test_call_success_error_and_options(private_fixture, capsys):
    _, binary = private_fixture
    assert headless.main(["--shum", str(binary), "--config", "cfg", "--installation", "inst",
                          "call", "--request", json.dumps(request())]) == 0
    response = json.loads(capsys.readouterr().out)
    assert response == {"version": 1, "id": "case", "result": {
        "method": "installation.status", "args": ["--config", "cfg", "--installation", "inst", "api"]
    }}
    assert headless.main(["--shum", str(binary), "call", "--request", json.dumps(request("error"))]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "UNKNOWN_METHOD"
    assert headless.main(["--shum", str(binary), "call", "--request", json.dumps(request("error-exit"))]) == 1
    assert json.loads(capsys.readouterr().out)["error"] == {"code": "INVALID_METHOD", "message": "Rejected"}


def test_call_file_stdin_and_validation(private_fixture, capsys, monkeypatch):
    directory, binary = private_fixture
    path = directory / "request.json"
    path.write_text(json.dumps(request()), encoding="utf-8")
    assert headless.main(["--shum", str(binary), "call", "--request-file", str(path)]) == 0
    capsys.readouterr()
    class Input:
        buffer = None
    import io
    Input.buffer = io.BytesIO(json.dumps(request("project.status")).encode())
    monkeypatch.setattr(sys, "stdin", Input)
    assert headless.main(["--shum", str(binary), "call"]) == 0
    assert json.loads(capsys.readouterr().out)["result"]["method"] == "project.status"
    for data in (b'{"version":1,"version":1,"id":"x","method":"a","params":{}}',
                 b'{"version":NaN,"id":"x","method":"a","params":{}}',
                 b'{"version":2,"id":"x","method":"a","params":{}}',
                 b'{"version":1,"id":"x","method":"a","params":[]}',
                 b" " * (headless.MAX_REQUEST + 1)):
        assert headless.main(["--shum", str(binary), "call", "--request", data.decode()]) == 1
        assert "error" in json.loads(capsys.readouterr().out)


def test_dynamic_asr_selection_passes_through_headless(private_fixture, capsys):
    _, binary = private_fixture
    for roles in (["solo"], ["left", "right"], ["a", "b", "c", "d", "e"]):
        payload = request("workflow.start")
        payload["params"] = {
            "input": "/private/recording.mp3", "project": "dynamic", "asr": roles,
        }
        assert headless.main(["--shum", str(binary), "call",
                              "--request", json.dumps(payload)]) == 0
        assert json.loads(capsys.readouterr().out)["result"]["params"] == payload["params"]


@pytest.mark.parametrize("method,code", [
    ("bad-json", "INVALID_RESPONSE"), ("bad-id", "INVALID_RESPONSE"),
    ("bad-version", "INVALID_RESPONSE"), ("exit", "INVALID_RESPONSE"),
    ("oversize", "RESPONSE_TOO_LARGE"),
])
def test_bad_child_response(private_fixture, capsys, method, code):
    _, binary = private_fixture
    assert headless.main(["--shum", str(binary), "call", "--request", json.dumps(request(method))]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["error"]["code"] == code
    if method == "exit":
        assert "failure detail" in captured.err


def test_timeout(private_fixture):
    _, binary = private_fixture
    with pytest.raises(headless.TransportError, match="timeout"):
        headless.call_api(request("slow"), str(binary), timeout=0.1)


def test_timeout_kills_process_group(private_fixture):
    directory, binary = private_fixture
    pid_file = directory / "child.pid"
    payload = request("tree")
    payload["params"]["pid_file"] = str(pid_file)
    with pytest.raises(headless.TransportError, match="timeout"):
        headless.call_api(payload, str(binary), timeout=0.3)
    assert pid_file.exists()
    pid = int(pid_file.read_text())
    # Reaped children can briefly remain as zombies in container PID namespaces.
    for _ in range(20):
        status = Path(f"/proc/{pid}/stat")
        if not status.exists() or status.read_text().split(") ")[1][0] == "Z":
            break
        time.sleep(0.05)
    else:
        pytest.fail("grandchild remained running after the API timeout")


def test_stderr_is_clipped(private_fixture, capsys):
    _, binary = private_fixture
    payload = request("stderr")
    payload["params"]["secret"] = "diagnostic"
    assert headless.main(["--shum", str(binary), "call", "--request", json.dumps(payload)]) == 0
    captured = capsys.readouterr()
    assert "diagnostic" in captured.err
    assert "truncated at 64 KiB" in captured.err
    assert len(captured.err) < 66 * 1024


@pytest.fixture
def running_server(private_fixture):
    directory, binary = private_fixture
    options = type("Options", (), {"shum": str(binary), "config": None,
                                    "installation": None, "timeout": None})()
    token_file = directory / "token"
    token = headless._token(str(token_file))
    server = headless.HeadlessServer("127.0.0.1", 0, token, options)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield server.server_port, token, token_file
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        assert not thread.is_alive()


def http(port, method, path, data=None, headers=None):
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path, body=data, headers=headers or {})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def test_http_auth_health_methods_and_limits(running_server):
    port, token, token_file = running_server
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
    auth = {"Authorization": "Bearer " + token}
    body = json.dumps(request()).encode()
    assert http(port, "GET", "/health")[0] == 401
    assert http(port, "GET", "/health", headers={"Authorization": "Bearer café"})[0] == 401
    assert http(port, "GET", "/health", headers=auth)[0] == 200
    assert http(port, "GET", "/health?token=" + token, headers=auth)[0] == 404
    assert http(port, "GET", "/health", headers=auth) == (200, {"version": 1, "ready": True})
    assert http(port, "POST", "/v1/requests", body, {"Content-Type": "application/json"})[0] == 401
    status, result = http(port, "POST", "/v1/requests", body, {**auth, "Content-Type": "application/json"})
    assert (status, result["id"], result["result"]["method"]) == (200, "case", "installation.status")
    assert http(port, "POST", "/v1/requests", json.dumps(request("error")), {
        **auth, "Content-Type": "application/json"
    })[1]["error"]["code"] == "UNKNOWN_METHOD"
    assert http(port, "POST", "/v1/requests", body, {**auth, "Origin": "http://localhost",
        "Content-Type": "application/json"})[0] == 403
    assert http(port, "POST", "/v1/requests", body, {**auth, "Content-Type": "text/plain"})[0] == 400
    assert http(port, "POST", "/v1/requests", b"x" * (headless.MAX_REQUEST + 1), {
        **auth, "Content-Type": "application/json"})[0] == 413
    assert http(port, "POST", "/v1/requests", json.dumps({**request(), "version": 2}), {
        **auth, "Content-Type": "application/json"})[0] == 400
    assert http(port, "GET", "/not-here", headers=auth)[0] == 404
    assert http(port, "OPTIONS", "/v1/requests", headers=auth)[0] == 404
    assert http(port, "PUT", "/v1/requests", headers=auth)[0] == 404
    secret = request("stderr")
    secret["params"]["secret"] = token
    assert http(port, "POST", "/v1/requests", json.dumps(secret), {
        **auth, "Content-Type": "application/json"
    })[0] == 200


def test_health_remains_available_during_workflow(running_server):
    port, token, _ = running_server
    result = []
    worker = threading.Thread(target=lambda: result.append(http(
        port, "POST", "/v1/requests", json.dumps(request("slow")),
        {"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )))
    worker.start()
    try:
        start = time.monotonic()
        assert http(port, "GET", "/health", headers={"Authorization": "Bearer " + token})[0] == 200
        assert time.monotonic() - start < 0.8
    finally:
        worker.join(timeout=5)
    assert result[0][0] == 200


def test_http_rejects_duplicate_length_and_transfer_encoding(running_server):
    port, token, _ = running_server
    for extra in ("Content-Length: 3\r\nContent-Length: 3\r\n",
                  "Transfer-Encoding: chunked\r\nContent-Length: 3\r\n"):
        import socket
        with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
            sock.sendall(("POST /v1/requests HTTP/1.1\r\nHost: localhost\r\n"
                          f"Authorization: Bearer {token}\r\n"
                          "Content-Type: application/json\r\n" + extra +
                          "Connection: close\r\n\r\n{}!").encode())
            response = sock.recv(1024)
            assert b" 400 " in response


def test_token_permissions_and_loopback(private_fixture):
    directory, _ = private_fixture
    token_file = directory / "token"
    assert len(headless._token(str(token_file))) == 64
    assert headless._token(str(token_file)) == token_file.read_text()
    token_file.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        headless._token(str(token_file))
    token_file.unlink()
    token_file.symlink_to(directory / "other")
    with pytest.raises(OSError):
        headless._token(str(token_file))
    with pytest.raises(ValueError, match="absolute"):
        headless._token("relative-token")
    directory.chmod(0o755)
    with pytest.raises(ValueError, match="private"):
        headless._token(str(directory / "new-token"))
    directory.chmod(0o700)
    with pytest.raises(SystemExit):
        headless.main(["serve", "--host", "0.0.0.0", "--token-file", str(token_file)])
