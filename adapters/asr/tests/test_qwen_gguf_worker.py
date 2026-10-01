import hashlib
import io
import json
import socketserver
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from unittest.mock import patch

import pytest

import qwen_gguf_worker as worker


def wav_file(path: Path) -> Path:
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 1600)
    return path


def test_rejects_truncation_and_repetition():
    with pytest.raises(worker.AdapterError, match="finish"):
        worker.decode_result(
            {"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]}, 1
        )
    with pytest.raises(worker.AdapterError, match="repeated"):
        worker.decode_result(
            {
                "choices": [{
                    "finish_reason": "stop",
                    "message": {"content": "one two three four " * 20},
                }]
            },
            1,
        )
    result = worker.decode_result(
        {"choices": [{"finish_reason": "stop", "message": {
            "content": "language German<asr_text>Guten Tag"
        }}]},
        2,
    )
    assert result == {
        "text": "Guten Tag",
        "language": "German",
        "segments": [{"start": 0.0, "end": 2, "text": "Guten Tag"}],
    }


def test_checks_artifact_hash_and_audio_format(tmp_path):
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"model")
    digest = hashlib.sha256(b"model").hexdigest()
    assert worker.verify_file(str(artifact), digest) == artifact
    with pytest.raises(worker.AdapterError, match="mismatch"):
        worker.verify_file(str(artifact), "0" * 64)
    wav = wav_file(tmp_path / "audio.wav")
    assert worker.validate_input(str(wav)) == wav
    with pytest.raises(worker.AdapterError, match="absolute"):
        worker.validate_input("audio.wav")


def test_serves_out_of_order_responses_with_indices(tmp_path):
    files = [wav_file(tmp_path / f"{i}.wav") for i in range(3)]
    requests = "".join(
        json.dumps({"input": str(file), "index": i}) + "\n"
        for i, file in enumerate(files)
    )
    output = io.StringIO()
    barrier = threading.Barrier(3)

    def transcribe(_socket, file, _language):
        index = int(file.stem)
        barrier.wait(timeout=2)
        time.sleep(0.03 if index == 0 else 0)
        return {"text": str(index), "language": "German", "segments": []}

    with patch.object(worker, "transcribe", side_effect=transcribe):
        worker.serve("unused", 3, "German", io.StringIO(requests), output)
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert {row["index"] for row in rows} == {0, 1, 2}
    assert rows[0]["index"] != 0
    assert all(row["text"] == str(row["index"]) for row in rows)


def test_worker_reports_indexed_error_and_exits_nonzero(tmp_path):
    wav = wav_file(tmp_path / "audio.wav")
    output = io.StringIO()
    with patch.object(worker, "transcribe", side_effect=worker.AdapterError("decode failed")):
        with pytest.raises(worker.AdapterError, match="Qwen windows failed"):
            worker.serve(
                "unused", 2, None,
                io.StringIO(json.dumps({"input": str(wav), "index": 7}) + "\n"),
                output,
            )
    assert json.loads(output.getvalue()) == {"index": 7, "error": "decode failed"}


def test_posts_audio_only_to_private_unix_socket(tmp_path):
    wav = wav_file(tmp_path / "audio.wav")
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(payload)
            response = {"choices": [{
                "finish_reason": "stop",
                "message": {"content": "Guten Tag"},
            }]}
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    class Server(socketserver.UnixStreamServer):
        allow_reuse_address = False

    socket_path = str(tmp_path / "asr.sock")
    with Server(socket_path, Handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            result = worker.transcribe(socket_path, wav, "German")
        finally:
            server.shutdown()
            thread.join()
    assert result["text"] == "Guten Tag"
    assert result["language"] == "German"
    assert seen[0]["messages"][0]["content"][0]["input_audio"]["data"]
    assert seen[0]["chat_template_kwargs"] == {"asr_language": "German"}
