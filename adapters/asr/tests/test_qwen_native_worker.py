import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import qwen_native_worker as worker
from test_qwen_gguf_worker import wav_file


def args(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3_asr"}')
    return SimpleNamespace(
        binary="/binary", binary_sha256="0" * 64,
        model=str(model), threads=16, language="German",
    )


def test_bridge_preserves_text_and_reports_unknown_detected_language(tmp_path):
    options = args(tmp_path)
    audio = wav_file(tmp_path / "clip.wav")
    with patch.object(worker.subprocess, "run", return_value=SimpleNamespace(
        returncode=0, stdout="  words\nmore  \n",
    )) as run:
        result = worker.transcribe(options, audio)
    assert result["text"] == "  words\nmore  "
    assert result["language"] is None
    assert result["segments"][0]["end"] == 0.1
    command = run.call_args.args[0]
    assert command[-2:] == ["--language", "German"]
    assert run.call_args.kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"


@pytest.mark.parametrize("code,text", [(1, "\n"), (0, "incomplete")])
def test_bridge_never_turns_failed_cli_into_empty_success(tmp_path, code, text):
    with patch.object(worker.subprocess, "run", return_value=SimpleNamespace(
        returncode=code, stdout=text,
    )):
        with pytest.raises(worker.AdapterError):
            worker.transcribe(args(tmp_path), wav_file(tmp_path / "clip.wav"))


def test_empty_native_result_has_no_invented_segments(tmp_path):
    with patch.object(worker.subprocess, "run", return_value=SimpleNamespace(
        returncode=0, stdout="\n",
    )):
        result = worker.transcribe(args(tmp_path), wav_file(tmp_path / "clip.wav"))
    assert result == {"text": "", "language": None, "segments": []}


def test_bridge_echoes_identity_and_loads_per_request(tmp_path):
    options = args(tmp_path)
    audio = wav_file(tmp_path / "clip.wav")
    requests = "".join(json.dumps({"index": i, "input": str(audio)}) + "\n" for i in (23, 8))
    output = io.StringIO()
    with patch.object(worker, "verify_file"), patch.object(worker, "transcribe", return_value={
        "text": "words", "language": None, "segments": [],
    }) as transcribe, patch.object(worker.sys, "stdin", io.StringIO(requests)), patch.object(
        worker.sys, "stdout", output,
    ):
        worker.serve(options)
    assert transcribe.call_count == 2
    assert [json.loads(line)["index"] for line in output.getvalue().splitlines()] == [23, 8]


@pytest.mark.parametrize("index", [True, -1, None, "23"])
def test_bridge_rejects_invalid_identity(tmp_path, index):
    with patch.object(worker, "verify_file"), patch.object(worker.sys, "stdin", io.StringIO(
        json.dumps({"index": index, "input": "/clip.wav"}) + "\n",
    )):
        with pytest.raises(worker.AdapterError, match="index"):
            worker.serve(args(tmp_path))
