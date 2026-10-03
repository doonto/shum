"""Small offline contract tests; no ONNX model or runtime is loaded."""
import hashlib
import io
import json
import sys
import types
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import primeline_onnx_worker as worker


def fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "checkpoint.nemo").write_bytes(b"source")
    (source / worker.SOURCE_MANIFEST).write_text(json.dumps({
        "repo": worker.SOURCE_REPO,
        "revision": worker.SOURCE_REVISION,
        "files": {"checkpoint.nemo": hashlib.sha256(b"source").hexdigest()},
    }))
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    for name in ("encoder-model.onnx", "decoder_joint-model.onnx", "vocab.txt"):
        (graphs / name).write_bytes(name.encode())
    (graphs / "config.json").write_text(json.dumps(worker.CONFIG))
    audit = {
        "source_sha256": hashlib.sha256(b"source").hexdigest(),
        "checkpoint_float_dtypes": {"torch.float32": 701},
        "onnx_initializer_dtypes": {
            "encoder-model.onnx": {"FLOAT": 612},
            "decoder_joint-model.onnx": {"FLOAT": 13},
        },
        "files": {p.name: p.stat().st_size for p in graphs.iterdir()},
    }
    (graphs / "export-check.json").write_text(json.dumps(audit))
    (graphs / worker.MANIFEST).write_text(
        json.dumps(worker.create_manifest(source / "checkpoint.nemo", graphs))
    )
    clip = tmp_path / "clip.wav"
    with wave.open(str(clip), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0" * 32000)
    args = ["--engine", "nemo", "--model", str(source), "--onnx-model", str(graphs), "--threads", "6"]
    return source, graphs, clip, args


class Decoder:
    calls = 0

    def recognize(self, path):
        self.calls += 1
        return type("Result", (), {"text": "Hallo Welt", "timestamps": [0.1, 0.4]})()


def test_single_clip_preserves_measured_coarse_timing(tmp_path):
    _, _, clip, args = fixture(tmp_path)
    output = tmp_path / "result.json"
    model = Decoder()
    assert worker.run(args + ["--input", str(clip), "--output", str(output)],
                      factory=lambda directory, threads: model) == 0
    assert json.loads(output.read_text()) == {
        "text": "Hallo Welt", "language": None,
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hallo Welt"}],
    }
    assert model.calls == 1


def test_serve_loads_once_and_indexes_errors(tmp_path):
    _, _, clip, args = fixture(tmp_path)
    loads = []
    model = Decoder()

    def factory(directory, threads):
        loads.append((directory, threads))
        return model

    stdout = io.StringIO()
    requests = io.StringIO(json.dumps({"input": str(clip)}) + "\n" +
                           json.dumps({"input": str(clip)}) + "\n" + '{"input":false}\n')
    assert worker.run(args + ["--serve"], factory=factory, stdin=requests, stdout=stdout) == 1
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert len(loads) == 1 and loads[0][1] == 6
    assert [row["text"] for row in lines[:2]] == ["Hallo Welt", "Hallo Welt"]
    assert lines[2]["error"].startswith("request[2]:")
    assert model.calls == 2


def test_rejects_changed_source_and_artifact(tmp_path):
    source, graphs, clip, args = fixture(tmp_path)
    cli = args + ["--input", str(clip), "--output", str(tmp_path / "out.json")]
    (graphs / "vocab.txt").write_bytes(b"x" * (graphs / "vocab.txt").stat().st_size)
    assert worker.run(cli, factory=lambda *_: Decoder()) == 1
    (graphs / "vocab.txt").write_bytes(b"vocab.txt")
    (source / "checkpoint.nemo").write_bytes(b"other!")
    assert worker.run(cli, factory=lambda *_: Decoder()) == 1
    assert not (tmp_path / "out.json").exists()


def test_model_relative_pointer_and_manifest_command(tmp_path):
    source, graphs, clip, args = fixture(tmp_path)
    (source.parent / worker.MANIFEST).write_text(json.dumps({"artifact_dir": "../graphs"}))
    cli = [arg for i, arg in enumerate(args) if i not in (4, 5)]
    assert worker.run(cli + ["--serve"], factory=lambda *_: Decoder(),
                      stdin=io.StringIO(json.dumps({"input": str(clip)}) + "\n"),
                      stdout=io.StringIO()) == 1
    (source.parent / worker.MANIFEST).write_text(json.dumps({"artifact_dir": str(graphs)}))
    output = io.StringIO()
    assert worker.run(args + ["--print-manifest"], stdout=output) == 0
    assert json.loads(output.getvalue())["artifact_sha256"]
    response = io.StringIO()
    assert worker.run(cli + ["--serve"], factory=lambda *_: Decoder(),
                      stdin=io.StringIO(json.dumps({"input": str(clip)}) + "\n"),
                      stdout=response) == 0
    assert json.loads(response.getvalue())["text"] == "Hallo Welt"


def test_invalid_pcm_and_decode_failure_are_indexed(tmp_path):
    _, _, clip, args = fixture(tmp_path)
    with wave.open(str(clip), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(1)
        wav.setframerate(16000)
        wav.writeframes(b"\0" * 16000)
    output = io.StringIO()
    assert worker.run(args + ["--serve"], factory=lambda *_: Decoder(),
                      stdin=io.StringIO(json.dumps({"input": str(clip)}) + "\n"),
                      stdout=output) == 1
    assert "request[0]" in json.loads(output.getvalue())["error"]
    with wave.open(str(clip), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0" * 32000)

    class Broken:
        def recognize(self, path):
            raise RuntimeError("decoder crashed")

    output = io.StringIO()
    assert worker.run(args + ["--serve"], factory=lambda *_: Broken(),
                      stdin=io.StringIO(json.dumps({"input": str(clip)}) + "\n"),
                      stdout=output) == 1
    assert "decoder crashed" in json.loads(output.getvalue())["error"]


def test_runtime_uses_cpu_tdt_and_six_threads(tmp_path, monkeypatch):
    _, graphs, _, _ = fixture(tmp_path)
    seen = {}

    class SessionOptions:
        pass

    class Session:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    class Model:
        asr = types.SimpleNamespace(_encoder=Session(), _decoder_joint=Session())

        def with_timestamps(self):
            seen["timestamps"] = True
            return self

    def load_model(kind, **kwargs):
        seen.update(kind=kind, **kwargs)
        return Model()

    monkeypatch.setitem(sys.modules, "onnx_asr", types.SimpleNamespace(load_model=load_model))
    monkeypatch.setitem(sys.modules, "onnxruntime",
                        types.SimpleNamespace(SessionOptions=SessionOptions))
    worker.load_model(graphs, 6)
    assert seen["kind"] == "nemo-conformer-tdt"
    assert seen["path"] == graphs
    assert seen["providers"] == ["CPUExecutionProvider"]
    assert seen["sess_options"].intra_op_num_threads == 6
    assert seen["sess_options"].inter_op_num_threads == 1
    assert seen["timestamps"]


def test_missing_provenance_fails_before_runtime_load(tmp_path):
    _, graphs, clip, args = fixture(tmp_path)
    (graphs / worker.MANIFEST).unlink()
    loaded = []
    output = io.StringIO()
    assert worker.run(args + ["--serve"], factory=lambda *_: loaded.append(True),
                      stdin=io.StringIO(json.dumps({"input": str(clip)}) + "\n"),
                      stdout=output) == 1
    assert not loaded
    assert "provenance" in json.loads(output.getvalue())["error"]


def test_rejects_wrong_rust_manifest_schema_and_revision(tmp_path):
    source, graphs, _, _ = fixture(tmp_path)
    checkpoint = source / "checkpoint.nemo"
    source_manifest = source / worker.SOURCE_MANIFEST
    invalid = {"repo": worker.SOURCE_REPO, "revision": worker.SOURCE_REVISION,
               "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    source_manifest.write_text(json.dumps(invalid))
    try:
        worker.create_manifest(checkpoint, graphs)
    except worker.common.AdapterError as exc:
        assert "schema" in str(exc)
    else:
        assert False, "missing Rust files map accepted"
    invalid["files"] = {"checkpoint.nemo": invalid.pop("sha256")}
    invalid["revision"] = "wrong-revision"
    source_manifest.write_text(json.dumps(invalid))
    try:
        worker.create_manifest(checkpoint, graphs)
    except worker.common.AdapterError as exc:
        assert "revision" in str(exc)
    else:
        assert False, "wrong source revision accepted"
