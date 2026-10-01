import hashlib
import json
import os
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

import onnx_worker


class EmptyAnnotation:
    def itertracks(self, *, yield_label):
        assert yield_label
        return iter(())


def make_args(tmp_path: Path) -> list[str]:
    model = tmp_path / "model"
    (model / "plda").mkdir(parents=True)
    (model / "config.yaml").write_text("pipeline: {}\n")
    (model / "plda/plda.npz").write_bytes(b"plda")
    (model / "plda/xvec_transform.npz").write_bytes(b"transform")
    wav = tmp_path / "input.wav"
    with wave.open(str(wav), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0" * 320)
    seg = tmp_path / "seg.onnx"
    emb = tmp_path / "emb.onnx"
    seg.write_bytes(b"segmentation")
    emb.write_bytes(b"embedding")
    return [
        "--model", str(model),
        "--segmentation-model", str(seg),
        "--segmentation-sha256", hashlib.sha256(seg.read_bytes()).hexdigest(),
        "--embedding-model", str(emb),
        "--embedding-sha256", hashlib.sha256(emb.read_bytes()).hexdigest(),
        "--input", str(wav),
        "--output", str(tmp_path / "output.json"),
        "--threads", "4",
    ]


def test_offline_onnx_adapter_writes_same_diarization_schema(tmp_path):
    args = make_args(tmp_path)
    annotation = EmptyAnnotation()
    output = SimpleNamespace(
        speaker_diarization=annotation,
        exclusive_speaker_diarization=annotation,
    )

    def factory(model, segmentation, embedding, threads, threshold):
        assert model == str(tmp_path / "model")
        assert segmentation == str(tmp_path / "seg.onnx")
        assert embedding == str(tmp_path / "emb.onnx")
        assert threads == 4
        assert threshold == 0.7
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        return lambda audio_path, *, num_speakers: output

    assert onnx_worker.run(args, pipeline_factory=factory) == 0
    assert json.loads((tmp_path / "output.json").read_text()) == {
        "turns": [], "exclusive_turns": [],
    }


def test_weight_hash_mismatch_fails_before_model_load(tmp_path, capsys):
    args = make_args(tmp_path)
    args[args.index("--embedding-sha256") + 1] = "0" * 64
    assert onnx_worker.run(
        args, pipeline_factory=lambda *_: pytest.fail("must not load the model"),
    ) == 1
    assert not (tmp_path / "output.json").exists()
    assert "SHA-256 mismatch" in capsys.readouterr().err


def test_count_and_threshold_options_are_forwarded(tmp_path):
    args = make_args(tmp_path) + [
        "--num-speakers", "3", "--clustering-threshold", "0.65",
    ]
    annotation = EmptyAnnotation()
    output = SimpleNamespace(
        speaker_diarization=annotation,
        exclusive_speaker_diarization=annotation,
    )

    def factory(_model, _seg, _emb, _threads, threshold):
        assert threshold == 0.65

        def diarize(_audio, *, num_speakers):
            assert num_speakers == 3
            return output

        return diarize

    assert onnx_worker.run(args, pipeline_factory=factory) == 0


def test_invalid_speaker_count_fails_before_loading(tmp_path, capsys):
    args = make_args(tmp_path) + ["--num-speakers", "0"]
    assert onnx_worker.run(
        args, pipeline_factory=lambda *_: pytest.fail("must not load the model"),
    ) == 1
    assert "--num-speakers must be positive" in capsys.readouterr().err
