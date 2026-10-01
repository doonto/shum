"""Unit tests for flix_fast_common.py (no torch/ctranslate2 required)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import flix_fast_common as common


def make_whisper_checkpoint_dir(tmp_path: Path, *, name: str = "model", nested_processor: bool = True) -> Path:
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]})
    )
    (model_dir / "generation_config.json").write_text(json.dumps({"language": "de", "task": "transcribe"}))
    if nested_processor:
        (model_dir / "processor_config.json").write_text(
            json.dumps(
                {
                    "feature_extractor": {
                        "feature_size": 128,
                        "sampling_rate": 16000,
                        "hop_length": 160,
                        "chunk_length": 30,
                        "n_fft": 400,
                        "dither": 0.0,
                    },
                    "processor_class": "WhisperProcessor",
                }
            )
        )
    else:
        (model_dir / "preprocessor_config.json").write_text(
            json.dumps({"feature_size": 80, "sampling_rate": 16000, "hop_length": 160, "chunk_length": 30, "n_fft": 400})
        )
    return model_dir


# ---------------------------------------------------------------------------
# validate_whisper_model_dir
# ---------------------------------------------------------------------------

def test_validate_whisper_model_dir_accepts_valid_checkpoint(tmp_path):
    model_dir = make_whisper_checkpoint_dir(tmp_path)
    assert common.validate_whisper_model_dir(str(model_dir)) == model_dir


def test_validate_whisper_model_dir_rejects_relative_path():
    with pytest.raises(common.AdapterError, match="absolute"):
        common.validate_whisper_model_dir("relative/path")


def test_validate_whisper_model_dir_rejects_missing_dir(tmp_path):
    with pytest.raises(common.AdapterError, match="does not exist"):
        common.validate_whisper_model_dir(str(tmp_path / "nope"))


def test_validate_whisper_model_dir_rejects_missing_config(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(common.AdapterError, match="config.json"):
        common.validate_whisper_model_dir(str(model_dir))


def test_validate_whisper_model_dir_rejects_non_whisper_architecture(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_asr", "architectures": ["Qwen3ASRForConditionalGeneration"]}))
    with pytest.raises(common.AdapterError, match="Whisper checkpoint"):
        common.validate_whisper_model_dir(str(model_dir))


# ---------------------------------------------------------------------------
# manifest / identity / generation defaults
# ---------------------------------------------------------------------------

def test_read_manifest_returns_none_when_absent(tmp_path):
    model_dir = make_whisper_checkpoint_dir(tmp_path)
    assert common.read_manifest(model_dir) is None


def test_read_manifest_parses_present_file(tmp_path):
    model_dir = make_whisper_checkpoint_dir(tmp_path)
    (model_dir / common.MANIFEST_FILENAME).write_text(json.dumps({"repo": "Flix-AI/flix-swissgerman-full", "revision": "abc123"}))
    manifest = common.read_manifest(model_dir)
    assert manifest == {"repo": "Flix-AI/flix-swissgerman-full", "revision": "abc123"}


def test_model_identity_uses_manifest_revision_when_present(tmp_path):
    model_dir = make_whisper_checkpoint_dir(tmp_path)
    (model_dir / common.MANIFEST_FILENAME).write_text(json.dumps({"repo": "r", "revision": "deadbeef"}))
    assert common.model_identity(model_dir) == "deadbeef"


def test_model_identity_falls_back_to_path_hash_without_manifest(tmp_path):
    model_dir = make_whisper_checkpoint_dir(tmp_path)
    identity = common.model_identity(model_dir)
    assert identity == common.model_identity(model_dir)  # stable
    assert len(identity) == 16


def test_read_generation_defaults_reads_language_and_task(tmp_path):
    model_dir = make_whisper_checkpoint_dir(tmp_path)
    assert common.read_generation_defaults(model_dir) == ("de", "transcribe")


def test_read_generation_defaults_defaults_without_file(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    assert common.read_generation_defaults(model_dir) == (None, "transcribe")


# ---------------------------------------------------------------------------
# flatten_feature_extractor_config
# ---------------------------------------------------------------------------

def test_flatten_feature_extractor_config_from_nested_processor_config(tmp_path):
    model_dir = make_whisper_checkpoint_dir(tmp_path, nested_processor=True)
    flat = common.flatten_feature_extractor_config(model_dir)
    assert flat == {"feature_size": 128, "sampling_rate": 16000, "hop_length": 160, "chunk_length": 30, "n_fft": 400}


def test_flatten_feature_extractor_config_from_flat_preprocessor_config(tmp_path):
    model_dir = make_whisper_checkpoint_dir(tmp_path, nested_processor=False)
    flat = common.flatten_feature_extractor_config(model_dir)
    assert flat["feature_size"] == 80


def test_flatten_feature_extractor_config_errors_without_any_config(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(common.AdapterError, match="processor_config"):
        common.flatten_feature_extractor_config(model_dir)


def test_flatten_feature_extractor_config_errors_without_feature_size(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "preprocessor_config.json").write_text(json.dumps({"sampling_rate": 16000}))
    with pytest.raises(common.AdapterError, match="feature_size"):
        common.flatten_feature_extractor_config(model_dir)


# ---------------------------------------------------------------------------
# cache resolution
# ---------------------------------------------------------------------------

def test_resolve_ct2_output_dir_uses_cache_root_and_identity(tmp_path):
    model_dir = make_whisper_checkpoint_dir(tmp_path)
    (model_dir / common.MANIFEST_FILENAME).write_text(json.dumps({"repo": "r", "revision": "rev123"}))
    cache_root = str(tmp_path / "cache")
    output = common.resolve_ct2_output_dir(model_dir, cache_root=cache_root, quantization="int8")
    assert output == tmp_path / "cache" / "rev123" / "int8"


def test_default_cache_root_honors_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("SHUM_FLIX_FAST_CACHE", str(tmp_path / "custom"))
    assert common.default_cache_root() == tmp_path / "custom"


def test_default_cache_root_rejects_relative_env_override(monkeypatch):
    monkeypatch.setenv("SHUM_FLIX_FAST_CACHE", "relative/cache")
    with pytest.raises(common.AdapterError, match="absolute"):
        common.default_cache_root()


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def test_load_provenance_errors_when_missing(tmp_path):
    ct2_dir = tmp_path / "ct2"
    ct2_dir.mkdir()
    with pytest.raises(common.AdapterError, match="flix-fast-provenance.json"):
        common.load_provenance(ct2_dir)


def test_verify_provenance_matches_model_passes_for_matching_identity(tmp_path):
    model_dir = make_whisper_checkpoint_dir(tmp_path)
    (model_dir / common.MANIFEST_FILENAME).write_text(json.dumps({"repo": "r", "revision": "rev123"}))
    common.verify_provenance_matches_model({"source_identity": "rev123"}, model_dir)  # no raise


def test_verify_provenance_matches_model_rejects_mismatch(tmp_path):
    model_dir = make_whisper_checkpoint_dir(tmp_path)
    (model_dir / common.MANIFEST_FILENAME).write_text(json.dumps({"repo": "r", "revision": "rev123"}))
    with pytest.raises(common.AdapterError, match="provenance mismatch"):
        common.verify_provenance_matches_model({"source_identity": "other-rev"}, model_dir)


# ---------------------------------------------------------------------------
# WAV validation & result building (parity with main.py's contract)
# ---------------------------------------------------------------------------

def write_wav(path: Path, *, channels: int = 1, rate: int = 16000, sampwidth: int = 2, n_frames: int = 1600) -> None:
    import wave

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(rate)
        wav_file.writeframes(b"\x00" * n_frames * sampwidth * channels)


def test_validate_input_wav_accepts_16khz_mono(tmp_path):
    wav_path = tmp_path / "clip.wav"
    write_wav(wav_path, n_frames=16000)
    path, duration = common.validate_input_wav(str(wav_path))
    assert path == wav_path
    assert duration == pytest.approx(1.0)


def test_validate_input_wav_rejects_wrong_rate(tmp_path):
    wav_path = tmp_path / "clip.wav"
    write_wav(wav_path, rate=8000)
    with pytest.raises(common.AdapterError, match="16000 Hz"):
        common.validate_input_wav(str(wav_path))


def test_validate_input_wav_rejects_stereo(tmp_path):
    wav_path = tmp_path / "clip.wav"
    write_wav(wav_path, channels=2)
    with pytest.raises(common.AdapterError, match="channel"):
        common.validate_input_wav(str(wav_path))


def test_build_result_normalizes_segments():
    result = common.build_result("hi", "de", [{"start": 0, "end": 1.5, "text": "hi"}])
    assert result == {"text": "hi", "language": "de", "segments": [{"start": 0.0, "end": 1.5, "text": "hi"}]}


def test_build_result_rejects_end_before_start():
    with pytest.raises(common.AdapterError, match="before start"):
        common.build_result("hi", None, [{"start": 2.0, "end": 1.0, "text": "hi"}])


def test_write_result_is_atomic(tmp_path):
    output_path = tmp_path / "result.json"
    common.write_result({"text": "hi", "language": None, "segments": []}, output_path)
    assert json.loads(output_path.read_text())["text"] == "hi"
    assert list(tmp_path.glob("*.tmp")) == []
