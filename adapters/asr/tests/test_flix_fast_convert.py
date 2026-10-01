"""Unit tests for flix_fast_convert.py.

The real ``ctranslate2.converters.TransformersConverter`` is never invoked
here (that would require the heavy deps and would load the checkpoint into
RAM); ``converter_factory`` is injected with a fake that just materializes
a placeholder ``model.bin``, matching how ``tests/test_main.py`` injects
fake pipelines/models to test main.py's engines without torch/transformers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import flix_fast_common as common
import flix_fast_convert as convert_module


def make_model_dir(tmp_path: Path, *, revision: str | None = "rev123") -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]})
    )
    (model_dir / "generation_config.json").write_text(json.dumps({"language": "de", "task": "transcribe"}))
    (model_dir / "processor_config.json").write_text(
        json.dumps({"feature_extractor": {"feature_size": 128, "sampling_rate": 16000, "hop_length": 160, "chunk_length": 30, "n_fft": 400}})
    )
    if revision is not None:
        (model_dir / common.MANIFEST_FILENAME).write_text(json.dumps({"repo": "Flix-AI/flix-swissgerman-full", "revision": revision}))
    return model_dir


class FakeConverter:
    def __init__(self, model_dir, *, copy_files, low_cpu_mem_usage):
        self.model_dir = model_dir
        self.copy_files = copy_files
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.convert_calls: list[dict] = []

    def convert(self, output_dir, *, quantization, force):
        self.convert_calls.append({"output_dir": output_dir, "quantization": quantization, "force": force})
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "model.bin").write_bytes(b"fake-ct2-model")
        (Path(output_dir) / "config.json").write_text("{}")
        return output_dir


def make_factory():
    created: list[FakeConverter] = []

    def factory(model_dir, *, low_cpu_mem_usage):
        converter = FakeConverter(model_dir, copy_files=list(convert_module.DEFAULT_COPY_FILES), low_cpu_mem_usage=low_cpu_mem_usage)
        created.append(converter)
        return converter

    factory.created = created
    return factory


def test_convert_writes_provenance_and_preprocessor_config(tmp_path):
    model_dir = make_model_dir(tmp_path)
    output_dir = tmp_path / "ct2-out"
    factory = make_factory()

    result = convert_module.convert(
        model_dir, output_dir, quantization="int8", force=False, low_cpu_mem_usage=True, converter_factory=factory
    )

    assert result == output_dir
    assert len(factory.created) == 1
    assert factory.created[0].convert_calls == [{"output_dir": str(output_dir), "quantization": "int8", "force": True}]

    preprocessor = json.loads((output_dir / common.PREPROCESSOR_CONFIG_FILENAME).read_text())
    assert preprocessor == {"feature_size": 128, "sampling_rate": 16000, "hop_length": 160, "chunk_length": 30, "n_fft": 400}

    provenance = json.loads((output_dir / common.PROVENANCE_FILENAME).read_text())
    assert provenance["source_identity"] == "rev123"
    assert provenance["quantization"] == "int8"
    assert provenance["source_manifest"] == {"repo": "Flix-AI/flix-swissgerman-full", "revision": "rev123"}
    assert provenance["generation_defaults"] == {"language": "de", "task": "transcribe"}


def test_convert_skips_reconversion_when_already_matching(tmp_path):
    model_dir = make_model_dir(tmp_path)
    output_dir = tmp_path / "ct2-out"
    factory = make_factory()

    convert_module.convert(model_dir, output_dir, quantization="int8", force=False, low_cpu_mem_usage=True, converter_factory=factory)
    convert_module.convert(model_dir, output_dir, quantization="int8", force=False, low_cpu_mem_usage=True, converter_factory=factory)

    assert len(factory.created) == 1  # second call skipped the (expensive) conversion step


def test_convert_force_reconverts_even_when_matching(tmp_path):
    model_dir = make_model_dir(tmp_path)
    output_dir = tmp_path / "ct2-out"
    factory = make_factory()

    convert_module.convert(model_dir, output_dir, quantization="int8", force=False, low_cpu_mem_usage=True, converter_factory=factory)
    convert_module.convert(model_dir, output_dir, quantization="int8", force=True, low_cpu_mem_usage=True, converter_factory=factory)

    assert len(factory.created) == 2


def test_convert_reconverts_when_quantization_changes(tmp_path):
    model_dir = make_model_dir(tmp_path)
    output_dir = tmp_path / "ct2-out"
    factory = make_factory()

    convert_module.convert(model_dir, output_dir, quantization="int8", force=False, low_cpu_mem_usage=True, converter_factory=factory)
    convert_module.convert(model_dir, output_dir, quantization="float32", force=False, low_cpu_mem_usage=True, converter_factory=factory)

    assert len(factory.created) == 2


def test_already_converted_false_when_provenance_missing(tmp_path):
    model_dir = make_model_dir(tmp_path)
    output_dir = tmp_path / "ct2-out"
    output_dir.mkdir()
    (output_dir / "model.bin").write_bytes(b"stale")
    assert convert_module.already_converted(output_dir, model_dir, "int8") is False


def test_run_end_to_end_with_explicit_output(tmp_path):
    model_dir = make_model_dir(tmp_path)
    output_dir = tmp_path / "ct2-out"
    factory = make_factory()

    exit_code = convert_module.run(
        ["--model", str(model_dir), "--output", str(output_dir), "--quantization", "int8"],
        converter_factory=factory,
    )

    assert exit_code == 0
    assert (output_dir / common.PROVENANCE_FILENAME).is_file()


def test_run_rejects_non_whisper_model(tmp_path, capsys):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "bert"}))

    exit_code = convert_module.run(["--model", str(model_dir), "--output", str(tmp_path / "out")])

    assert exit_code == 1
    assert "Whisper checkpoint" in capsys.readouterr().err
