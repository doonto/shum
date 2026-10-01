#!/usr/bin/env python3
"""Convert an original pinned HF Whisper checkpoint to CTranslate2 (offline).

Prototype tool for the Flix SwissGerman speed investigation: converts a
local Hugging Face Transformers Whisper checkpoint directory (the *same*
``--model`` directory ``main.py --engine whisper`` accepts, e.g. the
pinned ``models/flix/<revision>`` snapshot) into a quantized CTranslate2
model directory that ``flix_fast_worker.py`` can serve much faster on CPU.

    python flix_fast_convert.py \
        --model /absolute/models/flix/<pinned-revision> \
        --quantization int8

By default the converted copy is written *outside* this repository and
outside the original model snapshot, under
``~/.cache/shum/flix-fast-ct2/<revision>/<quantization>/`` (override with
``--output`` or ``--cache-root``/``SHUM_FLIX_FAST_CACHE``) -- the original
pinned checkpoint directory is only ever read, never modified.

This uses CTranslate2's own officially supported, generic
``TransformersConverter`` (registered "WhisperConfig" loader), which works
for arbitrary Hugging Face Whisper checkpoints including fine-tunes such as
Flix-AI/flix-swissgerman-full -- not a Flix-specific hack. See
``flix_fast_README.md`` for the compatibility/quality notes and current
status of this investigation.

IMPORTANT (memory): converting the real ~2.9 GiB checkpoint loads its
weights into RAM (bfloat16, ``low_cpu_mem_usage`` by default) for the
duration of the conversion. Do not run this against the real pinned model
while other memory-heavy jobs are active; see the README for measured
figures and the coordination rule this prototype was built under.
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import flix_fast_common as common
from flix_fast_common import AdapterError

# Files copied verbatim from the original checkpoint into the CTranslate2
# output directory alongside the converted weights. ``processor_config.json``
# is intentionally excluded here: its feature-extractor section is instead
# read, flattened, and re-written as preprocessor_config.json (see
# ``common.flatten_feature_extractor_config``) because faster-whisper does
# not understand the nested processor_config.json format.
DEFAULT_COPY_FILES = ("tokenizer.json", "tokenizer_config.json", "generation_config.json")

ConverterFactory = Callable[..., Any]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flix_fast_convert.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Absolute path to the original local HF Transformers Whisper checkpoint directory.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Absolute path for the converted CTranslate2 model directory. "
            "If omitted, resolved from --cache-root/SHUM_FLIX_FAST_CACHE + "
            "the model's pinned revision (or a path hash) + --quantization."
        ),
    )
    parser.add_argument(
        "--cache-root",
        default=None,
        help="Absolute cache root used to resolve --output when --output is omitted.",
    )
    parser.add_argument(
        "--quantization",
        default="int8",
        choices=common.QUANTIZATION_CHOICES,
        help="CTranslate2 weight quantization type (default: int8).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-convert even if a matching provenance.json already exists at the output.",
    )
    parser.add_argument(
        "--no-low-cpu-mem-usage",
        action="store_true",
        help=(
            "Disable from_pretrained(low_cpu_mem_usage=True) during conversion. "
            "Not recommended: this roughly doubles peak RAM while loading the "
            "checkpoint for conversion."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def already_converted(output_dir: Path, model_dir: Path, quantization: str) -> bool:
    """True if ``output_dir`` already holds a matching, current conversion."""
    if not (output_dir / "model.bin").is_file():
        return False
    try:
        provenance = common.load_provenance(output_dir)
    except AdapterError:
        return False
    return (
        provenance.get("source_identity") == common.model_identity(model_dir)
        and provenance.get("quantization") == quantization
    )


def default_converter_factory(model_dir: str, *, low_cpu_mem_usage: bool) -> Any:
    try:
        from ctranslate2.converters import TransformersConverter
    except ImportError as exc:
        raise AdapterError(
            "Missing dependency: install ctranslate2 + transformers (+ torch) "
            "in an isolated environment, e.g.:\n"
            "  uv venv --python 3.11 .venv-flix-fast\n"
            "  uv pip install --python .venv-flix-fast/bin/python "
            "-r flix_fast_requirements.txt --extra-index-url "
            "https://download.pytorch.org/whl/cpu\n"
            "(see flix_fast_README.md)"
        ) from exc
    return TransformersConverter(
        model_dir,
        copy_files=list(DEFAULT_COPY_FILES),
        low_cpu_mem_usage=low_cpu_mem_usage,
    )


def convert(
    model_dir: Path,
    output_dir: Path,
    *,
    quantization: str,
    force: bool,
    low_cpu_mem_usage: bool,
    converter_factory: ConverterFactory | None = None,
) -> Path:
    """Convert ``model_dir`` to CTranslate2 at ``output_dir``.

    Idempotent: skips the (expensive, RAM-loading) conversion step when
    ``output_dir`` already holds a matching provenance record for this
    exact ``model_dir``/``quantization`` pair, unless ``force`` is set. The
    lightweight preprocessor-config/provenance files are always
    (re)written so an interrupted or older-format output gets repaired
    without triggering a full re-conversion.
    """
    feature_config = common.flatten_feature_extractor_config(model_dir)
    manifest = common.read_manifest(model_dir)
    language, task = common.read_generation_defaults(model_dir)
    source_identity = common.model_identity(model_dir)

    if force or not already_converted(output_dir, model_dir, quantization):
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        factory = converter_factory or default_converter_factory
        converter = factory(str(model_dir), low_cpu_mem_usage=low_cpu_mem_usage)
        converter.convert(str(output_dir), quantization=quantization, force=True)
    else:
        print(f"{output_dir} already matches --model/--quantization; skipping conversion (use --force to redo)")

    (output_dir / common.PREPROCESSOR_CONFIG_FILENAME).write_text(
        json.dumps(feature_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    try:
        import ctranslate2

        ctranslate2_version = ctranslate2.__version__
    except ImportError:
        ctranslate2_version = None

    provenance = {
        "tool": "adapters/asr/flix_fast_convert.py",
        "converted_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_model_dir": str(model_dir),
        "source_identity": source_identity,
        "source_manifest": manifest,
        "quantization": quantization,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "ctranslate2_version": ctranslate2_version,
        "feature_extractor_config": feature_config,
        "generation_defaults": {"language": language, "task": task},
    }
    (output_dir / common.PROVENANCE_FILENAME).write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def run(argv: Sequence[str] | None = None, *, converter_factory: ConverterFactory | None = None) -> int:
    args = parse_args(argv)
    try:
        model_dir = common.validate_whisper_model_dir(args.model)
        if args.output is not None:
            output_dir = common.require_absolute(args.output, "--output")
        else:
            output_dir = common.resolve_ct2_output_dir(
                model_dir, cache_root=args.cache_root, quantization=args.quantization
            )
        common.configure_offline_environment()
        convert(
            model_dir,
            output_dir,
            quantization=args.quantization,
            force=args.force,
            low_cpu_mem_usage=not args.no_low_cpu_mem_usage,
            converter_factory=converter_factory,
        )
    except AdapterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {output_dir}")
    return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
