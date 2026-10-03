#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
project_root="$(pwd -P)"
remaps=("--remap-path-prefix=$project_root=/shum-source")
sensitive_paths=("$project_root")
if [[ -n "${CARGO_HOME:-}" && "$CARGO_HOME" == /* ]]; then
  cargo_home="$(realpath -m -- "$CARGO_HOME")"
  remaps+=("--remap-path-prefix=$cargo_home=/cargo")
  sensitive_paths+=("$cargo_home")
elif [[ -n "${HOME:-}" && "$HOME" == /* ]]; then
  cargo_home="$(realpath -m -- "$HOME/.cargo")"
  remaps+=("--remap-path-prefix=$cargo_home=/cargo")
  sensitive_paths+=("$cargo_home")
fi
if [[ -n "${HOME:-}" && "$HOME" == /* && "$HOME" != / ]]; then
  build_home="$(realpath -m -- "$HOME")"
  remaps+=("--remap-path-prefix=$build_home=/build-home")
  sensitive_paths+=("$build_home")
fi
separator=$'\x1f'
if [[ ${CARGO_ENCODED_RUSTFLAGS+x} ]]; then
  encoded="${CARGO_ENCODED_RUSTFLAGS:-}"
else
  encoded=""
  read -r -a existing_flags <<< "${RUSTFLAGS:-}"
  for flag in "${existing_flags[@]}"; do
    encoded+="${encoded:+$separator}$flag"
  done
fi
for flag in "${remaps[@]}"; do
  encoded+="${encoded:+$separator}$flag"
done
export CARGO_ENCODED_RUSTFLAGS="$encoded"

version="$(cargo metadata --no-deps --format-version 1 | sed -n 's/.*"name":"shum","version":"\([^"]*\)".*/\1/p')"
if [[ -z "$version" ]]; then
  echo "Cannot determine Shum version" >&2
  exit 1
fi

output="${1:-dist/shum-${version}}"
if [[ -e "$output" ]]; then
  echo "Refusing to overwrite distribution at $output" >&2
  exit 1
fi
cargo build --release --locked
mkdir -p "$output/bin" "$output/runtime/adapters/asr" "$output/runtime/adapters/diarization" \
  "$output/runtime/scripts" "$output/runtime/docs"
cp "${CARGO_TARGET_DIR:-target}/release/shum" "$output/bin/shum"
cp README.md "$output/runtime/README.md"
cp shum.example.toml "$output/runtime/shum.example.toml"
cp adapters/asr/{main.py,flix_fast_common.py,flix_fast_convert.py,flix_fast_worker.py,flix_onnx_probe.py,flix_onnx_pilot.py,flix_onnx_audit.py,qwen_gguf_worker.py,qwen_native_worker.py,pyproject.toml,uv.lock,flix_fast_requirements.txt,README.md,flix_fast_README.md,flix_onnx_README.md} "$output/runtime/adapters/asr/"
cp adapters/asr/{primeline_onnx_worker.py,primeline_onnx_README.md,primeline_onnx_requirements.txt,turbo_worker.py,turbo_ct2_worker.py,turbo_onnx_worker.py,turbo_README.md,turbo_ct2_README.md,turbo_onnx_README.md,turbo_requirements.txt} "$output/runtime/adapters/asr/"
cp adapters/diarization/{main.py,onnx_worker.py,requirements.txt,README.md} "$output/runtime/adapters/diarization/"
cp scripts/{flix_study.py,flix_matrix.py,asr_engine_study.py,shum_headless.py,onboarding-smoke.sh} "$output/runtime/scripts/"
cp docs/{USAGE.ru.md,API.md} "$output/runtime/docs/"
if command -v strip >/dev/null 2>&1; then
  strip --strip-debug "$output/bin/shum"
fi
for path in "${sensitive_paths[@]}"; do
  if grep -aFq -- "$path" "$output/bin/shum"; then
    echo "Release binary still embeds build path: $path" >&2
    exit 1
  fi
done
echo "Portable distribution: $output"
