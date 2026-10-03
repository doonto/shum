# primeLine FP32 ONNX worker

`primeline_onnx_worker.py` runs the audited primeLine NeMo TDT export with
`onnx-asr` and ONNX Runtime's CPU provider. It does not load NeMo, export
models, download files, or change the standard adapter. It needs `onnx-asr`
and `onnxruntime` in a dedicated Python environment. The measured setup used
six inference threads and one worker/in-flight clip: three complete
67-window corpus passes took 109.219, 112.804, and 112.541 seconds.
The measured worker's peak RSS was about 3.1 GiB; allow memory headroom for
the host and other engines. These are measurements, not a per-machine guarantee.

## Inputs and provenance

Pass `--model` as the absolute **source directory** containing exactly one
original `.nemo` file, and `--onnx-model` as the absolute derived ONNX directory.
The validated export contains `encoder-model.onnx`, `decoder_joint-model.onnx`,
`vocab.txt`, `config.json`, `export-check.json`, and its externally stored
weights. The exporter used NeMo 3.0.0 / torch 2.14.0+cpu. Its original source
revision was `3f1a9bcb611dfeda53fe74fe5f1a3d5701e8023e`, with source
SHA-256 `1b7f6e4f5dcffabd44464a4f8afbb0688f71ea972c960e19efce36e73494c80f`.
The original export is not bundled with the repository. A portable layout
might put it at `$SHUM_ROOT/models-derived/primeline-onnx/` and the source
at `$SHUM_ROOT/models/primeline/3f1a9bcb611dfeda53fe74fe5f1a3d5701e8023e/`.
The measured runtime used Python 3.13, `onnx-asr` 0.12.0, and
`onnxruntime` 1.30.0; an isolated environment could live under
`$XDG_CACHE_HOME/shum/prime-engine-onnx/.venv`.
`primeline_onnx_requirements.txt` pins all seven packages found in that
environment; install it in a new isolated environment if needed.
These are provenance references, **not** defaults embedded in the worker.
Copy the export and model together to other absolute locations if needed.

Generate a manifest **once** after export, in the derived artifact directory,
without overwriting the source checkpoint:

```sh
python adapters/asr/primeline_onnx_worker.py --engine nemo \
  --model "$SOURCE_MODEL_DIR" --onnx-model "$ONNX_ARTIFACT_DIR" \
  --threads 6 --print-manifest > "$ONNX_ARTIFACT_DIR/.shum-primeline-onnx.json"
```

The exporter creates `export-check.json` with original-source SHA-256,
FP32 initializer types, and the inventory of graph and external-weight sizes.
`--print-manifest` checks the source `.shum-manifest.json` against the Rust
model-manager schema (`repo`, `revision`, `files: {relative-path: sha256}`),
hashes every source file, checks the export's original checkpoint digest, and
generates SHA-256 hashes for every inventoried derived artifact and the audit.
At worker startup, the same source inventory, audit, and **all artifact hashes**
are checked before loading ONNX Runtime. This requires reading several GiB
once per worker start. Do not modify the export after generating its manifest.
An optional pointer **beside the source revision directory** (not inside it;
extra files inside a managed revision invalidate Rust's inventory), named
`.shum-primeline-onnx.json`, can contain
`{"artifact_dir": "/absolute/export/directory"}` (or a safe relative child
path, without `.` or `..`); then `--onnx-model` can be omitted. Use explicit
`--onnx-model` to leave the source model tree untouched.

## Protocol

```sh
python adapters/asr/primeline_onnx_worker.py --engine nemo \
  --model "$SOURCE_MODEL_DIR" --onnx-model "$ONNX_ARTIFACT_DIR" \
  --threads 6 --serve
```

Send one `{"input": "/absolute/mono-pcm16-16khz.wav"}` per stdin line.
The worker loads ONNX once, processes one request at a time, and writes one
JSON result per request. A malformed request, unsupported WAV, or decoder
failure emits `{"error":"request[N]: ..."}` and exits nonzero. stdout contains
only protocol JSON; diagnostics go to stderr. For one clip, replace `--serve`
with `--input /absolute/clip.wav --output /absolute/result.json`. Output uses
`{"text": "...", "language": null, "segments": [{"start": 0,
"end": seconds, "text": "..."}]}`. Silence returns an empty segment list.

`onnx-asr`'s TDT decoder exposes **token onset timestamps**, but does not
expose reliable token end times or VAD speech boundaries. The measured
implementation emitted one segment spanning the actual WAV duration. This
worker deliberately preserves that coarse, genuine window timing instead of
inventing word or speech-boundary intervals. It does not silently resample,
downmix, or select a different engine.

Run isolated tests with `pytest -q adapters/asr/tests/test_primeline_onnx_worker.py`
from the repository root with `adapters/asr` on `PYTHONPATH`.
