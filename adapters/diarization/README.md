# Diarization adapter (pyannote Community-1)

Isolated, self-contained Python CLI adapter that runs offline speaker
diarization on CPU using a locally installed
[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
pipeline. It has no dependency on, and is not imported by, any other adapter
or the Rust codebase — it communicates only through its CLI arguments and
the JSON file it writes.

## Guarantees

* **Offline**: sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
  `HF_DATASETS_OFFLINE=1` before any network-capable library is imported, so
  any accidental attempt to reach the network fails loudly instead of
  silently downloading a model or falling back to another pipeline.
* **CPU-only**: the pipeline is explicitly moved to `torch.device("cpu")`
  regardless of GPU availability, and thread counts are pinned from
  `--threads` via both `torch.set_num_threads` and the standard
  `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` /
  `NUMEXPR_NUM_THREADS` environment variables.
* **No audio upload**: the adapter only ever calls
  `Pipeline.from_pretrained(<local directory>)` with no API token, so there
  is no code path that can reach pyannoteAI's hosted `precision-2` cloud
  pipeline or upload audio anywhere.
* **No silent fallback**: input that isn't 16 kHz mono WAV, a model
  directory missing `config.yaml`, or any pipeline/network failure is
  reported as an explicit error on stderr with a non-zero exit code. The
  adapter never resamples, downmixes, or substitutes a different model
  behind your back, and never writes a partial/corrupt output file.

## Install

Requires Python 3.10+ and `uv`. Run `bash scripts/install.sh` from the
source checkout and choose the installation, settings, projects, and models
directories. The onboarding wizard offers to create this environment after
you consent to the dependency download:

`<installation-root>/runtime/environments/diarization/bin/python`

The shipped script and manifest are under
`<installation-root>/runtime/<release-id>/adapters/diarization/`; the
installed `[diarization].script` config value gives the exact script path.
If you declined setup, rerun `shum --onboarding` or use the shipped
requirements file to prepare only this environment:

```bash
SHUM_ROOT=/absolute/path/to/selected/installation
DIAR_DIR=/absolute/path/to/installed/runtime/adapters/diarization
uv venv "$SHUM_ROOT/runtime/environments/diarization"
uv pip install --python "$SHUM_ROOT/runtime/environments/diarization/bin/python" \
    -r "$DIAR_DIR/requirements.txt"
```

### Download the model (once, requires network + accepting user conditions)

1. Accept the user conditions on the
   [model page](https://huggingface.co/pyannote/speaker-diarization-community-1)
   with a Hugging Face account.
2. Clone the full pipeline repository to a local directory (this bundles all
   submodel weights so the pipeline can run fully offline afterwards):

   ```bash
   git lfs install
   git clone https://hf.co/pyannote/speaker-diarization-community-1 \
       /absolute/path/to/selected/models/diarization
   ```

No further network access is required to run the standalone adapter after
this step. For Shum-managed models, use `shum --setup-models` instead: it
creates and verifies the model manifest and downloads only after your
explicit latest or pinned-revision choice.

## Usage

```bash
"$SHUM_ROOT/runtime/environments/diarization/bin/python" "$DIAR_DIR/main.py" \
    --model /absolute/path/to/selected/models/diarization \
    --input /absolute/16khz/mono.wav \
    --output /absolute/result.json \
    --threads 4
```

* `--model`: absolute path to the local model directory from the step above
  (must contain `config.yaml`).
* `--input`: absolute path to a 16 kHz mono WAV file. Other formats are
  rejected explicitly (no silent resampling/downmixing).
* `--output`: absolute path to write the result JSON to.
* `--threads`: number of CPU threads to use for inference.
* `--segmentation-batch-size` and `--embedding-batch-size`: optional
  pyannote inference batch sizes (both default to 1 for unchanged behavior).
  Larger batches can improve CPU utilization but need more RAM; benchmark
  them on the target machine rather than assuming they meet a time target.
* `--profile`: log stage duration, ten-percent progress markers for long
  stages, total inference time, and structured profile records (below) to stderr without logging audio or
  transcription text. Even without `--profile`, the adapter emits
  `{"event":"shum.progress","version":1,"phase":"embeddings","completed":N,"total":T}`
  as NDJSON on stderr when pyannote's embedding hook reports a new completed
  batch count. Segmentation counters use the same format if its hook provides
  them. These are measured batches, not an elapsed-time estimate. For a 30-minute
  test through `shum benchmark-diarization`, set `args = [
  "--segmentation-batch-size", "16", "--embedding-batch-size", "32",
  "--profile"]` in the `[diarization]` adapter configuration.
* `--preload-audio`: load the full validated WAV once through `torchaudio`
  and pass one in-memory waveform to Community-1. This avoids decoding the
  same file again for every speaker-embedding crop while preserving one
  global speaker timeline. A 30-minute mono 16-kHz float32 waveform uses
  about 110 MiB, plus the model and intermediate activations. Preloading is
  opt-in until its end-to-end CPU timing and turn quality are validated.

### Structured profiling contract

With `--profile`, stderr includes flushed, single-line JSON records with
`event: "shum.profile"`, `version: 1`, `phase`, and `status`. Other stderr
lines remain human-readable; a parent meter must parse JSON lines and filter
by event/version. Stdout and the result JSON schema are unchanged.

* `parameters`: `status: "requested"` before model loading, then
  `"configured"` after batch configuration. `parameters` contains `threads`,
  `requested_segmentation_batch_size`, `requested_embedding_batch_size`, and
  `preload_audio`. The configured record also contains the effective pipeline
  attributes `segmentation_batch_size` and `embedding_batch_size`; an absent
  attribute is `null`, not an assumed value. The default CLI value of 1
  preserves the loaded pipeline's defaults; only non-default CLI values
  override them. `threads` is the CPU thread count applied to the environment
  and, by the default factory, PyTorch. An injected factory must honor it.
  Preload is the selected mode, not proof that waveform loading succeeded.
* `model_loading`, `batch_configuration`, optional `waveform_loading`,
  `result_conversion`, and `result_writing`: `status: "completed"` with
  `elapsed_seconds` measured using `time.perf_counter()` around the operation.
  Model loading includes lazy imports and CPU placement; result writing
  includes atomic file replacement.
* `segmentation` and `embeddings`: `status: "progress"` contains integer
  `completed`/`total` counters from the hook, with duplicate pairs suppressed.
  These records have no duration. The existing `shum.progress` records remain
  available with or without profiling and retain their schema.
* A segmentation or embeddings hook **without** `total` emits a `"completed"`
  record with measured `elapsed_seconds` and `since`, the preceding hook
  boundary (or `"pipeline_start"`). These are wall-clock hook intervals,
  including preparation between boundaries, not isolated neural-network
  timings. The last observed counters are included if available; completion
  does not fabricate a final counter.
* `clustering_assembly`: `"completed"` measures the interval from the
  embeddings completion hook through pipeline return, with `since: "embeddings"`.
  It includes the pipeline's remaining clustering/reconstruction/assembly,
  not result conversion or file writing. It is omitted without that boundary.
* `pipeline_total`: `"completed"` measures the entire pipeline call and
  overlaps its stage intervals; do not sum it with those intervals.

An operation that raises emits `"failed"` with measured `elapsed_seconds`
for its unfinished interval and re-raises the original error. Pipeline
failures emit a failed `pipeline_total` plus the active hook phase when known,
including its last partial counters and `since` boundary. Without an observed
stage, only the total is reported. Completed records already emitted remain
valid; failure never produces a completed result-writing record or a success
fallback. Hard kills cannot emit failure records, so retain prior progress.
Missing hooks produce missing stage durations, never zero placeholders or
estimated timings. A measured zero duration is valid for a short operation.

Profiling uses the existing Community-1 hook (`total`/`completed` counters and
artifact callbacks); it adds no dependency or inference wrapper. It changes
neither the model, speaker algorithm, precision, decoding, language behavior,
nor offline CPU execution. Batch sizes and preloading remain explicit
experimental options. Hook detail depends on the installed pyannote version;
missing detail is not synthesized.

### Experiment-only detailed CPU observer

[`scripts/diarization_cpu_observer.py`](../../scripts/diarization_cpu_observer.py)
adds opt-in call-level measurements for the installed Community-1 PyTorch
WeSpeaker path. It imports no Torch, pyannote, TorchCodec, or model code.
The experiment worker owns CLI wiring; the ordinary adapter does not enable it.

```python
observer = diarization_cpu_observer.install(pipeline, emit)
try:
    result = pipeline(audio_input, hook=existing_hook)
finally:
    observer.close()
```

`install(pipeline, emit) -> Observer` requires an already-loaded CPU
`PyannoteAudioPretrainedSpeakerEmbedding`, `BaseWeSpeakerResNet` with
`resnet.forward`, pyannote `Audio.crop`, and the loaded TorchCodec
`AudioDecoder` class. Unsupported backends or missing boundaries raise
`ObserverError` before installing any wrappers. Installation failures roll
back prior wrappers. Do not use it with compiled models or concurrent
pipeline execution; use one observer per process.

`close() -> None` restores wrappers before calling
`emit(phase, started, **values)` once for each scope:

| Phase | Exact boundary | Nested in |
| --- | --- | --- |
| `detail_audio_crop` | This pipeline's embedding `Audio.crop` | None |
| `detail_audio_decoder_init` | TorchCodec `AudioDecoder.__init__` during that crop | `detail_audio_crop` |
| `detail_audio_range_read` | `AudioDecoder.get_samples_played_in_range` during that crop | `detail_audio_crop` |
| `detail_embedding_forward` | The embedding model's `forward` | None |
| `detail_embedding_fbank` | Its `compute_fbank`, including waveform scaling, Kaldi fbank via `torch.vmap`, and centering | `detail_embedding_forward` |
| `detail_embedding_network` | Its `resnet.forward`, including convolutions, weighted statistics pooling, and embedding layers | `detail_embedding_forward` |

The instance wrappers leave Torch's module dispatch and existing hooks intact;
network-forward timing excludes hooks outside that method. Decoder methods
are temporarily wrapped at class level, but collect data only inside the
observed crop's context. Other decoder calls delegate unchanged and are not
counted. All arguments, masks, outputs, and exceptions pass through unchanged.
No weights, seeds, dtype, batching, decoding, task, or installed files change.

Records use `detail_version: 1`, `measurement: "sum_of_calls"`,
`scope: "inclusive"`, and `nested_in` as above. `elapsed_seconds` is the sum
of measured monotonic call intervals; `cpu_seconds` is the corresponding sum
of `process_time()` intervals across **all process threads**. Nested scopes
overlap: never add parent and child totals or treat their difference as an
exclusive measurement. `observer_elapsed_seconds` is the observation window.
The explicit `elapsed_seconds` value overrides the worker emitter's
time-since-`started` field; no stage duration is inferred from that window.

Each record includes `calls`, `completed_calls`, `failed_calls`, and `status`
(`completed`, `failed`, `partial_failure`, or `not_observed`). Uncalled scopes
have null elapsed/CPU durations, not fabricated zero timings; preload normally
leaves decoder scopes uncalled. Failed calls retain their measured partial
intervals. Lazy minimum-length model probes can be caught by pyannote and
appear here as failed forward calls even when the pipeline succeeds; detailed
scope status is **not** the overall pipeline outcome.
`nested_in` names the expected enclosing boundary; `nested_calls` counts
calls actually made inside that boundary, while `standalone_calls` counts
the others (including direct feature calls and all top-level scope calls).

`shapes` is a histogram of actual input/output shapes and observed call
outcomes, capped at 16 distinct signatures per scope.
`unlisted_shape_calls` counts overflow; `input_items` sums known leading
input dimensions over successful **and failed attempts**, and
`shaped_input_calls` identifies how many attempts contributed. Crop records
distinguish `input_kind: "file"` from `"waveform"`. These records contain no
audio values, filenames, or speech. Metadata extraction errors are reported
as `metadata_errors` and `metadata_error_types` without replacing an original
result or exception.

Close in a `finally` block to retain failed-call measurements. Closing during
an observed call raises an error. Concurrent replacement of an observed
method is reported, not overwritten. A successful restoration makes close
idempotent, including when the emitter subsequently raises; emission errors
propagate rather than masking them with a success fallback. Hard termination
cannot restore hooks or emit these aggregates. This observer measures CPU
wall/process time only; it does not synchronize accelerators or establish
bitwise equivalence between file-backed and preloaded decoding.
Wrappers add timing and metadata overhead; compare against an unobserved
baseline rather than treating instrumentation as free.

### Experiment-only CPU channels-last factor

[`scripts/diarization_cpu_layout.py`](../../scripts/diarization_cpu_layout.py)
provides a separate, opt-in `install(pipeline, emit) -> Handle` /
`handle.close() -> None` API. It does not enable itself in the ordinary
adapter, import Torch, or modify the observer/worker. The parent must pass an
already-loaded CPU/eval/uncompiled PyTorch WeSpeaker ResNet pipeline and wire
its explicit experiment flag separately.

The factor uses `torch.nn.utils.convert_conv2d_weight_memory_format` on the
embedding ResNet only, then registers one removable `conv1` pre-hook with
`with_kwargs=True`. The hook preserves all other arguments/keywords, performs
`input.contiguous(memory_format=torch.channels_last)`, and keeps logical
`(B,1,F,T)` dimensions. It does not permute axes, change masks, fbank,
clustering, batches, seeds, task, dtype, device, thread settings, or precision
policy. Existing hooks retain their order and remain installed. This can
coexist with the fine-grained observer; input conversion is inside its
inclusive network-forward timing.

All registered embedding parameters and buffers are checked for dense CPU
storage, FP32 parameters/floating buffers, unchanged identities/metadata,
absent gradients, and distinct tensor/storage identities. Shared modules,
tied registered tensors, or shared storage (even disjoint views) are rejected.
External aliases not registered in the model cannot be discovered; require
exclusive synchronous inference ownership, no optimizer/backward activity,
and no concurrent model mutation. Inputs must be dense CPU FP32 and arrive
without gradient recording; the factor does not change gradient mode or
`requires_grad`. Unsupported backends, compiled paths, or missing utility
raise `LayoutError`; there is no fallback.

Before conversion, after conversion, after inference, and after restoration,
verification hashes canonical logical-order raw bytes, including signed zero,
and compares tensor names, Parameter identities, shapes, dtypes, devices,
gradient flags and storage metadata. Only converted Conv2d weight layout and
storage may differ during the experiment. The baseline retains each original
weight `.data` view/storage. Close removes only the new hook, restores those
exact views (pointers, strides, offsets), and verifies again, even when
post-inference verification fails. Partial installation rolls back. Unexpected
changes to unrelated buffers/parameters fail verification rather than being
silently repaired. Close is idempotent after successful cleanup, including
when verification or emission raises.

Always close in a `finally` block. Weight conversion temporarily retains both
original and converted storage; budget for that extra FP32 weight memory and
transient hashing copies. A hard kill cannot perform cleanup or final checks.

The supplied `emit(phase, started, **values)` receives:

* `layout_verification`: measured wall/process CPU overhead, `checkpoint`
  (`before_conversion`, `after_conversion`, `after_inference`, `after_restore`,
  or `after_rollback`), `status`, and on success a combined logical
  `state_sha256` / `tensor_count`.
* `layout_conversion`: measured weight-conversion overhead and status.
* `layout_restoration`: measured normal-close restoration overhead and status.
* `layout_input`: aggregate measured pre-hook overhead, successful/failed call
  counts, and up to 16 actual input signatures with shape, before/after
  strides, channels-last flags, and whether the original tensor was reused.
  Overflow calls are counted; an uncalled hook has null durations.

Records use `layout_version: 1`; explicit `elapsed_seconds` values override
the worker emitter's timer, and CPU intervals include all process threads.
No predicted speed or accuracy is emitted. FP32 parameter-byte preservation
does **not** guarantee identical embeddings: layout may select different
kernels/reduction orders. Compare canonical diarization output against a
fresh plain control before accepting the factor.

### Experimental ONNX CPU engine

[`onnx_worker.py`](./onnx_worker.py) implements the same JSON output schema
using the independent, third-party
[`pyannote-onnx-community`](https://github.com/ben91lin/pyannote-onnx-community)
engine. It passes the entire recording to one pipeline, so speaker IDs remain
global. This is **not** the default adapter; runtime and diarization quality
must be measured on the target machine. Install the engine at pinned commit
`86a95ebb6792c90cf2f29a76ed8743f635ebadcf` in a separate Python 3.11
environment, and download these public ONNX weights at the pinned revisions:

| Weight | Revision | SHA-256 of `onnx/model.onnx` |
| --- | --- | --- |
| [segmentation](https://huggingface.co/onnx-community/pyannote-segmentation-3.0) | `733a93b6473d019a773298e08cefa686894b1854` | `057ee564753071c0b09b5b611648b50ac188d50846bff5f01e9f7bbf1591ea25` |
| [WeSpeaker embedding](https://huggingface.co/onnx-community/wespeaker-voxceleb-resnet34-LM) | `6a61a1833ff2583aabeba044f5c8221f00b67ceb` | `3955447b0499dc9e0a4541a895df08b03c69098eba4e56c02b5603e9f7f4fcbb` |

Supply their **absolute local file paths** using
`--segmentation-model PATH --embedding-model PATH` and the matching
`--segmentation-sha256 DIGEST --embedding-sha256 DIGEST`. The `--model`
argument remains the pinned Community-1 checkout, which supplies the local
PLDA artifacts. The worker validates every path and hash, forces the Hugging
Face libraries offline, and configures ONNX Runtime's CPU thread counts from
`--threads`. For a controlled trial, pass
`--program /path/to/onnx/venv/bin/python --script /path/to/onnx_worker.py`
to `benchmark-diarization` along with repeatable `--adapter-arg=VALUE`
entries for the four ONNX weight flags, using a new output directory.
The optional `--clustering-threshold` defaults to 0.7 and can be tuned
against a labeled reference; a lower seed threshold does **not** guarantee
more final speakers after VBx pruning. `--num-speakers N` forces a count
only when that count is independently known; it can produce three labels
without assigning the right speech to each person. On the first local
30-minute trial, the default ONNX engine finished in 121.3 seconds but
returned two speakers against three in the existing Community-1 run, with
only 62% label agreement on mutually detected speech after optimal label
permutation. Forcing three speakers raised agreement to 84.8%; leaving
the count automatic and setting the threshold to 0.65 yielded three
labels and 92% agreement, but recovered only 11.5% of the baseline's
shortest speaker (13 seconds total). The comparison is against a model,
not human ground truth. This quality result blocks replacing the default
adapter until the disagreements are checked against audio.
Model weights and transcripts must remain outside version control.
The engine is MIT/Apache-2.0; the public WeSpeaker ONNX model is CC-BY-4.0
and requires attribution.

### Experimental Nemotron 3 CPU engine

The [Nemotron 3 Diarization GGUF](https://huggingface.co/nvidia/Nemotron-3-Diarization)
Q8_0 model at revision `f667ed73aee57d40cc39428eb768b4fd87a0a29e`
(SHA-256 `08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1`)
was tested on the same private 1800-second WAV. Official NeMo-Speech.cpp
**v0.1.0** reports the model as compatible via `model info` but fails at
inference with `sortformer: pre_ln transformer variant is not supported`.
A CPU source build pinned to official commit
[`8c15060`](https://github.com/NVIDIA/NeMo-Speech.cpp/commit/8c15060bf0a71d7ac4f089931f2de65418b37128)
supports this V3 architecture:

```bash
git clone https://github.com/NVIDIA/NeMo-Speech.cpp.git /private/nemo-speech-src
cd /private/nemo-speech-src
git checkout --detach 8c15060bf0a71d7ac4f089931f2de65418b37128
git submodule update --init ggml
scripts/build_sentencepiece_static.sh
scripts/configure.sh cpu-diar
cmake --build --preset cpu-diar --parallel 8
```

On this host, the pinned SentencePiece dependency required
`-DCMAKE_POLICY_VERSION_MINIMUM=3.5` for CMake 4 and a compiler
`-include cstdint` flag for GCC 15; both were set in its local CMake
build cache rather than changing upstream model/runtime source. If
`build_sentencepiece_static.sh` fails for either reason after fetching
its source, configure its cache once, then rerun the script:

```bash
cmake -G Ninja -S .deps/sentencepiece-build/source \
  -B .deps/sentencepiece-build/build \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_CXX_FLAGS='-include cstdint' \
  -DCMAKE_BUILD_TYPE=Release -DSPM_BUILD_TEST=OFF \
  -DSPM_ENABLE_SHARED=OFF -DSPM_ENABLE_TCMALLOC=OFF
scripts/build_sentencepiece_static.sh
```

```bash
/private/nemo-speech-src/build/cpu-diar/bin/nemo-speech \
  diarize /private/1800-second.wav \
  --model /absolute/path/to/Nemotron-3-Diarization.q8_0.gguf \
  --device cpu --preset v3-offline --format json \
  --output /private/nemotron-speakers.json
```

The `v3-streaming` geometry **timed out after 610 seconds**. The
`v3-offline` **geometry preset**, which still streams in chunks, finished
in 60.2 seconds and yielded three labels. Do **not** use the separate
`--offline` full-attention flag on a 30-minute recording: its positional
table is limited to about 6.6 minutes. After best label permutation, the
fast preset agreed with the prior PyTorch output on only 71.7% of the
935.8 seconds where both marked a single speaker. It assigned only 1.5
of the PyTorch output's 13.5 seconds for the shortest speaker to the
corresponding label. These are comparisons between *models*, not DER or
proof of which one is right.

The native CLI returns `{"segments":[{"start", "end", "speaker"}]}` with
numeric speaker IDs, **not** this adapter's `turns`/`exclusive_turns`
schema or its verified model-manager protocol. It cannot be selected by
changing only the configured script path. Keep it experimental pending
audio review and a typed adapter with pinned source/model provenance.

### Output format

```json
{
  "turns": [
    {"start": 0.512, "end": 3.221, "speaker": "SPEAKER_00"}
  ],
  "exclusive_turns": [
    {"start": 0.512, "end": 3.221, "speaker": "SPEAKER_00"}
  ]
}
```

* `turns` mirrors pyannote's regular `speaker_diarization` output (overlap
  between speakers is possible).
* `exclusive_turns` mirrors Community-1's `exclusive_speaker_diarization`
  output (non-overlapping, easier to reconcile with transcript timestamps).
* All `start`/`end` values are in seconds relative to the start of the input
  file, rounded to milliseconds.

## Tests

Unit tests are dependency-free (they do not require `torch`/`pyannote.audio`
to be installed): pipeline loading is injected, and format validation is
exercised with real small WAV files built via the standard-library `wave`
module.

```bash
cd adapters/diarization
python3 -m pip install pytest   # if not already available
python3 -m pytest tests/
```
