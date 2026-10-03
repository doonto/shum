# ASR adapters (NeMo, Whisper/Transformers, Qwen3-ASR)

A single, self-contained Python CLI adapter (`main.py`) that runs offline,
CPU-only speech-to-text transcription with one of three model families,
selected with `--engine`. It has no dependency on, and is not imported by,
any other adapter or the Rust codebase — it communicates only through its
CLI arguments and the JSON file it writes.

```
python main.py --engine {nemo,whisper,qwen} \
    --model /absolute/local/model/path \
    --input /absolute/16khz/mono.wav \
    --output /absolute/result.json \
    --threads N
```

`--model` is always an absolute path to a *directory* for all three
engines, consistent with how a model manager typically lays out one
directory per local model checkout:

| `--engine` | `--model` expects | Original checkpoint format |
| --- | --- | --- |
| `nemo` | a directory containing exactly one `*.nemo` file | Original NVIDIA NeMo checkpoint (e.g. Parakeet, Conformer) |
| `whisper` | a directory | Local Hugging Face Transformers Whisper checkpoint (e.g. Flix SwissGerman) |
| `qwen` | a directory | Local Hugging Face Qwen3-ASR checkpoint (`Qwen/Qwen3-ASR-*-hf`) |

## Persistent worker mode (`--serve`)

Loading a NeMo/Whisper/Qwen3-ASR checkpoint is expensive (seconds, not
milliseconds). A caller that transcribes many short chunks — e.g. a
pipeline splitting long audio into ~28s clips — should **not** invoke
`main.py` once per chunk, since that reloads the model every single time.
Instead, start one persistent worker process per engine/model with
`--serve` and feed it one request per chunk over stdin:

```
python main.py --engine {nemo,whisper,qwen}     --model /absolute/local/model/path     --threads N     --serve
```

* `--input`/`--output` are not used (and are rejected if given) in `--serve`
  mode; there is no single input/output file.
* The model is loaded **exactly once**, immediately after `--model` is
  validated and before the first request is read — not once per request.
* The worker then reads newline-delimited JSON requests from stdin, one
  clip per line:
  ```
  {"input": "/absolute/path/to/clip-0001.wav"}
  ```
  and writes exactly one JSON-lines response per request to stdout, using
  the same `{text, language, segments}` schema as single-clip mode (see
  *Output format* below), e.g.:
  ```
  {"text": "hallo welt", "language": null, "segments": [{"start": 0.0, "end": 1.2, "text": "hallo welt"}]}
  ```
* **stdout carries JSON only** — one compact line per request, flushed
  immediately so a caller reading the pipe sees each response as soon as
  it's ready. All human-readable diagnostics go to stderr instead.
* **On any error** (malformed request line, a request missing/with a
  non-string `"input"`, an input file that fails the 16 kHz mono WAV check,
  or a transcription failure), the worker writes a single
  `{"error": "..."}` JSON line to stdout, logs the same message to stderr,
  and **exits with a non-zero status** — it does not try to keep serving
  after a failure. A caller that wants to keep processing further chunks
  after an error must start a new worker process.
* Blank/whitespace-only stdin lines are ignored (treated as line-protocol
  padding, not requests).
* Closing stdin (EOF) after zero or more successful requests ends the
  worker cleanly with exit code `0`.

Single-clip mode (`--input`/`--output`, no `--serve`) is unchanged and
still supported for one-off transcriptions.

## Experimental shared-server Qwen3-ASR GGUF worker

`qwen_gguf_worker.py` is an **alternative** to the Transformers Qwen
worker. It launches one offline CPU-only `llama-server` on a private Unix
socket, sends 16 kHz WAV data over that socket, and serves up to four
indexed requests concurrently. The temporary socket directory is private
(mode 0700); the server is supervised with a Linux parent-death signal
and stopped when the worker exits. No audio URL or public HTTP port is
used. The Python worker has no additional package dependencies.

This profile is **not enabled by default**. Set the Qwen model spec to
`repo = "Qwen/Qwen3-ASR-1.7B"` with
`revision = "7278e1e70fe206f11671096ffdd38061171dd6e5"` and a
separate, initially empty model base path in the selected models directory
(for example `/absolute/selected/models/qwen-original`), then run
`shum models install-pinned --role qwen` with that config. Do not
identify a GGUF converted from this **original** checkpoint as the
existing `Qwen/Qwen3-ASR-1.7B-hf` revision: they are distinct repos
and model layouts. The model manager verifies the original snapshot;
the worker hashes the server binary and both derived GGUF files on
every startup. The original model is never rewritten.

`qwen_native_worker.py` is a separate experimental bridge to an already
built original-checkpoint Qwen C CLI. It requires `--binary`,
`--binary-sha256`, an absolute `--model`, positive `--threads` and
`--serve`; optional `--language` is `German` or `Russian`. It serves
indexed JSON-lines requests, but launches and reloads the native model
for **each** clip. Its timing is therefore not persistent-model timing.
The bridge preserves the CLI text, reports failures explicitly, and
does not invent detected language or word timestamps. Nonempty text
has one whole-window segment. This is not a production-default adapter.

For offline comparisons outside the model manager, the bundled
`scripts/asr_engine_study.py` validates the fixed 67-window corpus and an
explicit source-description JSON, runs a persistent indexed worker with
one to four requests in flight, and saves partial/error/resource-stop
reports. The source description identifies engine/checkpoint/runtime,
their SHA-256 digests, and an absolute-path artifact checksum map under
`runtime_artifacts_sha256` (or `files`). Include all source shards,
tokenizer/config files, derived weights and backend binaries. Files are
checked before and after execution; this does not prove numerical
conversion equivalence.

Each run must use a new output directory and a deadline no greater than
600 seconds. Preparation hashing is measured separately; the deadline
starts before worker launch. The shared guard retains 4 GiB effective
headroom, rejects growth in worker-tree swap, and stops on new system
swap-out together with memory PSI full avg10 at least 1%. Global swap-out
alone is telemetry. A successful speed gate requires all 67 windows
strictly below 600 seconds; quality remains unverified without reference
text. Use a separately verified user-systemd memory/swap cap when running
real models; the helper itself does not install or configure such a cap.

### Serial CPU-profile measurements

`scripts/cpu_profile_study.py` measures fresh primeLine ONNX FP32,
Turbo CTranslate2 INT8 and Community-1 workers without updating installed
settings or selecting a production default:

```sh
python3 scripts/cpu_profile_study.py \
  --spec /absolute/trial.json \
  --passport /absolute/passport.json \
  --output /absolute/new-trial-directory
```

The spec supplies `component` (`primeline`, `turbo`, or `diarization`),
`processes`, `threads`, `estimated_peak_mib`, `limit_seconds` (at most 600),
and the worker `command` array. ASR `inputs` contain `index`, `path` and
`duration` for each frozen WAV. Each process has only one outstanding
request. primeLine's production protocol is unindexed; the meter
correlates its response with that process's sole pending input and
retains the unmodified stdout separately. Turbo responses must echo
their request indices. Diarization runs one continuous WAV, supplies
`duration_seconds`, and writes canonical turns to the `{output}` command
placeholder; its stdout receipt is not an ASR result.

The passport supplies a fixed `session_deadline_epoch` and previously
verified artifact `file_stats` mapping absolute filenames to
`[size, mtime_ns, inode]`. These metadata checks detect changes after
preparation; they do not replace the preparation SHA-256 manifest or
the adapter's mandatory startup validation. The experiment-only
`scripts/cpu_profile_worker.py` wraps the existing adapters, measures
model loading and inference, and controls auxiliary native thread pools
without changing decoding. Use Community-1's `--profile` to capture its
real segmentation, embedding and assembly hooks.
The wrapper accepts an opt-in `--detail-profile` for diarization only.
It observes the pinned embedding backend without changing the algorithm;
unsupported backends fail explicitly. Hooks are restored even when the
pipeline fails. See the diarization adapter documentation for the exact
nested measurement scopes.
The separate `--embedding-channels-last` factor changes only runtime
Conv2d storage/input layout. It verifies unchanged logical FP32 bytes
and restores original weight storage; it is never enabled by default.
For the current diarization CLI, requested batch size `1` preserves the
loaded default rather than forcing an effective size of one. Report the
`parameters` / `configured` event's effective sizes, not just the profile
identifier; the pinned Community-1 segmentation default is 32.

Reports include raw stdout/stderr, per-input results, measured phase
events, CPU time, owned processes and threads, sampled aggregate RSS,
own swap, global swap-out and memory PSI. Aggregate RSS can double-count
shared pages and miss peaks between samples; it is not proportional-set
memory. The deadline includes worker startup, validation and cleanup.
The same 4 GiB reserve and swap/PSI protections remain mandatory.
Foreign model work blocks a run; sustained foreign heavy CPU interrupts
it rather than silently treating interference as model performance.

`full_speed_gate` is a **single-run technical speed gate**, not profile
admission or an accuracy claim. Admission needs three independent fresh
full runs strictly below 600 seconds, output/completion diagnostics and
comparisons against a control. Short pilots never qualify a profile.
Full ASR runs require the frozen 67 inputs; full diarization requires
the corresponding continuous 1800 seconds. Empty text, repetition,
token limits, speaker counts and speech coverage need interpretation
against that control. Without reference annotations, accuracy remains
unconfirmed.

Set `[asr.qwen]` to a Python 3.11+ interpreter and
`script = "adapters/asr/qwen_gguf_worker.py"`, `threads = 16`,
`max_in_flight = 4`. The adapter `args` must supply absolute paths and
their lowercase SHA-256 hex strings:

| Flag | Artifact / value |
| --- | --- |
| `--server`, `--server-sha256` | Pinned CPU-built `llama-server` executable and its local SHA-256 |
| `--gguf`, `--gguf-sha256` | Q8_0 text GGUF derived from the pinned **original** Qwen checkpoint; the measured file's hash is `160537d1960d8367b38a3720aeaab8bc670fc5eb7b22fd58f0f94773af5e2832` |
| `--mmproj`, `--mmproj-sha256` | Matching Q8_0 audio projector; the measured file's hash is `8adf1c469db468b96d1cdbd4a90ce15fa8d98b8a0560d868d984ba4927926f87` |
| `--slots` | `4` (or fewer, matching `max_in_flight`) |
| `--language` | Optional `German` or `Russian` decoder prefix; omit only when automatic language detection is acceptable |

The measured `llama.cpp` source commit was
`4da6337767f973e2b4d0797e5b323d77d8565e4a`; compute the server
binary hash **after** building it (`sha256sum /absolute/llama-server`),
since compilation changes the binary. Remove `qwen` from
`resources.offload_roles`: GGUF is already quantized and does not support
Transformers disk offload. Set `resources.estimated_peak_mib.qwen` to a
peak actually measured for the shared server **plus** worker on your
machine, rather than the Transformers estimate. The model and artifact
hashes, paths, language setting and in-flight limit become part of the
job manifest; changing them requires a new output directory.

A separate 30-minute CPU experiment (four slots, 16 threads, forced
German) finished 67 windows in 401.2 seconds including server startup,
without a length-truncated or repeating answer. This is **not** the
formal Rust benchmark, and speed alone does not establish accuracy:
some windows without diarized speech still yielded text, including one
117-character hallucination candidate. Autodetection also selected
several wrong languages and hit a generation limit. Validate difficult
clips against the audio before relying on this alternative. Set an
appropriate language for recordings containing Russian rather than
forcing German on their Russian speech.

## Other locally tested CPU engines (not integrated adapters)

The [Swiss German Turbo Whisper CT2 weights](https://huggingface.co/gcoli/whisper-large-v3-turbo-swiss-german-mit)
at revision `540fb2e7a9135048963c5de9822579c96f8962e5` were loaded
directly with `faster-whisper`/CTranslate2 int8. On the same 67 windows,
three processes with six threads each, German transcription, VAD and
beam 3 with **default decoding fallback** took 282.9 seconds. No window
exceeded a 0.5 repeated-four-gram fraction. Disabling fallback and
reducing to beam 2 took 269.1 seconds but caused severe repetition in
three windows. This is a *different* source checkpoint: do not pass its
CT2 directory to `flix_fast_worker.py` with the Flix model manifest.
That adapter intentionally rejects mismatched provenance. A separate
pinned model definition and a quality-checked adapter would be needed
before using Turbo in a normal job.

[NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
Q8_0 GGUF at revision `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`
ran locally in the official NeMo-Speech.cpp v0.1.0 **CPU** CLI:

```bash
nemo-speech transcribe /private/67-clips \
  --model /absolute/path/to/parakeet-tdt-0.6b-v3.q8_0.gguf \
  --device cpu --format json --concurrency 1 \
  --output-dir /private/parakeet-results
```

One model load plus all 67 clips took 213.2 seconds, with 67 JSON
responses, about four CPU cores active and no large four-gram repeats.
The output JSON has words with timestamps but does **not** implement
shum's persistent JSON-lines worker protocol or its model-revision
checks. It cannot replace `[asr.primeline]` by changing only `script`.
Some windows disagree sharply with the existing primeLine output;
without an audio-grounded transcript these comparisons do not
establish WER or suitability for Swiss German. See the
[method and all engine results](../../README.md#cpu-engine-comparison-on-one-recording).

## Swiss German Whisper Turbo experiment

The separate [Turbo CTranslate2 repeatability guide](turbo_ct2_README.md)
documents the pinned publisher checkpoint, three-process INT8 beam-3/fallback
profile, frozen-corpus deadline and private diagnostics. This is not Flix,
an automatically selected production engine, or a word-timestamp adapter.
The [Turbo ONNX guide](turbo_onnx_README.md) covers the same pinned checkpoint's
streamed FP32 export, cached CPU decoder, optimized thread policy and native
Whisper fallback/VAD profile. The measured full ONNX profile failed the
600-second gate; an INT8 inference profile was not completed.

## Optional CPU/disk offload (`--offload-dir`)

For the separate Qwen3-ASR CPU ONNX Runtime INT4 and local OpenVINO
export experiments, see [the alternative-engine guide](qwen_onnx_README.md).
These use explicit paths and indexed diagnostic workers, not automatic
production adapter selection. Source-weight auditing, conversion
success, the full-corpus speed gate and audio-grounded quality acceptance
are separate checks.

Whisper and Qwen3-ASR can opt in to Accelerate CPU/disk offload. The default
load path is unchanged; NeMo does not support this flag. Explicitly provide
an **absolute, private** cache root outside the original model snapshot:

```bash
SHUM_ROOT=/absolute/path/to/selected/installation
ASR_DIR=/absolute/path/to/installed/runtime/adapters/asr
SHUM_MODELS=/absolute/path/to/selected/models
FLIX_REVISION=your-pinned-model-revision
mkdir -m 700 -p /absolute/private/asr-offload
"$SHUM_ROOT/runtime/environments/flix/bin/python" "$ASR_DIR/main.py" --engine whisper \
    --model "$SHUM_MODELS/flix/$FLIX_REVISION" --threads 4 \
    --offload-dir /absolute/private/asr-offload --serve
```

The same flag works for `--engine qwen` and in single-clip mode. Shum's
Rust scheduler enables it only when `resources.offload_dir` is configured
and its memory policy selects an offloaded worker. The adapter creates a private mode-0700
directory per engine and model revision/path under that root and refuses
symlinks, non-private cache directories, and cache roots inside the source
model snapshot. It never writes into the original checkpoint. Hugging Face
loading remains `local_files_only=True` with the offline environment flags
below. The 2 GiB Accelerate **model placement** budget on CPU is not a
process RSS limit; processors, activations, and temporary weights also use
RAM. Accelerate may read disk-mapped safetensors directly, leaving the cache
directory empty even when `hf_device_map` contains disk-offloaded modules.
Provide enough free disk space for other checkpoint formats.

This mode was checked offline on the pinned original Flix Whisper
(`a9c347a904af5d31b28c537ac732e0618aed6136`) and Qwen3-ASR
(`bcd2b5b7f32b480ab5790554cfa8347f246a14f3`) with Accelerate 1.15,
Transformers 5.17, and CPU PyTorch 2.14. Both generated *exactly* the same
text, language and segments as their existing baseline on the same private
28-second pilot clip; no clip/transcript is included here. With 4 threads:

| Engine | Default peak RSS / total elapsed | Offload peak RSS / total elapsed |
| --- | --- | --- |
| Flix Whisper | ~9.0 GiB / 111 s | ~5.0 GiB / 205 s |
| Qwen3-ASR | ~11.9 GiB / 50 s | ~5.2 GiB / 245 s |

These are single-process peak RSS and wall-clock measurements, not memory
guarantees or parallel-worker benchmarks. Disk offload substantially reduces
resident memory but slows every clip, especially Qwen3-ASR.

## Output format

```json
{
  "text": "full transcription",
  "language": "german",
  "segments": [
    {"start": 0.0, "end": 1.2, "text": "hallo"},
    {"start": 1.2, "end": 2.4, "text": "welt"}
  ]
}
```

* `start`/`end` are seconds relative to the start of the input file.
* `language` is `null` when the model/engine cannot report one (NeMo
  checkpoints are monolingual and expose no language detection).
* `segments` covers the whole file: engines that don't expose sub-segment
  timestamps (Qwen3-ASR on its own, or a NeMo model whose decoder doesn't
  support timestamps) report a single segment spanning `[0, duration]`
  rather than inventing sub-segment timing.

## Guarantees

* **Offline**: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
  `HF_DATASETS_OFFLINE=1` and `HF_HUB_DISABLE_TELEMETRY=1` are set before any
  network-capable library is imported. Hugging Face model and processor
  loading also passes `local_files_only=True` explicitly; NeMo restores
  the given local `.nemo` checkpoint. There is no code path that reaches
  the network to check for or download model updates.
* **CPU-only, PyTorch CPU builds**: `pyproject.toml` pins `torch` to the
  `https://download.pytorch.org/whl/cpu` index for every engine (see
  *Install* below), and each engine explicitly loads its model on
  `"cpu"`/`torch.device("cpu")` by default; in offload mode, Accelerate
  dispatches layers between CPU and local disk (never GPU). Both modes pin
  the intra-op thread count from `--threads` via `torch.set_num_threads`, plus the standard
  `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` /
  `NUMEXPR_NUM_THREADS` environment variables.
* **No silent fallback**: input that isn't 16 kHz mono WAV, a `--model` path
  that doesn't match the selected engine's expected checkpoint shape, or any
  loading/inference failure is reported as a single explicit `error: ...`
  message on stderr with a non-zero exit code. The adapter never resamples,
  downmixes, or substitutes a different model/engine behind your back, and
  never writes a partial/corrupt output file (the JSON is written to a
  temporary file in the same directory and atomically renamed into place).
* **Exact, documented HF/NeMo APIs**: every model-loading and inference call
  below (`ASRModel.restore_from`, `transcribe(..., timestamps=True)`, the
  `automatic-speech-recognition` pipeline's `return_timestamps`/
  `return_language`, `AutoModelForMultimodalLM` +
  `processor.apply_transcription_request` + `processor.decode(...,
  return_format="parsed")`) was verified against the actual `nemo_toolkit`
  and `transformers` source and confirmed importable at runtime; nothing
  here is guessed.

## Install (isolated per-engine `uv` environments)

Requires Python 3.10+ and [`uv`](https://docs.astral.sh/uv/) 0.5.3+.
The Qwen extra retains `librosa>=0.11.0` because librosa 1.0.0 requires
Python 3.12+, while these engines still support Python 3.10.

The three engines pull in large, and sometimes mutually incompatible,
dependency trees (`nemo_toolkit[asr]` bundles its own PyTorch
Lightning/Hydra/OmegaConf stack; `transformers` for Qwen3-ASR requires a
recent `>=5.18.0` release that may not be what `nemo_toolkit` expects).
Run `bash scripts/install.sh` from a source checkout to build the portable
binary and choose its installation, settings, projects, and models directories.
The onboarding wizard offers to prepare separate environments with `uv` after
you consent to the downloads. It writes them under the selected installation
root, **not** under the source checkout:

| Engine | Interpreter after setup |
| --- | --- |
| NeMo | `<installation-root>/runtime/environments/primeline/bin/python` |
| Whisper | `<installation-root>/runtime/environments/flix/bin/python` |
| Qwen3-ASR | `<installation-root>/runtime/environments/qwen/bin/python` |

The active adapter scripts and `pyproject.toml`/`uv.lock` live under
`<installation-root>/runtime/<release-id>/adapters/asr/`. The installed
`config.toml` records their exact paths under `[asr.*].script`. If you
declined environment setup, you can later run `shum --onboarding` and
accept it, or install one environment manually using the shipped manifests.
Run `shum --setup-models` to select supported revisions and confirm each
model download. It leaves custom and experimental adapters unchanged.
Set `ASR_DIR` to the parent of the installed `[asr.flix].script` path:

```bash
SHUM_ROOT=/absolute/path/to/selected/installation
ASR_DIR=/absolute/path/to/installed/runtime/adapters/asr
cd "$ASR_DIR"
uv venv "$SHUM_ROOT/runtime/environments/flix"
VIRTUAL_ENV="$SHUM_ROOT/runtime/environments/flix" uv sync --frozen --no-dev --extra whisper --active
"$SHUM_ROOT/runtime/environments/flix/bin/python" "$ASR_DIR/main.py" --engine whisper \
    --model /absolute/path/to/whisper-checkpoint-dir \
    --input /absolute/path/to/audio.wav --output /absolute/path/to/result.json --threads 4
```

Use `--extra nemo` with `primeline` or `--extra qwen` with `qwen` in the
same commands for the other engines. Set `ASR_DIR` from the current config,
not by sorting runtime release directories: older releases can coexist.

`pyproject.toml` routes `torch`, `torchaudio`, and `torchcodec` to PyTorch's
CPU-only wheel index (`download.pytorch.org/whl/cpu`) for every extra, so a
future transitive audio dependency cannot pull a CUDA wheel from PyPI alongside
CPU Torch. The current NeMo `[asr]` lock contains no TorchAudio or TorchCodec;
on Linux it resolves `torch==2.14.1+cpu`. This checks package provenance, not
NeMo model inference or native imports.

### Obtaining the checkpoints

This adapter never downloads a model itself; obtain each checkpoint ahead
of time with whatever tooling you prefer (`huggingface-cli download`,
`git lfs clone`, NGC, etc.) and point `--model` at the local result:

* **nemo**: a directory containing exactly one original `.nemo` file (a
  single archive), e.g. downloaded from NGC or a Hugging Face NeMo
  checkpoint repository and placed in its own model directory. The
  adapter looks for exactly one `*.nemo` file directly inside `--model`
  and fails explicitly (rather than guessing) if there are zero or more
  than one.
* **whisper**: a full local Transformers checkpoint directory (must contain
  `config.json`, model weights, and the processor/tokenizer files), e.g. a
  `Flix-AI/flix-swissgerman-full`-style local checkout.
* **qwen**: a full local `Qwen/Qwen3-ASR-{0.6B,1.7B}-hf` checkpoint
  directory (must contain `config.json`, model weights, and the processor
  files).

## Usage notes per engine

* **nemo**: loads the checkpoint with
  `nemo.collections.asr.models.ASRModel.restore_from(restore_path=..., map_location="cpu")`
  and transcribes with `model.transcribe([...], batch_size=1, timestamps=True)`.
  Segment-level timestamps are currently only produced by NeMo for
  Parakeet-family models (CTC/RNNT/TDT decoders); for other architectures
  (or older NeMo releases that reject the `timestamps` kwarg) the adapter
  falls back to a single segment spanning the whole file instead of
  guessing sub-segment timing. NeMo checkpoints are monolingual, so
  `language` is always `null`.
* **whisper**: loads the local checkpoint and processor, then uses the
  `automatic-speech-recognition` pipeline (with `device="cpu"` by default;
  Accelerate controls placement in offload mode)
  and calls it with `chunk_length_s=30, return_timestamps=True,
  return_language=True`, which yields chunk-level segments plus a per-chunk
  detected/forced language (Whisper is the only architecture the pipeline
  supports language detection for). If the very last chunk's closing
  timestamp is unresolved (a known Whisper decoding edge case), its `end`
  is filled in with the audio's total duration rather than dropped.
* **qwen**: loads the checkpoint with `AutoProcessor`/
  `AutoModelForMultimodalLM.from_pretrained(..., dtype=torch.float32,
  local_files_only=True)` and transcribes with the documented
  `processor.apply_transcription_request(audio=<local wav path>)` →
  `model.generate(...)` → `processor.decode(generated_ids,
  return_format="parsed")` flow, which returns the detected `language` and
  `transcription` directly. Qwen3-ASR alone does not produce sub-segment
  timestamps (that requires a separate `Qwen3-ForcedAligner` checkpoint,
  out of scope for this single-model `--model` contract), so the adapter
  reports one segment spanning the whole file.

## Tests

Unit tests are dependency-free (they do not require `torch`/`nemo`/
`transformers` to be installed): model loading and inference are injected
via factory/pipeline objects passed into `main.run()`, and WAV format
validation is exercised against small real WAV files built with the
standard-library `wave` module. A few tests intentionally call the real
`default_*_model_factory`/`default_whisper_pipeline_factory` functions to
confirm they raise a clear `AdapterError` (not a raw traceback) when the
corresponding extra isn't installed.

```bash
cd adapters/asr
python3 -m pip install pytest   # or: uv run --with pytest pytest tests/
python3 -m pytest tests/
```
