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
  network-capable library is imported, and every `from_pretrained`/`restore_from`
  call also passes `local_files_only=True` explicitly as an independent
  second guard. There is no code path that reaches the network to check for
  or download model updates.
* **CPU-only, PyTorch CPU builds**: `pyproject.toml` pins `torch` to the
  `https://download.pytorch.org/whl/cpu` index for every engine (see
  *Install* below), and each engine explicitly loads its model on
  `"cpu"`/`torch.device("cpu")` and pins the intra-op thread count from
  `--threads` via `torch.set_num_threads`, plus the standard
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

The three engines pull in large, and sometimes mutually incompatible,
dependency trees (`nemo_toolkit[asr]` bundles its own PyTorch
Lightning/Hydra/OmegaConf stack; `transformers` for Qwen3-ASR requires a
recent `>=4.57.6` release that may not be what `nemo_toolkit` expects).
**Install each engine you need into its own virtual environment** rather
than a single shared one. All three share the same `pyproject.toml` /
`uv.lock`; `UV_PROJECT_ENVIRONMENT` just tells `uv sync` which venv to
populate for a given `--extra`:

```bash
cd adapters/asr

# NeMo engine
UV_PROJECT_ENVIRONMENT=.venv-nemo uv sync --extra nemo
.venv-nemo/bin/python main.py --engine nemo \
    --model /absolute/path/to/model-dir \
    --input /absolute/path/to/audio.wav \
    --output /absolute/path/to/result.json \
    --threads 4

# Whisper (Transformers) engine
UV_PROJECT_ENVIRONMENT=.venv-whisper uv sync --extra whisper
.venv-whisper/bin/python main.py --engine whisper \
    --model /absolute/path/to/whisper-checkpoint-dir \
    --input /absolute/path/to/audio.wav \
    --output /absolute/path/to/result.json \
    --threads 4

# Qwen3-ASR engine
UV_PROJECT_ENVIRONMENT=.venv-qwen uv sync --extra qwen
.venv-qwen/bin/python main.py --engine qwen \
    --model /absolute/path/to/qwen3-asr-checkpoint-dir \
    --input /absolute/path/to/audio.wav \
    --output /absolute/path/to/result.json \
    --threads 4
```

`pyproject.toml` pins `torch` to PyTorch's CPU-only wheel index
(`download.pytorch.org/whl/cpu`) for every extra, so `uv sync` never pulls a
CUDA build, on any platform (verified: `uv sync --extra whisper` on this
repo resolves `torch==2.14.0+cpu`).

Each `.venv-*` directory is already covered by the repository's
`.gitignore` (`**/.venv*/`).

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
* **whisper**: loads the checkpoint with the `automatic-speech-recognition`
  `pipeline(..., device="cpu", dtype=torch.float32, local_files_only=True)`
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
