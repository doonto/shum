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

Requires Python 3.10+.

```bash
cd adapters/diarization
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
       /absolute/local/model/directory
   ```

No further network access is required to run the adapter after this step.

## Usage

```bash
python main.py \
    --model /absolute/local/model/directory \
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
  stages, and total inference time to stderr without logging audio or
  transcription text. For a 30-minute
  test through `shum benchmark-diarization`, set `args = [
  "--segmentation-batch-size", "16", "--embedding-batch-size", "32",
  "--profile"]` in the `[diarization]` adapter configuration.
* `--preload-audio`: load the full validated WAV once through `torchaudio`
  and pass one in-memory waveform to Community-1. This avoids decoding the
  same file again for every speaker-embedding crop while preserving one
  global speaker timeline. A 30-minute mono 16-kHz float32 waveform uses
  about 110 MiB, plus the model and intermediate activations. Preloading is
  opt-in until its end-to-end CPU timing and turn quality are validated.

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
