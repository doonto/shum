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
