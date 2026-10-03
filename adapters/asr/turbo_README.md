# Swiss German Turbo CT2 worker

`turbo_worker.py` serves the published CTranslate2 assets for
`gcoli/whisper-large-v3-turbo-swiss-german-mit` at revision
`540fb2e7a9135048963c5de9822579c96f8962e5`. It runs on CPU, offline,
with one loaded model and one request at a time per process. It does not
download or convert a model. Run it with the existing `adapters/asr/.venv-flix-fast`
environment; `turbo_requirements.txt` pins the measured runtime versions.
Do not use `turbo_ct2_worker.py` as a
production adapter: that study prototype starts its own three-process pool
and returns whole-window timestamps.

The Rust pipeline should start **three separate workers** through
`resources.asr_worker_counts`, with `--threads 6` on each. Do not add a pool
inside a worker. Budget about **2.1 GiB RSS per worker**; account for all
three workers and other models when setting memory limits.

## Provisioning and provenance

Pass the absolute original source directory as `--model`. Its
`.shum-manifest.json` must contain these fields:

```json
{
  "repo": "gcoli/whisper-large-v3-turbo-swiss-german-mit",
  "revision": "540fb2e7a9135048963c5de9822579c96f8962e5",
  "files": {
    "model.safetensors": "b37f10e1b905e7c6a0a7d3eb7462083b6e12caa71c06e26c83245d33337d246d"
  }
}
```

Put the publisher's CT2 directory at `<model>/ct2`, or supply its absolute
path with `--ct2-model`. It must contain `model.bin`, `config.json`,
`tokenizer.json`, `preprocessor_config.json` and `.shum-turbo-ct2.json`:

```json
{
  "repo": "gcoli/whisper-large-v3-turbo-swiss-german-mit",
  "revision": "540fb2e7a9135048963c5de9822579c96f8962e5",
  "files": {
    "model.bin": "baecb116482ebb80a434d4a71ab50d6642808b258b5ce0f82cfde571aca3dfa2",
    "config.json": "<verified 64-character SHA-256>",
    "tokenizer.json": "<verified 64-character SHA-256>",
    "preprocessor_config.json": "<verified 64-character SHA-256>",
    "vocabulary.json": "<verified 64-character SHA-256>"
  }
}
```

The source manifest may list additional source assets in `files`; the CT2
manifest must list all publisher model, configuration and tokenizer assets.
The worker hashes the source weights and every listed CT2 asset before
loading the model. The CT2 weights are **publisher-provided**, not a
locally converted copy. No independent numerical audit of the publisher's
conversion or transcription-accuracy claim has been completed. Preserve
the original model license and provenance; never record a local conversion
as verified.

```bash
adapters/asr/.venv-flix-fast/bin/python -u adapters/asr/turbo_worker.py \
  --engine whisper --model /absolute/pinned-source \
  --ct2-model /absolute/published-ct2 --threads 6 --serve
```

In `--serve` mode, send one JSON request per line with an absolute `input`
path and optional nonnegative integer `index`. Each line returns a JSON
`{text, language, segments}` result, including the index when supplied.
An indexed error response stops the worker. For a single clip, omit `--serve`
and add `--input /absolute/clip.wav --output /absolute/result.json`. Inputs
must be 16 kHz mono WAV files. Stdout in serve mode contains protocol
responses only.

Decoding forces German transcription, beam 3, best-of 1, Silero VAD,
no previous-text conditioning and the standard faster-whisper temperature
fallback `(0, .2, .4, .6, .8, 1)`, compression-ratio threshold 2.4,
log-probability threshold -1.0 and no-speech threshold 0.6. A stochastic
fallback can produce different text between runs.

Segments retain individual decoder onsets and valid ends. A start up to
40 ms before zero is clipped to zero. An overlapping segment's end can
extend into Whisper's padded 30-second final decoding window; up to
30.5 seconds of end overshoot is clipped to the actual WAV boundary.
This also handles a measured 0.14–30.12-second interval on a 2.4-second
final clip without inventing a whole-window onset or discarding text.
Every change emits a structured `asr_timestamp_adjustment` event to stderr
with request index, raw bounds, WAV duration and output bounds. A clipped
end is the recording boundary, **not** a verified natural utterance end.
Invalid, entirely outside or more extreme intervals fail the request.
Neither clipping nor decoder timing establishes transcription accuracy.
