# Swiss German Whisper Turbo CPU experiment

This is an opt-in benchmark, not an installed production adapter. The model is
`gcoli/whisper-large-v3-turbo-swiss-german-mit`, pinned to
`540fb2e7a9135048963c5de9822579c96f8962e5`. It is not Flix. Use the publisher's
`ct2/` assets from that exact snapshot and retain its license. The model binary
used in the experiment has SHA-256
`baecb116482ebb80a434d4a71ab50d6642808b258b5ce0f82cfde571aca3dfa2`.
An independent numerical audit of the publisher's conversion has not been
performed.

## Repeatability profile

`turbo_ct2_worker.py` starts three fresh CPU model processes, six threads each,
requesting INT8 (the runtime reports `int8_float32`). Each request has a unique
integer `index`; responses preserve it and may arrive out of order.

The profile follows the earlier successful trial: forced German, transcribe,
beam 3, best-of 1, Silero VAD, no previous-text conditioning, no word timestamps.
It retains faster-whisper's standard fallback temperatures
`[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`, compression-ratio threshold 2.4,
log-probability threshold -1.0, and no-speech threshold 0.6.
Sampling seeds remain engine-controlled, matching the historical experiment;
identical text across runs is measured, not assumed.

The installed experimental environment used faster-whisper 1.2.1,
CTranslate2 4.8.2, ONNX Runtime 1.30.0 for VAD, NumPy 2.4.6, and tokenizers
0.23.2. Models are loaded offline once per child process; there is no download,
conversion, precision fallback or disk-offload path in this worker.

```bash
python scripts/asr_engine_study.py \
  --manifest /absolute/frozen-corpus.json \
  --output /absolute/new-full-run \
  --full-corpus --source-description /absolute/source.json \
  --estimated-peak-mib 9216 --max-in-flight 3 --worker-command \
  /absolute/ct2-env/bin/python -u \
  /absolute/shum/adapters/asr/turbo_ct2_worker.py \
  --model /absolute/pinned-turbo/ct2 --threads 6 --workers 3 \
  --diagnostics /absolute/new-full-run/diagnostics
```

The source description must hash model/config/tokenizer assets, the actual
Python/native libraries, VAD graph, worker and shared helpers. Use a new
description if code or dependencies change; never rewrite historical evidence.
Model/corpus hashing is preparation and excluded from the separately recorded
600-second inference deadline. Loading all three processes and their shutdown
are included. Every one of 67 frozen windows must finish strictly before
600 seconds in each of three fresh runs.

Wrap a measurement in the existing user-systemd resource policy when available:
24 GiB `MemoryMax`, zero `MemorySwapMax`, and CPU quota 1800% for this profile.
The runner also retains its 4 GiB effective headroom reserve, own-process swap
growth stop and memory PSI plus new global swap-out stop. Verify active scope
properties and actual responses after startup; do not interpret a disappeared
scope's default values as its former limits.

## Timing and quality diagnostics

Returned segment timestamps cover the **whole input window**, not individual
utterances. Raw faster-whisper segment timing is retained in private diagnostic
files together with out-of-bounds flags. An initial protocol attempt stopped
because window 8's raw end was 28.34 seconds on a 28-second WAV. Whole-window
output avoids advertising those raw predictions as valid clip-relative timing;
it does not remove text, alter decoding or suppress the diagnostic defect.
Do not promote this bridge as a production segment-timing adapter.

Diagnostics record effective compute type, package versions, actual fallback
options, chosen segment temperatures, tokens, no-speech probability and VAD
retained duration. The library does not expose exact fallback-attempt history
or a definitive truncation status; neither is silently declared successful.
Per-request diagnostic numbers are not corpus indices; use the runner's mapping.
Keep token/content diagnostics private.

Compare all full text hypotheses, empty-window sets, repeated-four-gram flags,
fallback temperatures, raw-timestamp defects, loading and peak aggregate RSS.
No repetition flag or agreement with primeLine establishes WER or accuracy.
Application tests, builds and production installation remain deferred during
this experiment round.

## Three fresh full runs

| Run | Time including loading/cleanup | Completed | Aggregate peak RSS |
| --- | --- | --- | --- |
| 1 | 329.682 s | 67/67 | 6240 MiB |
| 2 | 344.623 s | 67/67 | 6239 MiB |
| 3 | 362.082 s | 67/67 | 6239 MiB |

All three runs passed the strict speed gate and produced the same empty-window
set: 0, 32, 40, 44, 45. None had repeated-four-gram fractions above 0.5 or
segment token counts at least the decoder limit. These are limited proxies,
not definitive truncation/accuracy acceptance.

**Text is not identical across runs:** 23 windows in run 2 and 19 in run 3
differ from run 1. Diagnostics show fallback selecting nonzero temperatures;
sampling is enabled by that profile, so do not promise deterministic text.
The exact cause of each difference has not been independently established.
Raw timing exceeded input bounds in 16, 14 and 16 windows respectively and
remains flagged rather than promoted as valid utterance timing.

Thus speed repeatability is confirmed, while transcription accuracy and
segment timing are still unverified. No production defaults were changed.
