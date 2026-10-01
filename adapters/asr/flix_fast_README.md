# `flix_fast_*` — CTranslate2/faster-whisper speed prototype for Flix SwissGerman

**Status: converted and benchmarked on real audio, but not production-ready.**
This is a standalone, not-yet-adopted prototype investigating a faster
CPU-only inference path for the *original pinned* Flix SwissGerman Whisper
checkpoint (`Flix-AI/flix-swissgerman-full`, see the pinned
`<storage-dir>/models/flix/<revision>/.shum-manifest.json`), aiming for 30 minutes
of audio in under 10 minutes on this machine (Intel i7-12850HX, 24 logical
CPUs, CPU-only, no GPU). Conversion does not modify the original HF model
snapshot or any existing job artifact.

The pinned Flix weights were converted to an isolated CT2/int8 cache and
tested on a 30-minute prefix of a real recording. With experimental
`--fast-decode`, three workers with six threads each completed only 62 of
67 windows by the strict 600-second deadline; four workers with four
threads each also hit the deadline. Adding `--vad-filter` to the three
workers completed all 67 windows in 573.5 seconds (RTF 0.319), but window
52 repeated a four-gram 0.57 of the time. With the same VAD and a beam
width of 2, all windows took 461.0 seconds: window 52 improved, but
window 8 repeated (0.58). Beam 3 took 489.2 seconds and still repeated
window 8. Single-worker and three-worker short replays of that **same**
WAV do not reproduce the full-run repetition. Thus the cause is unknown;
short-window fixes cannot certify the complete run. Without VAD the first,
mostly silent window also produced a repeated hallucination. Do not
switch production to this adapter without an audio-grounded accuracy
check and a safe response to suspect windows. See the
[complete engine comparison](../../README.md#cpu-engine-comparison-on-one-recording).

## Why CTranslate2/faster-whisper (and not whisper.cpp)

- CTranslate2's `TransformersConverter` has a generically-registered
  `WhisperConfig` model loader (see
  `ctranslate2/converters/transformers.py`, `WhisperLoader`) that copies
  weights straight from any Hugging Face `WhisperForConditionalGeneration`
  checkpoint — including fine-tunes, and including this checkpoint's
  128-mel-bin (`large-v3`-style) encoder — with no code changes required.
  This was confirmed by reading the installed library source, not just
  documentation.
- `faster-whisper` wraps that CTranslate2 model with the standard Whisper
  decoding loop (chunking, language/task forcing, segment timestamps),
  giving a result shape that maps directly onto `main.py`'s
  `{text, language, segments}` contract.
- `whisper.cpp` (GGML) was considered per the task brief ("optionally"):
  its hand-tuned CPU SIMD kernels can sometimes beat CTranslate2 on
  CPU-only hardware, and it does have official large-v3 (128-mel) support.
  However, converting an arbitrary HF fine-tune to GGUF is a less
  officially supported path than `ct2-transformers-converter`/
  `TransformersConverter`, and it would need its own separate quantization
  and tokenizer/vocab-compatibility validation for this exact checkpoint.
  Given "choose supported conversion", CTranslate2 was chosen as the one
  actually prototyped here; whisper.cpp remains a documented option for a
  possible follow-up investigation, not implemented.

## Files

| File | Purpose |
| --- | --- |
| [`flix_fast_common.py`](/home/d/ya/Docs/01.SYY/50.Code/shum/adapters/asr/flix_fast_common.py) | Shared, dependency-light helpers (no `torch`/`ctranslate2` import): model-dir validation, manifest/provenance reading, cache-path resolution, WAV validation, atomic JSON writing. Used by both scripts below so their idempotency/cache logic can never disagree. |
| [`flix_fast_convert.py`](/home/d/ya/Docs/01.SYY/50.Code/shum/adapters/asr/flix_fast_convert.py) | CLI: converts an original pinned HF Whisper checkpoint directory to a quantized CTranslate2 model directory, writes `preprocessor_config.json` + `flix-fast-provenance.json`. Idempotent (skips re-conversion when a matching provenance record already exists). |
| [`flix_fast_worker.py`](/home/d/ya/Docs/01.SYY/50.Code/shum/adapters/asr/flix_fast_worker.py) | CLI-compatible worker: same `--model`/`--input`/`--output`/`--threads`/`--serve` contract and JSON output schema as `main.py --engine whisper`, backed by `faster-whisper`. Requires a pre-converted CTranslate2 directory (does **not** auto-convert, to avoid silently loading the full model into RAM on worker startup). |
| [`flix_fast_requirements.txt`](/home/d/ya/Docs/01.SYY/50.Code/shum/adapters/asr/flix_fast_requirements.txt) | Pinned deps for an isolated `.venv-flix-fast` (not added to `pyproject.toml`, which is parent-owned). |
| [`tests/test_flix_fast_common.py`](/home/d/ya/Docs/01.SYY/50.Code/shum/adapters/asr/tests/test_flix_fast_common.py), [`tests/test_flix_fast_convert.py`](/home/d/ya/Docs/01.SYY/50.Code/shum/adapters/asr/tests/test_flix_fast_convert.py), [`tests/test_flix_fast_worker.py`](/home/d/ya/Docs/01.SYY/50.Code/shum/adapters/asr/tests/test_flix_fast_worker.py) | Pure-Python unit tests (no `torch`/`ctranslate2`/`faster_whisper` required to run them — heavy calls are injected via factory parameters, same pattern as `tests/test_main.py`). |

## Setup

```bash
cd adapters/asr
uv venv --python 3.11 .venv-flix-fast
uv pip install --python .venv-flix-fast/bin/python \
    -r flix_fast_requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

This was done once in this investigation (installed cleanly:
`ctranslate2==4.8.2`, `faster-whisper==1.2.1`, `torch==2.14.0+cpu`,
`transformers==5.17.0`) purely to read library source and confirm API
shapes; no model weights were loaded.

## Conversion

```bash
.venv-flix-fast/bin/python flix_fast_convert.py \
    --model /absolute/models/flix/<pinned-revision> \
    --quantization int8
```

Writes to `~/.cache/shum/flix-fast-ct2/<revision>/int8/` by default
(override with `--output` or `--cache-root`/`SHUM_FLIX_FAST_CACHE`) —
deliberately outside this repository and outside the original model
snapshot, so it is never mistaken for, and can never modify, the pinned
HF checkpoint. `--force` re-converts even if a matching conversion already
exists.

**Known compatibility fix baked into the converter:** this checkpoint's
`transformers_version: 5.0.0` means its feature-extractor settings live in
a unified `processor_config.json` with a *nested* `"feature_extractor"`
object (`{"feature_extractor": {"feature_size": 128, ...}}`), not the flat
`preprocessor_config.json` faster-whisper expects. Copying the file
verbatim would silently leave faster-whisper on its 80-mel-bin default and
break every transcription with a mel-bin shape mismatch against this
checkpoint's 128 mel bins (`config.json`: `num_mel_bins: 128`, matching
`large-v3`'s feature extractor, not the 80-bin earlier Whisper sizes).
`flix_fast_convert.py` reads whichever file is present, flattens it, and
writes a compatible `preprocessor_config.json` — see
`flix_fast_common.flatten_feature_extractor_config`.

**Provenance:** every conversion also writes `flix-fast-provenance.json`
into the output directory (source model path + identity/revision, the
`generation_config.json`-derived default language/task, quantization,
whether `low_cpu_mem_usage` was used, and the installed `ctranslate2`
version). `flix_fast_worker.py` reads this back and refuses to serve a
CTranslate2 directory whose recorded `source_identity` does not match the
`--model` directory it was given, so a stale/mismatched cache entry fails
loudly instead of silently transcribing with the wrong weights/vocabulary.

## Serving

```bash
.venv-flix-fast/bin/python flix_fast_worker.py \
    --model /absolute/models/flix/<pinned-revision> \
    --threads N --serve
```

Same wire protocol as `main.py`'s `--serve` mode: newline-delimited
`{"input": "/abs/clip.wav"}` requests on stdin, one
`{"text", "language", "segments"}` (or `{"error": ...}`) JSON line per
request on stdout, stderr for diagnostics, model loaded exactly once. By
default it uses `language`/`task` from the checkpoint's own
`generation_config.json` (this checkpoint: `language="de"`,
`task="transcribe"`) and `beam_size=1`/`best_of=1` (greedy, matching
`main.py`'s whisper engine, which does not set `num_beams` and so also
runs greedy) — override with `--language`/`--task`/`--beam-size`/
`--best-of`. `--vad-filter` defaults to off, for parity with `main.py`'s
whisper engine (which does not filter silence).

## Feasibility analysis (target: 30 min audio in <10 min, i.e. RTF < 0.33)

- **Baseline:** the Transformers adapter spent about 185 minutes on the
  35-minute recording. This is not an acceptable production speed.
- **Converted checkpoint:** one CT2/int8 worker at eight threads processed
  about 30 of 67 windows before its run was stopped; ordinary decoding
  with three workers at six threads was also stopped early. The strict
  `--fast-decode` 30-minute benchmarks ran to their deadline: three
  workers at six threads completed 62/67 windows, and four workers at
  four threads also failed. Four CT2 workers occupied roughly 15 CPU
  cores with about 2.3 GiB RSS each. More workers did not guarantee more
  throughput on this CPU. Three six-thread workers with `--fast-decode
  --vad-filter` then passed in 573.5 seconds, including model load and
  every window. Preparation remained outside the measured model time.
- **Quality:** a low-speech window had 335 output characters and a
  repeated-fourgram fraction above 0.96 in fast mode, whereas two other
  ASR engines returned empty output. VAD removed that case but window 52
  still repeated phrases (fourgram fraction 0.57 versus 0 in the two
  earlier outputs). On that window a normal CT2 decode without
  `--fast-decode` returned 146 characters versus 350/341 in the earlier
  Flix/primeLine hypotheses; which words are correct needs an audio check.
  `--fast-decode` and `--vad-filter` remain opt-in.

**Initial result:** one CT2/int8-with-VAD run passed the measured speed
target on this recording, but had a small margin and an unresolved
quality outlier. It remains experimental, not the default Flix adapter.

**Later repeatability check:** the original Flix snapshot is BF16. This
CPU lacks AVX-512 BF16 and AMX BF16 instructions; a separate,
provenance-checked CTranslate2 FP32 conversion was also tried without
quantization. With a hard 600-second cutoff, the original BF16
Transformers path finished 4/67 windows, the default FP32 Transformers
path 5/67 in a 571-second probe, and a single CT2 FP32 process 20/67.
Three later, identical CT2/int8 runs with `--fast-decode --vad-filter
--beam-size 2`, three six-thread workers and the same 67 WAV windows
finished **61/67, 62/67 and 65/67** by the deadline. All three
produced a strong repeated phrase in window 23, and VAD marked five
windows empty, some with short, unverified diarization turns. Therefore
the earlier 461-second pass is **not reproducible** in these new trials;
speed and preservation of quiet speech still need to be demonstrated.
One further full trial with normal CT2 decoding (allowing its fallback
on unstable outputs) removed the window-23 text loop but finished only
59/67 windows by 600 seconds, with about 8.6 GiB combined peak RSS.
It left the same five VAD-empty windows; this is not yet a viable
alternative.
See the full [same-checkpoint comparison](../../README.md#same-checkpoint-flix-cpu-precision-trials).

## Still to verify

- Repeat the passing speed configuration under realistic competing workloads
  and wire same-model parallel workers into the production scheduler; a
  benchmark-only multiworker pass does not accelerate `transcribe` yet.
- Audio-grounded accuracy, especially silence, dialect and names. Textual
  comparison with a previous run cannot determine which hypothesis is right.
- Production adoption only after both requirements pass. The worker already
  uses the Rust CLI's adapter contract and can be selected in the config;
  it has deliberately not replaced the default Flix adapter.
