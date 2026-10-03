# `flix_fast_*` — CTranslate2/faster-whisper speed prototype for Flix SwissGerman

**Status: converted and benchmarked on real audio, but not production-ready.**
This is a standalone, not-yet-adopted prototype investigating a faster
CPU-only inference path for the *original pinned* Flix SwissGerman Whisper
checkpoint (`Flix-AI/flix-swissgerman-full`, see the pinned
`<selected-models-dir>/flix/<revision>/.shum-manifest.json`), aiming for 30 minutes
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

## Repeatability diagnostics

The second diagnostic round verified all source-model and CT2 hashes and
verified that all 67 clip PCM hashes match the corresponding intervals in
the existing recording's first 1,800 seconds. A fresh **single** six-thread
worker, INT8/VAD/beam 2 with fast decode, reproduced window 23's historical
160-character answer exactly (repeated-four-gram fraction 0.71). That
one-window pilot took 10.85 seconds including startup, with about 2.6 GiB
sampled peak RSS. This is not a 30-minute benchmark or an accuracy result.
Concurrent workers are therefore not necessary to reproduce this loop;
its underlying cause and an audio-verified correction remain unresolved.

`--diagnostics-dir /absolute/new/empty/directory` writes local `ready.json`
and numbered request sidecars without changing the ordinary text/segments
JSON. Sidecars include actual compute type, versions, thread settings,
effective decode options, segment tokens/probabilities and retained VAD
duration. The installed faster-whisper API does not expose retained VAD
intervals, the number of fallback attempts, or a definitive truncation
flag; these fields explicitly remain unavailable. Timestamp decoding is
enabled in the current CT2 profile, even though the source generation
config requests text-only output; cross-engine comparisons must account
for that difference.

The packaged [`flix_study.py`](../../scripts/flix_study.py) helper freezes
an existing corpus and runs one persistent worker with a limit no greater
than 600 seconds. It uses new private output directories, checks the
frozen model and corpus hashes, and reserves at least 4 GiB. It stops on
a reserve breach, growth in the worker process tree's own VmSwap, or new
system swap-out together with memory PSI full avg10 at least 1.0%.
System swap-in/out alone is recorded, not attributed to the worker.
An earlier, overly strict swap-out-only guard stopped a sequence before
returning a window; that report is preserved, not treated as a quality
result. Missing counters fail safely; transient fork/exec status races
are retried briefly rather than reported as zero memory.

```bash
python3 /absolute/runtime/scripts/flix_study.py freeze \
  --source-model /absolute/models/flix/PINNED_REVISION \
  --ct2-model /absolute/cache/flix/int8 \
  --audio /absolute/prior-benchmark/audio.wav \
  --clips-dir /absolute/prior-benchmark/clips \
  --chunks-json /absolute/source-project/chunks.json \
  --output /absolute/projects/new-flix-study
python3 /absolute/runtime/scripts/flix_study.py run \
  --manifest /absolute/projects/new-flix-study/corpus.json \
  --output /absolute/projects/new-flix-study/fresh-window23 \
  --indices 23 --timeout-seconds 600 \
  --worker-command /absolute/environments/flix-fast/bin/python \
  /absolute/runtime/adapters/asr/flix_fast_worker.py \
  --model /absolute/models/flix/PINNED_REVISION \
  --ct2-model /absolute/cache/flix/int8 \
  --quantization int8 --threads 6 --serve --vad-filter --fast-decode --beam-size 2
```

`freeze` intentionally accepts only the pinned Flix revision
`a9c347a904af5d31b28c537ac732e0618aed6136`, not an arbitrary model.
`review --manifest CORPUS --study-dir STUDY --output NEW_JSON` creates a
local audio-review list with time ranges and hypothesis variants. An
optional `--historical-run DIRECTORY` requires all 67 historical WAV
hashes to match; historical engine/weight attestations are explicitly
not claimed. Review output remains `unverified`, with no WER/CER until a
human reference is supplied.

`run --full-corpus` selects all 67 windows instead of `--indices`.
`full_speed_gate` is true only after all 67 finish strictly before 600
seconds, including process/model startup. This is a speed result, not
an accuracy claim; quality remains unverified.

[`flix_matrix.py`](../../scripts/flix_matrix.py) runs selected short
profiles sequentially with fresh model processes between profiles:

```bash
python3 /absolute/runtime/scripts/flix_matrix.py \
  --manifest /absolute/projects/new-flix-study/corpus.json \
  --output /absolute/projects/new-flix-study/paired \
  --python /absolute/environments/flix-fast/bin/python \
  --worker /absolute/runtime/adapters/asr/flix_fast_worker.py \
  --threads 6 --profiles fast-repeat fallback-repeat
```

The repeated 8/23/52 sequence completed three times in each profile:
120.26 seconds with fast decode (about 2.57 GiB peak RSS), and 140.94
seconds with fallback (about 2.90 GiB). Fast decode repeated window 23
every time, fraction 0.71. Fallback removed that loop but returned three
different hypotheses, using final temperatures 0.2, 0.4, and 1.0.
Windows 8 and 52 stayed identical in these repeats.

`--seed` sets the CTranslate2 seed before model construction; optional
`--reset-seed-per-request` requests another API call before each WAV,
but does not guarantee native-worker reproducibility. With seed 0 and
the reset request, three **fresh** processes on window 23 returned the
same 141-character hypothesis at temperature 0.2, in 10.55, 10.59, and
11.03 seconds. Do not infer that reseeding a persistent model clears its
sampling state. Numerical CPU effects and native RNG lifecycle have not
been causally isolated.

Library VAD defaults include an unbounded positive-infinity duration.
Diagnostics encode non-finite values as explicit strings with field
paths; that known option is not an error. Unexpected non-finite metrics
also emit a warning instead of silently becoming zero or dropping text.

On the five VAD-empty windows, both fast and fallback profiles retained
zero audio and produced empty text. Without VAD, fallback produced short
hypotheses of 10/5/19/3/12 characters for windows 0/32/40/44/45; these may
be real faint speech or hallucinations. The no-VAD six-window pilot took
69.69 seconds; human listening is needed to distinguish those cases.
The paired filtered quiet/control profiles took 25.94 and 26.84 seconds,
with identical control-window text.

A full single-worker, six-thread fallback/VAD run completed **59/67**
before the strict 600-second deadline. Windows 59–66 remained missing;
this profile fails the speed gate despite completing the short probes.
Subsequent worker/thread configurations require fresh admission using
the measured single-worker peak plus margin and the 4 GiB reserve.

The next engine candidate is ONNX Runtime/Optimum ONNX, not a new model.
The [metadata-only ONNX probe](./flix_onnx_README.md) originally found
existing graphs without pinned-source export provenance and declined
under conservative memory estimates. A subsequent bounded streaming audit
verified every graph initializer against direct BF16-to-FP32 source values:
all 1,259 source parameters are covered, including transposed weights,
without loading a full model (about 190 MiB peak audit RSS).
This verifies weights, not runtime operators or generation behavior.
The separate diagnostic pilot uses a verified user-systemd cgroup with
at most 9 GiB `MemoryMax`, zero swap and a 120-second window limit;
it is not a production adapter or a full-corpus speed result.

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
| [`flix_fast_common.py`](./flix_fast_common.py) | Shared, dependency-light helpers (no `torch`/`ctranslate2` import): model-dir validation, manifest/provenance reading, cache-path resolution, WAV validation, atomic JSON writing. Used by both scripts below so their idempotency/cache logic can never disagree. |
| [`flix_fast_convert.py`](./flix_fast_convert.py) | CLI: converts an original pinned HF Whisper checkpoint directory to a quantized CTranslate2 model directory, writes `preprocessor_config.json` + `flix-fast-provenance.json`. Idempotent (skips re-conversion when a matching provenance record already exists). |
| [`flix_fast_worker.py`](./flix_fast_worker.py) | CLI-compatible worker: same `--model`/`--input`/`--output`/`--threads`/`--serve` contract and JSON output schema as `main.py --engine whisper`, backed by `faster-whisper`. Requires a pre-converted CTranslate2 directory (does **not** auto-convert, to avoid silently loading the full model into RAM on worker startup). |
| [`flix_fast_requirements.txt`](./flix_fast_requirements.txt) | Pinned deps for an isolated environment under the selected installation root (not part of the default ASR extras). |
| [`tests/test_flix_fast_common.py`](./tests/test_flix_fast_common.py), [`tests/test_flix_fast_convert.py`](./tests/test_flix_fast_convert.py), [`tests/test_flix_fast_worker.py`](./tests/test_flix_fast_worker.py) | Pure-Python unit tests (no `torch`/`ctranslate2`/`faster_whisper` required to run them — heavy calls are injected via factory parameters, same pattern as `tests/test_main.py`). |

## Setup

This experiment is **not** installed by the normal onboarding wizard.
Use the `flix_fast_requirements.txt` shipped with the installed adapter
runtime. Set `ASR_DIR` to the parent of the installed `[asr.flix].script`
path in `config.toml`; do not install packages in the source checkout.

```bash
SHUM_ROOT=/absolute/path/to/selected/installation
ASR_DIR=/absolute/path/to/installed/runtime/adapters/asr
SHUM_MODELS=/absolute/path/to/selected/models
FLIX_REVISION=your-pinned-model-revision
uv venv --python 3.11 "$SHUM_ROOT/runtime/environments/flix-fast"
uv pip install --python "$SHUM_ROOT/runtime/environments/flix-fast/bin/python" \
    -r "$ASR_DIR/flix_fast_requirements.txt" \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

This was done once in this investigation (installed cleanly:
`ctranslate2==4.8.2`, `faster-whisper==1.2.1`, `torch==2.14.0+cpu`,
`transformers==5.17.0`) purely to read library source and confirm API
shapes; no model weights were loaded.

## Conversion

```bash
"$SHUM_ROOT/runtime/environments/flix-fast/bin/python" "$ASR_DIR/flix_fast_convert.py" \
    --model "$SHUM_MODELS/flix/$FLIX_REVISION" \
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
"$SHUM_ROOT/runtime/environments/flix-fast/bin/python" "$ASR_DIR/flix_fast_worker.py" \
    --model "$SHUM_MODELS/flix/$FLIX_REVISION" \
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
