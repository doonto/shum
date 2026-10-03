# Swiss German Turbo ONNX comparison (opt-in, not an accepted engine)

This experiment uses the public `gcoli/whisper-large-v3-turbo-swiss-german-mit`
revision `540fb2e7a9135048963c5de9822579c96f8962e5`. The separate
`/home/d/.cache/shum/turbo-onnx-source/<revision>/` directory holds the
original Hugging Face FP16 checkpoint, tokenizer, model card (MIT license)
and `.hf-api.json` from the pinned Hub revision. The already published CT2
`model.bin` is **not** an export source. `turbo_onnx_export.py` checks sizes,
the Hub's LFS SHA-256 and non-LFS Git blob SHA-1 before export. Provenance
also records local SHA-256 for each original artifact.

The exporter and worker need an **isolated** Python environment with
`optimum-onnx`, `onnx`, `onnxruntime`, `transformers` and CPU PyTorch.
Never install these into the production or existing Whisper environments.
The worker is not wired into `main.py`; it speaks the frozen-study indexed
line protocol (`{"index":0,"input":"/absolute/clip.wav"}`). Exports and
inference are deliberately **not** performed during the concurrent CT2
measurements.

The successful **structure-only + streamed initializer** export is at:

```sh
SOURCE=/home/d/.cache/shum/turbo-onnx-source/540fb2e7a9135048963c5de9822579c96f8962e5
GRAPHS=/home/d/.cache/shum/turbo-onnx-graphs/540fb2e7a9135048963c5de9822579c96f8962e5/fp32-cached-stream-v1
PY=/home/d/.cache/shum/turbo-onnx-runtime/bin/python
```

[`turbo_onnx_stream_export.py`](../../scripts/turbo_onnx_stream_export.py)
uses the public Optimum cached Whisper wrappers and parameterless PyTorch
ONNX export. A separate process verifies every graph weight input against
the pinned safetensors name, shape, and FP16 source dtype, then widens one
tensor at a time and writes FP32 external data with `fdatasync`. It attaches
ONNX initializers only after validating all mappings. The tied LM head is
explicitly mapped to the source decoder embedding. The with-past graph
omits the unused `encoder_hidden_states` input because its encoder KV
cache is already supplied; all other runtime inputs match Optimum's
declared names. No model graph has yet passed an ORT loading or audio
pilot: **export completion is not runtime acceptance**.

The three completed stages ran in separate 24 GiB zero-swap scopes with
CPUQuota 1600%, a monitored 9 GiB stage-growth budget, effective 4 GiB
reserve, PSI/swap/OOM stops, and a 15-minute deadline per stage. The
encoder took 65.907 seconds (structure 56.075, attachment 6.510);
decoder took 17.398 seconds and cached decoder finished in its corrected
second attempt. Stage and phase JSON reports are in this session's private
`files/turbo-onnx-stream-*.json` and `files/turbo-stream-cached-retry/`.
The export provenance is `"$GRAPHS/.shum-turbo-onnx.json"` and contains
SHA-256 of each graph, external data file, processor artifact, and the
original pinned source. To reproduce in a **fresh** directory:

```sh
REPORTS=/absolute/private/existing/report-directory
GRAPHS=/absolute/new/unused/fp32-cached-stream
"$PY" scripts/turbo_onnx_stream_export.py \
  --source "$SOURCE" --output "$GRAPHS" --report-dir "$REPORTS" --prepare
systemd-run --user --scope --collect -p MemoryMax=24G -p MemorySwapMax=0 -p CPUQuota=1600% \
  "$PY" /home/d/Code/shum/scripts/turbo_onnx_stream_export.py \
  --source "$SOURCE" --output "$GRAPHS" --report-dir "$REPORTS" --stage encoder
systemd-run --user --scope --collect -p MemoryMax=24G -p MemorySwapMax=0 -p CPUQuota=1600% \
  "$PY" /home/d/Code/shum/scripts/turbo_onnx_stream_export.py \
  --source "$SOURCE" --output "$GRAPHS" --report-dir "$REPORTS" --stage decoder
systemd-run --user --scope --collect -p MemoryMax=24G -p MemorySwapMax=0 -p CPUQuota=1600% \
  "$PY" /home/d/Code/shum/scripts/turbo_onnx_stream_export.py \
  --source "$SOURCE" --output "$GRAPHS" --report-dir "$REPORTS" --stage decoder-past
"$PY" scripts/turbo_onnx_stream_export.py \
  --source "$SOURCE" --output "$GRAPHS" --report-dir "$REPORTS" --finalize
```

The scripts refuse to start a stage without cgroup v2 `MemoryMax <=24 GiB` and
`MemorySwapMax=0`. It checks at least 13 GiB effective available memory
before each stage, taking the minimum of host `MemAvailable` and the
headroom at **every cgroup ancestor**, not the scope's 9 GiB stage budget.
It enforces a 15-minute stage deadline and aborts if the scope's usage
grows by more than 9 GiB, or if the effective 4 GiB reserve is lost.
It also watches cgroup swap/OOM and system swap-out with full
memory PSI during the run. If any gate fails, **stop**; do not retry without
a new resource assessment. FP32 is needed for portable CPU execution from
the source's FP16 weights. If the installed Optimum/Transformers combination
rejects the source, produces a different graph layout, hits a memory cap,
or produces an ONNX graph that ORT cannot load, stop rather than substitute
another model. Discard an incomplete output directory before retrying.
Do not proceed to the next stage unless its previous command exits zero
and the corresponding `.stage-encoder.json`, `.stage-decoder.json`, or
`.stage-decoder-past.json`
exists. Each stage refuses to replace an existing graph. Complete provenance
records source and graph SHA-256; it is not a numerical ONNX/PyTorch
parity audit. A separately supervised single-clip pilot is required.

Two supervised encoder attempts on October 1, 2026 stopped under the
unchanged swap/PSI guard, despite isolated selective loading. The first
stopped in 7.88 seconds and the second, after pressure briefly cleared,
in 52.56 seconds. The controller observed host `pswpout` growth with
full-memory PSI `1.54%` and `1.44%`, respectively. Timed process peak
RSS was about 3.89 GiB, worker cgroup swap remained zero, and available
memory stayed above 4 GiB. A single fresh attempt with the corrected
24 GiB enclosing scope and tighter 9 GiB stage growth budget stopped
after 53.684 seconds: full-memory PSI reached `4.79%` with new host
swap-out, despite zero scope swap, 5.317 GiB maximum scope growth and a
13 GiB last-sample effective available value. Its incomplete graph was
removed. Reports for all three failed in-memory serialization attempts
are in this session's private `files/turbo-onnx-encoder-attempt-{1,2,3}.json`.
Do not repeat that unchanged serialization path under those host conditions.

After a successful export, pass this **worker command directly** to the
trusted study runner inside its existing 24 GiB zero-swap scope. Do not
nest a `systemd-run` wrapper inside the runner's owned process group.
Start with one known-good frozen clip before any full corpus run:

```sh
"$PY" /home/d/Code/shum/adapters/asr/turbo_onnx_worker.py \
  --source "$SOURCE" --graphs "$GRAPHS" --threads 16
```

The worker requires absolute, mono 16-kHz PCM16 WAV paths of at most
30 seconds, checks the export and source hashes, verifies its <=24 GiB
zero-swap enclosing scope and at least 13 GiB effective available
(9 GiB process-tree RSS budget plus 4 GiB reserve) before loading.
The headroom check includes the current scope and every ancestor.
A local monitor stops the worker if its process-tree RSS exceeds 9 GiB,
if swap or full-memory PSI with swap-out appears, or if effective
available memory drops below 4 GiB; the trusted runner independently
enforces its own pressure, timeout, and owned-group cleanup guards.
The worker requires the
separate encoder, decoder and decoder-with-past graphs, with no merged
decoder. It returns an error
for missing EOS or excessive duration and uses Optimum's
Whisper `generate` and the source processor with German transcription.
The historical default `--decode-profile beam-only` uses beam 3,
no sampling, no VAD, cached decoding, and a 400-new-token cap. A single
whole-clip interval in `segments` is **not** a word or speech timestamp.
The first guarded pilot stopped after 6.829 seconds at about 420 MiB RSS,
before graph loading: Optimum's `from_pretrained` file finder reported
`Could not find any ONNX files with target file name
encoder/encoder_model.onnx`. It compares basenames and does not accept the
stage subdirectory in `encoder_file_name`. That result remains a measured
failure under the original worker manifest. The worker now opens the three
graphs at their exact verified paths with ONNX Runtime and passes those
sessions, plus local Whisper and generation configs, to Optimum's
Whisper-specific public constructor. It does not copy or re-export graphs.
The v2 guarded pilot loaded all three graphs and returned 4/4
responses, but took 455.521 seconds for windows 23, 8, 52, 45,
peaking at 7,202 MiB RSS without a resource stop. It used three
independent 16-thread ORT sessions plus Torch's 16-thread pool.
The worker keeps ORT intra-op at the requested `--threads 16`
for matmuls, disables intra/inter-op idle spinning using public ORT
session settings, and limits Torch generation/beam bookkeeping to
one intra-op and one inter-op thread. Diagnostics record those values,
the attention-mask source, and valid mask frames. The mask is obtained
from the original 16-kHz processor with `return_attention_mask=True`
and passed to Whisper's supported `generate(attention_mask=...)`;
it is not fabricated. The historical default has no CT2 temperature
fallback or VAD.
The v3 control pilot improved window 8 to 32.233 seconds and window 45
to 7.461 seconds, but its full attempt stopped after 232.546 seconds:
3/67 windows completed before window 3 exhausted the 400-new-token cap;
window 1 had repetition 0.8645. The v3 worker used
`ORT_DISABLE_ALL` and `session.disable_prepacking=1`, so its latency
is not a verdict on normally optimized ORT. The subsequent worker
uses `ORT_ENABLE_ALL` and default enabled prepacking. This can increase
startup memory; the existing 9 GiB worker RSS budget, effective 4 GiB
reserve, 24 GiB enclosing scope and swap/PSI stops remain unchanged.
The v4 `ORT_ENABLE_ALL`/prepacking full attempt ended in 64.237 seconds
with 3/67 windows completed: window 3 failed the unchanged 400-token
cap without EOS; window 1 again had repetition 0.8645, with completed
text identical to v3. That is a decoding failure, not a speed timeout.
Optimizing ORT reduced time to that failure from 232.546 to 64.237
seconds. Neither run establishes a final ORT speed verdict.
The v5 `ct2-fallback-vad` control pilot completed 3/3: window 3 took
51.367 seconds, window 23 took 11.405 seconds, neither had high
repetition, and window 45 was an explicit VAD skip in 0.037 seconds.
The native fallback profile repaired the pilot's beam-only decoding
failure, but the full 67-window v5 run **timed out** at
600.283724943 seconds including cleanup. It completed indices 0–28
(29/67), leaving 29–66 pending. Peak process-tree RSS was
6,482,888 KiB (about 6,331 MiB); worker exit code was -15, and
all owned Turbo scopes are inactive. No completed
selected result failed EOS or the 400-token cap. Completed window 20
was the sole 4-gram-repetition fraction above 0.5 (0.557692);
completed window 0 was empty. This is a measured full-run timeout,
not an extrapolation from the pilot or a decoding-cap failure.

For the current 600-second gate, reject **this FP32 native-fallback/VAD
profile**: it did not finish 67/67. Two unchanged repeats cannot
repair the observed timeout. The CT2 INT8 beam-3 standard-fallback
comparison did finish 67/67 in each of its three fresh runs
(329.682, 344.623, and 362.082 seconds). Decoder INT8 quantization
stopped at guarded preprocessing; no ONNX INT8 runtime was measured.
This result does **not** reject every ONNX Runtime configuration.

### Opt-in CT2-like timestamp/fallback/VAD decode

The separate `--decode-profile ct2-fallback-vad` uses public
Transformers 4.57.6 `Whisper.generate`: timestamps enabled, German
transcription, first-pass beam 3, temperatures
`[0, .2, .4, .6, .8, 1]`, compression ratio `2.4`, log probability
`-1.0`, no-speech `0.6`, and no previous-token conditioning.
The native fallback switches sampling retries to one beam. Hugging
Face has no `best_of` keyword; its one returned sampled sequence
approximates CT2's `best_of=1`, but the two decoders' thresholds and
segment decisions are not numerically identical. Faster-whisper 1.2.1
computes compression ratio as the UTF-8 byte length of
`tokenizer.decode(tokens).strip()` divided by the zlib-compressed
length of those **text bytes**; its tokenizer drops EOS, timestamps,
and other tokens at or above `eot` when decoding. Transformers 4.57.6
encodes the **generated token IDs** as fixed-width little-endian bytes
(two bytes per token for this vocabulary), including timestamp and
EOS tokens present at its fallback check, then divides that byte
length by its zlib-compressed length. The common `2.4` threshold
therefore tests different data and can choose different retries.
Average-log-probability calculation and no-speech handling can also
differ between libraries. Passing timestamp
arguments on short-form inputs is supported by the installed public
Whisper generation path; it can make multiple calls for a single
window. `force_unique_generate_call=False` preserves those calls.

Supply the existing faster-whisper 1.2.1 site-packages directory
explicitly. The worker appends it to `sys.path` (without replacing
its own Torch, Transformers or tokenizers), checks the imported
package version/location, then calls the public
`get_speech_timestamps` and `collect_chunks` with the exact
faster-whisper transcription defaults: 30-second max speech,
160-ms minimum silence, and the library's remaining `VadOptions`
defaults. It records speech intervals, chunk metadata, retained
duration, and raw native timestamp segments relative to collected
VAD chunks. An empty Silero result is a valid explicit VAD skip,
not fabricated ASR EOS. Output segment bounds remain the whole
original WAV duration.

The worker verifies the three-token timestamp prompt
`<|startoftranscript|><|de|><|transcribe|>` on each **exposed selected**
generation result, and rejects a selected result without EOS or
one consuming all 400 new tokens (even if EOS appears at the cap).
The public `return_segments` output retains selected results; it
does **not** expose discarded temperature-retry sequences, their
cap/EOS state, or the selected temperature. Diagnostics explicitly
mark this limit: EOS/cap checks cover each exposed selected result,
not every hidden fallback attempt. A forced EOS at the cap on a
selected result is rejected. VAD-empty and native no-speech skips have no ASR
EOS claim. This is closer to CT2, not exact CT2 decoding parity.

Run only through the trusted 24 GiB zero-swap runner, first on guarded
pilot windows; do not interpret the historical beam-only failures as
this profile's result:

```sh
/home/d/.cache/shum/turbo-onnx-runtime/bin/python \
  /home/d/Code/shum/adapters/asr/turbo_onnx_worker.py \
  --source /home/d/.cache/shum/turbo-onnx-source/540fb2e7a9135048963c5de9822579c96f8962e5 \
  --graphs /home/d/.cache/shum/turbo-onnx-graphs/540fb2e7a9135048963c5de9822579c96f8962e5/fp32-cached-stream-v1 \
  --decode-profile ct2-fallback-vad \
  --faster-whisper-site /home/d/Code/shum/adapters/asr/.venv-flix-fast/lib/python3.11/site-packages \
  --threads 16
```

### Optional dynamic INT8

The encoder's FP32 external data is about 2.4 GiB; ORT's public dynamic
quantizer eagerly loads it, so this experiment does **not** quantize
the encoder. The decoders' original external data are 0.640 and
0.592 GiB. A metadata-only inspection found 57 and 49 MatMul nodes,
with 41 and 33 weight inputs supplied through constant `Transpose`
nodes and zero direct initializer B inputs. The optional
[`turbo_onnx_quantize.py`](../../scripts/turbo_onnx_quantize.py)
first saves an `ORT_ENABLE_BASIC` optimized graph through the public
`InferenceSession` API, then runs `quant_pre_process` for ONNX shape
inference and checks for direct constant MatMul weights. It invokes
public `quantize_dynamic` with `MatMulConstBOnly=True`, QInt8 weights
and external output data. Zero converted `MatMulInteger` nodes or
INT8 initializers is an explicit failure. No graph surgery is used.
The result, if all stages finish, is a **mixed FP32 encoder / dynamic
INT8 decoder** pilot, not full INT8 or CT2 INT8 parity. None has
finished. The first guarded preprocessing attempt failed when
`quant_pre_process` swallowed an ORT optimization error and then tried
to open its missing temporary optimized graph (scope growth 4.73 GiB,
no pressure stop). A direct public ORT basic-optimization alternative
saved its graph, but the output referenced external initializer data
at the wrong relative path (scope growth 1.85 GiB). Supplying ORT's
documented `session.optimized_model_external_initializers_file_name`
created that external file, but the third attempt stopped under the
unchanged host swap-out/full-memory PSI guard after 2.618 seconds
(PSI 1.08%, scope growth 1.85 GiB, no cgroup swap or OOM). Reports
remain in the private `files/turbo-onnx-int8-v{1,2,3}/` directories;
none of their partial graphs is usable. **Do not repeat the third
attempt unchanged** or claim a mixed INT8 engine. Reassess host
pressure and the allocation profile before a fresh attempt.

The following is the guarded stage layout for a future, materially
reassessed attempt, **not a command to launch under the present stop**.
Run only when no parent inference is active, in a fresh directory.
Each stage starts in its own 24 GiB zero-swap cgroup with at most
1600% CPU quota. It uses the existing 9 GiB scope-growth budget,
13 GiB effective admission, 4 GiB effective reserve, swap/PSI/OOM
stops and 900-second deadline. The source FP32 graphs are read-only.

```sh
PY=/home/d/.cache/shum/turbo-onnx-runtime/bin/python
SOURCE=/home/d/.cache/shum/turbo-onnx-source/540fb2e7a9135048963c5de9822579c96f8962e5
GRAPHS=/home/d/.cache/shum/turbo-onnx-graphs/540fb2e7a9135048963c5de9822579c96f8962e5/fp32-cached-stream-v1
MIXED=/absolute/fresh/private/mixed-decoder-int8-assessed
REPORTS=/absolute/fresh/private/int8-stage-reports
Q=/home/d/Code/shum/scripts/turbo_onnx_quantize.py
"$PY" "$Q" --graphs "$GRAPHS" --output "$MIXED" --report-dir "$REPORTS" --prepare
for STAGE in decoder-preprocess decoder-quantize decoder-past-preprocess decoder-past-quantize; do
  systemd-run --user --scope --collect -p MemoryMax=24G -p MemorySwapMax=0 \
    -p CPUQuota=1600% "$PY" "$Q" --graphs "$GRAPHS" --output "$MIXED" \
    --report-dir "$REPORTS" --stage "$STAGE" || exit 1
done
# Finalize only if every guarded stage exited successfully.
"$PY" "$Q" --graphs "$GRAPHS" --output "$MIXED" --report-dir "$REPORTS" --finalize
```

Each phase writes a validation JSON beside its output and a separate
guard report in `REPORTS`; the final mixed manifest includes input
and output SHA-256, quantized operator coverage and pinned source
identity. Stop on any admission failure, pressure stop, missing direct
constant MatMul weight or missing converted INT8 operator; never
repeat an unchanged failed stage. Only after successful finalization
can the worker select the profile:

```sh
"$PY" /home/d/Code/shum/adapters/asr/turbo_onnx_worker.py \
  --source "$SOURCE" --graphs "$GRAPHS" --decoder-graphs "$MIXED" --threads 16
```

The worker verifies the base FP32 manifest and the separate decoder
manifest; it loads the original FP32 encoder and the two quantized
decoders. It never replaces the frozen FP32 graphs.
The CT2 beam-3 standard-fallback study uses faster-whisper with `best_of=1`,
temperatures `[0, .2, .4, .6, .8, 1]`, compression-ratio threshold `2.4`,
log-probability threshold `-1.0`, no-speech threshold `0.6`,
`condition_on_previous_text=False`, and Silero VAD enabled. The ONNX
beam-only default uses none of those retries, silence filtering, or
timestamp decoding. The opt-in profile uses public counterparts but
does not reproduce CT2's internal decoder scores. CT2's raw segments can
even overshoot clip duration after VAD (observed 28.34 seconds against a
28.0-second input window); both comparison workers therefore report an
honest whole-input-window interval. The opt-in profile also retains
native raw segment bounds relative to collected VAD audio, which
cannot be compared directly to CT2's restored original-waveform
timestamps. The 400-token cap can
itself cause explicit failures. Diagnostics record the returned prompt,
generated-token length, cap, EOS observation, and whether the cap was
exhausted. An EOS at the cap may have been inserted by beam finalization:
the worker rejects it rather than accepting a possibly incomplete
hypothesis. Do not count these failures as valid transcriptions.
Do not label FP32 results INT8 or any pilot a production engine.
Dynamic INT8 decoder accuracy parity is not established by conversion.
