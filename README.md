# shum

CPU-only CLI for long German, Swiss German and occasional Russian MP3 recordings. The Rust program splits the entire recording into overlapping audio windows without dropping quiet sections, runs three independent local ASR passes (concurrently when RAM allows), assigns speakers with local pyannote, and asks an authenticated LLM CLI to compare the resulting texts. The final German text uses Standard German; Russian stays Russian. Unrecoverable speech is marked `[пропуск]`. The three raw hypotheses remain in the JSON output.

**Status:** early implementation. The Rust unit tests and adapter mock tests do not establish accuracy or 1–2-hour throughput. Run a short representative recording before relying on output. An LLM that only reads ASR text cannot prove that nothing was missed in audio; inspect intervals marked as gaps.

## Install on Ubuntu

1. Install `ffmpeg`, Rust (`cargo`) and `uv` through their normal package channels. Install and authenticate an LLM CLI; the example uses `codex`. It sends **transcript text** to its configured provider, never the audio file. Never run the LLM against recordings without permission to share their text.
2. Install the three ASR engine environments and the diarization environment separately as described in [ASR installation](adapters/asr/README.md) and [diarization installation](adapters/diarization/README.md). Dependencies are isolated from the OS Python; each `[asr.<role>]` config section points to its own interpreter.
3. For pyannote, accept the conditions at [speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) and authenticate the Hugging Face CLI: `hf auth login`. This token is not copied into the repository.
4. Run `cargo build --release`, then `cargo run -- --onboarding`. Choose a writable directory for Shum data (the suggested default is `~/shum`); projects go under its `projects/` subdirectory and managed model snapshots under `models/`. Edit the generated per-user config to set adapter interpreter and CLI paths if they differ from the example.
5. `cargo run -- models update --role primeline`; repeat for `flix`, `qwen` and `diarization`. A download may take several gigabytes. The original repositories and immutable revisions are recorded in the private `config.toml`; snapshots never go in the source tree. Run `cargo run -- doctor` **after** installing the models; `doctor` treats missing model directories as errors.

`cargo run -- models check` reports available newer Hugging Face commits. `cargo run -- models update --role qwen` downloads the current revision separately and activates it only after validation. For an **already specified** immutable revision, `cargo run -- models install-pinned --role qwen` downloads that exact revision without choosing the latest. `cargo run -- transcribe recording.mp3` saves into the selected `projects/` folder under a name derived from the input; `--output my-recording` names a project there explicitly. An absolute `--output` must also be inside that folder; paths into the source tree are rejected. The command checks weights on startup, **without updating** them. If the network is down, the pinned local revision still works with a warning; without that revision, the command fails.

Onboarding also accepts `--storage-dir /path/to/data` for non-interactive use. It imports an existing `shum.toml`, moves the existing `jobs/` and `models/` directories outside the source tree without overwriting them, and saves private settings to `${XDG_CONFIG_HOME:-$HOME/.config}/shum/settings.toml`. The model/adapter config goes in `config.toml` beside the settings file. Rerun onboarding to choose another storage directory; existing projects and model snapshots move with it. A destination with conflicting files is rejected before migration. Only the storage-directory question is implemented for now; CLI, model and language choices can be added to the same setup flow later.

The scheduler reads available RAM and cgroup v2 limits before each batch. It selects up to three ASR processes using the configurable `[resources]` reserve and peak estimates in the private `config.toml`; these are **estimates, not memory limits or guarantees**. Defaults (6000/10500/13500 MiB for primeLine/Flix/Qwen, plus 4096 MiB reserved) come from short CPU runs on one machine. By default one process handles each role. To use multiple independent processes of the same model in a normal job, set `[resources.asr_worker_counts]` (for example `flix = 3`) **and** a measured per-process RAM estimate in `[resources.asr_worker_peak_mib]` (for example `flix = 2600`); the total is capped by `max_asr_workers`. The scheduler assigns disjoint windows to those processes, falls back to one when the requested replicas do not fit RAM, and retains completed windows if interrupted. This is opt-in: the faster Flix CT2 decode has not passed audio-grounded quality review. When even one estimated peak exceeds available RAM, it tries that process with a warning; Linux swap is a last resort and OOM remains possible. Use `--sequential-asr` to force at most one ASR process irrespective of the estimates. All three hypotheses are still required for the final transcript.

To opt in to verified Accelerate CPU/disk offload, set `resources.offload_dir` to a private directory outside model snapshots (for example an absolute path to `~/shum/projects/.asr-offload`). Shum creates it with mode 0700 if missing and requires at least `offload_min_free_mib` free disk space before starting an offloaded worker; the default is 8192 MiB. Once the scheduler has reduced a batch to one worker and that worker's standard RAM estimate exceeds the available budget, it starts Flix or Qwen in offload mode. After a confirmed OOM it also retries either of those workers with offload, if configured. The `resources.offload_roles` list identifies compatible adapters; remove `flix` when selecting an alternative Flix engine without Accelerate support. primeLine and pyannote do **not** support this mode; their processes rely on serial execution and available RAM/swap. Offload estimates (`resources.offload_estimated_peak_mib`, default 6500/7000 MiB) derive from short trials, **not** safe maximum RSS. Flix and Qwen used about 5.0/5.2 GiB peak RSS on one 28-second clip, versus roughly 9.0/11.9 GiB without offload; they ran about 1.8/4.9 times slower. The cache is kept outside pinned model directories, and each newly generated hypothesis records its mode and model revision. A missing or full offload volume fails explicitly without publishing an incomplete window. See [adapter measurements](adapters/asr/README.md).

On Ubuntu, set `resources.supervise_asr_with_systemd = true` to run each ASR adapter as an isolated `systemd --user` service. The CLI then distinguishes a **confirmed** `Result=oom-kill` from timeouts and other errors. After an OOM it retains completed windows, retries up to `oom_retries` times with only one worker and a short backoff, then writes `waiting-resources.json` and stops. A later invocation resumes once available RAM rises by over 512 MiB or `[resources]` changes. When a hard memory limit was set, changing `[resources]` is required: more free RAM cannot raise the cgroup limit. If systemd supervision is disabled or unavailable, worker death is an error and resumption requires a new invocation; the program does not infer OOM from an exit code. The optional `systemd_memory_limit_mib` imposes a hard per-worker cgroup RAM limit; set it only after measuring peaks. A too-small limit **causes** OOM; it does not offload model weights. Neither a systemd service nor the scheduler guarantees that the main CLI survives a global OOM or full swap.

For automatic recovery if **the CLI itself** is killed, run the CLI under a separately configured `systemd --user` service with `Restart=on-failure` and bounded `StartLimitBurst`. Point it at the **same** input, config and output directory so it can resume completed windows. Do not automatically restart jobs left in `waiting-resources.json` without freeing RAM or adjusting the configuration. Windows service integration is not implemented.

The output directory retains decoded audio, clips, adapter responses, stderr, reviews, resource decisions in `resource-events.jsonl` and a manifest so interrupted jobs can resume. The resource log records worker mode, duration and, where systemd exposes it, peak cgroup memory use (`null` if unavailable). It is not a guaranteed peak-RSS profile. `review-flags.json` records windows where an ASR output repeats extensively or where only one model detects text and diarization detects no overlapping turn. These are **advisory**, not a VAD or a reason to delete speech: the original hypotheses remain intact, and flagged jobs get `needs_audio_review`. `transcript.txt` is for reading; `transcript.json` (schema version 1) contains timestamps, speaker IDs, language, gap flags, each original ASR hypothesis, input SHA-256 and the model revisions. Do not publish the job directory: it contains sensitive audio and text.

To measure the strict per-model speed target without an LLM call, run `cargo run --release -- benchmark-asr recording.mp3 --role flix --output bench-flix`. Repeat with a **new output directory** for `primeline` and `qwen`; for the speaker model run `cargo run --release -- benchmark-diarization recording.mp3 --output bench-dia`. These names resolve under the selected `projects/` folder. For an alternate ASR engine supply both `--program /absolute/path/to/python` and `--script /absolute/path/to/worker.py`. Both benchmarks accept `--threads N` and repeatable `--adapter-arg=VALUE` to override adapter settings for that benchmark only; `benchmark-asr` also accepts `--in-flight N` for an indexed concurrent adapter. For example, to investigate pyannote batch sizes: `benchmark-diarization recording.mp3 --output bench-dia-batched --adapter-arg=--segmentation-batch-size --adapter-arg=16 --adapter-arg=--embedding-batch-size --adapter-arg=32 --adapter-arg=--profile`. `--workers N` runs up to four independent processes of the **same** ASR model on disjoint windows; for multiple workers you must provide a measured `--worker-peak-mib N`. The ASR benchmark rejects the run when `workers × peak + reserve` exceeds available RAM; estimates cannot guarantee safety. To opt in to a successful and quality-checked benchmark's process count in `transcribe`, set the matching `[resources.asr_worker_counts]` and `[resources.asr_worker_peak_mib]` values. The input must be at least 30 minutes long. Shum decodes and prepares the first 1800 seconds, then times model loading plus all 67 overlapping ASR windows (or one diarization pass). `benchmark.json` records elapsed seconds, real-time factor, threads, process count and requests in flight (ASR), adapter hash, pinned revision and `under_ten_minutes`; the command exits non-zero when elapsed time is **600 seconds or more**, or if the model fails. Preparation is outside model timing; a full job also includes review. Resource events record every completed ASR window's duration and audio length without writing speech into the log. Use `--offload` only to measure the explicitly configured disk mode; results from an incomplete or reused benchmark directory do not count.

### CPU engine comparison on one recording

The same private recording's first 1800 seconds were split into 67 windows
(up to 28 seconds, 0.8-second overlap) on an i7-12850HX with 24 logical CPUs
and about 30 GiB RAM. Times include loading the model and processing all
windows, but exclude MP3 decoding, window preparation, LLM review and output
assembly. Each row used a separate output directory. The primeLine and Flix
rows are Rust `benchmark-asr` reports; Turbo and Parakeet were standalone
local CPU trials using **the same WAV windows**; Qwen was a standalone
shared-server trial. These are not full-job times.

| ASR model / engine | CPU settings | Seconds | Completed | Observed quality flags |
| --- | --- | ---: | ---: | --- |
| primeLine / NeMo | 1 process, 6 threads | 124.6 | 67/67 | 6 empty windows; no repeated four-grams above 0.5 |
| Flix / CTranslate2 int8 | 3 processes x 6 threads, Silero VAD, fast greedy | 573.5 | 67/67 | Repetition fraction 0.57 in window 52 |
| Flix / CTranslate2 int8 | As above, beam 2 | 461.0 | 67/67 | Window 52 improved, but window 8 repeated (0.58) |
| Flix / CTranslate2 int8 | As above, beam 3 | 489.2 | 67/67 | Window 8 still repeated (0.58) |
| Swiss German Turbo / CTranslate2 int8 | 3 x 6 threads, VAD, beam 2, no fallback | 269.1 | 67/67 | Severe repetition in windows 3, 22 and 24 (0.81/0.80/0.57) |
| Swiss German Turbo / CTranslate2 int8 | 3 x 6 threads, VAD, beam 3, default decoding fallback | 282.9 | 67/67 | No four-gram repetition above 0.5; 5 empty windows |
| Parakeet TDT 0.6B v3 / NeMo-Speech.cpp Q8_0 | 1 shared CPU model, concurrency 1 (observed ~4 cores, 10 OS threads) | 213.2 | 67/67 | No four-gram repetition above 0.5; 5 empty windows; differs from primeLine on windows 1, 23, 46 |
| Qwen3-ASR 1.7B original / llama.cpp Q8_0 | 16 threads, 4 slots, German prefix, repetition penalty 1.15 | 401.2 | 67/67 | No truncation; 117 characters on apparently silent window 45 |
| Same Qwen / llama.cpp, automatic language | 16 threads, 4 slots | 353.3 | 66/67 usable | One answer hit token limit; several German windows identified as other languages |

The Flix rows use the **same** pinned `Flix-AI/flix-swissgerman-full`
revision `a9c347a904af5d31b28c537ac732e0618aed6136`. The
[MIT-licensed Swiss German Turbo model](https://huggingface.co/gcoli/whisper-large-v3-turbo-swiss-german-mit)
uses revision `540fb2e7a9135048963c5de9822579c96f8962e5` (CT2 weights
verified with SHA-256 `baecb116482ebb80a434d4a71ab50d6642808b258b5ce0f82cfde571aca3dfa2`).
The [Parakeet v3 GGUF](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
uses revision `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`. The Qwen
GGUF was converted from **original** `Qwen/Qwen3-ASR-1.7B` revision
`7278e1e70fe206f11671096ffdd38061171dd6e5`, not the configured `-hf`
repository. Qwen's 401.2 seconds include server startup; its 353.3-second
automatic-language run used an already running server. The old Transformers
Flix run took about 185 minutes on **35** minutes of audio, so it is not a
like-for-like row here.

| Diarization engine / setting | Seconds on 1800 seconds | Speakers | Observed disagreement |
| --- | ---: | ---: | --- |
| Community-1 ONNX, default clustering, 8 threads | 121.3 | 2 | PyTorch output had a third, short speaker |
| Community-1 ONNX, threshold 0.60, 8 threads | 259.3 | 2 | Still merges the short speaker |
| Community-1 ONNX, threshold 0.65, 8 threads | 205.4 | 3 | 91.9% mapped-label agreement on shared single-speaker speech with PyTorch, but only 1.5 of its 13.5 seconds for the shortest speaker |
| Community-1 ONNX, forced 3 speakers, 8 threads | 234.0 | 3 | 84.9% agreement on comparable shared speech; short speaker still uncertain |
| Nemotron 3 Q8_0 / NeMo-Speech.cpp V3 streaming preset | >610 (timed out) | Unknown | Did not meet the strict speed target |
| Nemotron 3 Q8_0 / NeMo-Speech.cpp V3 **offline geometry preset** | 60.2 | 3 | Only 71.7% mapped-label agreement on 935.8 seconds of speech common to both outputs; just 1.5 of 13.5 seconds of PyTorch's shortest speaker |

The Nemotron "offline geometry preset" (`--preset v3-offline`) still
**streams in chunks**; do not pass `--offline` on 30-minute audio, since
full-attention mode has an approximately 6.6-minute positional-table
limit. This comparison used the
[Nemotron 3 GGUF](https://huggingface.co/nvidia/Nemotron-3-Diarization)
revision `f667ed73aee57d40cc39428eb768b4fd87a0a29e` and an official
NeMo-Speech.cpp CPU **source build** pinned to commit
`8c15060bf0a71d7ac4f089931f2de65418b37128`. The v0.1.0 binary
reports this GGUF as compatible but fails inference with
`pre_ln transformer variant is not supported`; the later source commit
supports V3 pre-LN. The older PyTorch/pyannote CPU diarizer exceeded 600
seconds on 30 minutes. For the ONNX adapter's configuration, see
[diarization setup](adapters/diarization/README.md).

**None of these quality proxies is WER or DER.** There is no human
transcript or speaker annotation for this recording. Model-to-model
character similarity (median to primeLine, with case/punctuation ignored)
was 0.78 for Turbo beam 3 and 0.87 for Parakeet v3; primeLine itself is
another Parakeet-family model, not a ground truth. Repeated four-grams
measure a specific failure, not completeness. Flix window 8 transcribes
without repetition in three consecutive single-worker passes **and**
with three simultaneous workers, but repeats in the complete beam-2/3
runs on an identical WAV; the cause remains unknown, so short probes do
not clear the full-run failure. Inspect audio in disputed windows 1, 3,
8, 22, 23, 24, 45, 46 and 52, especially possible omissions and the
short speaker. Never delete speech because ASR or diarization disagrees.
These faster engines are not production defaults; the complete
three-ASR-plus-diarization job has **not** been shown to finish within
10 minutes.

### Same-weight primeLine CPU engine comparison

The original pinned `primeline/parakeet-primeline` revision
`3f1a9bcb611dfeda53fe74fe5f1a3d5701e8023e` was run on the **same
67 prepared WAV windows** from the first 1800 seconds above. Its original
`.nemo` checkpoint has SHA-256
`1b7f6e4f5dcffabd44464a4f8afbb0688f71ea972c960e19efce36e73494c80f`
and FP32 weights. NeMo exported that exact checkpoint to two
FP32 ONNX graphs without quantization; the export contained only FLOAT
floating-point initializers. `onnx-asr` supplied TDT decoding to both
ONNX runs. The source checkpoint and the WAV windows were not modified.
GGUF conversion was excluded because the available converter does not
support FP32 for ASR; the GGUF result above uses a *different model*.
The engines used NeMo 3.0.0 / CPU PyTorch 2.14.0, `onnx-asr` 0.12.0,
ONNX Runtime CPU 1.30.0, and ONNX Runtime OpenVINO 1.24.1.

Each row below shows **three sequential, complete CPU runs** per ONNX
backend and two NeMo baseline runs on this machine. A persistent worker
loaded the model once and processed all
windows in order, with a hard 600-second deadline per run. Times include
worker startup and loading, but exclude the one-time ONNX export and
earlier audio preparation. Each worker requested six inference threads;
the CPU-seconds/wall-time ratio confirms multi-core execution but is
not an exact thread count. Peak RSS was sampled about twice per second,
not measured with a guaranteed high-water-mark facility.

| Same primeLine weights / CPU backend | Seconds (successive runs) | Windows | Approximate CPU cores active | Sampled peak RSS | Text vs NeMo |
| --- | ---: | ---: | ---: | ---: | --- |
| Original `.nemo` / NeMo + PyTorch | 159.7 / 171.3 | 67/67 each | 5.3 | 5.5 GiB | Baseline |
| FP32 ONNX / ONNX Runtime CPU EP | 146.3 / 128.9 / 144.5 | 67/67 each | 8.3–8.4 | 3.0 GiB | 65/67 windows identical; two small differences |
| Same FP32 ONNX / OpenVINO CPU EP | 125.1 / 104.7 / 129.5 | 67/67 each | 7.3 | 5.5 GiB | 67/67 windows identical |

OpenVINO was requested with FP32 precision and CPU fallback disabled.
A profiling probe attributed encoder and decoder execution to the
OpenVINO provider, not the CPU provider. The previous NeMo benchmark
finished in 124.6 seconds using the same windows but a different
benchmark harness and machine conditions; these times should not be
interpreted as a controlled throughput guarantee. All eight new runs
met the strict `<600 s` per-model target.
Across the three ONNX repeats, OpenVINO took 104.7–129.5 s (mean 119.8 s);
the CPU provider took 128.9–146.3 s (mean 139.9 s). OpenVINO was faster
in each of the three paired rounds, but used about 2.4 GiB more sampled
peak RSS: roughly 5.5 GiB versus 3.0 GiB. The extra memory matters when
other local ASR models must run concurrently, whereas both primeLine
backends have ample headroom under the per-model time limit. For future
primeLine integration, **ONNX Runtime CPU FP32 is the selected candidate**:
its roughly 2.4 GiB lower sampled RSS leaves more room for the other
ASR models, and its slowest measured run still finished in 146.3 s.
This is a resource/throughput decision, not a claim that its two
different words are more accurate than NeMo's.

I read every window's transcript to check coherence as well as
machine-detectable repetition. Across all three backends, the broad
conversation remains followable, but several windows contain
ungrammatical phrases, implausible word choices, or abrupt incomplete
thoughts; names and medical terminology are especially uncertain.
None of the 67 windows exceeded a 0.3 repeated-four-gram fraction.
Each backend repeated its own text exactly across all three ONNX runs;
the same six windows were empty with every backend. Independent
diarization marked short speech-like intervals in three of those six,
so they need audio review rather than automatic deletion or an
assertion that words were missed. In one of the two ONNX Runtime
differences it omitted a short, seemingly incoherent tail; in the
other it chose a different word. Without listening to the audio we
cannot tell whether either change helped or removed real speech.
These judgments measure textual plausibility, **not WER or verified
completeness**. The experimental ONNX workers returned one
window-level segment each; their TDT decoder does produce token
timestamps, but segment-level alignment and integration into the
normal Shum adapter have not been validated. The active primeLine
configuration remains NeMo.

### Same-checkpoint Flix CPU precision trials

The pinned Flix checkpoint `a9c347a904af5d31b28c537ac732e0618aed6136`
contains **BF16** weights (1,259 BF16 tensors), not FP32 weights.
This i7-12850HX exposes AVX2 and AVX-VNNI, but no AVX-512 BF16 or
AMX BF16. Its original BF16 snapshot was left untouched. Separately
derived CTranslate2 FP32 and INT8 copies have the *same pinned source
manifest and model file hash*; the FP32 copy preserves the BF16 numeric
values without recovering any lost precision, while INT8 changes them.
The CTranslate2 CPU build here supports FP32 and INT8, not native BF16.

These additional trials used the exact **same 67 WAV windows** as the
primeLine comparison. Runs had a 600-second cutoff including worker
loading, except the clearly labelled FP32/PyTorch preliminary probe,
which was stopped after 571 seconds. Incomplete runs cannot establish
30-minute quality or compliance with the `<600 s` target. BF16/PyTorch used the
original BF16 weights and FP32 audio features; the installed
Transformers feature extractor cannot convert BF16 features to NumPy.
The regular Shum Transformers adapter instead loads this checkpoint
into FP32. CTranslate2 runs kept their converted weights outside the
source snapshot. VAD was enabled **only** in the explicitly marked
rows; it may discard faint speech and is not a quality-preserving
speed optimization until checked against audio.

| Flix CPU backend / precision | Processes × threads | VAD / decode | Completed by 600 s | Sampled RSS | Observed issue |
| --- | ---: | --- | ---: | ---: | --- |
| Transformers/PyTorch, original BF16 | 1 × 6 | No VAD, default | 4/67 | 5.1 GiB | Too slow; long suspect output on quiet window 0 |
| Transformers/PyTorch, FP32 load | 1 × 6 | No VAD, default | 5/67 by 571 s, then stopped | 8.8 GiB | Too slow; preliminary run stopped before its deadline |
| CTranslate2 FP32 derived from BF16 | 1 × 6 | No VAD, beam 1 | 20/67 | 12.2 GiB | Too slow, high RAM |
| CTranslate2 INT8 derived from BF16 | 3 × 6 | No VAD, beam 1 | 51/67 | 8.3 GiB combined | Incomplete |
| Same INT8 | 4 × 5 | No VAD, beam 1 | 52/67 | 11.1 GiB combined | More processes did not fix throughput |
| Same INT8, three identical trials | 3 × 6 | Silero VAD, beam 2, fast decode | **61 / 62 / 65 of 67** | **8.4 / 8.4 / 8.2 GiB combined** | All timed out; same repeating output in window 23 |
| Same INT8, normal decoding | 3 × 6 | Silero VAD, beam 2, fallback enabled | 59/67 | 8.6 GiB combined | Timed out; window 23 did not repeat |

The three VAD trials ran on identical input hashes, used about 16.2–16.4
CPU cores in aggregate, and returned exactly the same text on all 61
windows common to all runs. In every run window 23 contained a
four-gram repetition fraction of about 0.71. Five windows were empty;
independent diarization marked about 2.8 seconds of possible speech
in one of them and 1.1 seconds in another. Another low-speech window
yielded much shorter text with VAD than without it. These are
**audio-review candidates**, not proof of VAD omissions or correctness
of the unfiltered text. No human reference transcript exists, so no
WER was measured. Earlier CT2 INT8/VAD beam-2 trials passed in 461.0
seconds but repeated a different window. The new three-run failure
means that single result does not establish reliable throughput or
quality; high CPU package temperatures were observed after some of
the new runs, but their effect was not isolated. **Do not adopt any
Flix engine from these trials yet.** A further normal-decoding trial
removed the obvious text loop in window 23 (156 characters, no
repeated four-grams versus 160 characters, 0.71 with fast decode)
but produced awkward wording there and finished only 59 windows. It retained the same five empty
windows, so the VAD-completeness question remains open. That mode
also fails the speed gate; two more full repeats would not make it
eligible under the present hardware and deadline.

Whisper may report text with timestamps beyond the end of a clip. Such segments are excluded from the text sent to the LLM, while the unmodified hypothesis is retained in `raw_text` and the original intervals in `out_of_bounds_segments` for manual inspection.
The JSON quality status is `needs_audio_review` when gaps or advisory review flags remain, otherwise `text_reviewed_audio_not_verified`; neither claims the LLM has checked the recording itself.

## Adapter protocol

Each persistent ASR process receives `--engine nemo|whisper|qwen --model PATH --threads N --serve`. It reads newline-delimited `{"input":"/absolute/clip.wav"}` requests on stdin and sends one JSON line per clip on stdout: `{"text":"…","language":"de","segments":[{"start":0.0,"end":1.0,"text":"…"}]}`. Times are clip-relative seconds. Optional `max_in_flight = 2..4` on an ASR adapter sends `{"index":N,"input":"…"}` requests concurrently; that adapter **must** return the same integer `index` with each response, including errors, so out-of-order completions are saved against the correct window. Existing adapters default to one in-flight request. A model is loaded once per process (additional same-role processes load their own copy). The diarization process receives `--model PATH --input AUDIO.wav --output RESULT.json --threads N` and writes `{"turns":[…],"exclusive_turns":[{"start":0.0,"end":1.0,"speaker":"SPEAKER_00"}]}` with absolute times. Non-zero exits fail the job; no subprocess may silently fetch a model.

The LLM receives a JSON prompt on stdin and must return JSON `{"segments":[{"index":0,"text":"…","language":"de|ru|mixed|unknown","gap":false}]}` for every requested index. For gaps, set `gap:true` and include `[пропуск]` in text. The example uses the `codex exec` `--output-last-message` option, replacing `{output}` in its arguments with the batch result path. Alternative authorized CLI programs may be configured; authentication and model choice belong to the CLI.

To add subject-specific spelling or formatting rules, set `review_rules = "/absolute/private/rules.txt"` at the **top level** of the per-user `config.toml`. The file is read as text, capped at 64 KiB and included in each LLM request. Rules must not override the no-invention requirement.

The process boundary keeps the Rust scheduler independent from model frameworks. Changing a model within a compatible engine requires a different repo/revision/path in the config, not a Rust change. FFmpeg, Python adapters and the LLM CLI remain independent installations. HTTP API, voice enrollment and Windows setup are later work.
