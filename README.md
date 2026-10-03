# shum

CPU-only CLI for long German, Swiss German and occasional Russian MP3 recordings. The Rust program splits the entire recording into overlapping audio windows without dropping quiet sections, runs the configured local ASR models (concurrently when RAM allows), assigns speakers with local pyannote, and asks an authenticated LLM CLI to compare the resulting texts. The final German text uses Standard German; Russian stays Russian. Unrecoverable speech is marked `[пропуск]`. The selected models' raw hypotheses remain in the JSON output.

**Status:** early implementation. The Rust unit tests and adapter mock tests do not establish accuracy or 1–2-hour throughput. Run a short representative recording before relying on output. An LLM that only reads ASR text cannot prove that nothing was missed in audio; inspect intervals marked as gaps.

See the [Russian usage guide](docs/USAGE.ru.md) for installation, complete
workflow, stage-only execution and recovery. The [agent API and headless
frontend](docs/API.md) provide versioned JSON-lines and authenticated
loopback HTTP transports; they do not require a GUI or interactive TTY.

## Install on Ubuntu

1. Install `ffmpeg`, Rust (`cargo`) and `uv` through their normal package channels.
2. Run `bash scripts/install.sh` from the checkout. It builds the release program, then starts onboarding. Plain `cargo build` only builds: it never prompts or changes your home directory.
3. Choose the local installation directory, then the locations for binaries, settings, projects and models. The suggested layout keeps them together, but they can be separate. An existing directory with `projects/` and `models/` can be adopted without overwriting its data. The interactive wizard registers `shum` in a directory already on `PATH` when possible (usually `~/.local/bin`, or `~/bin` if it is already on `PATH`). If needed, it asks before changing your Bash or Zsh startup files.
4. Configure the isolated adapter environments and model snapshots. Downloads may take several gigabytes and require confirmation. For pyannote, accept the conditions at [speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) and authenticate `hf`. See [ASR installation](adapters/asr/README.md) and [diarization installation](adapters/diarization/README.md).
5. Choose Codex CLI, GitHub Copilot CLI or a custom LLM CLI, or skip this step. Shum checks the selected tool, not just `gh auth status`. Authentication remains managed by that CLI: Shum does not copy its tokens.

Use `shum` for subsequent commands after registration. The checkout is source code, not the installation: runtime adapters, environments, model snapshots, personal settings and projects belong in the chosen locations. A first launch without settings starts onboarding in a terminal; a non-interactive launch gives explicit setup instructions instead of waiting for input. The current shell cannot inherit a `PATH` change made by its child; follow the printed `export PATH=...` instruction for this terminal or open a new one. Changing directories never makes a command available by name: `./shum` runs a file in the current directory, while `shum` searches `PATH`.

For a non-interactive layout, pass explicit locations to the installer:

```bash
bash scripts/install.sh --storage-dir /srv/shum \
  --bin-dir /srv/shum/bin --settings-dir /srv/shum/config \
  --projects-dir /srv/shum-projects --models-dir /srv/shum-models
```

Headless layout setup does **not** register a command or edit shell files by default. To opt in, add `--command-dir /absolute/directory` (a directory on `PATH` is convenient). Add `--setup-shell` only together with `--command-dir` to persist a missing `PATH` entry in supported Bash or Zsh startup files. If `shum` is installed but not found by name, repair command registration without reinstalling, downloading models, or probing an LLM:

```bash
/absolute/installed/bin/shum setup-command --command-dir /absolute/directory
```

Omit `--command-dir` to use the installed binary's directory if it is already on `PATH`, otherwise `~/.local/bin` (or `~/bin` when it is already on `PATH`). Add `--installation /absolute/installation/root` before `setup-command` if installation discovery needs help. Add `--setup-shell` with an explicit `--command-dir` to request persistent shell setup. Existing unrelated `shum` files or links are not overwritten.

`--install-dir` is an alias for `--storage-dir`. `SHUM_SETTINGS=/absolute/path/settings.toml` selects an installation explicitly; otherwise Shum uses metadata beside its executable or the small XDG installation locator. The chosen settings directory holds personal configuration. Re-run `shum --setup-llm` to change only the review provider, model or access settings.

### Updating an existing installation

Run `scripts/install.sh` again with the existing layout. Without a terminal,
provide all five locations and redirect stdin from `/dev/null`:

```bash
ROOT="$HOME/shum"
bash scripts/install.sh --storage-dir "$ROOT" \
  --bin-dir "$ROOT/bin" --settings-dir "$ROOT/config" \
  --projects-dir "$ROOT/projects" --models-dir "$ROOT/models" < /dev/null
```

Use the actual directories if your layout differs. The installer publishes
the runtime bundled with the new binary before replacing the executable,
backs up configuration, and retains model revisions, provider settings,
projects and Python environments. Custom adapters remain custom; managed
adapters are updated to the new runtime. This non-interactive command does not
download models or dependencies or change shell files.

Do not update by copying only a newly built executable: each binary expects
its matching runtime version, and startup rejects missing or damaged files.
Do not reinstall while a transcription or other Shum task is running.

`shum models check` reports available newer Hugging Face commits. `shum models update --role qwen` downloads the current revision separately and activates it only after validation. `shum models install-pinned --role qwen` installs the exact immutable revision already configured. Models are never automatically updated on startup.

`shum --setup-models` reopens only model setup and saves each successful choice immediately. The guided download flow uses the four supported production model/adapter combinations; it can keep, install a pinned revision or explicitly resolve the latest revision. Custom and experimental engines remain manual opt-ins and are not silently converted to production defaults.

`shum transcribe recording.mp3` creates a project under the chosen projects directory. `--output my-recording` selects its name; an absolute output path must be inside that directory. Preserve the entire project directory when moving a job: it contains checkpoints and provenance, not just the final text.

### Optional LLM and availability

The LLM receives **transcript text**, never the audio file. Do not enable it without permission to share the text with its provider. A readiness probe sends only a fixed technical request, not project data, and may consume provider quota. Local login status alone does not prove that a token or model access is still valid.

You can finish onboarding without an LLM. Local ASR and diarization results remain available, but the project waits for review; it does not claim to have produced a reviewed final transcript. A missing CLI, authentication problem, network failure or quota error produces an explicit warning and preserves completed work. Reconfigure the LLM or restore its access, then resume the project. Authentication repair does not repeat path selection or model downloads.

### Project status, resume and rerun

```bash
shum project status my-recording
shum project resume my-recording
shum project resume my-recording --input /new/location/recording.mp3
shum project rerun my-recording --step review
shum project rerun my-recording --step asr:flix
shum project rerun my-recording --step diarization
shum project rerun my-recording --step export
shum project run my-recording --step asr:flix
shum project run new-recording --step chunks --input recording.mp3
```

`resume` continues missing work using valid checkpoints. `rerun` starts a new attempt of the selected stage and makes its downstream results stale. Supported stages are audio preparation, chunks, each ASR role, diarization, review flags, LLM review and export. Rerunning Flix does not repeat primeLine or Qwen; rerunning diarization does not repeat ASR; changing the LLM does not invalidate audio or ASR. An export rerun uses existing review results without calling an LLM.

`project run --step` runs only that stage and its missing prerequisites,
not its descendants. A new project requires `--input`; an existing
project can use its original recording or rebind identical content.
For example, `--step asr:flix` prepares audio/chunks if needed, runs Flix
only, then stops before diarization or review. `--step diarization`
requires audio only, not ASR. Exit 0 for a selected stage means its
checkpoints are ready, not that a final transcript exists. Local
stage-only execution does not run LLM readiness probes.

If the original recording was moved, `resume --input` rebinds its path only after verifying the original input hash. It cannot replace the recording with different audio in the same project.

Previous attempts are retained for inspection, but a stale transcript is not presented as the current result. Review failure leaves completed review batches available for `resume`; use `rerun --step review` when you intentionally want a fresh review. Project state records waiting and failed stages explicitly, and concurrent writers to the same project are rejected.

Workflow commands exit with `0` for a complete transcript, `2` when local results are saved but LLM review is pending, and `1` on an error. Inspect `project status` for the persistent per-stage state; a pending review is not a completed transcript.

The scheduler reads available RAM and cgroup v2 limits before each batch. It selects up to `max_asr_workers` ASR processes using the configurable `[resources]` reserve and per-active-ID peak estimates in the private `config.toml`; these are **estimates, not memory limits or guarantees**. Historical estimates for inactive IDs may remain in the map. Optional `max_parallel_asr_models` caps distinct models active at once; `sequential_asr_roles` serializes distinct models without limiting replicas within one ID. Defaults (6000/10500/13500 MiB for primeLine/Flix/Qwen, plus 4096 MiB reserved) come from short CPU runs on one machine. By default one process handles each role. To use multiple independent processes of the same model in a normal job, set `[resources.asr_worker_counts]` (for example `flix = 3`) **and** a measured per-process RAM estimate in `[resources.asr_worker_peak_mib]` (for example `flix = 2600`); the total is capped by `max_asr_workers`. The scheduler assigns disjoint windows to those processes, falls back to one when the requested replicas do not fit RAM, and retains completed windows if interrupted. This is opt-in: the faster Flix CT2 decode has not passed audio-grounded quality review. When even one estimated peak exceeds available RAM, it tries that process with a warning; Linux swap is a last resort and OOM remains possible. Use `--sequential-asr` to force at most one ASR process irrespective of the estimates. The selected hypotheses, even if only one model was selected, appear in the final transcript; single-source accuracy remains unverified.

To opt in to verified Accelerate CPU/disk offload, set `resources.offload_dir` to a private directory outside model snapshots (for example an absolute path to `~/shum/projects/.asr-offload`). Shum creates it with mode 0700 if missing and requires at least `offload_min_free_mib` free disk space before starting an offloaded worker; the default is 8192 MiB. Once the scheduler has reduced a batch to one worker and that worker's standard RAM estimate exceeds the available budget, it starts Flix or Qwen in offload mode. After a confirmed OOM it also retries either of those workers with offload, if configured. The `resources.offload_roles` list identifies compatible adapters; remove `flix` when selecting an alternative Flix engine without Accelerate support. primeLine and pyannote do **not** support this mode; their processes rely on serial execution and available RAM/swap. Offload estimates (`resources.offload_estimated_peak_mib`, default 6500/7000 MiB) derive from short trials, **not** safe maximum RSS. Flix and Qwen used about 5.0/5.2 GiB peak RSS on one 28-second clip, versus roughly 9.0/11.9 GiB without offload; they ran about 1.8/4.9 times slower. The cache is kept outside pinned model directories, and each newly generated hypothesis records its mode and model revision. A missing or full offload volume fails explicitly without publishing an incomplete window. See [adapter measurements](adapters/asr/README.md).

On Ubuntu, set `resources.supervise_asr_with_systemd = true` to run each ASR adapter as an isolated `systemd --user` service. The CLI then distinguishes a **confirmed** `Result=oom-kill` from timeouts and other errors. After an OOM it retains completed windows, retries up to `oom_retries` times with only one worker and a short backoff, then writes `waiting-resources.json` and stops. A later invocation resumes once available RAM rises by over 512 MiB or `[resources]` changes. When a hard memory limit was set, changing `[resources]` is required: more free RAM cannot raise the cgroup limit. If systemd supervision is disabled or unavailable, worker death is an error and resumption requires a new invocation; the program does not infer OOM from an exit code. The optional `systemd_memory_limit_mib` imposes a hard per-worker cgroup RAM limit; set it only after measuring peaks. A too-small limit **causes** OOM; it does not offload model weights. Neither a systemd service nor the scheduler guarantees that the main CLI survives a global OOM or full swap.

For automatic recovery if **the CLI itself** is killed, run the CLI under a separately configured `systemd --user` service with `Restart=on-failure` and bounded `StartLimitBurst`. Point it at the **same** input, config and output directory so it can resume completed windows. Do not automatically restart jobs left in `waiting-resources.json` without freeing RAM or adjusting the configuration. Windows service integration is not implemented.

The output directory retains decoded audio, clips, adapter responses, stderr, reviews, resource decisions in `resource-events.jsonl` and a manifest so interrupted jobs can resume. The resource log records worker mode, duration and, where systemd exposes it, peak cgroup memory use (`null` if unavailable). It is not a guaranteed peak-RSS profile. `review-flags.json` records windows where an ASR output repeats extensively or where only one model detects text and diarization detects no overlapping turn. These are **advisory**, not a VAD or a reason to delete speech: the original hypotheses remain intact, and flagged jobs get `needs_audio_review`. Final filenames use the original audio stem: `recording.mp3` produces `recording.txt` for reading and `recording.json` (schema version 1) containing timestamps, speaker IDs, language, gap flags, each original ASR hypothesis, input SHA-256 and the model revisions. Do not publish the job directory: it contains sensitive audio and text.

The source stem is stored in workflow state and survives moving/renaming the
input with `--input`. Files normally live in the project root; names
colliding with internal JSON files (such as `workflow.json`) use its
`exports/` subdirectory instead. CLI completion prints the text path; API
workflow/status responses expose both paths in `exports`. Resuming an old
project archives its generic `transcript.*` files and rebuilds only export
when upstream checkpoints are valid.

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
| primeLine / ONNX Runtime FP32, guarded fresh repeats | 1 process, 6 threads, one window in flight | 109.2 / 112.8 / 112.5 | 67/67 in each of 3 runs | Identical full hypotheses across runs; 6 empty windows; no repeated four-grams above 0.5 |
| Flix / CTranslate2 int8 | 3 processes x 6 threads, Silero VAD, fast greedy | 573.5 | 67/67 | Repetition fraction 0.57 in window 52 |
| Flix / CTranslate2 int8 | As above, beam 2 | 461.0 | 67/67 | Window 52 improved, but window 8 repeated (0.58) |
| Flix / CTranslate2 int8 | As above, beam 3 | 489.2 | 67/67 | Window 8 still repeated (0.58) |
| Swiss German Turbo / CTranslate2 int8 | 3 x 6 threads, VAD, beam 2, no fallback | 269.1 | 67/67 | Severe repetition in windows 3, 22 and 24 (0.81/0.80/0.57) |
| Swiss German Turbo / CTranslate2 int8 | 3 x 6 threads, VAD, beam 3, default decoding fallback | 282.9 | 67/67 | No four-gram repetition above 0.5; 5 empty windows |
| Swiss German Turbo / CTranslate2 int8, guarded fresh repeats | 3 x 6 threads, VAD, beam 3, standard fallback | 329.7 / 344.6 / 362.1 | 67/67 in each of 3 runs | No four-gram repetition above 0.5; same 5 empty windows; text varies across runs |
| Swiss German Turbo / ONNX Runtime FP32, optimized cached decoder | 1 process, ORT 16 threads, Torch 1, beam 3, no VAD/fallback | 64.2 (decode error) | 3/67 | Window 3 hit 400 new tokens without EOS; window 1 repeated |
| Swiss German Turbo / ONNX Runtime FP32, native timestamp/fallback profile | 1 process, ORT 16 threads, Torch 1, beam 3, same Silero VAD | 600.3 (timeout) | 29/67 | Failed strict speed gate; window 20 repetition fraction 0.56 |
| Parakeet TDT 0.6B v3 / NeMo-Speech.cpp Q8_0 | 1 shared CPU model, concurrency 1 (observed ~4 cores, 10 OS threads) | 213.2 | 67/67 | No four-gram repetition above 0.5; 5 empty windows; differs from primeLine on windows 1, 23, 46 |
| Qwen3-ASR 1.7B original / llama.cpp Q8_0 | 16 threads, 4 slots, German prefix, repetition penalty 1.15 | 401.2 | 67/67 | No truncation; 117 characters on apparently silent window 45 |
| Qwen3-ASR 1.7B original / llama.cpp Q8_0, guarded fresh repeats | 16 threads, 4 slots, German prefix | 473.2 / 434.0 / 427.7 | 67/67 in each of 3 runs | Text varies across runs; window 45 has 116/116/117 characters |
| Qwen3-ASR 1.7B original / native C, BF16 source weights | 16 threads, German, model reloaded per clip | 600.0 (timeout) | 19/67 | No full speed pass; separate four-window pilot completed in 105.5 s |
| Qwen3-ASR 1.7B-hf / Transformers FP32, guarded fresh load | 1 process, 6 threads | 6.8 (resource stop) | 0/67 | New system swap-out plus memory PSI full avg10 1.26%; not a throughput result |
| Qwen3-ASR 1.7B-hf / Transformers CPU/disk offload | 1 process, 6 threads | 600.2 (timeout) | 6/67 | Lower memory, but failed the full-corpus speed gate |
| Same Qwen / llama.cpp, automatic language | 16 threads, 4 slots | 353.3 | 66/67 usable | One answer hit token limit; several German windows identified as other languages |
| Flix / CTranslate2 int8, guarded fallback rerun | 1 process x 6 threads, VAD, beam 2, seed 0 | 600.1 (timeout) | 59/67 | Failed the strict speed gate |
| Flix / CTranslate2 int8, guarded fallback rerun | 2 processes x 3 threads, VAD, beam 2, seed 0 | 600.2 (timeout) | 66/67 | Window 66 missing; failed the strict speed gate |

The fresh primeLine ONNX repeats used the recorded export of
`primeline/parakeet-primeline` revision
`3f1a9bcb611dfeda53fe74fe5f1a3d5701e8023e`. All graph files,
external weights and worker/runtime fingerprints were unchanged before
and after the repeats. This is recorded export provenance, **not a fresh
independent numerical graph-to-checkpoint audit**. Peak worker RSS was
3095/3096/3096 MiB. Each fresh process completed all 67 frozen windows
strictly below 600 seconds, including worker loading; corpus verification
and preparation remained outside that deadline.

The whole ONNX benchmark ran in a user-systemd scope with a 12 GiB hard
memory limit and zero swap, retaining the existing 4 GiB effective
headroom reserve and own-swap/PSI guards. An initial 9 GiB scope failed
admission before worker launch after source hashing charged page cache
to the scope; it is not a completed run. A separate fresh NeMo attempt
stopped during loading at 17.5 seconds, with zero results, when new system
swap-out coincided with memory PSI full avg10 of 1.2%. These stops do not
prove causal attribution to the workers.

The ONNX worker remains a standalone experiment with fixed graph paths,
not a production adapter. Raw Rust reports label its role as `nemo`;
the actual runtime here was ONNX Runtime 1.30.0 CPU, not NeMo. The six
empty windows were 0, 32, 40, 42, 44 and 45. Reproducibility and a speed
pass do not establish correctness; audio-grounded quality review remains
pending, and no installation defaults were changed.

The fresh Qwen GGUF repeats started a new shared server each time and
passed the strict 67/67, less-than-600-seconds gate in all three runs.
Peak worker-plus-server RSS was approximately 6.92 GiB. Runs were
sequential, with an actual 24 GiB user-systemd scope cap, zero scope swap
and the unchanged 4 GiB reserve/own-swap/PSI guards. Original source
shards matched their recorded LFS hashes; small source files matched
recorded Git blob hashes. Source/backend/derived-weight checksums were
checked before and after execution. The native C backend uses that same
recorded original source; it reloads the model for each window. Its
19/67 timeout is measured, not extrapolated from the short pilot.

This is **not** a same-weight comparison with the separately pinned
`Qwen/Qwen3-ASR-1.7B-hf` baseline: the repositories/layouts and language
controls differ. An independent numerical GGUF-to-source conversion
audit has not been performed. Quantization changes numerical values.
GGUF produced nonempty answers for every window, including windows
discarded by Flix VAD, and some answers changed across fresh runs.
Neither VAD nor agreement establishes silence or hallucination. Quality
remains unverified; no model was promoted to a production default.

The follow-up [Qwen ONNX/OpenVINO experiment tools](adapters/asr/qwen_onnx_README.md)
add a bounded source-weight audit. All 708 original BF16 tensors match
the 707 HF-packaged tensors bitwise after name mapping (the original
tied LM head maps to the HF embedding table). This resolves the source
parameter-identity question for those two pinned snapshots, not the
GGUF/ONNX quantization or preprocessing/decoding equivalence. The
historical language-control difference remains.

The alternative-engine round is complete. Published ONNX INT4 completed
5/67 windows in 93.702 seconds with forced German, then 1/67 in 58.862
seconds with automatic language and 16 threads; both stopped on missing
EOS, with repetitive generation confirmed by token diagnostics, not a
600-second timeout. Local OpenVINO audio/layerwise INT8 exports were
produced, but both inference pilots stopped under memory-pressure guards
before decoding a window. These profiles are rejected for this corpus
and resource budget. Qwen itself is not excluded on speed because the
original llama.cpp Q8_0 profile already passed three full fresh runs.
See the linked experiment guide for provenance, timings and limitations.
Application tests/builds were deferred for this round at the user's request.

### Selecting ASR roles

Select ASR roles for a new or existing project:

```sh
shum transcribe recording.mp3 --output meeting --asr primeline,flix
shum project resume meeting --asr primeline,flix
```

Each invocation without `--asr` uses every ASR role in the **current**
private config, including resume, run and rerun. An explicit `--asr` applies
only to that invocation. The project records the roles used for that run;
it does not save a default for later invocations. One role is supported.
Inactive roles are not launched, verified or sent to LLM
review, and their cached output is retained. Re-enabling a role runs only
missing recognition work and invalidates dependent review/export results.
`flix` names the configured workflow role (Turbo in the optimized profile).
An already-running command is not changed by another invocation: interrupt
it first, then resume its project with the desired selection.

The `[asr.ID]` tables in the private config define the available roles.
Each needs exactly one matching `[[models]]` record; keep exactly one
`diarization` model. Remove both entries to disable a model. IDs use ASCII
letters, digits, `_` or `-` and cannot conflict with project directories
such as `clips`, `reviews` or `exports`. Legacy `primeline`, `flix` and
`qwen` infer their engine; new IDs need `engine = "nemo"`, `"whisper"` or
`"qwen"`. The engine must support the chosen weights: the standard
`adapters/asr/main.py` loads NeMo, Transformers Whisper or Qwen3-ASR,
not arbitrary Hugging Face checkpoints. Keep private config and weights
outside the source checkout; [shum.example.toml](./shum.example.toml)
shows two distinct IDs sharing a Whisper engine. For a personal one-model
installation, keep one ASR `[asr.ID]` and matching `[[models]]` entry,
the diarization model, and an estimate for that ID in
`[resources.estimated_peak_mib]`. No other ASR entry is required.

### Execution plan and operator controls

An execution plan shows known steps, dependencies, attempts and their
current state. Progress measures completed *planned work*, not the
accuracy or completeness of a transcript. A missing duration estimate
must remain unknown; it must not become an invented percentage or ETA.
Parallel work can complete out of order, and adding a check or retrying
work changes the plan and any estimate. A failed or blocked prerequisite
is not a completed transcript even if all scheduled attempts have ended.

Read a project's plan and event journal with `shum project plan PROJECT`
and `shum project events PROJECT --after SEQUENCE`. The journal retains
the history of attempts and artifact selections; reconnecting clients
can request events after their last sequence. Operator controls are
explicit requests, not automatic retries based on diagnostic findings.
Preview a control request before confirming it. A confirmed skip can
leave required preparation or final verification unavailable; inspect
the resulting plan and exports rather than treating a progress percentage
as proof of success. Controls apply to a project, not to the user's
personal model or provider configuration.

Execution limits bound retries and graph extensions. They do not form a
smart controller and cannot establish which hypothesis matches the
audio. The [API reference](docs/API.md) describes the request format and
the [Russian usage guide](docs/USAGE.ru.md) explains terminal operation.

### Offline advisory quality diagnostics

The workflow produces a private `.quality/report.json` sidecar as artifacts
become available. It measures audio levels and flags suspicious ASR output,
timestamp defects and recorded resource failures. These checks do **not**
prove transcription accuracy, choose text, change audio, invoke another
model or automatically retry recognition. An incomplete report explicitly
lists missing inputs; unavailable telemetry is not treated as a passing check.
Each window's text analysis records `source_status` as `no_sources`,
`single_asr_source` or `multiple_asr_sources_unverified`; accuracy stays
`unverified` in every case. A single hypothesis has no cross-model
corroboration. Agreement among several models does not prove accuracy or
independence. The LLM review prompt states both limitations.
Diagnostic computation budgets, such as `quality.max_chunks` and
`quality.text.max_hypotheses`, limit the offline report only; they do not
cap the number of models in the workflow or discard hypotheses from
review and export. Text diagnostics compute per-role metrics for every
available hypothesis. The default `quality.text.max_hypotheses = 8`
limits only pairwise comparisons: with 12 sources the report contains
12 role metrics and 28 pairs, and lists the other four IDs in
`pairwise_omitted_roles` with a `pairwise_hypothesis_budget` info finding.

```sh
shum project check-quality PROJECT
shum --no-quality transcribe recording.mp3 --output PROJECT
```

Explicit `check-quality` rescans existing prepared audio offline without
re-running project stages. Automatic audio checks use a metadata-based fast
cache; the report records that limitation. Thresholds are configurable
preliminary heuristics, not calibrated accuracy scores. Set
`quality.enabled = false` to disable automatic checks persistently.
The agent API exposes `project.check_quality` and paginated
`project.quality_report`; the existing headless transport forwards both.
Old ASR, review and export checkpoints remain valid when diagnostics are added.

### Opt-in deployment of the speed-qualified ASR profile

The selected local profile uses primeLine FP32 ONNX (one six-thread worker),
Swiss German Turbo CTranslate2 INT8 (three six-thread workers, beam 3,
standard temperature fallback and Silero VAD), and original Qwen3-ASR
llama.cpp Q8_0 (16 threads, one server with four in-flight slots).
Turbo occupies the existing `flix` workflow role; checkpoint and API role
names do not change. This is an explicit local configuration, not a change
to the default model set or a claim of verified accuracy.

The portable runtime bundles [primeLine's production worker](adapters/asr/primeline_onnx_README.md)
and [Turbo's production worker](adapters/asr/turbo_README.md), with separate
pinned requirements. Keep original source snapshots, derived model assets,
native backend libraries and Python environments in the installation, not
the source checkout. Their guides explain required provenance manifests.
Use `resources.sequential_asr_roles = true` to run different models
sequentially without disabling Turbo's measured three-worker profile.
`--sequential-asr` still deliberately forces one worker.

Before replacing an existing installation, back up its binary and private
configuration; retain the previous immutable runtime and environments for
rollback. Preserve LLM, diarization and project settings. Do not update only
an original source revision while retaining older derived artifacts:
re-export/re-provision and validate the complete profile together.
Production Turbo retains individual utterance onsets, logs WAV-end
clipping and rejects invalid intervals; it does not use the benchmark
bridge's whole-window timestamp workaround. primeLine's decoder exposes
only coarse window timing, which is reported explicitly.

The fresh [Swiss German Turbo CTranslate2 repeats](adapters/asr/turbo_ct2_README.md)
passed the three-run speed gate with effective `int8_float32` and approximately
6.10 GiB peak aggregate worker RSS. Empty windows were 0, 32, 40, 44 and 45
in every run. Text differed from run 1 in 23 windows in run 2 and 19 in run 3;
standard fallback selected nonzero sampling temperatures, so speed
repeatability is not deterministic-text or accuracy acceptance. An initial
protocol attempt rejected a raw decoder timestamp beyond its WAV bounds.
The experimental bridge now reports honest whole-window intervals while
retaining raw timing defects in diagnostics, not production utterance timing.

The [Turbo ONNX comparison](adapters/asr/turbo_onnx_README.md) used locally
exported FP32 encoder, decoder and cached-decoder graphs from the same pinned
repository revision. Separating graph tracing from bounded external-weight
streaming succeeded after guarded monolithic export failures. Proper ORT
optimization/prepacking and nonspinning pools reduced runtime overhead.
The beam-only profile still failed decoding, not the 600-second timeout.
The closer native Whisper timestamp/fallback plus Silero VAD profile then
completed only windows 0-28 before the actual 600-second deadline.

Keep Turbo/CTranslate2 INT8 with beam 3 and standard fallback as the measured
speed candidate; do not repeat unchanged failed ONNX profiles. This is not
exact decoder parity: HF and faster-whisper calculate compression ratios over
different byte representations, so the shared threshold 2.4 can trigger
different retries. Guarded decoder INT8 preprocessing did not produce a ready
model or inference measurement. Reject the tested FP32 ONNX profiles, not all
ONNX Runtime configurations. Quality acceptance and production integration
remain separate; application tests and builds were deferred as requested.

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

A later hash-verified diagnostic reproduced window 23's **exact** historical
fast-decode answer in a fresh single six-thread worker (INT8, VAD, beam 2):
four-gram repetition remained 0.71. That one-window run took 10.85 seconds,
including startup, and about 2.6 GiB peak RSS; it is not a full-corpus speed
result. The loop does not require concurrent workers. A persistent
sequence was stopped by an overly strict system-swap-out-only guard.
The revised guard retains the reserve and own-worker-swap checks, while
requiring memory PSI full avg10 at least 1.0% alongside system swap-out.
Short experiments then resumed: the three repeated suspect windows took
120.26 seconds with fast decode and 140.94 seconds with fallback.
Fast decode consistently looped on window 23; fallback avoided the loop
but varied its text and final temperature within a persistent worker,
even with requested seed resets. Three fresh seeded workers returned the
same fallback hypothesis instead. These are diagnostic observations,
not verified quality or full-corpus performance. The [diagnostic helper and local review
workflow](adapters/asr/flix_fast_README.md#repeatability-diagnostics) preserve
these failures rather than count them as successes.

`benchmark-asr --corpus-manifest CORPUS.json` can reuse the same verified
67 clips without renormalizing them. For a manifest made by the helper,
pass its `audio_path` WAV as the positional input. Timeout, worker failure
and resource-stop reports retain completed/missing indices, window/worker
timing and sampled memory. The 600-second clock starts before worker
launch; first-result timing is not model-readiness timing. At least 4 GiB
of reserve is enforced; estimated worker peak remains an admission
estimate, not a memory limit.
Benchmarks verify the selected model locally before inference; they do
not query remote revisions or require unrelated role models to be
installed. Use a separate benchmark config with LLM disabled for local
engine experiments; the normal installation config remains unchanged.

The next same-checkpoint engine candidate is ONNX Runtime through Optimum
ONNX. A [read-only feasibility probe](adapters/asr/flix_onnx_README.md) found
existing graphs without source-revision/precision provenance and
insufficient headroom under conservative export/load estimates. A bounded
streaming audit subsequently verified numerical BF16-to-FP32 identity
for all 1,259 source parameters represented in the existing graphs.
Runtime/generation compatibility is tested separately under a kernel RAM
limit and zero swap; no ONNX full-corpus speed or accuracy result is
claimed. Production defaults are unchanged.

Whisper may report text with timestamps beyond the end of a clip. Such segments are excluded from the text sent to the LLM, while the unmodified hypothesis is retained in `raw_text` and the original intervals in `out_of_bounds_segments` for manual inspection.
The JSON quality status is `needs_audio_review` when gaps or advisory review flags remain, otherwise `text_reviewed_audio_not_verified`; neither claims the LLM has checked the recording itself.

## Adapter protocol

Each persistent ASR process receives `--engine nemo|whisper|qwen --model PATH --threads N --serve`. It reads newline-delimited `{"input":"/absolute/clip.wav"}` requests on stdin and sends one JSON line per clip on stdout: `{"text":"…","language":"de","segments":[{"start":0.0,"end":1.0,"text":"…"}]}`. Times are clip-relative seconds. Optional `max_in_flight = 2..4` on an ASR adapter sends `{"index":N,"input":"…"}` requests concurrently; that adapter **must** return the same integer `index` with each response, including errors, so out-of-order completions are saved against the correct window. Existing adapters default to one in-flight request. A model is loaded once per process (additional same-role processes load their own copy). The diarization process receives `--model PATH --input AUDIO.wav --output RESULT.json --threads N` and writes `{"turns":[…],"exclusive_turns":[{"start":0.0,"end":1.0,"speaker":"SPEAKER_00"}]}` with absolute times. Non-zero exits fail the job; no subprocess may silently fetch a model.

The LLM receives a JSON prompt on stdin and must return JSON `{"segments":[{"index":0,"text":"…","language":"de|ru|mixed|unknown","gap":false}]}` for every requested index. For gaps, set `gap:true` and include `[пропуск]` in text. The example uses the `codex exec` `--output-last-message` option, replacing `{output}` in its arguments with the batch result path. Alternative authorized CLI programs may be configured; authentication and model choice belong to the CLI.

To add subject-specific spelling or formatting rules, set `review_rules = "/absolute/private/rules.txt"` at the **top level** of the per-user `config.toml`. The file is read as text, capped at 64 KiB and included in each LLM request. Rules must not override the no-invention requirement.

The process boundary keeps the Rust scheduler independent from model frameworks. Changing a model within a compatible engine requires a different repo/revision/path in the config, not a Rust change. FFmpeg, Python adapters and the LLM CLI remain independent installations. HTTP API, voice enrollment and Windows setup are later work.
