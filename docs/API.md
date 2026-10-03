# Agent API and headless frontend

Shum exposes a local, non-interactive JSON-lines API. The CLI and HTTP
frontend use the same workflow checkpoints, project locks and path
boundaries as interactive Shum. These are synchronous operations, not a
durable HTTP job queue. Models and provider authentication are never
installed or repaired implicitly by a read-only request.

## JSON-lines transport

Build/install Shum first, then select the installation or configuration:

```bash
printf '%s\n' \
  '{"version":1,"id":"status-1","method":"installation.status","params":{}}' \
  | shum --installation "$HOME/shum-local" api
```

One UTF-8 JSON object per line; each line produces one response on stdout.
Diagnostics go to stderr. A request has `version: 1`, a nonempty string
`id` of at most 128 UTF-8 bytes, a `method`, and an object `params`.
Unexpected request/parameter fields fail explicitly. Include `params`
even when empty for compatibility with the headless frontend.

Success:

```json
{"version":1,"id":"status-1","result":{"state":"configured"}}
```

The example omits additional diagnostic fields. `configured` means the
configuration and installation layout can be loaded, **not** that ASR,
all model files or external LLM access are ready.

Failure:

```json
{"version":1,"id":"status-1","error":{"code":"invalid_params","message":"unknown configured model role","data":null}}
```

Inspect `result` or `error`, not just process status. The API process exits
1 if any line fails, but keeps accepting later lines. An oversized or
unparseable line has `id: null` because correlation cannot be trusted.
Requests are limited to 64 KiB including the line ending; responses to
1 MiB. A single workflow operation may take hours.

## Methods

| Method | Parameters | Effect |
| --- | --- | --- |
| `installation.status` | `{}` | Read configuration/layout state; no network or LLM probe |
| `installation.setup` | `installation`, `bin`, `settings`, `projects`, `models`: absolute directory strings; optional `command_dir`: absolute directory string, `setup_shell`: boolean (requires `command_dir`) | Explicit non-interactive layout setup; no model downloads/provider login |
| `models.verify` | Optional `role` | Offline pinned-revision/file-integrity verification; inspect `all_verified` and each model |
| `project.status` | `project`: name or path under configured projects root | Read workflow stages |
| `project.plan` | `project` | Read known execution graph, attempts and progress without starting work |
| `project.events` | `project`, optional `after` (last seen event sequence), `limit` (page size) | Read durable events after a sequence; use `next_after` to reconnect |
| `workflow.control_preview` | `project`, `request`: versioned control request | Validate and preview its effects without applying them |
| `workflow.control` | `project`, `request`: versioned control request | Submit an explicitly confirmed control request |
| `project.check_quality` | `project` | Rescan existing project artifacts offline; returns a compact advisory summary, never invokes ASR/LLM |
| `project.quality_report` | `project`, optional `offset` (default 0), `limit` (default 10, 1–50) | Read a page of saved per-window diagnostics without recomputing |
| `workflow.start` | `input`: MP3 path, `project`, optional `sequential_asr` | Start or continue complete workflow |
| `workflow.resume` | `project`, optional `input`, `sequential_asr` | Resume valid checkpoints; replacement input must have identical content |
| `workflow.rerun` | `project`, `step`, optional `sequential_asr` | Invalidate selected stage and descendants, then continue complete workflow |
| `workflow.step` | `project`, `step`, optional `input`, `sequential_asr` | Run selected stage and missing prerequisites only |

Supported steps: `audio`, `chunks`, `asr:<role>` for each configured ASR
role, `diarization`, `review-flags`, `review`, `export`. Query
`installation.status` for the current `steps` list.
New projects created with `workflow.step` require `input`.
Workflow methods also accept optional `no_quality: true` to disable automatic
advisory diagnostics for that invocation. They never perform diagnostic-driven
ASR retries. `project.status` includes an optional `quality` summary; saved
reports are marked `not_revalidated`, not asserted to be current.

Workflow methods accept optional `asr: ["meeting", "interview"]`: a nonempty
array of unique configured ASR IDs. Each ID needs one `[asr.ID]` adapter and
exactly one matching `[[models]]` entry. There may be one, two, five or more
ASR IDs, including multiple IDs using the same engine; the engine and
adapter must support each model's weights. Keep one diarization model.
New IDs must specify `engine = "nemo"`, `"whisper"` or `"qwen"`; the legacy
`primeline`, `flix` and `qwen` IDs may omit it.

Without `asr`, **every invocation** (start, resume, step or rerun) selects
all ASR IDs in the current config, ordered `primeline`, `flix`, `qwen`
when present, then other IDs alphabetically. An explicit selection applies
only to that invocation. Project status records the selection actually used,
not a default for later runs, including when reading old checkpoints.
Changing the selection preserves cached ASR outputs and invalidates
dependent review and export results. Inactive IDs are absent from LLM
hypotheses and exported models; quality reports list them as `skipped_roles`
rather than `missing_roles`. New status files always include `asr_roles`
with the roles used on the last run. Older files without that field are
read as the legacy `primeline`, `flix`, `qwen` selection. Inactive stages
are `skipped`. A requested `asr:ID` step outside an explicit
selection fails; include that ID in `asr` to run it.

For example, the same API project can run only the `meeting` model now
and return to all configured models on its next invocation:

```json
{"version":1,"id":"one","method":"workflow.start","params":{"project":"meeting","input":"/absolute/path/recording.mp3","asr":["meeting"]}}
```

Set `[resources].max_asr_workers` for the total ASR process cap and optional
`max_parallel_asr_models` for the maximum number of distinct active IDs.
`sequential_asr_roles` serializes distinct models. Supply
`[resources.estimated_peak_mib]` for each active ID; estimates for removed
or inactive historical IDs may remain in the map. These memory figures are
admission estimates, not hard RAM limits.

Quality reports always mark transcription accuracy `unverified`.
Each window's text `analysis.value.source_status` distinguishes
`no_sources`, `single_asr_source` and
`multiple_asr_sources_unverified`; none establishes correctness.
One hypothesis has no cross-model corroboration. Agreement between
multiple hypotheses does not establish accuracy or model independence.
The review prompt tells the LLM both limitations and contains only the
selected IDs' hypotheses.
Quality computation budgets (including `quality.max_chunks` and
`quality.text.max_hypotheses`) bound diagnostics, **not** the number of
models a workflow can run. Text analysis keeps per-role metrics for
every available source; `quality.text.max_hypotheses` (default 8) limits
only pairwise comparisons. `analysis.value.pairwise_omitted_roles` names
sources outside that comparison budget, and the
`pairwise_hypothesis_budget` info finding explains the reduced pairwise
coverage. For 12 sources at the default budget, the report has 12 role
metrics, 28 pairwise comparisons, and four omitted pairwise roles.
Review and export still receive all 12 hypotheses.
`execution_status` describes diagnostics only (`complete`, `partial`, `failed`);
missing inputs and optional telemetry are explicit. `project.check_quality`
forces a fresh prepared-audio scan even if automatic diagnostics are disabled.
It does not run missing prerequisites, contact the network or rewrite ASR,
review, workflow checkpoints or source-named exports. Diagnostic failures use
`diagnostics_failed`; suspicious-output findings are advisory, not API failures.
Report schema/rules versions, numerical evidence and policy/source fingerprints
are stored in a private project `.quality/` sidecar. Automatic audio cache
validation uses file metadata; explicit checks bypass it.
`workflow.rerun` does not accept `input`: rebind through resume first.

### Plans, progress and manual controls

`project.plan` returns `{"project": "...", "plan": {...}}`. The plan
contains `schema_version`, `revision`, `stages`, `attempts`, `decisions`,
`skipped`, `budget`, `queued`, `metadata` and `project_root`.
Metadata lists the selected engines, external services and concurrency settings.
Each stage has `status`,
`selected_attempt`, `active_attempt`, `units`, `done`, `unit`,
`numeric_done`, `numeric_total` and `operation`. Progress covers
computation in the *current known plan*, not transcription accuracy.
A revised plan or an added check may
change the denominator; a missing estimate is unknown, not zero remaining
time or a fabricated ETA. A completed attempt does not make blocked
prerequisites or an unverified final transcript complete.
The ASR stages follow the selected project roles rather than a fixed
three-model list: one or five configured roles produce different plans.
For diarization, `numeric_done` and `numeric_total` reflect the existing
pyannote embedding hook's completed and total batches when reported.
The adapter emits
`{"event":"shum.progress","version":1,"phase":"embeddings","completed":N,"total":T}`
as one JSON line on stderr. If it has not reported a total, Shum cannot
calculate embedding completion; elapsed time is not evidence that the
speaker model has stalled. The progress signal measures inference
batches, not the quality of speaker assignments. Segmentation may report
the same event shape with `"phase":"segmentation"` when its hook supplies
counts; absent counts must remain unknown.

`project.events` returns `{"project":"...","after":0,"next_after":5,
"events":[...]}`. Each event has `version: 1`, `sequence`, `kind`,
`step`, `attempt_id`, `message` and `payload` (the three identifiers/text
fields may be null). Payloads carry the affected stage, attempt and decision
when applicable. To reconnect, first fetch `project.plan`, then fetch events
after that snapshot's `revision`. Re-read the snapshot for fine-grained
progress; the journal does not store terminal redraws.
Default `after` is 0 and `limit` is 100; `limit` must be 1–500.
After processing a page, pass its `next_after` as `after` when
reconnecting. Events record attempts and selections so history is not
silently replaced when work is retried. A short or empty page is normal;
for an empty page `next_after` equals the requested `after`.

Controls use a nested `request` object. Its version is independent of
the API envelope version:

```json
{"version":1,"id":"preview-1","method":"workflow.control_preview","params":{"project":"meeting","request":{"version":1,"request_id":"retry-window-3","expected_revision":4,"initiator":{"kind":"user"},"reason":"Inspect window 3 again","evidence":[{"window":3}],"confirmed":false,"action":{"action":"retry","step":"asr:flix","fragments":[3],"parameters":{"threads":2,"max_in_flight":1}}}}}
```

Take `expected_revision` from `project.plan.plan.revision` and choose a
unique `request_id` (ASCII letters, digits, `-` or `_`, at most 128 bytes).
`reason` must be nonempty; `evidence` is a list of JSON values.
`initiator.kind` is `user`, `rule` (with `rule`) or `model` (with `model`,
`policy_version`, optional probability `confidence`). Supported tagged
actions are `retry` (`step`, nonempty `fragments`, `parameters` with
optional positive `threads` and `max_in_flight` of 1–4), `skip` (`step`),
`cancel_attempt` (`attempt_id`), `restore` (`step`), `add_check` (`step`,
`dependencies` of attempt IDs, `fragments`), `select` (`attempt_id`) and
`invalidate` (`step`), plus `dependencies` (`attempt_id`, `dependencies`).
Dependencies can change only while an attempt is pending and may reference
only earlier attempts, so execution dependencies remain acyclic.
Retry currently supports ASR fragments only. `add_check` targets an ASR
stage or `review-flags` and creates a separate `check:asr-ID` or
`check:review-flags` logical stage. Repeated checks create new attempts
of that stage; skipping it prevents automatic recreation until restored.
Its inputs are frozen ASR artifact versions; changing the selected result
does not silently change an already scheduled check's inputs.
Empty check `fragments` means all known target fragments.
The `confirmed` flag is required when *submitting* a user `skip` or
`cancel_attempt`; previews can show those consequences before confirmation.
The preview returns `preview.revision`,
`affected_steps`, `fragments` and `parameters`. Submit the same request
with method `workflow.control` to receive a `receipt` containing
`request_id`, `revision` and `state`; a queued receipt is not proof the
action has finished. Applied or rejected receipts may also include
`error`. Re-read plan and events to observe its application.

Control is operator-initiated. Preview first, then submit a confirmed
request at the expected graph revision. A stale revision, invalid/cyclic
dependencies or an exhausted work budget must fail explicitly rather than
modify the graph. Retry scopes its invalidation to requested work;
cancelling an attempt is not the same as skipping the logical step.
Skipping required preparation can block downstream work. Skipping
diarization leaves a documented limitation, while skipping LLM review
cannot assert a verified final transcript. Restore is an explicit
operator action; resume alone must not undo an explicit skip. A request
changes project execution state, not private model/provider settings.
The default automatic work budget is three attempts and 600 seconds across
all rule/model retries and supplementary checks. Measured elapsed time is
charged once per attempt and retained across resume. Existing OOM recovery
records a `resource-recovery` rule decision and restarts only missing
fragments; it does not reset the failed attempt's history. Exhaustion stops
automatic work even if a decision model requests another attempt.
This mechanism does not enable a quality controller or establish audio
accuracy. User requests still require explicit confirmation where applicable.

Neither progress nor advisory diagnostics establish which words were
spoken. The local ASR and diarization adapters use project audio; review
may transmit selected transcript text to the configured external
provider. Source-named exports retain the original audio stem and
legacy `transcript.*` files migrate on resume/export without discarding
valid upstream checkpoints.

Review responses accept pure JSON or exactly one complete triple-backtick
JSON block (labelled `json` or unlabelled). No surrounding prose or extra
blocks are allowed. Removing the wrapper does not relax schema or batch
validation. The original UTF-8 response is retained in a private
`*.response-*.txt` file alongside the review response, including on
validation failure. The API still returns one JSON response per request;
provider Markdown never leaks into stdout.
Resume accepts this same form in saved review batches from older versions.
After all batch checks pass, it preserves the fenced original and atomically
replaces the cache with normalized JSON, without repeating ASR or the
provider request. Invalid saved responses remain unchanged.

For `installation.setup`, omit global `--config`/`--installation`
selectors. Supply all five independent paths, outside the source tree.
Existing foreign layouts, orphan configurations and conflicting binary
metadata fail rather than being overwritten. Fully explicit setup ignores
ambient installation discovery, including inherited `SHUM_SETTINGS`.
Existing global installations remain selected in their global locator.
By default API setup does not register `shum`, create a command in a user
directory, or edit shell startup files. Set `command_dir` explicitly to
register it; set `setup_shell: true` only with `command_dir` to add a missing
`PATH` entry to supported Bash/Zsh startup files. After setup use the
returned settings/config path or installed executable:

```json
{
  "version": 1,
  "id": "install-1",
  "method": "installation.setup",
  "params": {
    "installation": "/srv/shum-local",
    "bin": "/srv/shum-local/bin",
    "settings": "/srv/shum-local/config",
    "projects": "/srv/shum-projects",
    "models": "/srv/shum-models"
  }
}
```

This establishes a layout, **not** a ready transcription installation.
Configure environments, pinned models and optional review provider using
the [usage guide](USAGE.ru.md). API setup does not prompt, accept model
terms, authenticate providers or infer consent to downloads.
To repair command registration for an existing installation without
reinstalling or probing models/providers, run
`/absolute/installed/bin/shum setup-command --command-dir /absolute/directory`.
Pass `--installation /absolute/installation/root` before `setup-command`
when needed. The CLI prints the command path and an `export PATH=...`
instruction if the directory is absent from the current terminal's `PATH`;
a child process cannot update its parent shell. A command named `shum`
resolves through `PATH`, not the current working directory (`./shum`
explicitly names a file in that directory).

## Workflow outcomes and errors

Workflow results and `project.status` include `exports: {"txt": "...",
"json": "..."}` with expected final paths. These filenames preserve the
original audio stem: `recording.mp3` produces `recording.txt` and
`recording.json`, not the project name. Rebinding identical audio under a
different filename does not rename the exports. Files normally live in
the project root; names colliding with internal JSON files use its
`exports/` subdirectory. Paths alone do not mean files are ready: inspect
the export stage and outcome. Legacy projects retain old paths until
resume/rerun migrates their export; valid upstream checkpoints are reused.

Workflow responses contain `outcome`, child `exit_code`, project path,
workflow `status`, and a private `log_directory` containing stdout/stderr.
These local diagnostic logs are not a public endpoint. Retain or remove
them according to your local policy; project checkpoints are separate.

- `complete`, child exit 0: full workflow completed.
- `stage_complete`, child exit 0: requested operation completed; it does
  **not** imply a final transcript. Check stage status when requesting export.
- `pending_review`, child exit 2: local results saved; external review is
  unavailable/disabled. API transport exits 0 because this is a valid,
  explicit waiting outcome, not a fabricated transcript.
- `workflow_failed`: child failed; error data contains child exit code,
  project/log locations. Inspect checkpoints before resuming.

Core error codes include `invalid_request`, `unsupported_version`,
`request_too_large`, `response_too_large`, `invalid_params`,
`method_not_found`, `setup_required`, `invalid_project`, `control_rejected`,
`execution_failed`, and `workflow_failed`. Transport errors use uppercase
codes such as `UNAUTHORIZED`, `INVALID_JSON`, `TIMEOUT`, and
`INVALID_RESPONSE`. Clients should preserve unknown future codes.

Read-only calls never test external LLM access. Workflow review can call
the configured provider and transmit transcript text; enable it only with
permission to share that text. Stage-only local ASR/audio calls do not run
LLM startup probes. Full workflow planning and local stages also defer any
provider readiness request until review actually needs the provider; a
plan read cannot transmit project text. The API offers no arbitrary shell,
configuration-edit,
model-update or file-read endpoint.

## Headless CLI frontend

The bundled stdlib Python frontend requires Linux and Python 3.11+.
Use the copy under the installed versioned runtime, or run from source:

```bash
python3 scripts/shum_headless.py --shum "$HOME/shum-local/bin/shum" \
  --config "$HOME/shum-local/config/config.toml" call \
  --request '{"version":1,"id":"p1","method":"project.status","params":{"project":"meeting"}}'
```

`call --request-file request.json` reads a file; plain `call` reads one JSON
request from stdin. JSON is the only stdout output. Exit 0 means a valid
result; exit 1 means a core or transport error. `--timeout SECONDS` is
optional: no short default is imposed on long transcription. An explicit
timeout stops the owned API process group; use project status/resume to
recover checkpoints. External systemd-managed adapters require separate
supervision and are not durable frontend jobs.

## Authenticated loopback HTTP frontend

Create an owner-only token directory, then run:

```bash
mkdir -m 700 "$HOME/shum-headless-auth"
python3 scripts/shum_headless.py --shum "$HOME/shum-local/bin/shum" \
  --config "$HOME/shum-local/config/config.toml" \
  serve --host 127.0.0.1 --port 8765 \
  --token-file "$HOME/shum-headless-auth/token"
```

The token file is created with mode 0600 and contains 64 lowercase hex
characters. Existing token files must be owner-only regular files, not
symlinks. The frontend does not print the token. Only loopback IPv4 binds
are accepted; never expose this service through a public reverse proxy.
Every route requires `Authorization: Bearer TOKEN`, including health.
Tokens in URLs/cookies are not accepted; browser `Origin` requests and
CORS are not supported.

```bash
TOKEN="$(cat "$HOME/shum-headless-auth/token")"
curl --fail-with-body http://127.0.0.1:8765/v1/requests \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  --data '{"version":1,"id":"h1","method":"installation.status","params":{}}'
```

`POST /v1/requests` takes one envelope. A valid core error envelope uses
HTTP 200: inspect `error`. Malformed/auth/transport requests use HTTP
4xx/5xx. `GET /health` reports transport readiness only; it does not
verify installation, models or login. There is no audio upload, browser
UI, server-side job listing or persistent background queue.
