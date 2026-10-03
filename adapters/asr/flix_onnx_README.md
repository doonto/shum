# Flix ONNX metadata-only feasibility probe

This standalone checker reads the pinned Flix source's small JSON metadata and
manifest, stats the source weight file, and optionally reads an existing ONNX
graph directory's JSON configs and file sizes. It writes a structured JSON
report **outside** both directories. It does not read model weights or ONNX
external data, load a runtime, export, transcribe, or change the source.
No dependency installation is required.

```sh
/usr/bin/python3 adapters/asr/flix_onnx_probe.py \
  --model /absolute/models/flix/a9c347a904af5d31b28c537ac732e0618aed6136 \
  --graphs /absolute/experiments/onnx-prototype/graphs \
  --report /absolute/separate/directory/flix-onnx-probe.json
```

The pinned source is `Flix-AI/flix-swissgerman-full` at
`a9c347a904af5d31b28c537ac732e0618aed6136`. The probe checks its declared
manifest identity, 128-mel/16 kHz processor, German/transcribe generation
defaults, and presence of tokenizer and weight files. It does **not** rehash
the source weights; the manifest is a declaration, not an integrity attestation.
Matching graph configs, dtype claims, or even a matching provenance declaration
cannot establish numerical BF16-to-FP32 preservation or generation parity.

The report contains export and, when `--graphs` is given, no-cache load gates:

- Export estimate: `max(24 GiB, 8 * source weight bytes) + 4 GiB reserve`.
- Existing no-cache load estimate: `max(16 GiB, 2 * (encoder + decoder external-data bytes)) + 4 GiB reserve`.

The probe compares these against current `/proc/meminfo` `MemAvailable`, capped
by cgroup v2 headroom when available. These are **conservative admission
estimates, not measured peak limits**; they must be rechecked immediately
before any future heavy operation. For the observed source and graphs the gates
require at least 28 GiB and 20 GiB available respectively. The existing graphs
have no pinned-source export provenance; they cannot be approved as the same
checkpoint from metadata alone. With `--graphs`, this metadata-only checker leaves compatibility blocked
even if memory and declared provenance pass. It does not perform the
separate numerical audit or load a runtime.

Exit 0 means `ready_for_export_review` (only when no existing graphs are
requested and the export gate passes); this is **not** export approval. Exit 2
means a `blocked` report was written. Exit 1 means invalid input or inability
to write the report. No INT8, timestamps, segments, or inference is supported.

## Audited diagnostic runtime pilot

[`flix_onnx_pilot.py`](./flix_onnx_pilot.py) is a separate one-window
compatibility experiment, not a persistent ASR adapter. It requires an
explicit local audit JSON covering all 1,259 source parameters and
rechecks source and graph hashes. A bounded streaming audit of the
existing encoder, decoder and decoder-with-past verified all initializers
bitwise against BF16 values widened directly to FP32, including
transpositions. The audit used about 190 MiB peak RSS; this does not
measure runtime memory or prove generation parity.

The pilot never exports, downloads, changes source snapshots, or installs
dependencies. It uses the available official Optimum Whisper loader.
Before loading, its worker verifies a user-systemd cgroup with
`MemoryMax` at most 9 GiB and `MemorySwapMax=0`; the controller enforces
the external 4 GiB reserve and pressure guards. No unsecured load is
attempted when systemd or limits are unavailable.

The initial profile is FP32, no cache/VAD, German transcription, greedy
beam 1, and text-only output. CPU arena, prepacking and graph optimization
are disabled to bound the initial load. Its 128-new-token and 120-second
limits are **artificial pilot caps**, not the normal Flix benchmark.
EOS and possible truncation are reported; text remains unverified.

Invoke the interpreter directly, preserving its virtualenv symlink:

```sh
/absolute/onnx-environment/bin/python /absolute/runtime/adapters/asr/flix_onnx_pilot.py \
  --source /absolute/models/flix/a9c347a904af5d31b28c537ac732e0618aed6136 \
  --graphs /absolute/experiments/onnx-prototype/graphs \
  --audit /absolute/private/flix-onnx-identity-audit.json \
  --input /absolute/corpus/00023.wav \
  --report /absolute/private/new-onnx-pilot.json \
  --threads 6 --seed 0 --run
```

Omit `--run` for metadata-only validation. The first actual launch exposed
a tokenizer serialization mismatch between the source's Transformers-5
format and installed Transformers 4.57.6. The explicit in-memory adapter
preserves all 107 nonempty extra special tokens and validates all 1,609
added-token IDs and the 51,866-token vocabulary; it does not overwrite
the source or discard tokens. The nested 128-mel feature configuration
is likewise constructed explicitly. Runtime decoding must still be
verified by a successful bounded pilot, not inferred from these checks.

The second secured launch passed model and processor loading (3.54
seconds) and began generation, then stopped on new system swap-out
together with memory PSI full avg10 of 1.44%. The worker had no own swap,
its cgroup peak was about 5.35 GiB, and cleanup verified it stopped.
No complete decoded hypothesis or speed/accuracy result was produced.
This resource stop is specific to that larger FP32 pilot, not a reason
to stop lower-memory CT2 experiments.

The installed public ORT dynamic quantizer eagerly loads external
FP32 weights and deep-copies the model. A decoder copy alone implies
over 7.2 GiB before quantization arrays, so it does not meet the bounded
4 GiB conversion target. `use_external_data_format` controls output,
not out-of-core loading; shape inference can also write beside its input.
No quantization was launched and source graphs remain unchanged.

[`flix_onnx_audit.py`](./flix_onnx_audit.py) preserves the reproducible
streaming identity check. It uses bounded reads rather than loading
model tensors, requires the pinned source, reports direct/transposed
matches and ambiguity, and enforces a 512 MiB address-space limit in
its standalone process:

```sh
python3 /absolute/runtime/adapters/asr/flix_onnx_audit.py \
  --source /absolute/models/flix/a9c347a904af5d31b28c537ac732e0618aed6136 \
  --graphs /absolute/experiments/onnx-prototype/graphs \
  --report /absolute/private/flix-onnx-identity-audit.json
```

Run the auditor in a fresh process, not inside a heavyweight notebook or
test host. Partial parameter coverage or ambiguous mappings cannot certify
the entire checkpoint. Weight identity still does not establish runtime
arithmetic, timestamp alignment, or audio-grounded quality.
