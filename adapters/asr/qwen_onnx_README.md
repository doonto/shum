# Qwen3-ASR alternative CPU engines

These are opt-in experiments, not production defaults. Neither a speed
pass nor plausible text proves audio accuracy. Application tests and
builds were intentionally deferred for this experiment round.

## ONNX Runtime INT4

`qwen_onnx_worker.py` bridges the Apache-2.0
[qwen3-asr-onnx runtime](https://github.com/andrewleech/qwen3-asr-onnx)
to the indexed diagnostic worker protocol. Keep its source and license
outside the checkout and pin its commit. It requires all three
`*.int4.onnx` graphs, `decoder_weights.int4.data`, `embed_tokens.bin`,
`config.json` and `tokenizer.json`. It never falls back to another
precision or downloads missing files.

The published
[1.7B ONNX model](https://huggingface.co/andrewleech/qwen3-asr-1.7b-onnx)
uses an FP32 audio encoder, INT4 decoder weights and FP16 token embeddings;
it is not an all-INT4 graph. The adapter memory-maps embeddings and casts
only selected token rows to FP32. Encoder inference, KV-cache prefill and
autoregressive decoding use `CPUExecutionProvider`. By default the experiment
forces German through tokenizer prefill. `--language auto` instead lets the
model generate its language header and requires the ASR text separator.
Both profiles reject generation without EOS.
Timestamps cover the whole WAV window, not individual words.

For this round the public model revision is
`df916193ac67e59347769891a21e10d81d12acdd`, runtime commit
`4aee2ae0c355c7bc33bd5c09befec9e9eabd4b50`. The publisher names the
original `Qwen/Qwen3-ASR-1.7B` as its base model but does not pin the
upstream weight revision. Verified LFS/artifact hashes identify the files
used; they do not establish numerical identity of the quantized weights
to our original checkpoint.

Supply explicit model/source/environment paths:

```bash
python scripts/asr_engine_study.py \
  --manifest /absolute/corpus.json --output /absolute/new-run \
  --full-corpus --source-description /absolute/source.json \
  --estimated-peak-mib 7168 --max-in-flight 1 --worker-command \
  /absolute/onnx-env/bin/python -u \
  /absolute/shum/adapters/asr/qwen_onnx_worker.py \
  --model /absolute/qwen3-asr-1.7b-int4 \
  --source /absolute/pinned-qwen3-asr-onnx \
  --dependency-site /absolute/torch-env/lib/python3.11/site-packages \
  --threads 6
```

Add `--diagnostics /absolute/new-run/diagnostics` to retain per-request token
IDs, raw decoded text, language policy and EOS status in a private directory,
including failed hypotheses. These files contain transcription content;
do not publish them as generic benchmark telemetry. Filenames use request
numbers, not corpus window indices; the runner's requests and summary map
between them. Missing EOS remains an error, never a truncated success.
Use a fresh directory for each run: diagnostic files are not overwritten.

The ONNX environment needs compatible NumPy and ONNX Runtime. The
dependency site supplies PyTorch, librosa and tokenizers without changing
an existing inference environment. Record all package versions and
native libraries in provenance. Do not resolve a virtualenv Python
symlink when invoking it.

Use the same frozen 67-window / 1800-second corpus for each engine.
The 600-second deadline includes worker/model loading and shutdown.
Corpus/artifact verification is preparation, reported separately.
Admission reserves 4 GiB; process swap growth and memory PSI combined
with new system swap-out stop unsafe runs. A stop during model loading
is a resource failure, not a measured 600-second throughput failure.

## Local OpenVINO export

`scripts/qwen_openvino_experiment.py` exports the locally pinned
Transformers `Qwen/Qwen3-ASR-1.7B-hf` packaging. Export audio first
(`--part audio`), then decoder layers (`--part layers --int8`).
The latter also writes memory-mapped FP16 token embeddings. Supply
`--model`, a fresh `--output`, and `--openvino-site` explicitly.
Only the independent OpenVINO/NNCF packages are appended to the existing
Qwen Python environment; they do not replace its Transformers version.

The single large decoder export exceeded safe memory headroom. Splitting
it into 28 cached layer graphs and a head reduced export memory, but
runtime compilation/inference still requires its own measurement. Do not
treat successful conversion of the audio encoder or individual layers as
successful full ASR. Audio-wrapper and uncompressed-graph numerical
checks are conversion checks, not an application regression suite or
post-quantization quality acceptance.

`scripts/qwen_checkpoint_compare.py` performs a bounded bitwise audit of
original BF16 safetensors and HF packaging, including the original tied
LM head. Matching all source parameters proves source parameter
identity only; it does not prove equal preprocessing, decoding, converted
weights or runtime output.

## Measured round outcome

All full attempts used the frozen 1800-second / 67-window corpus on the
i7-12850HX CPU. Times below include loading and owned-process cleanup;
model export and separately recorded preparation are excluded.

| Runtime/profile | Completed | Measured time | Result |
| --- | --- | --- | --- |
| ONNX Runtime 1.30.0 INT4, German, 6 threads | 5/67 | 93.702 s | Window 5 reached 448 tokens without EOS |
| ONNX Runtime 1.30.0 INT4, automatic language, 16 threads | 1/67 | 58.862 s | Window 1 reached 448 tokens without EOS |
| OpenVINO layerwise INT8, first four-window pilot | 0/4 | 10.334 s | Resource abort, peak RSS 10.68 GiB |
| OpenVINO layerwise INT8, FP16 mmap embedding mitigation | 0/4 | 10.365 s | Resource abort, peak RSS 9.84 GiB |
| Original llama.cpp Q8_0, historical three fresh-server full runs | 67/67 each | 473.199 / 433.998 / 427.721 s | Three-run speed gate passed; quality unverified |

A separate German window-5 diagnostic reproduced failure in 47.838 seconds:
one token repeated 405 consecutive times. The automatic-language profile
generated a Chinese header and repeated a four-token sequence 106 times
on window 1. Increasing the token limit is therefore not justified by a
legitimate long transcript. The exact cause of the published-graph/runtime
failure is not established; do not attribute it to quantization alone.
The ONNX attempts failed decoding, **not** the 600-second timeout.

Local OpenVINO conversion produced the FP32 audio graph and all 28 INT8
decoder layers plus the head. The layer export supervisor completed in
42.384 seconds at 2.73 GiB peak RSS. The audio conversion wrote its graph
in 5.809 seconds, but its supervisor was subsequently resource-aborted
during teardown, so that whole attempt was not successful. Full decoder
conversion exceeded safe headroom; an FP16 attention-mask dtype mismatch
was corrected before the subsequent resource-limited attempt. Both
layerwise inference pilots stopped on memory PSI plus new system swap-out
before producing a window. Export success is not full ASR acceptance.

Reject these ONNX/OpenVINO profiles for the current corpus and resource
constraints. Do **not** exclude Qwen3-ASR-1.7B itself on speed: the existing
llama.cpp profile meets the requested limit in three measured full runs.
Accuracy still needs an audio-grounded reference; no experimental profile
has been promoted or installed. Application tests and builds remain
deferred as requested.
