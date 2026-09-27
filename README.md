# shum

CPU-only CLI for long German, Swiss German and occasional Russian MP3 recordings. The Rust program prepares audio, starts three independent local ASR passes at the same time, assigns speakers with local pyannote, and asks an authenticated LLM CLI to compare the resulting texts. The final German text uses Standard German; Russian stays Russian. Unrecoverable speech is marked `[пропуск]`. The three raw hypotheses remain in the JSON output.

**Status:** early implementation. The Rust unit tests and adapter mock tests do not establish accuracy or 1–2-hour throughput. Run a short representative recording before relying on output. An LLM that only reads ASR text cannot prove that nothing was missed in audio; inspect intervals marked as gaps.

## Install on Ubuntu

1. Install `ffmpeg`, Rust (`cargo`) and `uv` through their normal package channels. Install and authenticate an LLM CLI; the example uses `codex`. It sends **transcript text** to its configured provider, never the audio file. Never run the LLM against recordings without permission to share their text.
2. Install the three ASR engine environments and the diarization environment separately as described in [ASR installation](adapters/asr/README.md) and [diarization installation](adapters/diarization/README.md). Dependencies are isolated from the OS Python; each `[asr.<role>]` config section points to its own interpreter.
3. For pyannote, accept the conditions at [speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) and authenticate the Hugging Face CLI: `hf auth login`. This token is not copied into the repository.
4. `cp shum.example.toml shum.toml`. Set adapter interpreter and CLI paths if they differ from the example. Run `cargo build --release`.
5. `cargo run -- models update --role primeline`; repeat for `flix`, `qwen` and `diarization`. A download may take several gigabytes. The original repositories and immutable revisions are recorded in `shum.toml`. Model weights and recordings are ignored by Git. Run `cargo run -- doctor` **after** installing the models; `doctor` treats missing model directories as errors.

`cargo run -- models check` reports available newer Hugging Face commits. `cargo run -- models update --role qwen` downloads the current revision separately and activates it only after validation. `cargo run -- transcribe recording.mp3 --output jobs/recording` checks on startup, **without updating** the weights. If the network is down, the pinned local revision still works with a warning; without that revision, the command fails.

The output directory retains decoded audio, clips, adapter responses, stderr, reviews and a manifest so interrupted jobs can resume. `transcript.txt` is for reading; `transcript.json` (schema version 1) contains timestamps, speaker IDs, language, gap flags, each original ASR hypothesis, input SHA-256 and the model revisions. Do not publish the job directory: it contains sensitive audio and text.
The JSON quality status is `needs_audio_review` when gaps remain, otherwise `text_reviewed_audio_not_verified`; neither claims the LLM has checked the recording itself.

## Adapter protocol

One persistent ASR process per model receives `--engine nemo|whisper|qwen --model PATH --threads N --serve`. It reads newline-delimited `{"input":"/absolute/clip.wav"}` requests on stdin and sends one JSON line per clip on stdout: `{"text":"…","language":"de","segments":[{"start":0.0,"end":1.0,"text":"…"}]}`. Times are clip-relative seconds. Each model is loaded once per job, not once per clip. The diarization process receives `--model PATH --input AUDIO.wav --output RESULT.json --threads N` and writes `{"turns":[…],"exclusive_turns":[{"start":0.0,"end":1.0,"speaker":"SPEAKER_00"}]}` with absolute times. Non-zero exits fail the job; no subprocess may silently fetch a model.

The LLM receives a JSON prompt on stdin and must return JSON `{"segments":[{"index":0,"text":"…","language":"de|ru|mixed|unknown","gap":false}]}` for every requested index. For gaps, set `gap:true` and include `[пропуск]` in text. The example uses the `codex exec` `--output-last-message` option, replacing `{output}` in its arguments with the batch result path. Alternative authorized CLI programs may be configured; authentication and model choice belong to the CLI.

To add subject-specific spelling or formatting rules, set `review_rules = "rules.txt"` at the **top level** of `shum.toml`. The file is read as text, capped at 64 KiB and included in each LLM request. Rules must not override the no-invention requirement.

The process boundary keeps the Rust scheduler independent from model frameworks. Changing a model within a compatible engine requires a different repo/revision/path in the config, not a Rust change. FFmpeg, Python adapters and the LLM CLI remain independent installations. HTTP API, voice enrollment and Windows setup are later work.
