#!/usr/bin/env python3
"""Opt-in, indexed ONNX Turbo worker for frozen PCM16 WAV clips.

The default preserves historical beam-only decoding. An opt-in profile uses
public Whisper timestamp/fallback generation and faster-whisper Silero VAD.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.turbo_onnx_export import GIB, REVISION, available_bytes, own_cgroup, sha256, source_manifest
import flix_fast_common as common

MAX_NEW_TOKENS = 400
TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def start_resource_monitor() -> threading.Event:
    stop = threading.Event()
    initial_pswpout = int(dict(line.split() for line in
                               Path("/proc/vmstat").read_text().splitlines())["pswpout"])

    def watch() -> None:
        while not stop.wait(0.2):
            try:
                rss_kib, swap_kib = common.process_tree_memory_kib(os.getpid())
                current_pswpout = int(dict(line.split() for line in
                                           Path("/proc/vmstat").read_text().splitlines())["pswpout"])
                psi = common.memory_full_avg10(
                    Path("/proc/pressure/memory").read_text())
                reason = None
                if rss_kib * 1024 > 9 * GIB:
                    reason = "9 GiB worker process-tree RSS budget exceeded"
                elif swap_kib:
                    reason = "worker process-tree swap detected"
                elif available_bytes() < 4 * GIB:
                    reason = "4 GiB effective cgroup/host reserve lost"
                elif current_pswpout < initial_pswpout:
                    reason = "system swap counter reset"
                elif current_pswpout > initial_pswpout and psi >= 1.0:
                    reason = "host swap-out with full-memory PSI"
            except (OSError, ValueError, RuntimeError, common.AdapterError) as exc:
                reason = f"resource monitoring failed: {type(exc).__name__}: {exc}"
            if reason:
                print(f"Turbo ONNX worker stopped: {reason}", file=sys.stderr, flush=True)
                os.kill(os.getpid(), signal.SIGTERM)
                return

    threading.Thread(target=watch, daemon=True, name="turbo-onnx-resource-guard").start()
    return stop


def verify_graphs(graphs: Path, source: Path) -> dict:
    provenance = json.loads((graphs / ".shum-turbo-onnx.json").read_text())
    if (provenance.get("complete") is not True or provenance.get("export_dtype") != "fp32"
            or provenance.get("export_strategy") not in (
                "selective_fp32_encoder_decoder_decoder_past_no_folding",
                "streamed_external_fp32_cached_no_constant_folding")):
        raise ValueError("ONNX export incomplete or not FP32")
    if provenance.get("source") != source_manifest(source):
        raise ValueError("ONNX export/source provenance mismatch")
    hashes = provenance.get("graphs_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Missing export graph hashes")
    if not all(f"{stage}/{name}.onnx" in hashes for stage, name in
               (("encoder", "encoder_model"), ("decoder", "decoder_model"),
                ("decoder-past", "decoder_with_past_model"))):
        raise ValueError("Missing one or more cached decoder graph hashes")
    for name, expected in hashes.items():
        file = graphs / name
        if (file.is_symlink() or not file.is_file()
                or not file.resolve().is_relative_to(graphs)
                or sha256(file) != expected):
            raise ValueError(f"Modified or missing ONNX artifact: {name}")
    return provenance


def verify_quantized_decoders(decoder_graphs: Path, graphs: Path,
                              fp32_provenance: dict) -> dict:
    manifest = json.loads((decoder_graphs / ".shum-turbo-onnx-mixed.json").read_text())
    if (manifest.get("complete") is not True
            or manifest.get("profile") != "fp32_encoder_dynamic_int8_decoders"
            or manifest.get("source") != fp32_provenance["source"]
            or manifest.get("fp32_graphs") != str(graphs)
            or manifest.get("fp32_graphs_sha256") != fp32_provenance["graphs_sha256"]):
        raise ValueError("Mixed ONNX decoder/source provenance mismatch")
    hashes = manifest.get("quantized_sha256")
    expected = {"decoder/decoder_model.onnx", "decoder/decoder_model.onnx.data",
                "decoder-past/decoder_with_past_model.onnx",
                "decoder-past/decoder_with_past_model.onnx.data"}
    if not isinstance(hashes, dict) or set(hashes) != expected:
        raise ValueError("Incomplete mixed ONNX decoder hash inventory")
    coverage = manifest.get("coverage")
    if (not isinstance(coverage, dict)
            or any(not isinstance(coverage.get(stage), dict)
                   or any(type(coverage[stage].get(metric)) is not int
                          or coverage[stage][metric] <= 0
                          for metric in ("matmul_integer", "int8_initializers"))
                   for stage in ("decoder", "decoder-past"))):
        raise ValueError("Mixed ONNX profile has no proven INT8 MatMul coverage")
    for name, expected_hash in hashes.items():
        file = decoder_graphs / name
        if (file.is_symlink() or not file.is_file()
                or not file.resolve().is_relative_to(decoder_graphs)
                or sha256(file) != expected_hash):
            raise ValueError(f"Modified or missing mixed ONNX decoder: {name}")
    return manifest


def audio_samples(path: Path):
    import numpy as np

    if not path.is_absolute() or not path.is_file():
        raise ValueError("Input must be an existing absolute WAV")
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(),
                wav.getcomptype()) != (1, 2, 16000, "NONE"):
            raise ValueError("Input must be mono PCM16 16-kHz WAV")
        frames = wav.getnframes()
        if not 0 < frames <= 30 * 16000:
            raise ValueError("Input must contain 0 < duration <= 30 seconds; no truncation allowed")
        audio = np.frombuffer(wav.readframes(frames), dtype="<i2").astype(np.float32) / 32768
    if len(audio) != frames:
        raise ValueError("Incomplete WAV payload")
    return audio


def load_ct2_vad(site: Path):
    if (not site.is_absolute() or not site.is_dir() or site.is_symlink()
            or not (site / "faster_whisper/vad.py").is_file()
            or not (site / "faster_whisper/assets/silero_vad_v6.onnx").is_file()):
        raise ValueError("CT2 fallback profile requires an absolute faster-whisper 1.2.1 site")
    sys.path.append(str(site))
    import faster_whisper
    from faster_whisper import vad

    if (faster_whisper.__version__ != "1.2.1"
            or not Path(vad.__file__).resolve().is_relative_to(site.resolve())):
        raise ValueError("VAD must come from the supplied faster-whisper 1.2.1 site")
    return vad


def selected_attempts(output, prompt: list[int], eos_ids: set[int],
                      chunk_length: int, processor) -> tuple[str, list[dict], list[dict]]:
    if not isinstance(output, dict) or len(output.get("segments", [])) != 1:
        raise ValueError("Whisper did not return one batch of native timestamp segments")
    segments = output["segments"][0]
    if not segments:
        if output["sequences"][0].numel():
            raise ValueError("Native no-speech skip returned tokens without segments")
        return "", [], []
    text = processor.tokenizer.decode(output["sequences"][0], skip_special_tokens=True)
    attempts = []
    raw = []
    seen = {}
    for segment in segments:
        result = segment["result"]
        key = id(result)
        if key not in seen:
            ids = result["sequences"].tolist()
            if ids[:len(prompt)] != prompt:
                raise ValueError("Timestamp decoder prompt differs from German transcription")
            generated_length = len(ids) - len(prompt)
            reached_eos = bool(eos_ids and any(token in eos_ids for token in ids[len(prompt):]))
            cap_exhausted = generated_length >= MAX_NEW_TOKENS
            attempt = {"token_ids": ids, "reached_eos": reached_eos,
                       "generated_length": generated_length, "max_new_tokens": MAX_NEW_TOKENS,
                       "cap_exhausted": cap_exhausted,
                       "accepted": reached_eos and not cap_exhausted,
                       "selected_temperature": None,
                       "segment_indices": []}
            seen[key] = len(attempts)
            attempts.append(attempt)
        attempt_index = seen[key]
        attempts[attempt_index]["segment_indices"].append(len(raw))
        start, end = float(segment["start"]), float(segment["end"])
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("Native timestamp segment contains non-finite bounds")
        raw.append({"start": start, "end": end, "tokens": segment["tokens"].tolist(),
                    "selected_attempt_index": attempt_index,
                    "outside_collected_chunk": start < 0 or end < start or end > chunk_length / 16000})
    return text, attempts, raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--decoder-graphs", type=Path,
                        help="Verified dynamic-INT8 decoders; the encoder remains FP32")
    parser.add_argument("--decode-profile", choices=("beam-only", "ct2-fallback-vad"),
                        default="beam-only")
    parser.add_argument("--faster-whisper-site", type=Path,
                        help="Existing faster-whisper 1.2.1 site-packages for opt-in Silero VAD")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if (args.decode_profile == "ct2-fallback-vad") != (args.faster_whisper_site is not None):
        parser.error("--faster-whisper-site is required only with --decode-profile ct2-fallback-vad")
    source, graphs = args.source.resolve(), args.graphs.resolve()
    provenance = verify_graphs(graphs, source)
    decoder_graphs = args.decoder_graphs.resolve() if args.decoder_graphs else graphs
    profile = "fp32_encoder_dynamic_int8_decoders" if args.decoder_graphs else "fp32"
    if args.decoder_graphs:
        verify_quantized_decoders(decoder_graphs, graphs, provenance)
    if provenance["source"]["revision"] != REVISION:
        parser.error("Wrong Turbo source revision")
    own_cgroup()
    if available_bytes() < 13 * GIB:
        parser.error("Worker requires 9 GiB estimated RSS plus 4 GiB effective reserve")
    guard = start_resource_monitor()
    if args.diagnostics:
        if not args.diagnostics.is_absolute() or args.diagnostics.exists():
            parser.error("--diagnostics must be an unused absolute directory")
        args.diagnostics.mkdir(mode=0o700)
    import onnxruntime as ort
    import torch
    from optimum.onnxruntime.modeling_seq2seq import ORTModelForWhisper
    from transformers import GenerationConfig, WhisperConfig, WhisperProcessor

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    vad = load_ct2_vad(args.faster_whisper_site) if args.faster_whisper_site else None
    processor = WhisperProcessor.from_pretrained(str(source), local_files_only=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    sessions = {
        name: ort.InferenceSession(
            str((graphs if name == "encoder" else decoder_graphs) / path),
            options, providers=["CPUExecutionProvider"])
        for name, path in (
            ("encoder", "encoder/encoder_model.onnx"),
            ("decoder", "decoder/decoder_model.onnx"),
            ("decoder_with_past", "decoder-past/decoder_with_past_model.onnx"),
        )
    }
    model = ORTModelForWhisper(
        config=WhisperConfig.from_pretrained(str(graphs), local_files_only=True),
        generation_config=GenerationConfig.from_pretrained(str(graphs), local_files_only=True),
        encoder_session=sessions["encoder"],
        decoder_session=sessions["decoder"],
        decoder_with_past_session=sessions["decoder_with_past"],
        use_io_binding=False, model_save_dir=graphs,
    )
    if (model.config.num_mel_bins, processor.feature_extractor.feature_size,
            processor.feature_extractor.sampling_rate) != (128, 128, 16000):
        raise ValueError("Loaded model/processor dimensions differ from source")
    print(f"Pinned Turbo ONNX {profile} CPU worker ready ({args.decode_profile}, beam 3; "
          f"ORT optimization=ORT_ENABLE_ALL, prepacking=enabled, "
          f"intra={args.threads}, inter=1, spin=off; Torch intra=1, inter=1)",
          file=sys.stderr, flush=True)
    try:
        for line in sys.stdin:
            request = {}
            try:
                request = json.loads(line)
                index, input_path = request["index"], Path(request["input"])
                if type(index) is not int or index < 0:
                    raise ValueError("index must be a nonnegative integer")
                audio = audio_samples(input_path)
                vad_spans = None
                vad_chunks = None
                if vad:
                    options = vad.VadOptions(max_speech_duration_s=30,
                                             min_silence_duration_ms=160)
                    vad_spans = vad.get_speech_timestamps(audio, options, sampling_rate=16000)
                    if any(span["start"] < 0 or span["end"] <= span["start"]
                           or span["end"] > len(audio) for span in vad_spans):
                        raise ValueError("Silero returned out-of-bounds speech intervals")
                    chunks, vad_chunks = vad.collect_chunks(
                        audio, vad_spans, sampling_rate=16000, max_duration=30)
                    retained_samples = sum(
                        span["end"] - span["start"] for span in vad_spans)
                    retained_duration = retained_samples / 16000
                    if sum(len(chunk) for chunk in chunks) != retained_samples:
                        raise ValueError("Silero collected audio differs from retained duration")
                    chunks = [chunk for chunk in chunks if len(chunk)]
                else:
                    chunks = [audio]
                    retained_duration = len(audio) / 16000
                eos = model.generation_config.eos_token_id
                eos_ids = {eos} if isinstance(eos, int) else set(eos or ())
                prompt = [model.generation_config.decoder_start_token_id]
                for position, token in processor.get_decoder_prompt_ids(
                        language="de", task="transcribe", no_timestamps=not bool(vad)):
                    if position != len(prompt):
                        raise ValueError("Noncontiguous Whisper decoder prompt")
                    prompt.append(token)
                if len(prompt) != (3 if vad else 4):
                    raise ValueError("Unexpected Whisper prompt length for decode profile")
                texts = []
                attempts = []
                raw_segments = []
                mask_frames = []
                native_no_speech_skips = 0
                for chunk_index, chunk in enumerate(chunks):
                    features = processor(
                        chunk, sampling_rate=16000, return_tensors="pt",
                        return_attention_mask=True,
                    )
                    if "attention_mask" not in features:
                        raise ValueError("Source processor did not return an attention mask")
                    mask_frames.append(int(features.attention_mask.sum().item()))
                    with torch.inference_mode():
                        if vad:
                            output = model.generate(
                                input_features=features.input_features,
                                attention_mask=features.attention_mask,
                                language="de", task="transcribe",
                                return_timestamps=True, temperature=TEMPERATURES,
                                compression_ratio_threshold=2.4, logprob_threshold=-1.0,
                                no_speech_threshold=0.6, condition_on_prev_tokens=False,
                                force_unique_generate_call=False,
                                num_beams=3, use_cache=True,
                                max_new_tokens=MAX_NEW_TOKENS, return_dict_in_generate=True,
                                return_segments=True,
                            )
                        else:
                            output = model.generate(
                                input_features=features.input_features,
                                attention_mask=features.attention_mask,
                                language="de", task="transcribe",
                                return_timestamps=False, num_beams=3, do_sample=False,
                                use_cache=True, max_new_tokens=MAX_NEW_TOKENS,
                                return_dict_in_generate=True,
                            )
                    if vad:
                        chunk_text, chunk_attempts, chunk_segments = selected_attempts(
                            output, prompt, eos_ids, len(chunk), processor)
                        if not chunk_attempts:
                            native_no_speech_skips += 1
                        raw_offset = len(raw_segments)
                        for attempt in chunk_attempts:
                            attempt["chunk_index"] = chunk_index
                            attempt["segment_indices"] = [
                                value + raw_offset for value in attempt["segment_indices"]]
                        offset = len(attempts)
                        for segment in chunk_segments:
                            segment["chunk_index"] = chunk_index
                            segment["selected_attempt_index"] += offset
                        attempts.extend(chunk_attempts)
                        raw_segments.extend(chunk_segments)
                        texts.append(chunk_text)
                    else:
                        ids = output.sequences[0].tolist()
                        if ids[:len(prompt)] != prompt:
                            raise ValueError("Returned decoder prompt differs from requested German transcription")
                        generated_length = len(ids) - len(prompt)
                        reached_eos = bool(eos_ids and any(
                            token in eos_ids for token in ids[len(prompt):]))
                        cap_exhausted = generated_length >= MAX_NEW_TOKENS
                        attempts.append({"chunk_index": chunk_index, "token_ids": ids,
                                         "reached_eos": reached_eos,
                                         "generated_length": generated_length,
                                         "max_new_tokens": MAX_NEW_TOKENS,
                                         "cap_exhausted": cap_exhausted,
                                         "accepted": reached_eos and not cap_exhausted})
                        texts.append(processor.tokenizer.decode(
                            ids, skip_special_tokens=True))
                accepted = all(attempt["accepted"] for attempt in attempts)
                if args.diagnostics:
                    details = {"index": index, "input": str(input_path),
                               "selected_attempts": attempts,
                               "max_new_tokens": MAX_NEW_TOKENS,
                               "accepted": accepted,
                               "language": "de", "task": "transcribe", "beam": 3,
                               "decode_profile": args.decode_profile,
                               "fallback": bool(vad), "vad": bool(vad),
                               "timestamps": bool(vad),
                               "temperatures": list(TEMPERATURES) if vad else None,
                               "compression_ratio_threshold": 2.4 if vad else None,
                               "logprob_threshold": -1.0 if vad else None,
                               "no_speech_threshold": 0.6 if vad else None,
                               "condition_on_prev_tokens": False if vad else None,
                               "native_no_speech_skips": native_no_speech_skips,
                               "vad_empty_skip": bool(vad and not chunks),
                               "vad_speech_samples": vad_spans,
                               "vad_chunk_metadata": vad_chunks,
                               "duration_after_vad_seconds": retained_duration,
                               "raw_segments": raw_segments,
                               "raw_timestamp_reference": "collected_vad_chunk" if vad else None,
                               "fallback_attempts_exposed": "selected_only" if vad else "single_call",
                               "discarded_fallback_attempts_observable": False if vad else None,
                               "precision_profile": profile,
                               "attention_mask_source": "source_processor",
                               "attention_mask_valid_frames_per_chunk": mask_frames,
                               "ort_intra_op_threads": args.threads,
                               "ort_inter_op_threads": 1,
                               "ort_idle_spinning": False,
                               "ort_graph_optimization": "ORT_ENABLE_ALL",
                               "ort_prepacking": "enabled",
                               "torch_intra_op_threads": torch.get_num_threads(),
                               "torch_inter_op_threads": torch.get_num_interop_threads(),
                               "raw_segments_available": bool(vad),
                               "output_segment_policy": "whole_input_window",
                               "input_duration_seconds": len(audio) / 16000}
                    if not vad:
                        details.update({
                            "token_ids": attempts[0]["token_ids"],
                            "reached_eos": attempts[0]["reached_eos"],
                            "generated_length": attempts[0]["generated_length"],
                            "cap_exhausted": attempts[0]["cap_exhausted"],
                            "attention_mask_valid_frames": mask_frames[0],
                        })
                    (args.diagnostics / f"{index:05d}.json").write_text(
                        json.dumps(details, ensure_ascii=False, indent=2) + "\n")
                if any(attempt["cap_exhausted"] for attempt in attempts):
                    raise ValueError("Generation exhausted the token cap; EOS may be forced at max length")
                if not accepted:
                    raise ValueError("Generation did not reach EOS; hypothesis incomplete")
                text = "".join(texts).strip()
                result = {"index": index, "text": text, "language": "de",
                          "segments": [{"start": 0.0, "end": len(audio) / 16000, "text": text}]}
            except (ValueError, RuntimeError, OSError, KeyError, IndexError, TypeError,
                    AttributeError, wave.Error,
                    json.JSONDecodeError) as exc:
                result = {"index": request.get("index") if isinstance(request, dict) else None,
                          "error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        guard.set()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, ImportError, json.JSONDecodeError) as exc:
        print(f"Turbo ONNX worker blocked: {exc}", file=sys.stderr)
        sys.exit(2)
