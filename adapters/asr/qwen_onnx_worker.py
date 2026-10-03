#!/usr/bin/env python3
"""Indexed offline diagnostic bridge to a pinned qwen3-asr-onnx implementation.

No graph or tokenizer fallback: the requested INT4 artifacts must all exist.
The published exporter/runtime and its Apache license remain in the supplied
source directory; this adapter does not vendor third-party implementation.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dependency-site", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=448)
    parser.add_argument("--language", choices=["German", "auto"], default="German")
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()
    if args.threads < 1 or args.max_tokens < 1:
        parser.error("threads and token limit must be positive")
    sys.path.append(str(args.dependency_site))
    sys.path.insert(0, str(args.source))
    import numpy as np
    import onnxruntime as ort
    import torch
    from tokenizers import Tokenizer
    from src.inference import greedy_decode_onnx
    from src.mel import log_mel_spectrogram
    from src.prompt import build_prompt_ids, EOS_TOKEN_IDS

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    config = json.loads((args.model / "config.json").read_text())
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    for text, expected in (("<|audio_pad|>", 151676), ("<asr_text>", 151704),
                           ("<|im_end|>", 151645), ("<|endoftext|>", 151643)):
        if tokenizer.token_to_id(text) != expected:
            raise ValueError(f"tokenizer special token mismatch: {text}")
    shape = config.get("embed_tokens_shape") or [
        config["decoder"]["vocab_size"], config["decoder"]["hidden_size"],
    ]
    storage = config.get("embed_tokens_dtype", "float32")
    if storage not in ("float16", "float32"):
        raise ValueError(f"unsupported embedding storage: {storage}")
    dtype = np.float16 if storage == "float16" else np.float32
    embedding_file = args.model / "embed_tokens.bin"
    if embedding_file.stat().st_size != int(np.prod(shape)) * np.dtype(dtype).itemsize:
        raise ValueError("embedding table shape/length mismatch")
    embeddings = np.memmap(embedding_file, mode="r", dtype=dtype, shape=tuple(shape))
    # Narrow only the selected token rows, not the whole embedding table.
    class Embeddings:
        def __getitem__(self, index):
            return embeddings[index].astype(np.float32)

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sessions = {
        name: ort.InferenceSession(str(args.model / f"{name}.int4.onnx"), options,
                                   providers=["CPUExecutionProvider"])
        for name in ("encoder", "decoder_init", "decoder_step")
    }
    if args.diagnostics:
        args.diagnostics.mkdir(mode=0o700, parents=True, exist_ok=True)
    print(f"Qwen3-ASR ONNX INT4 CPU sessions ready; language={args.language}",
          file=sys.stderr, flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        try:
            with wave.open(request["input"]) as wav:
                if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 16000:
                    raise ValueError("expected mono signed-16-bit 16-kHz WAV")
                samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
            features = log_mel_spectrogram(samples).numpy()
            encoder = sessions["encoder"]
            names = [port.name for port in encoder.get_inputs()]
            if names == ["mel"]:
                encoder_inputs = {"mel": features}
            elif names == ["mel", "mel_len"]:
                encoder_inputs = {"mel": features,
                                  "mel_len": np.array([features.shape[-1]], dtype=np.int64)}
            else:
                raise ValueError(f"unsupported encoder input schema: {names}")
            audio = encoder.run(None, encoder_inputs)[0]
            prompt = build_prompt_ids(audio.shape[1])
            if args.language != "auto":
                prompt.extend(tokenizer.encode(f"language {args.language}<asr_text>",
                                                add_special_tokens=False).ids)
            tokens = greedy_decode_onnx(sessions, Embeddings(), audio, prompt, args.max_tokens)
            if args.diagnostics:
                details = {"request_index": request["index"], "input": request["input"],
                           "language_policy": args.language, "token_ids": tokens,
                           "token_count": len(tokens), "reached_eos": bool(tokens and tokens[-1] in EOS_TOKEN_IDS),
                           "raw_decoded": tokenizer.decode(tokens, skip_special_tokens=False)}
                destination = args.diagnostics / f'{request["index"]:05}.json'
                with destination.open("x", encoding="utf-8") as stream:
                    json.dump(details, stream, ensure_ascii=False, indent=2)
            if not tokens or tokens[-1] not in EOS_TOKEN_IDS:
                raise ValueError("generation did not reach EOS; incomplete hypothesis")
            language = "de"
            if args.language == "auto":
                if 151704 not in tokens:
                    raise ValueError("automatic-language output has no ASR text separator")
                separator = tokens.index(151704)
                prefix = tokenizer.decode(tokens[:separator], skip_special_tokens=True)
                language = "de" if prefix.strip() == "language German" else "unknown"
                tokens = tokens[separator + 1:]
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            response = {"index": request["index"], "text": text, "language": language,
                        "segments": [{"start": 0.0, "end": len(samples) / 16000, "text": text}]}
        except (ValueError, RuntimeError, OSError, KeyError) as exc:
            response = {"index": request.get("index"), "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
