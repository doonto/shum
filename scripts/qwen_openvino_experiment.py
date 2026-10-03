#!/usr/bin/env python3
"""Opt-in Qwen3-ASR CPU OpenVINO export and indexed diagnostic worker.

Uses local Transformers Qwen3-ASR weights only. Explicit paths keep the
experiment independent of production configuration and provider access.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Model, Qwen3DecoderLayer, Qwen3RMSNorm, apply_rotary_pos_emb, repeat_kv,
)
from transformers.models.qwen3_asr.modeling_qwen3_asr import (
    Qwen3ASREncoder, Qwen3ASRMultiModalProjector,
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while part := stream.read(4 * 1024 * 1024):
            value.update(part)
    return value.hexdigest()


def load_component(module, prefix, checkpoint, dtype=torch.float32):
    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        state = {
            name: source.get_tensor(prefix + name).to(dtype)
            for name in module.state_dict()
        }
    module.load_state_dict(state, strict=True, assign=True)
    return module.eval()


class Audio(torch.nn.Module):
    def __init__(self, encoder, projector):
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, features, valid_indices, window_mask):
        encoder = self.encoder
        chunks = features.reshape(1, 128, -1, 100).permute(0, 2, 1, 3).reshape(-1, 1, 128, 100)
        x = torch.nn.functional.gelu(encoder.conv2d1(chunks))
        x = torch.nn.functional.gelu(encoder.conv2d2(x))
        x = torch.nn.functional.gelu(encoder.conv2d3(x))
        x = encoder.conv_out(x.permute(0, 3, 1, 2).reshape(-1, 13, 480 * 16))
        x = x + encoder.positional_embedding.positional_embedding[:13]
        x = torch.index_select(x.reshape(-1, 1024), 0, valid_indices)
        for layer in encoder.layers:
            residual = x
            normalized = layer.self_attn_layer_norm(x)
            attention = layer.self_attn
            q = attention.q_proj(normalized).reshape(-1, 16, 64).transpose(0, 1).unsqueeze(0)
            k = attention.k_proj(normalized).reshape(-1, 16, 64).transpose(0, 1).unsqueeze(0)
            v = attention.v_proj(normalized).reshape(-1, 16, 64).transpose(0, 1).unsqueeze(0)
            attended = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=window_mask)
            x = residual + attention.out_proj(attended.squeeze(0).transpose(0, 1).reshape(-1, 1024))
            x = x + layer.fc2(layer.activation_fn(layer.fc1(layer.final_layer_norm(x))))
        return self.projector(encoder.ln_post(x))


class Decoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, embeds, positions, mask, past_keys, past_values):
        x = embeds
        cosine, sine = self.model.rotary_emb(x, positions)
        keys, values = [], []
        for index, layer in enumerate(self.model.layers):
            residual = x
            normalized = layer.input_layernorm(x)
            attention = layer.self_attn
            shape = (x.shape[0], x.shape[1], -1, 128)
            q = attention.q_norm(attention.q_proj(normalized).reshape(shape)).transpose(1, 2)
            k = attention.k_norm(attention.k_proj(normalized).reshape(shape)).transpose(1, 2)
            v = attention.v_proj(normalized).reshape(shape).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cosine, sine)
            k = torch.cat((past_keys[index], k), dim=2)
            v = torch.cat((past_values[index], v), dim=2)
            keys.append(k)
            values.append(v)
            attended = torch.nn.functional.scaled_dot_product_attention(
                q, repeat_kv(k, 2), repeat_kv(v, 2), attn_mask=mask.to(q.dtype),
            )
            x = residual + attention.o_proj(attended.transpose(1, 2).reshape(x.shape))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        logits = torch.nn.functional.linear(self.model.norm(x[:, -1:, :]), self.model.embed_tokens.weight)
        return logits, torch.stack(keys), torch.stack(values)


class Block(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, cosine, sine, mask, past_key, past_value):
        layer = self.layer
        residual = x
        normalized = layer.input_layernorm(x)
        attention = layer.self_attn
        shape = (x.shape[0], x.shape[1], -1, 128)
        q = attention.q_norm(attention.q_proj(normalized).reshape(shape)).transpose(1, 2)
        k = attention.k_norm(attention.k_proj(normalized).reshape(shape)).transpose(1, 2)
        v = attention.v_proj(normalized).reshape(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cosine, sine)
        k = torch.cat((past_key, k), dim=2)
        v = torch.cat((past_value, v), dim=2)
        attended = torch.nn.functional.scaled_dot_product_attention(
            q, repeat_kv(k, 2), repeat_kv(v, 2), attn_mask=mask,
        )
        x = residual + attention.o_proj(attended.transpose(1, 2).reshape(x.shape))
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        return x, k, v


class Head(torch.nn.Module):
    def __init__(self, norm, embedding):
        super().__init__()
        self.norm = norm
        self.embedding = torch.nn.Parameter(embedding, requires_grad=False)

    def forward(self, x):
        return torch.nn.functional.linear(self.norm(x[:, -1:, :]), self.embedding)


def export_layers(args, ov):
    import nncf
    config = AutoConfig.from_pretrained(args.model, local_files_only=True).text_config
    checkpoint = args.model / "model.safetensors"
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "layers-export.json").exists():
        raise ValueError("refusing to replace existing layerwise export")
    started = time.monotonic()
    reports = []
    for index in range(config.num_hidden_layers + 1):
        name = f"layer-{index:02}" if index < config.num_hidden_layers else "head"
        target = output / (name + ".xml")
        if target.exists() or target.with_suffix(".bin").exists():
            raise ValueError(f"refusing to overwrite {target}")
        if index < config.num_hidden_layers:
            with torch.device("meta"):
                layer = Qwen3DecoderLayer(config, index)
            load_component(layer, f"model.language_model.layers.{index}.", checkpoint)
            module = Block(layer).eval()
            shape = torch.randn(1, 3, 2048)
            cosine = torch.ones(1, 3, 128)
            sine = torch.zeros_like(cosine)
            mask = torch.triu(torch.full((1, 1, 3, 3), -1e9), diagonal=1)
            empty = torch.zeros(1, 8, 0, 128)
            examples = (shape, cosine, sine, mask, empty, empty.clone())
            names = ["embeds", "cosine", "sine", "mask", "past_key", "past_value"]
            shapes = [[1, -1, 2048], [1, -1, 128], [1, -1, 128],
                      [1, 1, -1, -1], [1, 8, -1, 128], [1, 8, -1, 128]]
            outputs = ["hidden", "present_key", "present_value"]
        else:
            with torch.device("meta"):
                norm = Qwen3RMSNorm(2048, eps=config.rms_norm_eps)
            load_component(norm, "model.language_model.norm.", checkpoint)
            with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
                embedding = source.get_tensor("model.language_model.embed_tokens.weight").float()
            embedding_path = output / "embeddings-f16.npy"
            if embedding_path.exists():
                raise ValueError("refusing to overwrite existing embedding table")
            table = np.lib.format.open_memmap(
                embedding_path, mode="w+", dtype=np.float16, shape=tuple(embedding.shape),
            )
            for start in range(0, embedding.shape[0], 512):
                table[start:start + 512] = embedding[start:start + 512].numpy().astype(np.float16)
            table.flush()
            del table
            module = Head(norm, embedding).eval()
            examples = (torch.randn(1, 3, 2048),)
            names, shapes, outputs = ["embeds"], [[1, -1, 2048]], ["logits"]
        with torch.inference_mode():
            graph = ov.convert_model(module, example_input=examples)
        for port, port_name in zip(graph.inputs, names):
            port.get_tensor().set_names({port_name})
        for port, port_name in zip(graph.outputs, outputs):
            port.get_tensor().set_names({port_name})
        graph.reshape({n: ov.PartialShape(s) for n, s in zip(names, shapes)})
        if args.int8:
            graph = nncf.compress_weights(graph, mode=nncf.CompressWeightsMode.INT8_ASYM)
        ov.save_model(graph, target, compress_to_fp16=False)
        reports.append({"name": name, "xml_sha256": digest(target),
                        "bin_sha256": digest(target.with_suffix(".bin"))})
        print(json.dumps({"event": "layer_exported", "name": name}), flush=True)
        del module, graph, examples
        if index < config.num_hidden_layers:
            del layer
        else:
            del norm, embedding
        gc.collect()
    report = {"status": "exported", "format": "OpenVINO IR, layerwise KV cache",
              "precision": "INT8_ASYM weight-only" if args.int8 else "FP32",
              "numerical_parity": "not yet verified end-to-end",
              "model": str(args.model), "graphs": reports,
              "elapsed_seconds": time.monotonic() - started}
    (output / "layers-export.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def audio_inputs(features, feature_mask):
    lengths = feature_mask.reshape(-1, 100).sum(-1).numpy().astype(np.int64)
    lengths = (lengths + 7) // 8
    valid = np.concatenate([np.arange(length) + 13 * index for index, length in enumerate(lengths)])
    groups = np.arange(len(valid)) // 104
    mask = np.where(groups[:, None] == groups[None, :], 0.0, -1e9).astype(np.float32)[None, None]
    return features.float().numpy(), valid.astype(np.int64), mask


def export(args, ov):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    target = output / (args.part + ".xml")
    if target.exists() or target.with_suffix(".bin").exists():
        raise ValueError("refusing to overwrite existing OpenVINO artifacts")
    started = time.monotonic()
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if config.text_config.num_hidden_layers != 28 or config.audio_config.encoder_layers != 24:
        raise ValueError("this experiment requires the local Qwen3-ASR-1.7B architecture")
    checkpoint = args.model / "model.safetensors"
    if args.part == "audio":
        with torch.device("meta"):
            encoder = Qwen3ASREncoder(config.audio_config)
            projector = Qwen3ASRMultiModalProjector(config)
        load_component(encoder, "model.audio_tower.", checkpoint)
        load_component(projector, "model.multi_modal_projector.", checkpoint)
        # This deterministic sinusoidal buffer is not stored in safetensors.
        from transformers.models.qwen3_asr.modeling_qwen3_asr import SinusoidsPositionEmbedding
        encoder.positional_embedding = SinusoidsPositionEmbedding(13, 1024)
        module = Audio(encoder, projector).eval()
        features = torch.randn(1, 128, 200)
        feature_mask = torch.ones(1, 200, dtype=torch.long)
        prepared = audio_inputs(features, feature_mask)
        examples = tuple(torch.from_numpy(item) for item in prepared)
        with torch.inference_mode():
            reference = projector(encoder(features, feature_mask).last_hidden_state)
            actual = module(*examples)
        difference = float((reference - actual).abs().max())
        if not torch.allclose(reference, actual, atol=0.002, rtol=0.002):
            raise ValueError(f"audio wrapper differs from native model: max error {difference}")
        names = ["features", "valid_indices", "window_mask"]
        shapes = [[1, 128, -1], [-1], [1, 1, -1, -1]]
        outputs = ["audio_embeds"]
    else:
        with torch.device("meta"):
            model = Qwen3Model(config.text_config)
        dtype = torch.float16 if args.decoder_fp16 else torch.float32
        load_component(model, "model.language_model.", checkpoint, dtype)
        # RoPE frequency buffers are regenerated from the checkpoint config.
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
        model.rotary_emb = Qwen3RotaryEmbedding(config.text_config)
        module = Decoder(model).eval()
        embeds = model.embed_tokens(torch.tensor([[1, 2, 3]]))
        positions = torch.arange(3)[None]
        mask = torch.triu(torch.full((1, 1, 3, 3), -1e9), diagonal=1)
        empty = torch.zeros(28, 1, 8, 0, 128, dtype=dtype)
        examples = (embeds.detach(), positions, mask, empty, empty.clone())
        with torch.inference_mode():
            native = model(inputs_embeds=embeds, attention_mask=mask,
                           position_ids=positions, use_cache=False).last_hidden_state
            reference = torch.nn.functional.linear(native[:, -1:], model.embed_tokens.weight)
            actual = module(*examples)[0]
        difference = float((reference - actual).abs().max())
        if not torch.allclose(reference, actual, atol=0.02, rtol=0.01):
            raise ValueError(f"decoder wrapper differs from native model: max error {difference}")
        names = ["embeds", "positions", "mask", "past_keys", "past_values"]
        shapes = [[1, -1, 2048], [1, -1], [1, 1, -1, -1],
                  [28, 1, 8, -1, 128], [28, 1, 8, -1, 128]]
        outputs = ["logits", "present_keys", "present_values"]
    print(json.dumps({"event": "wrapper_verified", "part": args.part,
                      "max_abs_error": difference}), flush=True)
    with torch.inference_mode():
        graph = ov.convert_model(module, example_input=examples)
    for port, name in zip(graph.inputs, names):
        port.get_tensor().set_names({name})
    for port, name in zip(graph.outputs, outputs):
        port.get_tensor().set_names({name})
    graph.reshape({name: ov.PartialShape(shape) for name, shape in zip(names, shapes)})
    # Confirm runtime graph before weight compression; not an application test suite.
    compiled = ov.Core().compile_model(graph, "CPU", {
        "INFERENCE_PRECISION_HINT": "f32", "INFERENCE_NUM_THREADS": args.threads,
    })
    with torch.inference_mode():
        native = module(*examples)
    runtime = compiled(dict(zip(names, (x.numpy() for x in examples))))
    native_output = native if args.part == "audio" else native[0]
    ov_output = runtime[compiled.output(outputs[0])]
    error = float(np.max(np.abs(native_output.numpy() - ov_output)))
    if not np.allclose(native_output.numpy(), ov_output, atol=0.02, rtol=0.01):
        raise ValueError(f"OpenVINO conversion parity failed: max error {error}")
    del compiled
    gc.collect()
    if args.int8:
        import nncf
        graph = nncf.compress_weights(graph, mode=nncf.CompressWeightsMode.INT8_ASYM)
    ov.save_model(graph, target, compress_to_fp16=False)
    report = {
        "status": "exported", "part": args.part, "model": str(args.model),
        "format": "OpenVINO IR", "precision": "INT8_ASYM weight-only" if args.int8 else "FP32",
        "wrapper_max_abs_error": difference, "uncompressed_graph_max_abs_error": error,
        "elapsed_seconds": time.monotonic() - started,
        "graph_sha256": digest(target), "weights_sha256": digest(target.with_suffix(".bin")),
        "torch_version": torch.__version__, "openvino_version": ov.__version__,
        "decoder_source_conversion": "BF16-to-FP16" if args.decoder_fp16 else "BF16-to-FP32",
    }
    (output / (args.part + "-export.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def serve(args, ov):
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    core = ov.Core()
    properties = {"INFERENCE_NUM_THREADS": args.threads, "PERFORMANCE_HINT": "LATENCY",
                  "INFERENCE_PRECISION_HINT": "f32"}
    audio_model = core.compile_model(args.output / "audio.xml", "CPU", properties)
    layerwise = (args.output / "layers-export.json").is_file()
    if layerwise:
        layer_models = [core.compile_model(args.output / f"layer-{i:02}.xml", "CPU", properties)
                        for i in range(28)]
        layer_requests = [model.create_infer_request() for model in layer_models]
        head = core.compile_model(args.output / "head.xml", "CPU", properties)
    else:
        decoder_model = core.compile_model(args.output / "decoder.xml", "CPU", properties)
        decoder_request = decoder_model.create_infer_request()
    embeddings = np.load(args.output / "embeddings-f16.npy", mmap_mode="r")
    eos = set(AutoConfig.from_pretrained(args.model, local_files_only=True).eos_token_id)
    print("OpenVINO Qwen3-ASR loaded; sequential indexed worker ready", file=sys.stderr, flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        try:
            with wave.open(request["input"]) as wav:
                if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 16000:
                    raise ValueError("expected mono signed-16-bit 16-kHz WAV")
                samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
            features = processor.apply_transcription_request(
                samples, language="German",
            )
            inputs = audio_inputs(features["input_features"], features["input_features_mask"])
            audio = audio_model(dict(zip(["features", "valid_indices", "window_mask"], inputs)))[audio_model.output(0)]
            ids = features["input_ids"].numpy()
            embeds = embeddings[ids].astype(np.float32)
            placeholder = ids[0] == processor.audio_token_id if hasattr(processor, "audio_token_id") else ids[0] == 151676
            if int(placeholder.sum()) != audio.shape[0]:
                raise ValueError("audio placeholder/embedding length mismatch")
            embeds[0, placeholder] = audio
            keys = np.zeros((28, 1, 8, 0, 128), dtype=np.float32)
            values = np.zeros_like(keys)
            tokens = []
            past = 0
            for _ in range(args.max_tokens):
                count = embeds.shape[1]
                positions = np.arange(past, past + count, dtype=np.int64)[None]
                mask = np.where(
                    np.arange(past + count)[None, :] <= positions[0, :, None], 0.0, -1e9,
                ).astype(np.float32)[None, None]
                if layerwise:
                    frequencies = 1.0 / (1000000 ** (np.arange(0, 128, 2, dtype=np.float32) / 128))
                    angles = positions[..., None] * frequencies
                    angles = np.concatenate((angles, angles), axis=-1)
                    cosine, sine = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
                    hidden = embeds
                    new_keys, new_values = [], []
                    for index, (model, infer) in enumerate(zip(layer_models, layer_requests)):
                        result = infer.infer({
                            "embeds": hidden, "cosine": cosine, "sine": sine, "mask": mask,
                            "past_key": keys[index], "past_value": values[index],
                        })
                        hidden = result[model.output("hidden")]
                        new_keys.append(result[model.output("present_key")].copy())
                        new_values.append(result[model.output("present_value")].copy())
                    logits = head({"embeds": hidden})[head.output(0)]
                    token = int(logits.argmax())
                else:
                    result = decoder_request.infer({
                        "embeds": embeds, "positions": positions, "mask": mask,
                        "past_keys": keys, "past_values": values,
                    })
                    token = int(result[decoder_model.output("logits")].argmax())
                if token in eos:
                    break
                tokens.append(token)
                if layerwise:
                    keys, values = np.stack(new_keys), np.stack(new_values)
                else:
                    keys = result[decoder_model.output("present_keys")].copy()
                    values = result[decoder_model.output("present_values")].copy()
                past += count
                embeds = embeddings[np.array([[token]])].astype(np.float32)
            else:
                raise ValueError("max token limit reached without EOS; not a complete hypothesis")
            text = processor.tokenizer.decode(tokens, skip_special_tokens=True)
            response = {"index": request["index"], "text": text, "language": "de",
                        "segments": [{"start": 0.0, "end": len(samples) / 16000, "text": text}]}
        except (ValueError, RuntimeError, OSError, KeyError) as exc:
            response = {"index": request.get("index"), "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["export", "serve"])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--openvino-site", type=Path, required=True)
    parser.add_argument("--part", choices=["audio", "decoder", "layers"])
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--decoder-fp16", action="store_true")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=448)
    args = parser.parse_args()
    if args.threads < 1 or args.max_tokens < 1:
        parser.error("threads and max-tokens must be positive")
    sys.path.append(str(args.openvino_site))
    import openvino as ov
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    if args.action == "export":
        if args.part is None:
            parser.error("export requires --part")
        if args.part == "layers":
            export_layers(args, ov)
        else:
            export(args, ov)
    else:
        serve(args, ov)


if __name__ == "__main__":
    main()
