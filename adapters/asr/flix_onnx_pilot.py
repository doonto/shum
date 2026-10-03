#!/usr/bin/env python3
"""One-window, text-only Flix ONNX compatibility pilot; no export or fallback engine.

Dry-run verifies local identity and resource admission without loading ONNX.
Run uses installed Optimum's ORTModelForSpeechSeq2Seq and requires a verified
systemd user scope with a kernel-enforced RAM limit and zero swap allowance.
Results are diagnostic text, never timestamped or production ASR segments.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Any

import flix_fast_common as common

GIB = 1024**3
RESERVE = 4 * GIB
MAX_BUDGET = 9 * GIB
WINDOW_SECONDS = 120
REVISION = "a9c347a904af5d31b28c537ac732e0618aed6136"
GRAPH_NAMES = ("encoder_model", "decoder_model")


class PilotError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise PilotError(f"Expected JSON object: {path}")
    return result


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def local_file(root: Path, name: str) -> Path:
    path = root / name
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise PilotError(f"Missing or linked model file: {path}")
    return path


def source_token_mapping(source: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    tokenizer_config = read_json(local_file(source, "tokenizer_config.json"))
    entries = read_json(local_file(source, "tokenizer.json")).get("added_tokens")
    extras = tokenizer_config.get("extra_special_tokens")
    if not isinstance(extras, list) or not isinstance(entries, list):
        raise PilotError("Source fast tokenizer requires a special-token list and added-token inventory")
    if any(not isinstance(token, str) or not token for token in extras) or len(set(extras)) != len(extras):
        raise PilotError("Source special-token list has invalid or repeated entries")
    by_content = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("content"), str) or (
            not isinstance(entry.get("id"), int)
        ) or entry["content"] in by_content:
            raise PilotError("Invalid tokenizer.json added-token inventory")
        by_content[entry["content"]] = entry
    if any(token not in by_content or by_content[token].get("special") is not True
           for token in extras):
        raise PilotError("A source special token is missing or not special in tokenizer.json")
    mapping = {f"flix_extra_special_{index:03d}": token for index, token in enumerate(extras)}
    return mapping, entries


def validate_fast_token_ids(tokenizer: Any, entries: list[dict[str, Any]],
                            mapping: dict[str, str], expected_vocab: int) -> None:
    if len(tokenizer) != expected_vocab:
        raise PilotError("Fast tokenizer vocabulary size differs from source model")
    if any(tokenizer.convert_tokens_to_ids(entry["content"]) != entry["id"]
           for entry in entries):
        raise PilotError("Fast tokenizer changed an added-token ID")
    if any(token not in tokenizer.all_special_tokens for token in mapping.values()):
        raise PilotError("Fast tokenizer dropped a source special token")


def load_source_processor(source: Path, expected_vocab: int) -> Any:
    from transformers import WhisperFeatureExtractor, WhisperProcessor, WhisperTokenizerFast

    mapping, entries = source_token_mapping(source)
    tokenizer = WhisperTokenizerFast.from_pretrained(
        str(source), local_files_only=True, extra_special_tokens=mapping,
    )
    validate_fast_token_ids(tokenizer, entries, mapping, expected_vocab)
    nested = read_json(local_file(source, "processor_config.json")).get("feature_extractor")
    if not isinstance(nested, dict) or nested.get("feature_size") != 128 or (
        nested.get("sampling_rate") != 16000
    ):
        raise PilotError("Source processor has no matching 128-mel/16-kHz feature configuration")
    feature_extractor = WhisperFeatureExtractor(**nested)
    return WhisperProcessor(feature_extractor, tokenizer)


def verify_identity(source: Path, graphs: Path, audit_path: Path) -> dict[str, Any]:
    manifest = read_json(local_file(source, ".shum-manifest.json"))
    audit = read_json(audit_path)
    if (manifest.get("repo"), manifest.get("revision")) != (
        "Flix-AI/flix-swissgerman-full", REVISION
    ):
        raise PilotError("Source manifest is not the pinned Flix revision")
    if not audit.get("exact_identity") or audit.get("source_parameters_total") != 1259 or (
        audit.get("unique_source_parameters_matched") != 1259
    ):
        raise PilotError("Incomplete numerical identity audit")
    files = manifest.get("files", {})
    for name in ("model.safetensors", "config.json", "generation_config.json",
                 "processor_config.json", "tokenizer.json", "tokenizer_config.json"):
        expected = files.get(name)
        if not isinstance(expected, str) or digest(local_file(source, name)) != expected:
            raise PilotError(f"Source checksum mismatch: {name}")
    if audit["source"].get("sha256") != files["model.safetensors"]:
        raise PilotError("Audit and pinned source hashes disagree")
    source_config, graph_config = read_json(source / "config.json"), read_json(graphs / "config.json")
    if (source_config.get("model_type"), source_config.get("num_mel_bins"),
            source_config.get("dtype")) != ("whisper", 128, "bfloat16"):
        raise PilotError("Source must be BF16 128-mel Whisper")
    if (graph_config.get("model_type"), graph_config.get("num_mel_bins"),
            graph_config.get("dtype")) != ("whisper", 128, "float32"):
        raise PilotError("Graph must be FP32 128-mel Whisper")
    sg, gg = read_json(source / "generation_config.json"), read_json(graphs / "generation_config.json")
    if {k: v for k, v in sg.items() if k != "transformers_version"} != {
        k: v for k, v in gg.items() if k != "transformers_version"
    } or (sg.get("language"), sg.get("task")) != ("de", "transcribe"):
        raise PilotError("Generation defaults/suppression differ from pinned source")
    processor = read_json(graphs / "preprocessor_config.json")
    source_processor = read_json(source / "processor_config.json").get("feature_extractor", {})
    if processor.get("feature_size") != 128 or processor.get("sampling_rate") != 16000 or (
        source_processor.get("feature_size"), source_processor.get("sampling_rate")
    ) != (128, 16000):
        raise PilotError("Source and graph processor must be 128 mel / 16 kHz")
    tokens = read_json(source / "tokenizer_config.json")
    if (tokens.get("language"), tokens.get("task")) != ("de", "transcribe"):
        raise PilotError("Tokenizer language/task differ from pinned source")
    special_mapping, token_entries = source_token_mapping(source)
    for name in GRAPH_NAMES:
        entry = audit.get("graphs", {}).get(name, {})
        if entry.get("exact_initializers") != entry.get("initializer_count") or not entry.get(
            "initializer_count"
        ):
            raise PilotError(f"Unverified graph initializers: {name}")
        for suffix, sha_key in ((".onnx", "onnx_sha256"), (".onnx_data", "external_data_sha256")):
            if digest(local_file(graphs, name + suffix)) != entry.get(sha_key):
                raise PilotError(f"Graph checksum mismatch: {name}{suffix}")
    return {
        "revision": REVISION, "source_sha256": files["model.safetensors"],
        "graph_sha256": {name: audit["graphs"][name]["onnx_sha256"] for name in GRAPH_NAMES},
        "external_data_sha256": {
            name: audit["graphs"][name]["external_data_sha256"] for name in GRAPH_NAMES
        },
        "tokenizer_sha256": files["tokenizer.json"],
        "added_token_count": len(token_entries),
        "special_token_count": len(special_mapping),
        "generation_sha256": files["generation_config.json"],
        "suppressed_tokens": sg.get("suppress_tokens"),
        "begin_suppress_tokens": sg.get("begin_suppress_tokens"),
    }


def mem_available() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
            break
    else:
        raise PilotError("Cannot establish MemAvailable")
    headroom = common.cgroup_headroom_bytes()
    return min(available, headroom) if headroom is not None else available


def budget(available: int) -> int:
    admitted = min(MAX_BUDGET, ((available - RESERVE) // (1024**2)) * (1024**2))
    if admitted < GIB:
        raise PilotError("Less than 1 GiB remains after the 4 GiB reserve")
    return admitted


def runtime_requirements() -> dict[str, Any]:
    packages = {}
    for name in ("optimum", "optimum-onnx", "onnxruntime", "transformers", "torch", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    systemd = shutil.which("systemd-run")
    return {
        "packages": packages,
        "systemd_run": systemd,
        "systemctl": shutil.which("systemctl"),
        "available": all(packages.values()) and bool(systemd and shutil.which("systemctl")),
        "runtime_api": (
            "optimum.onnxruntime.ORTModelForSpeechSeq2Seq.from_pretrained("
            "graphs, export=False, local_files_only=True, use_cache=False, "
            "use_merged=False, encoder_file_name='encoder_model.onnx', "
            "decoder_file_name='decoder_model.onnx', session_options=...)"
        ),
        "scope_status": "not_probed_by_metadata_dry_run",
    }


def scope_memory(scope: Path, limit: int) -> dict[str, int]:
    maximum = (scope / "memory.max").read_text().strip()
    swap = (scope / "memory.swap.max").read_text().strip()
    if not maximum.isdecimal() or int(maximum) > limit or swap != "0":
        raise PilotError("Kernel cgroup RAM/swap limits do not match the requested hard cap")
    return {
        "memory_max_bytes": int(maximum),
        "memory_swap_max_bytes": 0,
        "memory_current_bytes": int((scope / "memory.current").read_text().strip()),
    }


def own_scope(limit: int, unit: str) -> Path:
    membership = Path("/proc/self/cgroup").read_text(encoding="ascii")
    relative = next((line[3:] for line in membership.splitlines() if line.startswith("0::/")), None)
    if relative is None or not relative.endswith("/" + unit):
        raise PilotError("Worker is not in the requested systemd cgroup-v2 scope")
    scope = Path("/sys/fs/cgroup") / relative.lstrip("/")
    scope_memory(scope, limit)
    return scope


def sample(scope: Path | None = None, pid: int | None = None) -> dict[str, float | int]:
    info = {
        line.split(":", 1)[0]: line.split(":", 1)[1].strip().split()[0]
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines() if ":" in line
    }
    vmstat = dict(line.split() for line in Path("/proc/vmstat").read_text(encoding="ascii").splitlines())
    own_rss = own_swap = 0
    if pid is not None:
        own_rss, own_swap = common.process_tree_memory_kib(pid)
    result: dict[str, float | int] = {
        "available_bytes": int(info["MemAvailable"]) * 1024,
        "own_rss_kib": own_rss, "own_vmswap_kib": own_swap,
        "system_pswpout_pages": int(vmstat["pswpout"]),
        "psi_memory_full_avg10_percent": common.memory_full_avg10(
            Path("/proc/pressure/memory").read_text(encoding="ascii")
        ),
    }
    if scope:
        result.update(scope_memory(scope, MAX_BUDGET))
        parent_membership = "0::/" + str(scope.parent.relative_to(Path("/sys/fs/cgroup")))
        outer_headroom = common.cgroup_headroom_bytes(membership=parent_membership)
        if outer_headroom is not None:
            result["outer_cgroup_headroom_bytes"] = outer_headroom
            result["available_bytes"] = min(result["available_bytes"], outer_headroom)
        result["cgroup_memory_peak_bytes"] = int((scope / "memory.peak").read_text().strip())
        result["cgroup_swap_current_bytes"] = int((scope / "memory.swap.current").read_text().strip())
        result["cgroup_oom_kill_count"] = int(dict(
            line.split() for line in (scope / "memory.events").read_text().splitlines()
        )["oom_kill"])
    return result


def pressure_reason(current: dict[str, float | int], initial: dict[str, float | int]) -> str | None:
    if current["available_bytes"] < RESERVE:
        return "4_gib_reserve_lost"
    if current["own_vmswap_kib"] or current.get("cgroup_swap_current_bytes", 0):
        return "own_swap_detected"
    if current["system_pswpout_pages"] < initial["system_pswpout_pages"]:
        return "system_swap_counter_reset"
    if current["system_pswpout_pages"] > initial["system_pswpout_pages"] and (
        current["psi_memory_full_avg10_percent"] >= 1.0
    ):
        return "swap_out_with_full_memory_psi"
    if current.get("cgroup_oom_kill_count", 0) > initial.get("cgroup_oom_kill_count", 0):
        return "cgroup_oom_kill"
    return None


def monitored_sample(scope: Path | None, pid: int | None,
                     process: subprocess.Popen[str]) -> dict[str, float | int] | None:
    for attempt in range(3):
        try:
            return sample(scope, pid)
        except common.AdapterError as exc:
            if not str(exc).startswith("Invalid VmRSS/VmSwap for worker "):
                raise
            if process.poll() is not None:
                return None
            if attempt == 2:
                raise
            time.sleep(0.02)
    return None


def validate_wav(path: Path) -> float:
    if not path.is_absolute() or not path.is_file():
        raise PilotError("--input must be an existing absolute WAV")
    try:
        with wave.open(str(path), "rb") as stream:
            if (stream.getnchannels(), stream.getframerate(), stream.getsampwidth(),
                    stream.getcomptype()) != (1, 16000, 2, "NONE"):
                raise PilotError("Input must be 16 kHz mono PCM16 WAV")
            duration = stream.getnframes() / 16000
    except wave.Error as exc:
        raise PilotError(f"Invalid WAV: {exc}") from exc
    if not 0 < duration <= 30:
        raise PilotError("Pilot accepts exactly one nonempty window of at most 30 seconds")
    return duration


def write_report(path: Path, report: dict[str, Any], protected: tuple[Path, Path]) -> None:
    if not path.is_absolute() or not path.parent.is_dir() or path.is_symlink():
        raise PilotError("--report must be a normal absolute path in an existing directory")
    if any(path.resolve().is_relative_to(root) for root in protected):
        raise PilotError("--report must be outside source and graph directories")
    with path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def termination_flags(ids: list[int], eos_token_id: int | list[int], cap: int = 128) -> dict[str, Any]:
    eos_ids = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id)
    if not ids or not eos_ids:
        raise PilotError("Generation returned no token IDs or EOS is unspecified")
    observed = any(token in eos_ids for token in ids)
    # Whisper includes decoder prompt tokens in its returned sequence, so this
    # threshold is a conservative *possible* cap hit, not a counted generation.
    near_cap = len(ids) >= cap
    return {
        "eos_observed": observed,
        "artificial_max_new_tokens": cap,
        "cap_reached_possible": near_cap,
        "possible_truncation": not observed,
        "termination": ("eos_observed_not_audio_completeness_proof" if observed else
                        "possible_128_token_cap_truncation" if near_cap else
                        "no_eos_termination_unverified"),
    }


def worker(args: argparse.Namespace) -> int:
    common.configure_offline_environment()
    common.configure_thread_environment(args.threads)
    scope = own_scope(args.budget_bytes, args.scope_unit)
    initial = sample(scope, os.getpid())
    if initial["available_bytes"] < RESERVE or initial["own_vmswap_kib"]:
        raise PilotError("Worker lost 4 GiB reserve or has VmSwap before loading")
    print(json.dumps({"phase": "scope_ready", "pid": os.getpid(),
                      "cgroup": str(scope), "memory": initial}), flush=True)
    t0 = time.monotonic()
    import numpy as np
    import onnxruntime as ort
    import torch
    from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

    torch.set_num_threads(args.threads)
    expected_vocab = read_json(args.source / "config.json")["vocab_size"]
    processor = load_source_processor(args.source, expected_vocab)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_prepacking", "1")
    model = ORTModelForSpeechSeq2Seq.from_pretrained(
        str(args.graphs), export=False, local_files_only=True, use_cache=False,
        use_merged=False, provider="CPUExecutionProvider", session_options=options,
        encoder_file_name="encoder_model.onnx", decoder_file_name="decoder_model.onnx",
    )
    if model.config.num_mel_bins != 128 or processor.feature_extractor.feature_size != 128:
        raise PilotError("Loaded processor/model mel count mismatch")
    if model.generation_config.suppress_tokens != args.suppress_tokens or (
        model.generation_config.begin_suppress_tokens != args.begin_suppress_tokens
    ):
        raise PilotError("Loaded generation suppression differs from source")
    print(json.dumps({"phase": "loaded", "startup_seconds": round(time.monotonic()-t0, 3),
                      "memory": sample(scope, os.getpid())}), flush=True)
    with wave.open(str(args.input), "rb") as wav:
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    if features.shape[1] != 128:
        raise PilotError("Unexpected feature dimension")
    torch.manual_seed(args.seed)
    td = time.monotonic()
    with torch.inference_mode():
        token_ids = model.generate(
            input_features=features, language="de", task="transcribe",
            return_timestamps=False, num_beams=1, do_sample=False,
            use_cache=False, max_new_tokens=128,
        )
    ids = token_ids[0].tolist()
    flags = termination_flags(ids, model.generation_config.eos_token_id)
    text = processor.batch_decode(token_ids, skip_special_tokens=True)[0]
    print(json.dumps({"phase": "decoded", "text": text, "tokens": len(ids), **flags,
                      "decode_seconds": round(time.monotonic()-td, 3),
                      "memory": sample(scope, os.getpid())}, ensure_ascii=False), flush=True)
    return 0


def scoped_activity(scope: Path | None, pid: int | None) -> dict[str, Any]:
    members: list[int] = []
    if scope is not None and scope.exists():
        members = [int(value) for value in (scope / "cgroup.procs").read_text().split()]
    pid_mismatch = False
    if pid is not None and Path(f"/proc/{pid}").exists():
        state = Path(f"/proc/{pid}/status").read_text(encoding="ascii")
        if not any(line.startswith("State:") and line.split()[1] in ("Z", "X")
                   for line in state.splitlines()):
            membership = Path(f"/proc/{pid}/cgroup").read_text(encoding="ascii")
            expected = "0::/" + str(scope.relative_to(Path("/sys/fs/cgroup"))) if scope else None
            pid_mismatch = expected not in membership.splitlines() if expected else True
            if not pid_mismatch and pid not in members:
                members.append(pid)
    return {"scoped_pids": sorted(set(members)), "pid_mismatch": pid_mismatch}


def stop_owned_job(process: subprocess.Popen[str], unit: str, scope: Path | None,
                   pid: int | None, report: dict[str, Any]) -> None:
    cleanup: dict[str, Any] = {"unit": unit + ".scope", "status": "pending",
                               "systemctl_attempted": False, "signals": []}
    report["cleanup"] = cleanup
    try:
        activity = scoped_activity(scope, pid)
        if process.poll() is None or activity["scoped_pids"] or activity["pid_mismatch"]:
            cleanup["systemctl_attempted"] = True
            try:
                stopped = subprocess.run(
                    ["systemctl", "--user", "stop", unit + ".scope"],
                    capture_output=True, text=True, timeout=10, check=False,
                )
                cleanup["systemctl_exit_code"] = stopped.returncode
                cleanup["systemctl_error"] = stopped.stderr[-1000:] if stopped.returncode else None
            except (OSError, subprocess.TimeoutExpired) as exc:
                cleanup["systemctl_error"] = f"{type(exc).__name__}: {exc}"

        for sig in (None, signal.SIGTERM, signal.SIGKILL):
            activity = scoped_activity(scope, pid)
            if sig is not None and pid in activity["scoped_pids"] and not activity["pid_mismatch"]:
                try:
                    os.kill(pid, sig)
                    cleanup["signals"].append(sig.name)
                except ProcessLookupError:
                    cleanup["signals"].append(sig.name + ":already_exited")
            for _ in range(10):
                activity = scoped_activity(scope, pid)
                if not activity["scoped_pids"] and not activity["pid_mismatch"]:
                    break
                time.sleep(0.2)
            if not activity["scoped_pids"] and not activity["pid_mismatch"]:
                break

        activity = scoped_activity(scope, pid)
        cleanup.update(activity)
        if process.poll() is None:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                cleanup["launcher_killed"] = True
                process.wait(timeout=2)
        cleanup["launcher_exit_code"] = process.poll()
        if activity["scoped_pids"] or activity["pid_mismatch"] or process.poll() is None:
            raise PilotError("Scoped worker termination could not be verified")
        cleanup["status"] = "verified_stopped"
    except (OSError, ValueError, PilotError, subprocess.TimeoutExpired) as exc:
        cleanup["status"] = "unverified_cleanup_failure"
        cleanup["error"] = f"{type(exc).__name__}: {exc}"
        raise PilotError(f"Owned scope cleanup failed: {exc}") from exc


def run(args: argparse.Namespace, report: dict[str, Any]) -> None:
    unit = "flix-onnx-pilot-" + uuid.uuid4().hex
    command = [
        "systemd-run", "--user", "--scope", "--quiet", f"--unit={unit}",
        "--property=MemoryAccounting=yes", f"--property=MemoryMax={report['budget_bytes']}",
        "--property=MemorySwapMax=0", sys.executable, "-u", str(Path(__file__).resolve()),
        "--worker", "--source", str(args.source), "--graphs", str(args.graphs),
        "--input", str(args.input), "--report", str(args.report), "--threads", str(args.threads),
        "--seed", str(args.seed), "--budget-bytes", str(report["budget_bytes"]),
        "--scope-unit", unit + ".scope",
        "--suppress-tokens", json.dumps(report["identity"]["suppressed_tokens"]),
        "--begin-suppress-tokens", json.dumps(report["identity"]["begin_suppress_tokens"]),
    ]
    report["scope_unit"] = unit + ".scope"
    events: queue.Queue[dict[str, Any]] = queue.Queue()
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               env={**os.environ, **common.OFFLINE_ENV_VARS})
    reader_done = threading.Event()
    stderr_tail: collections.deque[str] = collections.deque(maxlen=40)

    def consume() -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                try:
                    events.put(json.loads(line))
                except ValueError:
                    events.put({"phase": "unexpected_output", "message": line[:500]})
        finally:
            reader_done.set()

    def consume_errors() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            stderr_tail.append(line[-500:])

    reader = threading.Thread(target=consume, daemon=True)
    reader.start()
    errors_reader = threading.Thread(target=consume_errors, daemon=True)
    errors_reader.start()
    scope: Path | None = None
    pid: int | None = None
    initial = sample()
    report["initial_memory"] = initial
    try:
        while True:
            try:
                event = events.get(timeout=0.2)
            except queue.Empty:
                event = None
            if event:
                phase = event.get("phase")
                if phase == "scope_ready":
                    pid = event["pid"]
                    scope = Path(event["cgroup"])
                    if not scope.is_relative_to(Path("/sys/fs/cgroup")):
                        raise PilotError("Worker cgroup escaped v2 hierarchy")
                    scope_memory(scope, report["budget_bytes"])
                    report["scope_verified"] = True
                    try:
                        checked = monitored_sample(scope, pid, process)
                    except FileNotFoundError:
                        if process.poll() is None:
                            raise PilotError("Verified scope vanished while worker is live") from None
                        checked = None
                    initial = checked if checked is not None else event["memory"]
                    report["last_memory"] = initial
                    report["peak_cgroup_bytes"] = max(
                        report.get("peak_cgroup_bytes", 0), initial.get("cgroup_memory_peak_bytes", 0),
                    )
                elif phase == "loaded":
                    report["startup_seconds"] = event["startup_seconds"]
                    report["startup_memory"] = event["memory"]
                    report["peak_cgroup_bytes"] = max(
                        report.get("peak_cgroup_bytes", 0),
                        event["memory"].get("cgroup_memory_peak_bytes", 0),
                    )
                elif phase == "decoded":
                    for field in ("eos_observed", "cap_reached_possible", "possible_truncation"):
                        if type(event.get(field)) is not bool:
                            raise PilotError(f"Worker omitted termination flag: {field}")
                    report["decode_seconds"] = event["decode_seconds"]
                    report["text"] = event["text"]
                    report["token_count"] = event["tokens"]
                    report["decode_memory"] = event["memory"]
                    report["last_memory"] = event["memory"]
                    report["peak_cgroup_bytes"] = max(
                        report.get("peak_cgroup_bytes", 0),
                        event["memory"].get("cgroup_memory_peak_bytes", 0),
                    )
                    report["termination"] = {
                        key: event[key] for key in (
                            "eos_observed", "cap_reached_possible", "possible_truncation",
                            "artificial_max_new_tokens", "termination",
                        )
                    }
                else:
                    raise PilotError(f"Unexpected worker protocol event: {phase}")
            if time.monotonic() - started >= WINDOW_SECONDS:
                raise PilotError("120_second_window_timeout")
            finished = process.poll() is not None and reader_done.is_set() and events.empty()
            if finished:
                if process.returncode != 0:
                    errors_reader.join(timeout=1)
                    cause = stderr_tail[-1].strip() if stderr_tail else "no worker error detail"
                    raise PilotError(f"Worker exited {process.returncode}: {cause[:300]}")
                if not report.get("scope_verified") or "text" not in report:
                    raise PilotError("Worker finished without verified scope and diagnostic text")
                try:
                    current = monitored_sample(scope, pid, process)
                except FileNotFoundError:
                    report["cgroup_removed_after_completion"] = True
                else:
                    if current is not None:
                        report["last_memory"] = current
                        reason = pressure_reason(current, initial)
                        if reason:
                            raise PilotError(reason)
                    else:
                        report["controller_memory_sample_unavailable_after_exit"] = True
                report["status"] = "runtime_compatibility_observed_text_unverified"
                return
            try:
                current = monitored_sample(scope, pid, process)
            except FileNotFoundError as exc:
                if process.poll() is None:
                    raise PilotError("Verified scope vanished while worker is live") from exc
                continue
            if current is None:
                continue
            report["last_memory"] = current
            report["peak_cgroup_bytes"] = max(report.get("peak_cgroup_bytes", 0),
                                              current.get("cgroup_memory_peak_bytes", 0))
            reason = pressure_reason(current, initial)
            if reason:
                raise PilotError(reason)
    except (PilotError, OSError, ValueError, KeyError) as exc:
        report["operation_error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["total_seconds"] = round(time.monotonic() - started, 3)
        errors_reader.join(timeout=1)
        report["stderr_tail"] = "".join(stderr_tail)[-2000:]
        try:
            stop_owned_job(process, unit, scope, pid, report)
        finally:
            report["worker_exit_code"] = process.poll()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--graphs", type=Path, required=True)
    p.add_argument("--audit", type=Path)
    p.add_argument("--input", type=Path)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--run", action="store_true", help="Explicit opt-in; never implied by dry-run")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--budget-bytes", type=int, help=argparse.SUPPRESS)
    p.add_argument("--scope-unit", help=argparse.SUPPRESS)
    p.add_argument("--suppress-tokens", type=json.loads, help=argparse.SUPPRESS)
    p.add_argument("--begin-suppress-tokens", type=json.loads, help=argparse.SUPPRESS)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    report: dict[str, Any] = {"status": "blocked", "mode": "run" if args.run else "metadata_only",
                              "seed": args.seed, "threads": args.threads, "profile": {
        "cache": False, "vad": False, "language": "de", "task": "transcribe",
        "beam": 1, "timestamps": False, "max_new_tokens": 128,
        "artificial_cap_128": True, "normal_benchmark_comparable": False,
        "ort_cpu_arena": False, "ort_prepacking": False, "ort_graph_optimization": "disabled",
    }, "window_seconds": WINDOW_SECONDS}
    try:
        if args.threads < 1 or args.threads > 32 or args.seed < 0 or args.seed > 2**32-1:
            raise PilotError("threads must be 1..32 and seed 0..4294967295")
        if args.worker:
            if not args.budget_bytes or args.budget_bytes > MAX_BUDGET or not args.input or (
                not args.scope_unit or not args.scope_unit.startswith("flix-onnx-pilot-")
            ):
                raise PilotError("Scoped worker requires an input and a bounded budget")
            return worker(args)
        for path in (args.source, args.graphs):
            if not path.is_absolute() or not path.is_dir():
                raise PilotError("Source and graphs must be existing absolute directories")
        if not args.audit or not args.audit.is_absolute():
            raise PilotError("An absolute --audit identity JSON path is required")
        if args.input:
            report["input_seconds"] = validate_wav(args.input)
        elif args.run:
            raise PilotError("--run requires exactly one --input window")
        report["identity"] = verify_identity(args.source, args.graphs, args.audit)
        available = mem_available()
        report["available_before_bytes"] = available
        report["reserve_bytes"] = RESERVE
        report["budget_bytes"] = budget(available)
        report["requirements"] = runtime_requirements()
        report["scope_required"] = "systemd-run --user --scope MemoryMax/MemorySwapMax=0"
        if args.run:
            if not report["requirements"]["available"]:
                raise PilotError("Required installed ONNX/Optimum packages or systemd user tools are missing")
            refreshed = mem_available()
            report["available_at_launch_bytes"] = refreshed
            report["budget_bytes"] = min(report["budget_bytes"], budget(refreshed))
            run(args, report)
        else:
            report["status"] = "metadata_ready_runtime_unverified"
    except (PilotError, OSError, ValueError, KeyError, common.AdapterError, subprocess.SubprocessError) as exc:
        report["status"] = "blocked"
        report["error"] = f"{type(exc).__name__}: {exc}"
    try:
        write_report(args.report, report, (args.source, args.graphs))
    except (PilotError, OSError) as exc:
        print(f"error: cannot persist report: {exc}", file=sys.stderr)
        return 1
    print(f"{report['status']}: {args.report}")
    return 0 if report["status"] in ("metadata_ready_runtime_unverified",
                                     "runtime_compatibility_observed_text_unverified") else 2


if __name__ == "__main__":
    sys.exit(main())
