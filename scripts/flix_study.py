#!/usr/bin/env python3
"""Freeze an existing Flix corpus and run bounded, single-worker diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapters" / "asr"))
import flix_fast_common as common

REPO = "Flix-AI/flix-swissgerman-full"
REVISION = "a9c347a904af5d31b28c537ac732e0618aed6136"
RESERVE_MIB = 4096
MEMORY_FULL_AVG10_THRESHOLD_PERCENT = 1.0
RESOURCE_GUARD_SEMANTICS = (
    "Stop if effective MemAvailable/cgroup headroom falls below 4096 MiB, "
    "worker process-tree VmSwap grows above the prelaunch baseline, or Linux "
    "memory PSI full avg10 reaches 1.0% with new system pswpout. System "
    "pswpout alone is telemetry, not worker attribution. Counter/read errors "
    "stop for safety; thresholds do not establish causal attribution."
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    fd, name = tempfile.mkstemp(prefix=".flix-study-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wav_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getframerate(), audio.getsampwidth()) != (1, 16000, 2):
            raise ValueError(f"Expected 16 kHz mono PCM16 WAV: {path}")
        frames = audio.getnframes()
        while samples := audio.readframes(16000):
            digest.update(samples)
    return {"frames": frames, "pcm_sha256": digest.hexdigest()}


def freeze_corpus(args: argparse.Namespace) -> Path:
    source = args.source_model.resolve()
    ct2 = args.ct2_model.resolve()
    manifest = read_json(source / ".shum-manifest.json")
    if (manifest.get("repo"), manifest.get("revision")) != (REPO, REVISION):
        raise ValueError("The source is not the pinned Flix checkpoint")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Source manifest must contain file checksums")
    for name, expected in files.items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or sha256(path) != expected:
            raise ValueError(f"Source checksum or path mismatch: {name}")
    provenance = read_json(ct2 / "flix-fast-provenance.json")
    if provenance.get("source_identity") != REVISION or provenance.get("source_manifest") != manifest:
        raise ValueError("Converted model provenance does not match the source manifest")
    audio = args.audio.resolve()
    audio_info = wav_metadata(audio)
    audio_duration = audio_info["frames"] / 16000
    duration = 1800.0
    if audio_duration < duration:
        raise ValueError("The source recording must contain at least 1800 seconds")
    bounds = read_json(args.chunks_json)
    clips = []
    for index, path in enumerate(sorted(args.clips_dir.resolve().glob("*.wav"))):
        if path.name != f"{index:05d}.wav" or index >= len(bounds):
            raise ValueError(f"Missing or unexpected clip index: {path.name}")
        metadata = wav_metadata(path)
        start = float(bounds[index]["start"])
        end = min(float(bounds[index]["end"]), duration)
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= duration:
            raise ValueError(f"Invalid boundaries for clip {index}")
        if abs(metadata["frames"] / 16000 - (end - start)) > 1 / 16000 + 1e-9:
            raise ValueError(f"Clip duration does not match boundaries: {index}")
        with wave.open(str(audio), "rb") as recording:
            recording.setpos(round(start * 16000))
            pcm_digest = hashlib.sha256(recording.readframes(metadata["frames"])).hexdigest()
        if pcm_digest != metadata["pcm_sha256"]:
            raise ValueError(f"Clip PCM does not match the source recording: {index}")
        clips.append({
            "index": index, "path": str(path), "start": start, "end": end,
            "sha256": sha256(path), **metadata,
        })
    if len(clips) != 67:
        raise ValueError(f"Expected 67 clips, found {len(clips)}")
    ct2_files = {
        str(path.relative_to(ct2)): sha256(path)
        for path in sorted(ct2.rglob("*")) if path.is_file()
    }
    if "model.bin" not in ct2_files:
        raise ValueError("Converted model is missing model.bin")
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Refusing to replace an existing study: {output}")
    output.mkdir(mode=0o700, parents=True)
    result = output / "corpus.json"
    write_json(result, {
        "schema": 1, "audio_path": str(audio), "audio_sha256": sha256(audio),
        "duration_seconds": duration, "audio_duration_seconds": audio_duration, "clips": clips,
        "source_model": {"path": str(source), "repo": REPO, "revision": REVISION, "files": files},
        "ct2_model": {"path": str(ct2), "files": ct2_files, "provenance": provenance},
    })
    return result


def verify_corpus(path: Path) -> dict[str, Any]:
    corpus = read_json(path)
    if corpus.get("schema") != 1 or len(corpus.get("clips", [])) != 67:
        raise ValueError("Expected schema 1 corpus with 67 clips")
    if corpus.get("duration_seconds") != 1800.0:
        raise ValueError("Expected the 1800-second corpus")
    if sha256(Path(corpus["audio_path"])) != corpus["audio_sha256"]:
        raise ValueError("Corpus audio checksum mismatch")
    for index, clip in enumerate(corpus["clips"]):
        if clip["index"] != index or sha256(Path(clip["path"])) != clip["sha256"]:
            raise ValueError(f"Corpus clip identity/checksum mismatch: {index}")
        if wav_metadata(Path(clip["path"])) != {key: clip[key] for key in ("frames", "pcm_sha256")}:
            raise ValueError(f"Corpus PCM checksum mismatch: {index}")
    for key in ("source_model", "ct2_model"):
        model = corpus[key]
        root = Path(model["path"]).resolve()
        for name, expected in model["files"].items():
            artifact = (root / name).resolve()
            if not artifact.is_relative_to(root) or sha256(artifact) != expected:
                raise ValueError(f"{key} checksum/path mismatch: {name}")
    source = corpus["source_model"]
    if (source.get("repo"), source.get("revision")) != (REPO, REVISION):
        raise ValueError("The frozen corpus must select the pinned Flix checkpoint")
    if read_json(Path(source["path"]) / ".shum-manifest.json") != {
        "repo": source["repo"], "revision": source["revision"], "files": source["files"],
    }:
        raise ValueError("Source manifest changed after freezing")
    return corpus


def memory_sample(pid: int | None = None) -> dict[str, int | float]:
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        info[name] = int(value.split()[0])
    vmstat = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    headroom = common.cgroup_headroom_bytes()
    available_kib = info["MemAvailable"]
    if headroom is not None:
        available_kib = min(available_kib, headroom // 1024)
    rss_kib = own_swap_kib = 0
    if pid is not None:
        rss_kib, own_swap_kib = common.process_tree_memory_kib(pid)
    psi_full_avg10 = common.memory_full_avg10(
        Path("/proc/pressure/memory").read_text(encoding="ascii")
    )
    return {
        "available_kib": available_kib, "rss_kib": rss_kib,
        "own_swap_kib": own_swap_kib, "psi_memory_full_avg10_percent": psi_full_avg10,
        "swap_in_pages": int(vmstat["pswpin"]), "swap_out_pages": int(vmstat["pswpout"]),
    }


def resource_reason(sample: dict[str, int | float], initial: dict[str, int | float]) -> str | None:
    if sample["available_kib"] < RESERVE_MIB * 1024:
        return "available_memory_below_4_gib_reserve"
    if sample["own_swap_kib"] > initial["own_swap_kib"]:
        return "worker_process_tree_vmswap_increased"
    if sample["swap_out_pages"] < initial["swap_out_pages"]:
        return "system_pswpout_counter_reset"
    if (sample["swap_out_pages"] > initial["swap_out_pages"]
            and sample["psi_memory_full_avg10_percent"] >= MEMORY_FULL_AVG10_THRESHOLD_PERCENT):
        return "memory_psi_full_avg10_at_least_1_percent_with_new_system_swap_out"
    return None


def selected_indices(value: str) -> list[int]:
    result = [int(item) for item in value.split(",")]
    if not result or any(not 0 <= index < 67 for index in result):
        raise ValueError("Indices must be a comma-separated sequence from 0 through 66")
    return result


def command_artifacts(command: list[str]) -> dict[str, str]:
    artifacts = {}
    for argument in command:
        if argument.endswith(".py"):
            script = Path(argument)
            if script.is_file():
                artifacts[str(script.resolve())] = sha256(script)
                common_path = script.with_name("flix_fast_common.py")
                if common_path.is_file():
                    artifacts[str(common_path.resolve())] = sha256(common_path)
    executable = Path(command[0])
    if executable.is_file():
        artifacts[str(executable.resolve())] = sha256(executable)
        venv_config = executable.parent.parent / "pyvenv.cfg"
        if venv_config.is_file():
            artifacts[str(venv_config.resolve())] = sha256(venv_config)
    return artifacts


def validate_response(response: Any, index: int) -> None:
    if isinstance(response, dict) and "error" in response:
        raise ValueError(f"Worker failed: {response['error']}")
    if not isinstance(response, dict) or type(response.get("index")) is not int or response["index"] != index:
        raise ValueError(f"Unexpected worker response index; expected {index}")
    if not isinstance(response.get("text"), str) or not isinstance(response.get("segments"), list):
        raise ValueError("Worker response lacks text/segments")


def repetition_fraction(text: str) -> float:
    words = text.lower().split()
    grams = [tuple(words[index:index + 4]) for index in range(max(0, len(words) - 3))]
    return (len(grams) - len(set(grams))) / len(grams) if grams else 0.0


def review_report(args: argparse.Namespace) -> Path:
    corpus = read_json(args.manifest)
    runs = []
    review_windows: dict[int, dict[str, Any]] = {}
    selected = set(selected_indices(args.indices)) if getattr(args, "indices", None) else set(range(67))

    def add_variant(index: int, text: str, run_name: str, request_number: int | None, flags: list[str]) -> None:
        clip = corpus["clips"][index]
        item = review_windows.setdefault(index, {
            "index": index, "start": clip["start"], "end": clip["end"],
            "audio_path": clip["path"], "reference_text": None,
            "review_status": "pending_human_audio_review", "variants": [],
        })
        item["variants"].append({
            "run": run_name, "request_number": request_number, "text": text, "flags": flags,
        })

    study_root = args.study_dir.resolve()
    for summary_path in sorted(study_root.rglob("summary.json")):
        run_dir = summary_path.parent
        run_name = str(run_dir.relative_to(study_root))
        summary = read_json(summary_path)
        if summary["manifest_sha256"] != sha256(args.manifest):
            raise ValueError(f"Run uses a different corpus manifest: {run_dir}")
        ready_path = run_dir / "diagnostics" / "ready.json"
        ready = read_json(ready_path) if ready_path.exists() else None
        windows = []
        for path in sorted(run_dir.glob("result-*-window-*.json")):
            match = re.fullmatch(r"result-(\d+)-window-(\d+)\.json", path.name)
            if match is None:
                raise ValueError(f"Unexpected result filename: {path}")
            request_number, index = map(int, match.groups())
            if index not in selected:
                continue
            result = read_json(path)
            validate_response(result, request_number)
            clip = corpus["clips"][index]
            fraction = repetition_fraction(result["text"])
            flags = []
            if fraction > 0.5:
                flags.append("repeated_fourgrams")
            if not result["text"].strip():
                flags.append("empty_text_not_proof_of_omission")
            if any(segment["start"] < 0 or segment["end"] > clip["frames"] / 16000 for segment in result["segments"]):
                flags.append("segment_outside_clip")
            diagnostic_path = run_dir / "diagnostics" / f"request-{request_number:05d}.json"
            diagnostic = read_json(diagnostic_path) if diagnostic_path.exists() else None
            if diagnostic and any(
                field["path"].startswith("segments[") for field in diagnostic.get("nonfinite_fields", [])
            ):
                flags.append("nonfinite_engine_metrics")
            windows.append({
                "index": index, "request_number": request_number,
                "characters": len(result["text"]), "repeated_fourgram_fraction": fraction,
                "text_sha256": hashlib.sha256(result["text"].encode()).hexdigest(), "flags": flags,
                "final_temperatures": [segment["temperature"] for segment in diagnostic["segments"]] if diagnostic else None,
                "duration_after_vad_seconds": diagnostic.get("duration_after_vad_seconds") if diagnostic else None,
            })
            add_variant(index, result["text"], run_name, request_number, flags)
        runs.append({
            "run": run_name, "status": summary["status"], "reason": summary["reason"],
            "completed_indices": summary["completed_indices"],
            "elapsed_seconds": summary.get("elapsed_seconds"),
            "model_ready_seconds": summary["model_ready_seconds"],
            "peak_rss_kib": summary["peak_rss_kib"],
            "resource_guard_semantics": summary.get("resource_guard_semantics"),
            "memory_full_avg10_threshold_percent": summary.get("memory_full_avg10_threshold_percent"),
            "initial_memory": summary.get("initial_memory"),
            "last_memory": summary.get("last_memory"),
            "profile": ready, "windows": windows,
        })
    for run_dir in getattr(args, "historical_run", []):
        summary = read_json(run_dir / "summary.json")
        expected_hashes = {Path(clip["path"]).name: clip["sha256"] for clip in corpus["clips"]}
        if summary.get("input_sha256") != expected_hashes:
            raise ValueError(f"Historical run has different WAV inputs: {run_dir}")
        for index in sorted(selected):
            path = run_dir / f"{index:05d}.json"
            if not path.exists():
                continue
            result = read_json(path)
            flags = ["historical_engine_versions_and_weights_not_reattested"]
            if repetition_fraction(result["text"]) > 0.5:
                flags.append("repeated_fourgrams")
            if not result["text"].strip():
                flags.append("empty_text_not_proof_of_omission")
            add_variant(index, result["text"], "historical:" + run_dir.name, None, flags)
    if not runs:
        raise ValueError("No completed or stopped diagnostic runs found")
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Refusing to replace an existing review report: {output}")
    write_json(output, {
        "schema": 1, "manifest_sha256": sha256(args.manifest),
        "quality_status": "unverified", "wer": None, "cer": None, "runs": runs,
        "review_windows": [review_windows[index] for index in sorted(review_windows)],
        "warning": "Automatic flags and text agreement do not establish audio accuracy.",
    })
    return output


def run_diagnostic(args: argparse.Namespace) -> Path:
    if not math.isfinite(args.timeout_seconds) or not 0 < args.timeout_seconds <= 600:
        raise ValueError("Timeout must be finite, positive, and no greater than 600 seconds")
    if args.estimated_peak_mib < 4096:
        raise ValueError("Use a conservative single-worker estimate of at least 4096 MiB")
    full_corpus = getattr(args, "full_corpus", False)
    indices = list(range(67)) if full_corpus else selected_indices(args.indices)
    corpus = verify_corpus(args.manifest)
    command = list(args.worker_command)
    if not command or "--serve" not in command or "--diagnostics-dir" in command:
        raise ValueError("Supply a persistent worker command with --serve and without --diagnostics-dir")
    for flag, expected in (
        ("--model", corpus["source_model"]["path"]), ("--ct2-model", corpus["ct2_model"]["path"]),
    ):
        if command.count(flag) != 1 or command.index(flag) + 1 >= len(command):
            raise ValueError(f"Worker command must explicitly select {flag}")
        if Path(command[command.index(flag) + 1]).resolve() != Path(expected):
            raise ValueError(f"Worker command {flag} differs from the frozen model")
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Refusing to replace an existing run: {output}")
    output.mkdir(mode=0o700, parents=True)
    try:
        initial = memory_sample()
        sampling_error = None
    except (OSError, ValueError, KeyError, common.AdapterError) as exc:
        initial = None
        sampling_error = str(exc)
    summary: dict[str, Any] = {
        "status": "blocked", "reason": None, "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest), "selected_indices": indices,
        "completed_indices": [], "peak_rss_kib": 0, "limit_seconds": args.timeout_seconds,
        "quality_status": "unverified", "command": command, "initial_memory": initial,
        "resource_samples": [] if initial is None else [initial],
        "resource_guard_semantics": RESOURCE_GUARD_SEMANTICS,
        "memory_full_avg10_threshold_percent": MEMORY_FULL_AVG10_THRESHOLD_PERCENT,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "full_corpus": full_corpus, "full_speed_gate": False, "model_ready_seconds": None,
    }
    needed = (args.estimated_peak_mib + RESERVE_MIB) * 1024
    if sampling_error is not None:
        summary["reason"] = f"resource_monitor_error_before_launch: {sampling_error}"
        summary["missing_indices"] = indices
        write_json(output / "summary.json", summary)
        return output / "summary.json"
    assert initial is not None
    if initial["available_kib"] < needed:
        summary["reason"] = "insufficient_memory_for_estimated_peak_plus_reserve"
        summary["missing_indices"] = indices
        write_json(output / "summary.json", summary)
        return output / "summary.json"
    diagnostics = output / "diagnostics"
    command += ["--diagnostics-dir", str(diagnostics)]
    summary["command"] = command
    runtime_artifacts = command_artifacts(command)
    summary["runtime_artifacts_sha256"] = runtime_artifacts
    responses: queue.Queue[tuple[float, str | Exception | None]] = queue.Queue()
    started = time.monotonic()
    process = None
    try:
        with (output / "worker.stderr").open("x", encoding="utf-8") as error_log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=error_log, text=True)
            assert process.stdin is not None and process.stdout is not None

            def read_responses() -> None:
                try:
                    while line := process.stdout.readline(2 * 1024 * 1024):
                        if not line.endswith("\n"):
                            raise ValueError("Worker response exceeds limit or is incomplete")
                        responses.put((time.monotonic(), line))
                    responses.put((time.monotonic(), None))
                except (OSError, UnicodeError, ValueError) as exc:
                    responses.put((time.monotonic(), exc))

            reader = threading.Thread(target=read_responses, daemon=True)
            reader.start()
            summary["status"] = "running"
            with (output / "events.jsonl").open("x", encoding="utf-8") as events:
                for request_number, index in enumerate(indices):
                    if time.monotonic() - started >= args.timeout_seconds:
                        summary.update(status="timeout", reason="deadline_before_request")
                        break
                    sent = time.monotonic()
                    process.stdin.write(json.dumps({"index": request_number, "input": corpus["clips"][index]["path"]}) + "\n")
                    process.stdin.flush()
                    while True:
                        try:
                            sample = memory_sample(process.pid)
                            reason = resource_reason(sample, initial)
                        except (OSError, ValueError, KeyError, common.AdapterError) as exc:
                            summary.update(status="resource_abort", reason=f"resource_monitor_error: {exc}")
                            break
                        summary["peak_rss_kib"] = max(summary["peak_rss_kib"], sample["rss_kib"])
                        summary["last_memory"] = sample
                        summary["resource_samples"].append(sample)
                        if reason is not None:
                            summary.update(status="resource_abort", reason=reason)
                            break
                        ready_path = diagnostics / "ready.json"
                        if summary["model_ready_seconds"] is None and ready_path.exists():
                            ready = read_json(ready_path)
                            summary["model_ready_seconds"] = ready["ready_monotonic_seconds"] - started
                        if time.monotonic() - started >= args.timeout_seconds:
                            summary.update(status="timeout", reason="deadline_waiting_for_result")
                            break
                        try:
                            received, line = responses.get(timeout=0.1)
                        except queue.Empty:
                            continue
                        if received - started >= args.timeout_seconds:
                            summary.update(status="timeout", reason="late_result_excluded")
                            break
                        if isinstance(line, Exception):
                            raise line
                        if line is None:
                            raise ValueError("Worker exited before returning the requested window")
                        response = json.loads(line)
                        validate_response(response, request_number)
                        write_json(output / f"result-{request_number:05d}-window-{index:05d}.json", response)
                        event = {
                            "request_number": request_number, "index": index, "worker_id": 0,
                            "sent_seconds": sent - started, "completed_seconds": received - started,
                            "latency_seconds": received - sent, "characters": len(response["text"]),
                        }
                        events.write(json.dumps(event) + "\n")
                        events.flush()
                        summary["completed_indices"].append(index)
                        break
                    if summary["status"] != "running":
                        break
                else:
                    summary.update(status="complete", reason=None)
    except (OSError, common.AdapterError, KeyError) as exc:
        summary.update(status="resource_abort", reason=f"resource_monitor_error: {exc}")
    except (ValueError, UnicodeError) as exc:
        summary.update(status="error", reason=str(exc))
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            summary["worker_exit_code"] = process.returncode
            if process.stdout is not None:
                process.stdout.close()
            if process.stdin is not None:
                process.stdin.close()
        summary["missing_indices"] = indices[len(summary["completed_indices"]):]
        try:
            if any(sha256(Path(path)) != expected for path, expected in runtime_artifacts.items()):
                summary.update(status="error", reason="runtime_artifact_changed_during_run")
        except OSError as exc:
            summary.update(status="error", reason=f"cannot_reverify_runtime_artifact: {exc}")
        summary["full_speed_gate"] = (
            full_corpus and summary["status"] == "complete"
            and summary["completed_indices"] == list(range(67))
            and summary["elapsed_seconds"] < 600.0
        )
        summary["shutdown_seconds"] = time.monotonic() - started - summary["elapsed_seconds"]
        write_json(output / "summary.json", summary)
    return output / "summary.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    freeze = commands.add_parser("freeze")
    for name in ("source-model", "ct2-model", "clips-dir", "chunks-json", "audio", "output"):
        freeze.add_argument(f"--{name}", required=True, type=Path)
    run = commands.add_parser("run")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    selection = run.add_mutually_exclusive_group(required=True)
    selection.add_argument("--indices")
    selection.add_argument("--full-corpus", action="store_true", help="Process all 67 windows; a speed pass does not establish quality.")
    run.add_argument("--timeout-seconds", type=float, default=600)
    run.add_argument("--estimated-peak-mib", type=int, default=4096)
    run.add_argument("--worker-command", nargs=argparse.REMAINDER, required=True)
    review = commands.add_parser("review")
    for name in ("manifest", "study-dir", "output"):
        review.add_argument(f"--{name}", required=True, type=Path)
    review.add_argument("--indices", help="Optional fixed comma-separated audio-review selection")
    review.add_argument("--historical-run", type=Path, action="append", default=[])
    args = parser.parse_args()
    try:
        actions = {"freeze": freeze_corpus, "run": run_diagnostic, "review": review_report}
        path = actions[args.action](args)
        print(path)
        if args.action == "run" and read_json(path)["status"] != "complete":
            return 1
        return 0
    except (OSError, ValueError, KeyError, TypeError, wave.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
