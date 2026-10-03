#!/usr/bin/env python3
"""Bounded, engine-independent diagnostic of the frozen 67-clip/1800-second corpus.

The source description is JSON with nonempty ``engine``, ``checkpoint`` and
``runtime`` identities, their ``engine_sha256``, ``checkpoint_sha256`` and
``runtime_sha256`` (64 lowercase hex characters), and a nonempty ``files`` or
``runtime_artifacts_sha256`` object mapping absolute artifact paths to SHA-256
digests. If both maps are supplied, they must match. Each identity hash
must occur in the map; all mapped artifacts are verified before and after the
run. List *all* model, binary, and runtime artifacts needed to identify the
experiment in ``files``. Neither a speed pass nor automatic repetition flags
establish transcription quality (WER/CER remain unverified).

Example:
  python scripts/asr_engine_study.py --manifest /abs/corpus.json \
    --output /abs/new-run --full-corpus --source-description /abs/source.json \
    --estimated-peak-mib 8192 --max-in-flight 4 --worker-command \
    /abs/venv/bin/python -u /abs/worker.py --serve

Worker protocol: one JSON line {"index": request_number, "input": absolute_wav}
per request; one JSON line {"index": request_number, "text": string,
"segments": list} per response. Indices identify requests, not clip numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flix_study as study

HASH = re.compile(r"[0-9a-f]{64}\Z")
RESPONSE_LIMIT = 2 * 1024 * 1024


def verify_corpus(path: Path) -> dict[str, Any]:
    corpus = study.read_json(path)
    if not isinstance(corpus, dict) or corpus.get("schema") != 1 or corpus.get("duration_seconds") != 1800.0:
        raise ValueError("Expected a schema 1 frozen 1800-second corpus")
    clips = corpus.get("clips")
    if not isinstance(clips, list) or len(clips) != 67:
        raise ValueError("Expected exactly 67 frozen clips")
    audio = Path(corpus["audio_path"])
    if not audio.is_absolute() or study.sha256(audio) != corpus["audio_sha256"]:
        raise ValueError("Frozen audio path/checksum mismatch")
    audio_info = study.wav_metadata(audio)
    if audio_info["frames"] < 1800 * 16000:
        raise ValueError("Frozen audio is shorter than 1800 seconds")
    with wave.open(str(audio), "rb") as recording:
        for index, clip in enumerate(clips):
            file = Path(clip["path"])
            if type(clip["index"]) is not int or clip["index"] != index or not file.is_absolute():
                raise ValueError(f"Invalid clip identity/path: {index}")
            if study.sha256(file) != clip["sha256"]:
                raise ValueError(f"Clip WAV checksum mismatch: {index}")
            metadata = study.wav_metadata(file)
            if metadata != {key: clip[key] for key in ("frames", "pcm_sha256")}:
                raise ValueError(f"Clip PCM checksum mismatch: {index}")
            start, end = clip["start"], clip["end"]
            if (not isinstance(start, (int, float)) or isinstance(start, bool)
                    or not isinstance(end, (int, float)) or isinstance(end, bool)
                    or not math.isfinite(start) or not math.isfinite(end)
                    or not 0 <= start < end <= 1800
                    or abs(metadata["frames"] / 16000 - (end - start)) > 1 / 16000 + 1e-9):
                raise ValueError(f"Invalid clip bounds/duration: {index}")
            recording.setpos(round(start * 16000))
            if hashlib.sha256(recording.readframes(metadata["frames"])).hexdigest() != metadata["pcm_sha256"]:
                raise ValueError(f"Clip PCM differs from frozen audio: {index}")
    return corpus


def source_artifacts(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    description = study.read_json(path)
    if not isinstance(description, dict):
        raise ValueError("Source description must be an object")
    for field in ("engine", "checkpoint", "runtime"):
        if not isinstance(description.get(field), str) or not description[field].strip():
            raise ValueError(f"Source description requires {field} identity")
        if not isinstance(description.get(field + "_sha256"), str) or not HASH.fullmatch(description[field + "_sha256"]):
            raise ValueError(f"Source description requires {field}_sha256")
    files = description.get("files", description.get("runtime_artifacts_sha256"))
    if "files" in description and "runtime_artifacts_sha256" in description and files != description["runtime_artifacts_sha256"]:
        raise ValueError("Source files and runtime_artifacts_sha256 maps disagree")
    if not isinstance(files, dict) or not files:
        raise ValueError("Source description requires a nonempty files or runtime_artifacts_sha256 map")
    for name, digest in files.items():
        if not isinstance(name, str) or not Path(name).is_absolute() or not isinstance(digest, str) or not HASH.fullmatch(digest):
            raise ValueError(f"Invalid source artifact path or SHA-256: {name}")
    for field in ("engine", "checkpoint", "runtime"):
        if description[field + "_sha256"] not in files.values():
            raise ValueError(f"{field}_sha256 must identify a hashed file in source files")
    check_artifacts(files)
    return description, files


def check_artifacts(artifacts: dict[str, str]) -> None:
    for name, expected in artifacts.items():
        path = Path(name)
        if not path.is_file() or study.sha256(path) != expected:
            raise ValueError(f"Artifact checksum/path mismatch: {name}")


def worker_artifacts(command: list[str]) -> dict[str, str]:
    exe = Path(command[0])
    if not exe.is_absolute() or not exe.is_file() or not os.access(exe, os.X_OK):
        raise ValueError("Worker executable must be an existing executable absolute path")
    for argument in command[1:]:
        if argument.endswith(".py") and (not Path(argument).is_absolute() or not Path(argument).is_file()):
            raise ValueError(f"Worker script must be an existing absolute path: {argument}")
    artifacts = study.command_artifacts(command)
    if str(exe.resolve()) not in artifacts:
        raise ValueError("Cannot hash worker executable")
    return artifacts


def validate_response(response: Any, pending: dict[int, tuple[int, float]]) -> int:
    if not isinstance(response, dict):
        raise ValueError("Worker response must be a JSON object")
    if "error" in response:
        raise ValueError(f"Worker failed: {str(response['error'])[:500]}")
    number = response.get("index")
    if type(number) is not int or number not in pending:
        raise ValueError(f"Unexpected, duplicate or late worker response index: {number!r}")
    if not isinstance(response.get("text"), str) or not isinstance(response.get("segments"), list):
        raise ValueError("Worker response requires text string and segments list")
    for segment in response["segments"]:
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            raise ValueError("Worker segment requires text string")
        for field in ("start", "end"):
            value = segment.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"Worker segment {field} must be finite")
        if segment["start"] < 0 or segment["end"] < segment["start"]:
            raise ValueError("Worker segment has invalid bounds")
    return number


def group_alive(pgid: int) -> bool:
    """Ignore zombies: they have exited but may await their external reaper."""
    for task in Path("/proc").iterdir():
        if not task.name.isdecimal():
            continue
        try:
            stat = (task / "stat").read_text().rsplit(")", 1)[1].split()
            if int(stat[2]) == pgid and stat[0] != "Z":
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def stop_group(process: subprocess.Popen[str]) -> bool:
    pgid = process.pid
    for sig, interval in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        end = time.monotonic() + interval
        while time.monotonic() < end:
            process.poll()
            if not group_alive(pgid):
                process.wait(timeout=1)
                return True
            time.sleep(0.02)
    process.poll()
    return not group_alive(pgid)


def run_diagnostic(args: argparse.Namespace) -> Path:
    if not math.isfinite(args.timeout_seconds) or not 0 < args.timeout_seconds <= 600:
        raise ValueError("Timeout must be positive, finite and <= 600 seconds")
    if args.estimated_peak_mib < 4096 or not 1 <= args.max_in_flight <= 4:
        raise ValueError("Estimated peak must be >= 4096 MiB and max-in-flight must be 1..4")
    indices = list(range(67)) if args.full_corpus else study.selected_indices(args.indices)
    command = list(args.worker_command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("Supply a persistent --worker-command")
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Refusing to replace an existing run: {output}")
    output.mkdir(mode=0o700, parents=True)
    preparation_started = time.monotonic()
    summary: dict[str, Any] = {
        "status": "blocked", "reason": None, "manifest_path": str(args.manifest.resolve()),
        "source_description_path": str(args.source_description.resolve()),
        "selected_indices": indices, "completed_indices": [], "missing_indices": indices[:],
        "quality_status": "unverified", "wer": None, "cer": None,
        "full_corpus": args.full_corpus, "full_speed_gate": False,
        "limit_seconds": args.timeout_seconds, "max_in_flight": args.max_in_flight,
        "command": command, "peak_rss_kib": 0, "resource_samples": [],
        "resource_guard_semantics": study.RESOURCE_GUARD_SEMANTICS,
        "memory_full_avg10_threshold_percent": study.MEMORY_FULL_AVG10_THRESHOLD_PERCENT,
    }
    report = output / "summary.json"
    process: subprocess.Popen[str] | None = None
    reader: threading.Thread | None = None
    responses: queue.Queue[tuple[float, str | Exception | None]] = queue.Queue()
    started: float | None = None
    hashes: dict[str, str] = {}
    source_files: dict[str, str] = {}
    try:
        corpus = verify_corpus(args.manifest)
        description, source_files = source_artifacts(args.source_description)
        hashes = {
            str(args.manifest.resolve()): study.sha256(args.manifest),
            str(args.source_description.resolve()): study.sha256(args.source_description),
            **source_files, **worker_artifacts(command),
        }
        # Keep independently supplied source digests even when a command file overlaps.
        check_artifacts(source_files)
        summary.update(
            manifest_sha256=study.sha256(args.manifest),
            source_description_sha256=study.sha256(args.source_description),
            source_description=description, runtime_artifacts_sha256=study.command_artifacts(command),
            source_artifacts_sha256=source_files,
        )
        try:
            initial = study.memory_sample()
        except (OSError, ValueError, KeyError, study.common.AdapterError) as exc:
            summary.update(status="resource_abort", reason=f"resource_monitor_error_before_launch: {exc}")
            return report
        summary["initial_memory"] = initial
        summary["resource_samples"].append(initial)
        if initial["available_kib"] < (args.estimated_peak_mib + study.RESERVE_MIB) * 1024:
            summary["reason"] = "insufficient_memory_for_estimated_peak_plus_reserve"
            return report

        started = time.monotonic()  # Includes launch and model loading; hashing above is preparation.
        with (output / "worker.stderr").open("x", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                text=True, bufsize=1, start_new_session=True,
            )
            summary["worker_pid"] = process.pid
            assert process.stdout is not None and process.stdin is not None

            def read_responses() -> None:
                try:
                    while line := process.stdout.readline(RESPONSE_LIMIT + 1):
                        if not line.endswith("\n") or len(line) > RESPONSE_LIMIT:
                            raise ValueError("Worker response exceeds limit or is incomplete")
                        responses.put((time.monotonic(), line))
                    responses.put((time.monotonic(), None))
                except (OSError, UnicodeError, ValueError) as exc:
                    responses.put((time.monotonic(), exc))

            reader = threading.Thread(target=read_responses, daemon=True)
            reader.start()
            pending: dict[int, tuple[int, float]] = {}
            completed: set[int] = set()
            next_request = 0
            summary["status"] = "running"
            with (output / "requests.jsonl").open("x", encoding="utf-8") as requests, \
                    (output / "events.jsonl").open("x", encoding="utf-8") as events, \
                    (output / "responses.jsonl").open("x", encoding="utf-8") as response_log:
                while len(completed) < len(indices):
                    try:
                        sample = study.memory_sample(process.pid)
                    except (OSError, ValueError, KeyError, study.common.AdapterError) as exc:
                        summary.update(status="resource_abort", reason=f"resource_monitor_error: {exc}")
                        break
                    summary["resource_samples"].append(sample)
                    summary["last_memory"] = sample
                    summary["peak_rss_kib"] = max(summary["peak_rss_kib"], sample["rss_kib"])
                    reason = study.resource_reason(sample, initial)
                    if reason:
                        summary.update(status="resource_abort", reason=reason)
                        break
                    if time.monotonic() - started >= args.timeout_seconds:
                        summary.update(status="timeout", reason="deadline_waiting_for_result")
                        break
                    while next_request < len(indices) and len(pending) < args.max_in_flight:
                        number, index = next_request, indices[next_request]
                        sent = time.monotonic()
                        request = {"index": number, "input": corpus["clips"][index]["path"]}
                        process.stdin.write(json.dumps(request) + "\n")
                        process.stdin.flush()
                        requests.write(json.dumps({"request_number": number, "clip_index": index, **request}) + "\n")
                        requests.flush()
                        pending[number] = (index, sent)
                        next_request += 1
                    if time.monotonic() - started >= args.timeout_seconds:
                        summary.update(status="timeout", reason="deadline_after_dispatch")
                        break
                    try:
                        received, line = responses.get(timeout=min(0.1, max(0.001, started + args.timeout_seconds - time.monotonic())))
                    except queue.Empty:
                        if process.poll() is not None and responses.empty():
                            raise ValueError(f"Worker exited before completing requests (exit {process.returncode})")
                        continue
                    if received - started >= args.timeout_seconds:
                        summary.update(status="timeout", reason="late_result_excluded")
                        break
                    if isinstance(line, Exception):
                        raise line
                    if line is None:
                        raise ValueError(f"Worker exited before completing requests (exit {process.poll()})")
                    response_log.write(line)
                    response_log.flush()
                    response = json.loads(line)
                    number = validate_response(response, pending)
                    index, sent = pending.pop(number)
                    clip_duration = corpus["clips"][index]["frames"] / 16000
                    if any(segment["end"] > clip_duration + 1 / 16000 for segment in response["segments"]):
                        raise ValueError(f"Worker segment exceeds clip duration: {index}")
                    completed.add(number)
                    study.write_json(output / f"result-{number:05d}-window-{index:05d}.json", response)
                    event = {
                        "request_number": number, "index": index, "worker_id": 0,
                        "sent_seconds": sent - started, "completed_seconds": received - started,
                        "latency_seconds": received - sent, "characters": len(response["text"]),
                        "repeated_fourgram_fraction": study.repetition_fraction(response["text"]),
                    }
                    events.write(json.dumps(event, allow_nan=False) + "\n")
                    events.flush()
                    summary["completed_indices"] = [indices[i] for i in sorted(completed)]
                else:
                    process.stdin.close()
                    try:
                        received, extra = responses.get(timeout=min(0.1, max(0.001, started + args.timeout_seconds - time.monotonic())))
                        if extra is not None and received - started < args.timeout_seconds:
                            if isinstance(extra, str):
                                response_log.write(extra)
                                response_log.flush()
                            raise ValueError("Unsolicited or duplicate worker response")
                    except queue.Empty:
                        pass
                    summary.update(status="complete", reason=None)
    except (OSError, ValueError, KeyError, TypeError, wave.Error, json.JSONDecodeError, BrokenPipeError) as exc:
        summary.update(status="error", reason=str(exc)[:1000])
    finally:
        summary["preparation_seconds"] = (started or time.monotonic()) - preparation_started
        if process is not None:
            try:
                if process.stdin:
                    process.stdin.close()
                stopped = stop_group(process)
                if not stopped:
                    summary.update(status="error", reason="Worker process group survived shutdown")
            except (OSError, subprocess.TimeoutExpired) as exc:
                summary.update(status="error", reason=f"Worker shutdown failed: {exc}")
            summary["worker_exit_code"] = process.poll()
            if reader is not None:
                reader.join(timeout=0.2)
            while not responses.empty():
                received, extra = responses.get_nowait()
                if isinstance(extra, str) and received - started < args.timeout_seconds and summary["status"] == "complete":
                    summary.update(status="error", reason="Unsolicited or duplicate worker response")
            if process.stdout:
                process.stdout.close()
        if started is not None:
            summary["elapsed_seconds"] = time.monotonic() - started
            if summary["status"] == "complete" and summary["elapsed_seconds"] >= args.timeout_seconds:
                summary.update(status="timeout", reason="deadline_during_shutdown")
        summary["missing_indices"] = [
            index for number, index in enumerate(indices)
            if number not in {int(p.name.split("-")[1]) for p in output.glob("result-*-window-*.json")}
        ]
        try:
            if hashes:
                check_artifacts(hashes)
                check_artifacts(source_files)
            if "manifest_sha256" in summary:
                verify_corpus(args.manifest)
        except (OSError, ValueError, KeyError, TypeError, wave.Error) as exc:
            summary.update(status="error", reason=f"Artifact changed after run: {exc}")
        summary["full_speed_gate"] = (
            args.full_corpus and summary["status"] == "complete"
            and summary["completed_indices"] == list(range(67))
            and summary.get("elapsed_seconds", float("inf")) < 600
        )
        study.write_json(report, summary)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--indices")
    selection.add_argument("--full-corpus", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--estimated-peak-mib", type=int, default=4096)
    parser.add_argument("--source-description", required=True, type=Path)
    parser.add_argument("--max-in-flight", type=int, default=1)
    parser.add_argument("--worker-command", nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_diagnostic(args)
        print(report)
        return 0 if study.read_json(report)["status"] == "complete" else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
