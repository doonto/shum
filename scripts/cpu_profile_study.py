#!/usr/bin/env python3
"""Serial CPU experiments with owned-process accounting and strict safeguards."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import queue
import resource
import signal
import subprocess
import threading
import time

import asr_engine_study as engine
import flix_study as study


def cpu_limits():
    affinity = sorted(os.sched_getaffinity(0))
    relative = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
    leaf = Path("/sys/fs/cgroup") / relative.lstrip("/")
    ancestors = [p for p in (leaf, *leaf.parents) if p.is_relative_to("/sys/fs/cgroup")]
    rows, quota = [], float(len(affinity))
    for path in ancestors:
        row = {"path": str(path)}
        for name in ("cpu.max", "cpuset.cpus.effective", "memory.max",
                     "memory.current", "memory.pressure"):
            file = path / name
            if file.is_file():
                row[name] = file.read_text().strip()
        if "cpu.max" in row:
            amount, period = row["cpu.max"].split()
            if amount != "max":
                quota = min(quota, int(amount) / int(period))
        rows.append(row)
    return {"affinity": affinity, "logical_cpus": os.cpu_count(),
            "effective_cpu_capacity": quota, "ancestors": rows}


def process_table():
    result = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            result[int(directory.name)] = {
                "ppid": int(fields[1]), "pgid": int(fields[2]),
                "ticks": int(fields[11]) + int(fields[12]),
                "child_ticks": int(fields[13]) + int(fields[14]),
                "start": int(fields[19]), "state": fields[0],
                "name": (directory / "comm").read_text().strip(),
            }
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            continue
    return result


def owned_pids(table, roots):
    owned = {pid for pid, row in table.items() if pid in roots or row["pgid"] in roots}
    while True:
        descendants = {pid for pid, row in table.items() if row["ppid"] in owned}
        if descendants <= owned:
            return owned
        owned |= descendants


def sample(roots):
    table = process_table()
    pids = owned_pids(table, roots)
    measured = study.memory_sample()
    rss = swap = threads = 0
    for pid in pids:
        try:
            values = {}
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                key, _, value = line.partition(":")
                if key in ("VmRSS", "VmSwap", "Threads"):
                    values[key] = int(value.split()[0])
            rss += values.get("VmRSS", 0)
            swap += values.get("VmSwap", 0)
            threads += values.get("Threads", 0)
        except (FileNotFoundError, ProcessLookupError):
            continue
    measured.update(rss_kib=rss, own_swap_kib=swap, owned_pids=sorted(pids),
                    live_processes=sum(table[pid]["state"] != "Z" for pid in pids),
                    kernel_threads=threads,
                    cpu_seconds_sampled=sum(table[p]["ticks"] + table[p]["child_ticks"]
                                            for p in pids) / os.sysconf("SC_CLK_TCK"))
    measured["load_average"] = list(os.getloadavg())
    measured["cpu_frequency_khz"] = {}
    for cpu in os.sched_getaffinity(0):
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
        if path.is_file():
            measured["cpu_frequency_khz"][str(cpu)] = int(path.read_text())
    return measured


def foreign_model_processes(excluded):
    blocked = []
    for pid, row in process_table().items():
        if pid in excluded or row["state"] == "Z":
            continue
        if row["name"] not in ("python", "python3", "shum", "qwen-native-worker", "whisper-cli"):
            continue
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if ("--model" in command and ("--serve" in command or "--input" in command)
                and ("diarization" in command or "/adapters/asr/" in command
                     or "/models/" in command)):
            blocked.append({"pid": pid, "name": row["name"]})
    return blocked


class BackgroundCPU:
    def __init__(self):
        self.previous = process_table()
        self.when = time.monotonic()

    def observe(self, roots):
        current = process_table()
        now = time.monotonic()
        excluded = owned_pids(current, roots) | {os.getpid(), os.getppid()}
        seconds = now - self.when
        ticks = 0
        for pid, row in current.items():
            previous = self.previous.get(pid)
            if pid not in excluded and previous and previous["start"] == row["start"]:
                ticks += max(0, row["ticks"] - previous["ticks"])
        self.previous, self.when = current, now
        return ticks / os.sysconf("SC_CLK_TCK") / seconds if seconds > 0 else 0.0

    def initial(self):
        time.sleep(1)
        return self.observe([])


def validate_diarization(value, duration):
    if not isinstance(value, dict) or set(value) != {"turns", "exclusive_turns"}:
        raise ValueError("Invalid diarization result schema")
    for name in ("turns", "exclusive_turns"):
        if not isinstance(value[name], list):
            raise ValueError("Diarization turns must be an array")
        for turn in value[name]:
            if (not isinstance(turn, dict)
                    or not isinstance(turn.get("speaker"), str) or not turn["speaker"]
                    or any(isinstance(turn.get(k), bool)
                           or not isinstance(turn.get(k), (float, int))
                           or not math.isfinite(turn[k]) for k in ("start", "end"))
                    or not 0 <= turn["start"] < turn["end"] <= duration + 0.5):
                raise ValueError("Invalid diarization turn")


def telemetry(output):
    events = []
    for path in sorted(output.glob("worker-*.stderr")):
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("{"):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    events.append({"worker": path.stem, **value})
    return events


def verify_stats(passport):
    for name, expected in passport["file_stats"].items():
        stat = Path(name).stat()
        if [stat.st_size, stat.st_mtime_ns, stat.st_ino] != expected:
            raise ValueError(f"Pinned artifact metadata changed: {name}")


def run(spec, output, passport):
    if output.exists():
        raise ValueError(f"Refusing to overwrite trial: {output}")
    output.mkdir(parents=True, mode=0o700)
    limit = spec["limit_seconds"]
    if not 0 < limit <= 600:
        raise ValueError("Run deadline must be positive and at most 600 seconds")
    if (type(spec["processes"]) is not int or not 1 <= spec["processes"] <= 4
            or type(spec["threads"]) is not int or spec["threads"] < 1
            or spec["component"] not in ("primeline", "turbo", "diarization")):
        raise ValueError("Invalid component or process/thread counts")
    if spec.get("full_corpus") and (
            spec["component"] == "diarization" and spec["duration_seconds"] != 1800
            or spec["component"] != "diarization"
            and [item["index"] for item in spec["inputs"]] != list(range(67))):
        raise ValueError("Full runs require exactly 1800 seconds or the 67 fixed windows")
    if time.time() + limit + 10 >= passport["session_deadline_epoch"]:
        raise ValueError("Night budget cannot accommodate a full deadline and cleanup")
    limits = cpu_limits()
    if spec["processes"] * spec["threads"] > limits["effective_cpu_capacity"]:
        raise ValueError("Profile exceeds effective CPU capacity")
    if spec["component"] == "primeline" and spec["processes"] != 1:
        raise ValueError("primeLine must use one process and one request")
    preparation = time.monotonic()
    verify_stats(passport)
    initial = sample([])
    busy = foreign_model_processes({os.getpid(), os.getppid()})
    background = BackgroundCPU()
    initial_background = background.initial()
    report = {"schema": 1, "profile": spec, "status": "blocked", "reason": None,
              "quality_status": "unverified", "full_speed_gate": False,
              "initial_memory": initial, "limits": limits, "completed_indices": [],
              "resource_samples": [], "peak_rss_kib": 0, "peak_processes": 0,
              "peak_kernel_threads": 0, "owned_pids": [], "foreign_model_processes": busy}
    report["initial_background_cpu_cores"] = initial_background
    if busy or initial_background >= 1.5 or initial["available_kib"] < (spec["estimated_peak_mib"] + 4096) * 1024:
        report["reason"] = ("foreign_model_running" if busy else
                            "foreign_heavy_cpu" if initial_background >= 1.5 else
                            "insufficient_memory_plus_4_gib_reserve")
        study.write_json(output / "summary.json", report)
        return report
    report["preparation_seconds"] = time.monotonic() - preparation
    processes, readers, streams = [], [], []
    replies = queue.Queue()
    used_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    pending, completed, results = {}, set(), {}
    report["status"] = "running"
    next_input = 0
    last_save = 0.0
    items = spec.get("inputs", [])
    eof_closed = False
    background_busy_since = None
    try:
        for worker in range(spec["processes"]):
            error = (output / f"worker-{worker}.stderr").open("x")
            streams.append(error)
            command = [str(part).replace("{output}", str(output / "diarization.json"))
                       for part in spec["command"]]
            environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                           "CUDA_VISIBLE_DEVICES": "", "OMP_MAX_ACTIVE_LEVELS": "1"}
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=error, text=True, start_new_session=True, env=environment)
            processes.append(child)

            def read(child=child, worker=worker):
                try:
                    for line in child.stdout:
                        replies.put((worker, line))
                except (OSError, UnicodeError) as error:
                    replies.put((worker, error))
                finally:
                    replies.put((worker, None))

            thread = threading.Thread(target=read, daemon=True)
            thread.start()
            readers.append(thread)
        roots = {p.pid for p in processes}
        while True:
            elapsed = time.monotonic() - started
            current = sample(roots)
            current["background_cpu_cores"] = background.observe(roots)
            if current["background_cpu_cores"] >= 1.5:
                if background_busy_since is None:
                    background_busy_since = time.monotonic()
            else:
                background_busy_since = None
            current["elapsed_seconds"] = elapsed
            report["resource_samples"].append(current)
            report["peak_rss_kib"] = max(report["peak_rss_kib"], current["rss_kib"])
            report["peak_processes"] = max(report["peak_processes"], current["live_processes"])
            report["peak_kernel_threads"] = max(report["peak_kernel_threads"], current["kernel_threads"])
            report["owned_pids"] = sorted(set(report["owned_pids"]) | set(current["owned_pids"]))
            reason = study.resource_reason(current, initial)
            if reason:
                report.update(status="resource_abort", reason=reason)
                break
            other = foreign_model_processes(current["owned_pids"] + [os.getpid(), os.getppid()])
            if other:
                report.update(status="external_interference", reason="foreign_model_started",
                              interfering_processes=other)
                break
            if background_busy_since is not None and time.monotonic() - background_busy_since >= 5:
                report.update(status="external_interference", reason="sustained_foreign_heavy_cpu")
                break
            if elapsed >= limit:
                report.update(status="timeout", reason="absolute_worker_deadline")
                break
            if spec["component"] != "diarization":
                for worker, child in enumerate(processes):
                    if worker not in pending and next_input < len(items):
                        item = items[next_input]
                        child.stdin.write(json.dumps({"index": item["index"], "input": item["path"]}) + "\n")
                        child.stdin.flush()
                        pending[worker] = item
                        next_input += 1
                if len(completed) == len(items) and not eof_closed:
                    for child in processes:
                        child.stdin.close()
                    eof_closed = True
            elif not eof_closed:
                processes[0].stdin.close()
                eof_closed = True
            try:
                worker, line = replies.get(timeout=0.2)
                if isinstance(line, Exception):
                    raise line
                if line is not None:
                    with (output / f"worker-{worker}.stdout").open("a") as log:
                        log.write(line)
                if line is not None and spec["component"] != "diarization":
                    if worker not in pending:
                        raise ValueError("Unexpected worker stdout")
                    item = pending.pop(worker)
                    response = json.loads(line)
                    if spec["component"] == "primeline" and isinstance(response, dict) and "index" not in response:
                        response = {**response, "index": item["index"]}
                    engine.validate_response(response, {item["index"]: (item["index"], 0)})
                    if any(segment["end"] > item["duration"] + 1 / 16000 for segment in response["segments"]):
                        raise ValueError("ASR output timestamp exceeds WAV")
                    study.write_json(output / f"result-{item['index']:05d}.json", response)
                    completed.add(item["index"])
                    results[item["index"]] = response
                    report["completed_indices"] = sorted(completed)
            except queue.Empty:
                pass
            exited = [child.poll() for child in processes]
            if any(code not in (None, 0) for code in exited):
                raise ValueError(f"Worker failed: {exited}")
            if (all(code is not None for code in exited) and not current["live_processes"]
                    and replies.empty() and all(not reader.is_alive() for reader in readers)):
                if spec["component"] == "diarization":
                    value = study.read_json(output / "diarization.json")
                    validate_diarization(value, spec["duration_seconds"])
                    report["completed_indices"] = [0]
                    report["speakers"] = sorted({t["speaker"] for t in value["turns"]})
                    report["speech_coverage_seconds"] = union_duration(value["turns"])
                elif len(completed) != len(items):
                    raise ValueError("Worker exited with missing ASR results")
                report.update(status="complete", reason=None)
                break
            if elapsed - last_save >= 2:
                report["elapsed_seconds_live"] = elapsed
                study.write_json(output / "summary.json", report)
                last_save = elapsed
    except (OSError, ValueError, KeyError, TypeError) as error:
        report.update(status="error", reason=str(error))
    except KeyboardInterrupt:
        report.update(status="interrupted", reason="experiment_runner_interrupted")
        raise
    finally:
        for child in processes:
            if child.stdin and not child.stdin.closed:
                child.stdin.close()
            if engine.group_alive(child.pid):
                if not engine.stop_group(child):
                    report.update(status="cleanup_failed", reason="owned_process_group_survived")
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                report.update(status="cleanup_failed", reason="owned_worker_survived_kill_deadline")
        for reader in readers:
            reader.join(timeout=2)
        for child in processes:
            child.stdout.close()
        for stream in streams:
            stream.close()
        report["elapsed_seconds"] = time.monotonic() - started
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        report["cpu_seconds"] = after.ru_utime + after.ru_stime - used_before.ru_utime - used_before.ru_stime
        report["cpu_average_cores"] = report["cpu_seconds"] / report["elapsed_seconds"]
        report["exit_codes"] = [p.returncode for p in processes]
        report["cleanup_verified"] = all(not engine.group_alive(p.pid) for p in processes)
        if report["status"] == "complete" and report["elapsed_seconds"] >= limit:
            report.update(status="timeout", reason="deadline_during_validation_or_cleanup")
        report["telemetry"] = telemetry(output)
        report["empty_indices"] = [i for i, v in results.items() if not v["text"].strip()]
        report["strong_repetitions"] = {str(i): study.repetition_fraction(v["text"])
                                       for i, v in results.items()
                                       if study.repetition_fraction(v["text"]) > 0.5}
        report["full_speed_gate"] = (spec.get("full_corpus", False)
                                     and report["status"] == "complete"
                                     and report["elapsed_seconds"] < 600
                                     and report["cleanup_verified"]
                                     and (spec["component"] == "diarization"
                                          or len(completed) == 67))
        try:
            verify_stats(passport)
        except (OSError, ValueError) as error:
            report.update(status="artifact_changed", reason=str(error), full_speed_gate=False)
        study.write_json(output / "summary.json", report)
    return report


def union_duration(turns):
    total, end = 0.0, 0.0
    for turn in sorted(turns, key=lambda t: (t["start"], t["end"])):
        total += max(0.0, turn["end"] - max(end, turn["start"]))
        end = max(end, turn["end"])
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--passport", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(study.read_json(args.spec), args.output, study.read_json(args.passport))
    print(json.dumps({k: report.get(k) for k in
                      ("status", "reason", "elapsed_seconds", "cpu_seconds", "peak_rss_kib",
                       "completed_indices", "full_speed_gate")}))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
