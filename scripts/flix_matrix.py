#!/usr/bin/env python3
"""Run sequential, bounded CT2 diagnostic profiles on one frozen corpus."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import flix_study as study

SUSPECTS = "8,23,52"
REPEATS = ",".join([SUSPECTS] * 3)
QUIET_CONTROLS = "7,24,0,32,40,44,45"
PROFILES = {
    "fast-repeat": (REPEATS, ["--fast-decode", "--vad-filter", "--beam-size", "2"], 0),
    "fast-reverse": ("52,23,8", ["--fast-decode", "--vad-filter", "--beam-size", "2"], 0),
    "fallback-repeat": (REPEATS, ["--vad-filter", "--beam-size", "2"], 0),
    "fallback-seed42": (SUSPECTS, ["--vad-filter", "--beam-size", "2"], 42),
    "fast-quiet": (QUIET_CONTROLS, ["--fast-decode", "--vad-filter", "--beam-size", "2"], 0),
    "fallback-quiet": (QUIET_CONTROLS, ["--vad-filter", "--beam-size", "2"], 0),
    "fallback-no-vad": ("0,23,32,40,44,45", ["--beam-size", "2"], 0),
    "fallback-beam1": (SUSPECTS, ["--vad-filter", "--beam-size", "1"], 0),
    "fallback-beam3": (SUSPECTS, ["--vad-filter", "--beam-size", "3"], 0),
}


def build_worker_command(
    corpus: dict[str, Any], python: Path, worker: Path, profile: str, threads: int,
) -> list[str]:
    _, options, seed = PROFILES[profile]
    return [
        str(python.absolute()), str(worker.resolve()),
        "--model", corpus["source_model"]["path"],
        "--ct2-model", corpus["ct2_model"]["path"],
        "--quantization", corpus["ct2_model"]["provenance"]["quantization"],
        "--threads", str(threads), "--serve", "--seed", str(seed),
        "--reset-seed-per-request", *options,
    ]


def run_matrix(args: argparse.Namespace) -> Path:
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if len(set(args.profiles)) != len(args.profiles):
        raise ValueError("Profiles must not repeat; use a new matrix for independent repetitions")
    for path in (args.python, args.worker):
        if not path.is_file():
            raise ValueError(f"Missing executable/script: {path}")
    corpus = study.verify_corpus(args.manifest)
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Refusing to replace an existing matrix: {output}")
    output.mkdir(mode=0o700, parents=True)
    report: dict[str, Any] = {
        "schema": 1, "status": "running", "manifest_sha256": study.sha256(args.manifest),
        "profiles_requested": args.profiles, "runs": [], "quality_status": "unverified",
        "full_speed_gate": False,
    }
    report_path = output / "matrix.json"
    study.write_json(report_path, report)
    runner = Path(__file__).with_name("flix_study.py")
    try:
        for profile in args.profiles:
            run_dir = output / profile
            indices = PROFILES[profile][0]
            command = [
                sys.executable, str(runner), "run", "--manifest", str(args.manifest.resolve()),
                "--output", str(run_dir), "--indices", indices, "--timeout-seconds", "600",
                "--worker-command", *build_worker_command(corpus, args.python, args.worker, profile, args.threads),
            ]
            status = subprocess.run(command, check=False)
            summary_path = run_dir / "summary.json"
            if not summary_path.is_file():
                raise ValueError(f"Profile {profile} exited {status.returncode} without a terminal report")
            summary = study.read_json(summary_path)
            report["runs"].append({"profile": profile, "exit_code": status.returncode, "summary": summary})
            study.write_json(report_path, report)
            if summary["status"] in ("blocked", "resource_abort", "error"):
                report["status"] = "stopped"
                report["stop_reason"] = summary["reason"]
                break
        else:
            report["status"] = (
                "finished" if all(run["summary"]["status"] == "complete" for run in report["runs"])
                else "finished_with_failures"
            )
    except (OSError, ValueError) as exc:
        report["status"] = "error"
        report["stop_reason"] = str(exc)
    finally:
        completed = {run["profile"] for run in report["runs"]}
        report["profiles_not_run"] = [profile for profile in args.profiles if profile not in completed]
        study.write_json(report_path, report)
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output", "python", "worker"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES), required=True)
    args = parser.parse_args()
    try:
        report = run_matrix(args)
        print(report)
        return 0 if study.read_json(report)["status"] == "finished" else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
