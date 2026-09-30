"""Bounded restart supervisor for verified host-native GPU interruptions.

This program is deliberately outside the scientific training loop.  It only
relaunches the exact same development command after a process has terminated
with a previously observed NVIDIA native failure.  Patient-step resume state is
validated by ``train_prism.py`` before any optimizer update is continued.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


KNOWN_NATIVE_MARKERS = (
    "returned non-zero exit status 2147483651",  # Windows 0x80000003 breakpoint
    "nvcuda64.dll",
    "c10_cuda.dll",
    "torch_cpu.dll",
)


def is_known_native_failure(text: str) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in KNOWN_NATIVE_MARKERS)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--max-native-retries", type=int, default=96)
    parser.add_argument("--retry-delay-seconds", type=float, default=8.0)
    parser.add_argument("--attempt-prefix", type=str, default="")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise RuntimeError("the exact development command is required after --")
    if args.max_native_retries < 1:
        raise RuntimeError("max-native-retries must be positive")

    log_dir = args.runtime / "native_supervisor_attempts"
    status_path = args.runtime / "ictal_native_supervisor_status.json"
    log_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, args.max_native_retries + 1):
        stem = f"{args.attempt_prefix}_attempt_{attempt:03d}" if args.attempt_prefix else f"attempt_{attempt:03d}"
        stdout_path = log_dir / f"{stem}.log"
        stderr_path = log_dir / f"{stem}.err"
        atomic_json(status_path, {"status": "RUNNING", "attempt": attempt,
                                  "started_utc": utc_now(), "test_accessed": False})
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        if result.returncode == 0:
            atomic_json(status_path, {"status": "COMPLETE", "attempt": attempt,
                                      "completed_utc": utc_now(), "test_accessed": False})
            return
        failure_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        known_native = is_known_native_failure(failure_text)
        state = {"status": "RETRYING_NATIVE_FAILURE" if known_native else "STOPPED_NON_NATIVE_FAILURE",
                 "attempt": attempt, "returncode": int(result.returncode),
                 "known_native": known_native, "stderr_file": stderr_path.name,
                 "updated_utc": utc_now(), "test_accessed": False}
        atomic_json(status_path, state)
        if not known_native:
            raise RuntimeError(f"development stopped with non-native exit code {result.returncode}")
        if attempt == args.max_native_retries:
            raise RuntimeError("native retry budget exhausted without changing the scientific protocol")
        time.sleep(args.retry_delay_seconds)


if __name__ == "__main__":
    main()
