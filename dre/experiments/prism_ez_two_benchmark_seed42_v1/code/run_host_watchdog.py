"""Bounded outer recovery for host faults that terminate a child supervisor.

This helper is deliberately outside PRiSM-EZ's scientific computation.  It
only relaunches the same native-failure supervisor when Windows has terminated
that supervisor before it can record a terminal status.  It fingerprints the
private, patient-step resume state between launches and stops after a bounded
number of consecutive no-progress host interruptions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def resume_fingerprint(runtime: Path) -> str:
    """Hash only private resumable state; it never reads validation/test data."""
    digest = hashlib.sha256()
    for path in sorted((runtime / "ictal").glob("fold*/in_epoch.pt")):
        digest.update(str(path.relative_to(runtime)).encode("utf-8"))
        digest.update(path.read_bytes())
    for path in sorted((runtime / "ictal").glob("fold*/last.pt")):
        digest.update(str(path.relative_to(runtime)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_status(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def terminal_state(status: dict | None) -> str | None:
    if not status:
        return None
    state = status.get("status")
    if state == "COMPLETE":
        return "complete"
    if state == "STOPPED_NON_NATIVE_FAILURE":
        return "non_native_failure"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--supervisor", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--host-retries", type=int, default=96)
    parser.add_argument("--max-consecutive-no-progress", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=8.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise RuntimeError("the exact development command is required after --")
    if args.host_retries < 1 or args.max_consecutive_no_progress < 1:
        raise RuntimeError("retry limits must be positive")

    status_path = args.runtime / "ictal_native_supervisor_status.json"
    host_status_path = args.runtime / "ictal_host_watchdog_status.json"
    host_log_dir = args.runtime / "host_watchdog_attempts"
    host_log_dir.mkdir(parents=True, exist_ok=True)
    consecutive_no_progress = 0

    for host_attempt in range(1, args.host_retries + 1):
        before = resume_fingerprint(args.runtime)
        outer_stdout = host_log_dir / f"host_{host_attempt:03d}.log"
        outer_stderr = host_log_dir / f"host_{host_attempt:03d}.err"
        supervisor = [str(args.python), str(args.supervisor), "--runtime", str(args.runtime),
                      "--max-native-retries", "96", "--retry-delay-seconds", str(args.retry_delay_seconds),
                      "--attempt-prefix", f"host{host_attempt:03d}", "--", *command]
        atomic_json(host_status_path, {"status": "RUNNING", "host_attempt": host_attempt,
                                       "started_utc": utc_now(), "test_accessed": False})
        with outer_stdout.open("w", encoding="utf-8") as stdout, outer_stderr.open("w", encoding="utf-8") as stderr:
            result = subprocess.run(supervisor, stdout=stdout, stderr=stderr, check=False)
        state = terminal_state(read_status(status_path))
        if state == "complete" and result.returncode == 0:
            atomic_json(host_status_path, {"status": "COMPLETE", "host_attempt": host_attempt,
                                           "completed_utc": utc_now(), "test_accessed": False})
            return
        if state == "non_native_failure":
            atomic_json(host_status_path, {"status": "STOPPED_NON_NATIVE_FAILURE", "host_attempt": host_attempt,
                                           "returncode": int(result.returncode), "test_accessed": False})
            raise RuntimeError("inner supervisor stopped on a non-native failure")
        after = resume_fingerprint(args.runtime)
        consecutive_no_progress = 0 if after != before else consecutive_no_progress + 1
        record = {"status": "RETRYING_HOST_INTERRUPTION", "host_attempt": host_attempt,
                  "returncode": int(result.returncode), "progress_observed": after != before,
                  "consecutive_no_progress": consecutive_no_progress, "updated_utc": utc_now(),
                  "test_accessed": False}
        atomic_json(host_status_path, record)
        if consecutive_no_progress >= args.max_consecutive_no_progress:
            record["status"] = "STOPPED_NO_PROGRESS"
            atomic_json(host_status_path, record)
            raise RuntimeError("host recovery stopped after bounded consecutive no-progress interruptions")
        if host_attempt == args.host_retries:
            raise RuntimeError("host recovery retry budget exhausted")
        time.sleep(args.retry_delay_seconds)


if __name__ == "__main__":
    main()
