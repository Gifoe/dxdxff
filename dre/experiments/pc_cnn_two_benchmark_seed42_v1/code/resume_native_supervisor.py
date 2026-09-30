"""Bounded epoch-boundary resume for native Windows GPU process failures.

This supervisor never changes training arguments or checkpoint contents. It
restarts only known native process exits and stops after repeated no-progress
failures. Python-level exceptions require manual diagnosis.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


NATIVE_EXITS = {2147483651, 3221225477, 3221225501, 3221226505}  # Windows NTSTATUS


def stamp(paths):
    return tuple((path.name, path.stat().st_size, path.stat().st_mtime_ns)
                 if path.exists() else (path.name, None, None) for path in paths)


def publish(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--official-cnn", type=Path, required=True)
    parser.add_argument("--omni-cache", type=Path, required=True)
    parser.add_argument("--omni-official-split", type=Path, required=True)
    parser.add_argument("--omni-inner-split", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=12)
    args = parser.parse_args()
    work = args.runtime / "omni" / "fold1"
    marker = work / "PC_VALIDATION_SELECTION.json"
    checkpoints = [work / "stage_B_last.pt", work / "stage_C_last.pt", marker]
    status = args.runtime / "omni_native_resume_status.json"
    command = [str(args.python), str(args.code / "train_pccnn.py"),
               "--benchmark", "omni", "--fold", "1",
               "--official-cnn", str(args.official_cnn),
               "--omni-cache", str(args.omni_cache),
               "--omni-official-split", str(args.omni_official_split),
               "--omni-inner-split", str(args.omni_inner_split),
               "--protocol", str(args.protocol), "--runtime", str(args.runtime)]
    existing_attempts = [int(path.stem.rsplit("_", 1)[-1])
                         for path in args.runtime.glob("omni_pc_native_attempt_*.log")]
    first_attempt = max(existing_attempts, default=0) + 1
    no_progress = 0
    for attempt in range(first_attempt, args.max_attempts + 1):
        if marker.is_file():
            published = json.loads(marker.read_text(encoding="utf-8"))
            if published.get("status") == "TRAIN_VALIDATION_COMPLETE" and not published.get("test_accessed"):
                publish(status, {"state": "COMPLETE", "attempts": attempt - 1,
                                 "test_accessed": False})
                return
            raise RuntimeError("Unexpected PC completion marker")
        before = stamp(checkpoints)
        publish(status, {"state": "RUNNING", "attempt": attempt,
                         "checkpoint_stamp": before, "test_accessed": False})
        out = args.runtime / f"omni_pc_native_attempt_{attempt:02d}.log"
        err = args.runtime / f"omni_pc_native_attempt_{attempt:02d}.err"
        if out.exists() or err.exists():
            raise RuntimeError("Refusing to overwrite an existing native-attempt log")
        with out.open("wb") as output, err.open("wb") as errors:
            result = subprocess.run(command, stdout=output, stderr=errors,
                                    check=False)
        if marker.is_file():
            published = json.loads(marker.read_text(encoding="utf-8"))
            if result.returncode == 0 and published.get("status") == "TRAIN_VALIDATION_COMPLETE" and not published.get("test_accessed"):
                publish(status, {"state": "COMPLETE", "attempts": attempt,
                                 "test_accessed": False})
                return
        after = stamp(checkpoints)
        progressed = after != before
        no_progress = 0 if progressed else no_progress + 1
        exit_code = result.returncode & 0xFFFFFFFF
        publish(status, {"state": "NATIVE_FAILURE" if exit_code in NATIVE_EXITS else "NON_NATIVE_FAILURE",
                         "attempt": attempt, "exit_code_unsigned": exit_code,
                         "progressed": progressed, "consecutive_no_progress": no_progress,
                         "test_accessed": False})
        if exit_code not in NATIVE_EXITS:
            raise RuntimeError(f"Non-native trainer error {exit_code}; inspect {err}")
        if no_progress >= 3:
            raise RuntimeError("Three native failures without a completed epoch; stop for diagnosis")
        time.sleep(30)
    raise RuntimeError("Native failure retry budget exhausted")


if __name__ == "__main__":
    main()
