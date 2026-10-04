#!/usr/bin/env python3
"""Bounded resume-safe launcher for the ten predeclared training runs.

It never changes a scientific parameter.  A failed native process may be
retried only from the hash-bound ``last.pt`` checkpoint, and a Python-level
exception stops immediately for diagnosis.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def native_exit(code: int) -> bool:
    # Windows access violation, illegal instruction, stack-buffer overrun and
    # breakpoint exits observed previously in the GPU runtime.
    return int(code) in {3221225477, 3221225501, 3221225725, 3221225781, -1073741819}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--official-cnn", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--max-native-attempts", type=int, default=4)
    args = parser.parse_args()
    status_path = args.runtime / "TRAINING_SUPERVISOR_STATUS.json"
    order = [(fold, variant) for fold in range(1, 6) for variant in ("baseline", "plugin")]
    for fold, variant in order:
        completed = args.runtime / "checkpoints" / f"fold_{fold}" / variant / "TRAINING_COMPLETE.json"
        if completed.exists():
            continue
        for attempt in range(1, args.max_native_attempts + 1):
            log = args.runtime / f"train_fold{fold}_{variant}_attempt{attempt}.log"
            error_log = args.runtime / f"train_fold{fold}_{variant}_attempt{attempt}.err"
            command = [str(args.python), "-u", str(args.trainer), "--records", str(args.records),
                       "--private-manifest", str(args.private_manifest), "--protocol", str(args.protocol),
                       "--official-cnn", str(args.official_cnn), "--runtime", str(args.runtime),
                       "--fold", str(fold), "--variant", variant]
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(json.dumps({"status": "RUNNING", "fold": fold, "variant": variant,
                                                "attempt": attempt, "command": command}) + "\n", encoding="utf-8")
            with log.open("w", encoding="utf-8") as out, error_log.open("w", encoding="utf-8") as err:
                result = subprocess.run(command, stdout=out, stderr=err, check=False)
            if result.returncode == 0 and completed.exists():
                break
            if result.returncode == 0:
                raise RuntimeError("Trainer returned success without a complete marker")
            if not native_exit(result.returncode):
                raise RuntimeError(f"STOP_ENGINEERING_ERROR fold={fold} variant={variant} exit={result.returncode}")
            if attempt == args.max_native_attempts:
                raise RuntimeError(f"STOP_NATIVE_RUNTIME_UNSTABLE fold={fold} variant={variant}")
            time.sleep(5)
        else:
            raise AssertionError("unreachable")
    status_path.write_text(json.dumps({"status": "ALL_TRAINING_COMPLETE", "folds": 5,
                                        "variants": ["baseline", "plugin"]}) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
