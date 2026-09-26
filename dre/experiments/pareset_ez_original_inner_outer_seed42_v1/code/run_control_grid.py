"""Frozen seed-42 matched 36-D PRQ/BCR controls; never evaluates test."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock = {
        "status": "FROZEN_MATCHED_CONTROL_GRID",
        "branches": ["prq", "bcr"], "folds": [1, 2, 3, 4, 5], "seed": 42,
        "epochs": 45, "minimum_epochs": 18, "patience": 6,
        "learning_rate": 0.0001, "weight_decay": 0.001, "dropout": 0.4,
        "patient_batch_size": 4, "threads": 2,
        "repair": "exact-zero forward and finite-gradient variance std",
        "test_evaluated": False,
        "runner_sha256": digest(args.runner),
        "supplement_models_sha256": digest(args.source_root / "epilens" / "models.py"),
        "data_sha256": digest(args.data), "manifest_sha256": digest(args.manifest),
    }
    lock_path = args.output_root / "CONTROL_LOCK.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text(encoding="utf-8")) != lock:
            raise RuntimeError("Control lock differs")
    else:
        atomic_json(lock_path, lock)
    status_path = args.output_root / "CONTROL_STATUS.json"
    for fold in (1, 2, 3, 4, 5):
        for branch in ("prq", "bcr"):
            cell = args.output_root / f"fold{fold}" / branch
            selected_path = cell / "selection.json"
            if selected_path.exists():
                if json.loads(selected_path.read_text(encoding="utf-8")).get("test_evaluated") is not False:
                    raise RuntimeError("Test-tainted control cell")
                print(f"SKIP_COMPLETE fold={fold} branch={branch}", flush=True)
                continue
            if cell.exists() and any(cell.iterdir()):
                raise RuntimeError(f"Incomplete nonempty cell; do not overwrite: {cell}")
            cell.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(args.runner),
                   "--source-root", str(args.source_root), "--data", str(args.data),
                   "--manifest", str(args.manifest), "--output", str(cell),
                   "--fold", str(fold), "--seed", "42", "--branch", branch,
                   "--epochs", "45", "--minimum-epochs", "18", "--patience", "6",
                   "--learning-rate", "0.0001", "--weight-decay", "0.001", "--dropout", "0.4",
                   "--patient-batch-size", "4", "--threads", "2", "--device", "cuda"]
            print(f"START fold={fold} branch={branch}", flush=True)
            t0 = time.time()
            with (args.output_root / f"fold{fold}_{branch}.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False, env={**os.environ, "PYTHONUNBUFFERED": "1"})
            elapsed = time.time() - t0
            if completed.returncode != 0 or not selected_path.exists():
                atomic_json(status_path, {"status": "FAILED", "fold": fold, "branch": branch, "exit_code": completed.returncode, "elapsed_seconds": elapsed})
                raise RuntimeError(f"Control failed: fold={fold}, branch={branch}")
            atomic_json(status_path, {"status": "RUNNING", "last_completed_fold": fold, "last_completed_branch": branch, "elapsed_seconds": elapsed, "test_evaluated": False})
            print(f"DONE fold={fold} branch={branch} seconds={elapsed:.1f}", flush=True)
    atomic_json(status_path, {"status": "COMPLETE", "cells": 10, "test_evaluated": False})
    print("CONTROL_GRID_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
