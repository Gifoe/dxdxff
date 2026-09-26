"""Frozen seed-42 PaReSet-EZ development grid. No test evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


VARIANTS = ("base", "full", "no_reference", "uniform_reference", "no_pool", "bce_only")
FOLDS = (1, 2, 3, 4, 5)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


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
        "status": "FROZEN_DEVELOPMENT_GRID",
        "seed": 42,
        "folds": FOLDS,
        "variants": VARIANTS,
        "epochs": 45,
        "minimum_epochs": 18,
        "patience": 6,
        "learning_rate": 0.0001,
        "weight_decay": 0.001,
        "dropout": 0.4,
        "patient_batch_size": 4,
        "threads": 2,
        "selection": "supplied threshold grid and patient-equal validation metrics",
        "test_evaluated": False,
        "runner_sha256": digest(args.runner),
        "model_sha256": digest(args.runner.parent / "pareset_ez.py"),
        "data_sha256": digest(args.data),
        "manifest_sha256": digest(args.manifest),
        "supplement_features_sha256": digest(args.source_root / "epilens" / "features.py"),
        "supplement_evaluation_sha256": digest(args.source_root / "epilens" / "evaluation.py"),
    }
    lock_path = args.output_root / "DEVELOPMENT_LOCK.json"
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing != lock:
            raise RuntimeError("Development lock differs; refusing to continue")
    else:
        write_json(lock_path, lock)
    status_path = args.output_root / "DEVELOPMENT_STATUS.json"
    for fold in FOLDS:
        for variant in VARIANTS:
            cell = args.output_root / f"fold{fold}" / variant
            selection = cell / "selection.json"
            if selection.exists():
                saved = json.loads(selection.read_text(encoding="utf-8"))
                if saved.get("test_evaluated") is not False:
                    raise RuntimeError(f"Unexpected test evaluation in {selection}")
                print(f"SKIP_COMPLETE fold={fold} variant={variant}", flush=True)
                continue
            if cell.exists() and any(cell.iterdir()):
                raise RuntimeError(f"Incomplete nonempty cell; do not overwrite: {cell}")
            cell.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, str(args.runner),
                "--source-root", str(args.source_root),
                "--data", str(args.data),
                "--manifest", str(args.manifest),
                "--output", str(cell),
                "--seed", "42", "--fold", str(fold), "--variant", variant,
                "--epochs", "45", "--minimum-epochs", "18", "--patience", "6",
                "--learning-rate", "0.0001", "--weight-decay", "0.001",
                "--dropout", "0.4", "--patient-batch-size", "4", "--threads", "2",
                "--device", "cuda",
            ]
            t0 = time.time()
            print(f"START fold={fold} variant={variant}", flush=True)
            with (args.output_root / f"fold{fold}_{variant}.log").open("w", encoding="utf-8") as log:
                process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False, env={**os.environ, "PYTHONUNBUFFERED": "1"})
            elapsed = time.time() - t0
            if process.returncode != 0 or not selection.exists():
                write_json(status_path, {"status": "FAILED", "fold": fold, "variant": variant, "exit_code": process.returncode, "elapsed_seconds": elapsed})
                raise RuntimeError(f"Cell failed: fold={fold}, variant={variant}, code={process.returncode}")
            selected = json.loads(selection.read_text(encoding="utf-8"))
            write_json(status_path, {"status": "RUNNING", "last_completed_fold": fold, "last_completed_variant": variant, "elapsed_seconds": elapsed, "last_validation_patient_macro_f1": selected.get("patient_macro_f1"), "test_evaluated": False})
            print(f"DONE fold={fold} variant={variant} seconds={elapsed:.1f}", flush=True)
    write_json(status_path, {"status": "COMPLETE", "cells": len(FOLDS) * len(VARIANTS), "test_evaluated": False})
    print("DEVELOPMENT_GRID_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
