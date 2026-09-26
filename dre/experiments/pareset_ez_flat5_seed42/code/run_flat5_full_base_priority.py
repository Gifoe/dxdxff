"""Outcome-blind priority amendment: complete flat-fivefold full/base first.

Imports the frozen training/evaluation implementation without altering its
scientific rules. BCR is deferred, not replaced by old-protocol results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_flat_5fold as frozen


PRIORITY = ("full", "base")
METRICS = frozen.METRICS


def amendment(args, create: bool) -> None:
    folder = args.output
    path = folder / "FLAT5_PRIORITY_AMENDMENT.json"
    marker = folder / "FLAT5_PRIORITY_AMENDMENT.sha256"
    expected = {
        "status": "FROZEN_PRIORITY_AMENDMENT",
        "reason": "User requested completing full and base first; no new flat-fivefold outcomes inspected to set or change training rules",
        "original_protocol_lock_sha256": frozen.digest(folder / "FLAT5_PROTOCOL_LOCK.json"),
        "priority_methods": list(PRIORITY),
        "deferred_method": "bcr",
        "folds": list(frozen.FOLDS),
        "seed": frozen.SEED,
        "selection": "final epoch 45; fixed NEZ threshold 0.5; no inner validation",
        "scientific_rule_change": False,
        "frozen_runner_sha256": frozen.digest(Path(frozen.__file__)),
        "priority_runner_sha256": frozen.digest(Path(__file__)),
        "interpretation": "exploratory rerun on historical patients, not independent heldout confirmation",
    }
    if create:
        if path.exists() or marker.exists():
            raise RuntimeError("Priority amendment already exists")
        frozen.json_write(path, expected)
        marker.write_text(frozen.digest(path) + "\n", encoding="ascii")
        print(f"PRIORITY_AMENDMENT_SHA256={frozen.digest(path)}", flush=True)
        return
    if not path.exists() or not marker.exists():
        raise RuntimeError("Freeze priority amendment before continuing")
    if marker.read_text(encoding="ascii").strip() != frozen.digest(path):
        raise RuntimeError("Priority amendment digest mismatch")
    if json.loads(path.read_text(encoding="utf-8")) != expected:
        raise RuntimeError("Priority amendment changed")


def aggregate(args) -> None:
    frames = []
    fold_rows = []
    for fold in frozen.FOLDS:
        for method in PRIORITY:
            cell = args.output / f"fold{fold}" / method
            complete = cell / "DONE.json"
            if not complete.exists():
                raise RuntimeError(f"Incomplete method/fold: {cell}")
            saved = json.loads(complete.read_text(encoding="utf-8"))
            if saved["fold"] != fold or saved["method"] != method or saved["epoch"] != frozen.EPOCHS:
                raise RuntimeError(f"Invalid completion marker: {complete}")
            frame = pd.read_csv(cell / "test_patient_metrics_PRIVATE.csv")
            if len(frame) != saved["n_test"]:
                raise RuntimeError(f"Patient count mismatch: {cell}")
            frames.append(frame)
            fold_rows.append({"fold": fold, "method": method, "patients": len(frame),
                              **{key: float(frame[key].mean()) for key in METRICS}})
    patients = pd.concat(frames, ignore_index=True)
    summary_rows = []
    for method in PRIORITY:
        subset = patients.loc[patients.method == method]
        if len(subset) != 80 or subset.patient_id.nunique() != 80:
            raise RuntimeError(f"Expected 80 disjoint test patients: {method}")
        summary_rows.append({"method": method, "patients": 80, "seed": frozen.SEED,
                             **{key: float(subset[key].mean()) for key in METRICS}})
    left = patients.loc[patients.method == "full"].set_index("patient_id")
    right = patients.loc[patients.method == "base"].set_index("patient_id")
    if set(left.index) != set(right.index):
        raise RuntimeError("Full/base patients are not paired")
    rows = []
    for key in METRICS:
        delta = (left[key] - right[key]).to_numpy(dtype=float)
        rng = np.random.default_rng(4201)
        draws = rng.integers(0, 80, size=(10000, 80))
        samples = np.nanmean(delta[draws], axis=1)
        rows.append({"primary": "full", "comparator": "base", "metric": key,
                     "delta": float(np.nanmean(delta)),
                     "ci95_low": float(np.nanquantile(samples, .025)),
                     "ci95_high": float(np.nanquantile(samples, .975)),
                     "positive_patients": int(np.sum(delta > 0)),
                     "negative_patients": int(np.sum(delta < 0)),
                     "zero_patients": int(np.sum(delta == 0))})
    summary = pd.DataFrame(summary_rows)
    fold_table = pd.DataFrame(fold_rows)
    paired = pd.DataFrame(rows)
    summary.to_csv(args.output / "FLAT5_FULL_BASE_SUMMARY.csv", index=False)
    fold_table.to_csv(args.output / "FLAT5_FULL_BASE_FOLD_RESULTS.csv", index=False)
    paired.to_csv(args.output / "FLAT5_FULL_BASE_PAIRED_BOOTSTRAP.csv", index=False)
    lines = ["# PaReSet-EZ flat five-fold full/base priority result", "",
             "Exploratory rerun on the historical 80-patient cohort; previous outer outcomes had already been viewed. This is not an independent heldout confirmation.",
             "Every fold trains on frozen fit+validation and tests on frozen test; seed 42; final epoch 45; fixed threshold 0.5; no inner validation or test-based selection.",
             "BCR is deferred by the user-requested priority amendment; no BCR result is included in this report.", "",
             summary.to_markdown(index=False), "", paired.to_markdown(index=False), "",
             "Only compact aggregate results may be published. Patient/channel metrics, checkpoints and logs remain private on the server."]
    (args.output / "FLAT5_FULL_BASE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    frozen.json_write(args.output / "FLAT5_FULL_BASE_STATUS.json", {
        "status": "COMPLETE", "methods": list(PRIORITY), "deferred_method": "bcr",
        "patients_per_method": 80, "folds": 5, "seed": 42,
        "test_used_for_tuning": False, "historical_outcomes_previously_viewed": True,
        "original_protocol_lock_sha256": frozen.digest(args.output / "FLAT5_PROTOCOL_LOCK.json"),
        "priority_amendment_sha256": frozen.digest(args.output / "FLAT5_PRIORITY_AMENDMENT.json"),
    })
    print(summary.to_string(index=False), flush=True)
    print(paired.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run", "aggregate"))
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    dep = frozen.dependencies(args)
    groups = frozen.partition(args, dep)
    frozen.verify_or_create_lock(args, groups, create=False)
    amendment(args, create=args.phase == "prepare")
    if args.phase == "prepare":
        return
    if args.phase == "run":
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA required but unavailable")
        for fold in frozen.FOLDS:
            train, test = groups[fold]
            for method in PRIORITY:
                frozen.fit_one(args, dep, fold, method, train, test, device)
    aggregate(args)


if __name__ == "__main__":
    main()
