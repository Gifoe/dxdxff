"""Fit-only 88-D patient-z Logistic and LightGBM controls on frozen validation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    from epilens.data import load_records, load_partition_manifest, validate_protocol, select_records
    from epilens.features import feature_baseline_vector
    from epilens.feature_baselines import build_classical_estimator, patient_feature_zscore
    from epilens.evaluation import patient_metrics, select_threshold

    args.output_root.mkdir(parents=True, exist_ok=True)
    lock = {
        "status": "FROZEN_88D_TABULAR_CONTROLS",
        "folds": [1, 2, 3, 4, 5], "seed": 42,
        "models": ["logistic_regression", "lightgbm"],
        "representation": "supplementary 88D feature_baseline_vector plus label-free per-patient zscore",
        "sample_weight": "one over number of valid channels in each fit patient",
        "selection": "supplementary inner-validation threshold selector",
        "test_evaluated": False,
        "data_sha256": sha256(args.data), "manifest_sha256": sha256(args.manifest),
        "runner_sha256": sha256(Path(__file__)),
    }
    lock_path = args.output_root / "TABULAR_LOCK.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text(encoding="utf-8")) != lock:
            raise RuntimeError("Tabular lock mismatch")
    else:
        lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    records = load_records(args.data)
    manifest = load_partition_manifest(args.manifest)
    validate_protocol(records, manifest, expected_patients=80)
    features = [f"feature_{i:02d}" for i in range(88)]
    table_rows = []
    for record in records:
        values = feature_baseline_vector(record)
        valid_channels = record.valid.any(axis=(0, 1))
        for index in np.nonzero(valid_channels)[0].tolist():
            row = {"patient_id": record.patient_id, "center": record.center,
                   "channel_name": record.channel_names[index],
                   "label_nez": int(record.label_nez[index])}
            row.update(zip(features, values[index].astype(float).tolist()))
            table_rows.append(row)
    frame = patient_feature_zscore(pd.DataFrame(table_rows), features)
    if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite tabular control features")
    summary_rows = []
    for fold in range(1, 6):
        fit_ids = {r.patient_id for r in select_records(records, manifest, fold, "fit")}
        val_ids = {r.patient_id for r in select_records(records, manifest, fold, "validation")}
        fit = frame.loc[frame.patient_id.isin(fit_ids)].copy()
        val = frame.loc[frame.patient_id.isin(val_ids)].copy()
        patient_counts = fit.groupby("patient_id").size()
        sample_weight = fit.patient_id.map(lambda pid: 1.0 / patient_counts[pid]).to_numpy(dtype=float)
        for name in ("logistic_regression", "lightgbm"):
            output = args.output_root / f"fold{fold}" / name
            selection_path = output / "selection.json"
            if selection_path.exists():
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                if selection.get("test_evaluated") is not False:
                    raise RuntimeError("Test-tainted tabular cell")
                summary_rows.append(selection)
                continue
            if output.exists() and any(output.iterdir()):
                raise RuntimeError(f"Incomplete tabular output: {output}")
            output.mkdir(parents=True, exist_ok=True)
            estimator = build_classical_estimator(name, seed=42)
            if name == "lightgbm":
                estimator.set_params(model__n_jobs=2)
            t0 = time.perf_counter()
            estimator.fit(fit[features].to_numpy(dtype=float), fit.label_nez.to_numpy(dtype=int), model__sample_weight=sample_weight)
            predictions = estimator.predict_proba(val[features].to_numpy(dtype=float))[:, 1]
            ledger = val[["patient_id", "center", "channel_name", "label_nez"]].copy()
            ledger["probability_nez"] = predictions
            selected = select_threshold(ledger)
            ledger["threshold"] = selected.threshold
            ledger["predicted_nez"] = (predictions >= selected.threshold).astype(int)
            ledger["fold"] = fold
            ledger["model"] = name
            ledger.to_csv(output / "validation_predictions.csv", index=False)
            patient_metrics(ledger, selected.threshold).to_csv(output / "validation_patient_metrics.csv", index=False)
            selection = {"fold": fold, "model": name, "threshold": selected.threshold,
                         "patient_macro_f1": selected.patient_macro_f1,
                         "patient_ez_f1": selected.patient_ez_f1,
                         "patient_balanced_accuracy": selected.patient_balanced_accuracy,
                         "seconds": time.perf_counter() - t0, "test_evaluated": False}
            selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
            summary_rows.append(selection)
            print(json.dumps(selection), flush=True)
    pd.DataFrame(summary_rows).to_csv(args.output_root / "TABULAR_VALIDATION_SUMMARY.csv", index=False)
    (args.output_root / "TABULAR_STATUS.json").write_text(json.dumps({"status": "COMPLETE", "cells": len(summary_rows), "test_evaluated": False}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
