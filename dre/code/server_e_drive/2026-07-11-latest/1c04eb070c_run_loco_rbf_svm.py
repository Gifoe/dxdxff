#!/usr/bin/env python3
"""Train the fixed RBF-SVM LOCO baseline using an explicit patient split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from outcome_hifos.cache_schema import load_cache_contract
from task1_baselines.cache_io import task1_feature_records
from task1_baselines.feature_aggregation import build_channel_feature_table
from task1_baselines.models.feature_models import build_feature_estimator, feature_parameter_grid
from task1_baselines.thresholds import select_patient_macro_threshold


def _parse_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).rename(columns={"split_role": "partition", "fold_idx": "outer_fold"})
    required = {"subject_id", "center", "outer_fold", "partition"}
    if not required.issubset(frame.columns):
        raise ValueError(f"LOCO manifest missing columns: {sorted(required - set(frame.columns))}")
    frame["partition"] = frame["partition"].astype(str).str.lower()
    if frame.outer_fold.nunique() != 1 or int(frame.outer_fold.iloc[0]) != 1:
        raise ValueError("LOCO manifest must contain only outer fold 1")
    if frame.subject_id.duplicated().any() or set(frame.partition) != {"fit", "validation", "test"}:
        raise ValueError("LOCO manifest must assign every patient exactly once to fit/validation/test")
    if frame.subject_id.nunique() != 80:
        raise ValueError("LOCO RBF-SVM requires exactly 80 patients")
    target_centers = frame.loc[frame.partition.eq("test"), "center"].astype(str).str.lower().unique()
    if len(target_centers) != 1:
        raise ValueError("LOCO test partition must contain exactly one held-out center")
    source = frame.loc[~frame.partition.eq("test"), "center"].astype(str).str.lower()
    if target_centers[0] in set(source):
        raise ValueError("Held-out center leaked into fit or validation")
    return frame.sort_values("subject_id", kind="mergesort").reset_index(drop=True)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "subject_id", "center", "channel_name", "label_nez", "clinical_true_nez",
        "clinical_true_ez", "valid_seizure_count", "valid_window_count", "outer_fold", "partition",
    }
    return [name for name in frame if name not in excluded and pd.api.types.is_numeric_dtype(frame[name])]


def _patient_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject, group in frame.groupby("subject_id", sort=True):
        label = group.label_nez.to_numpy(int)
        predicted = group.predicted_nez.to_numpy(int)
        rows.append({
            "subject_id": subject,
            "center": str(group.center.iloc[0]),
            "outer_fold": 1,
            "n_channels": int(len(group)),
            "patient_macro_f1": float(f1_score(label, predicted, average="macro", labels=[0, 1], zero_division=0)),
            "patient_ez_f1": float(f1_score(label, predicted, pos_label=0, zero_division=0)),
            "patient_nez_f1": float(f1_score(label, predicted, pos_label=1, zero_division=0)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--loco-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    ledger_path = output / "oof_channel_ledger.csv"
    if ledger_path.is_file():
        if args.resume:
            print(json.dumps({"status": "skipped", "reason": "complete_ledger_exists", "output_dir": str(output)}))
            return
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = _parse_manifest(Path(args.loco_manifest))
    cache = load_cache_contract(args.feature_cache)
    records = task1_feature_records(cache, set(manifest.subject_id))
    table, feature_audit = build_channel_feature_table(records, profile="p2_matched_simple")
    table = table.merge(manifest[["subject_id", "center", "outer_fold", "partition"]], on=["subject_id", "center"], validate="many_to_one")
    if table.subject_id.nunique() != 80:
        raise RuntimeError("Feature cache did not produce the complete LOCO cohort")
    columns = _feature_columns(table)
    if not columns:
        raise RuntimeError("No numeric RBF-SVM features available")
    fit = table[table.partition.eq("fit")].reset_index(drop=True)
    validation = table[table.partition.eq("validation")].reset_index(drop=True)
    test = table[table.partition.eq("test")].copy()
    params = feature_parameter_grid("rbf_svm", compact=True)[0]
    model = build_feature_estimator("rbf_svm", params, seed=args.seed)
    model.fit(fit[columns], fit.label_nez)
    validation_scores = validation[["subject_id", "label_nez"]].copy()
    validation_scores["score_nez_probability"] = model.predict_proba(validation[columns])[:, 1]
    threshold = select_patient_macro_threshold(validation_scores, source="loco_validation_patient_macro_f1")
    test["score_nez_probability"] = model.predict_proba(test[columns])[:, 1]
    test["score_ez_probability"] = 1.0 - test.score_nez_probability
    test["selected_threshold"] = float(threshold.threshold)
    test["threshold_source"] = threshold.source
    test["predicted_nez"] = (test.score_nez_probability >= threshold.threshold).astype(int)
    test["predicted_ez"] = 1 - test.predicted_nez
    test["model"] = "rbf_svm"
    test["seed"] = int(args.seed)
    test["analysis_status"] = "LOCO_HELD_OUT_CENTER"
    keep = [
        "model", "seed", "outer_fold", "subject_id", "center", "channel_name", "label_nez",
        "score_nez_probability", "score_ez_probability", "selected_threshold", "threshold_source",
        "predicted_nez", "predicted_ez", "analysis_status",
    ]
    test[keep].sort_values(["subject_id", "channel_name"], kind="mergesort").to_csv(ledger_path, index=False)
    patients = _patient_metrics(test)
    patients.to_csv(output / "patient_metrics.csv", index=False)
    (output / "checkpoint").mkdir(exist_ok=True)
    joblib.dump({"model": model, "feature_columns": columns, "parameters": params}, output / "checkpoint" / "rbf_svm.joblib")
    payload = {
        "status": "passed", "model": "RBF-SVM", "seed": int(args.seed),
        "n_fit_patients": int(fit.subject_id.nunique()), "n_validation_patients": int(validation.subject_id.nunique()),
        "n_test_patients": int(test.subject_id.nunique()), "held_out_center": str(test.center.iloc[0]),
        "selected_threshold": float(threshold.threshold), "threshold_source": threshold.source,
        "patient_macro_f1": float(patients.patient_macro_f1.mean()), "feature_audit": feature_audit,
    }
    (output / "run_audit.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
