#!/usr/bin/env python3
"""Run one patient-relative feature control on one frozen Task 1 LOCO split.

The runner deliberately fits preprocessing, the classifier, and the decision
threshold from the LOCO fit/validation partitions only.  The held-out center is
used once, exclusively for final evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from outcome_hifos.cache_schema import load_cache_contract
from task1_baselines.cache_io import task1_feature_records
from task1_baselines.feature_aggregation import build_channel_feature_table
from task1_baselines.metrics import compute_task1_metrics
from task1_baselines.patient_controls import _Preprocessor, _classical_scores, feature_columns
from task1_baselines.thresholds import select_patient_macro_threshold


CONTROL_NAMES = {"patient_z_logistic", "patient_z_rbf_svm"}


def _manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).rename(columns={"split_role": "partition", "fold_idx": "outer_fold"})
    required = {"subject_id", "center", "outer_fold", "partition"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"LOCO manifest missing columns: {sorted(missing)}")
    frame = frame[list(required)].copy()
    frame["partition"] = frame["partition"].astype(str).str.strip().str.lower()
    if frame["subject_id"].duplicated().any() or frame["subject_id"].nunique() != 80:
        raise ValueError("LOCO manifest must assign each of the 80 patients exactly once.")
    if frame["outer_fold"].nunique() != 1 or int(frame["outer_fold"].iloc[0]) != 1:
        raise ValueError("LOCO manifest must contain exactly outer fold 1.")
    if set(frame["partition"]) != {"fit", "validation", "test"}:
        raise ValueError("LOCO manifest must contain fit, validation, and test partitions.")
    held_out = frame.loc[frame["partition"].eq("test"), "center"].astype(str).str.lower().unique()
    if len(held_out) != 1:
        raise ValueError("LOCO test partition must contain one held-out center.")
    if held_out[0] in set(frame.loc[~frame["partition"].eq("test"), "center"].astype(str).str.lower()):
        raise ValueError("Held-out center appears in fit or validation patients.")
    return frame.sort_values("subject_id", kind="stable").reset_index(drop=True)


def _safe_auc(y: np.ndarray, score: np.ndarray, fn) -> float:
    return float(fn(y, score)) if np.unique(y).size == 2 else float("nan")


def _patient_metrics(test: pd.DataFrame, *, experiment: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject, group in test.groupby("subject_id", sort=True):
        y = group["label_nez"].to_numpy(dtype=int)
        score_nez = group["score_nez_probability"].to_numpy(dtype=float)
        predicted = group["predicted_nez"].to_numpy(dtype=int)
        ez = 1 - y
        ez_score = 1.0 - score_nez
        ez_order = np.argsort(-ez_score, kind="stable")
        positive = np.flatnonzero(ez[ez_order] == 1)
        rows.append({
            "subject_id": subject,
            "center": str(group["center"].iloc[0]),
            "outer_fold": 1,
            "n_channels": int(len(group)),
            "true_ez_count": int(ez.sum()),
            "predicted_ez_count": int((1 - predicted).sum()),
            "true_ez_fraction": float(ez.mean()),
            "predicted_ez_fraction": float((1 - predicted).mean()),
            "patient_macro_f1": float(f1_score(y, predicted, average="macro", labels=[0, 1], zero_division=0)),
            "patient_ez_f1": float(f1_score(y, predicted, pos_label=0, zero_division=0)),
            "patient_nez_f1": float(f1_score(y, predicted, pos_label=1, zero_division=0)),
            "patient_balanced_accuracy": float(balanced_accuracy_score(y, predicted)) if np.unique(y).size == 2 else float("nan"),
            "patient_ez_auprc": _safe_auc(ez, ez_score, average_precision_score),
            "patient_ez_auroc": _safe_auc(ez, ez_score, roc_auc_score),
            "patient_ez_mrr": float(1.0 / (positive[0] + 1)) if positive.size else float("nan"),
            "top1_is_ez_rate": float(ez[ez_order[0]]),
            "predicted_ez_count_mae": float(abs((1 - predicted).sum() - ez.sum())),
            "predicted_ez_fraction_mae": float(abs((1 - predicted).mean() - ez.mean())),
            "truek_patient_macro_f1": float("nan"),
            "selected_validation_threshold": float(group["selected_threshold"].iloc[0]),
            "experiment": experiment,
            "analysis_status": "PRIMARY_CONFIRMATORY_LOCO_CONTROL",
            "true_count_used_for_prediction": False,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--loco-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", default="patient_z_logistic", choices=sorted(CONTROL_NAMES))
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
    manifest = _manifest(Path(args.loco_manifest))
    cache = load_cache_contract(args.feature_cache)
    table, feature_audit = build_channel_feature_table(task1_feature_records(cache, set(manifest["subject_id"])), profile="p2_matched_simple")
    table = table.merge(manifest, on=["subject_id", "center"], how="inner", validate="many_to_one")
    if table["subject_id"].nunique() != 80:
        raise RuntimeError("Feature cache did not produce all 80 LOCO patients.")
    columns = feature_columns(table)
    fit = table.loc[table["partition"].eq("fit")].reset_index(drop=True)
    validation = table.loc[table["partition"].eq("validation")].reset_index(drop=True)
    test = table.loc[table["partition"].eq("test")].copy().reset_index(drop=True)
    preprocessor = _Preprocessor(SimpleImputer(), StandardScaler(), patient_z=True)
    fit_x = preprocessor.fit_transform(fit, columns)
    validation_x = preprocessor.transform(validation, columns)
    test_x = preprocessor.transform(test, columns)
    validation_scores = _classical_scores(args.model, fit_x, fit["label_nez"].to_numpy(int), validation_x, args.seed)
    threshold_frame = validation[["subject_id", "label_nez"]].copy()
    threshold_frame["score_nez_probability"] = validation_scores
    threshold = select_patient_macro_threshold(threshold_frame, source="loco_validation_patient_macro_f1")
    test_scores = _classical_scores(args.model, fit_x, fit["label_nez"].to_numpy(int), test_x, args.seed)
    test["score_nez_probability"] = test_scores
    test["score_ez_probability"] = 1.0 - test_scores
    test["selected_threshold"] = float(threshold.threshold)
    test["threshold_source"] = threshold.source
    test["predicted_nez"] = (test_scores >= threshold.threshold).astype(int)
    test["predicted_ez"] = 1 - test["predicted_nez"]
    test["model"] = args.model
    test["seed"] = int(args.seed)
    test["analysis_status"] = "PRIMARY_CONFIRMATORY_LOCO_CONTROL"
    keep = [
        "model", "seed", "outer_fold", "subject_id", "center", "channel_name", "label_nez",
        "score_nez_probability", "score_ez_probability", "selected_threshold", "threshold_source",
        "predicted_nez", "predicted_ez", "analysis_status",
    ]
    test[keep].sort_values(["subject_id", "channel_name"], kind="stable").to_csv(ledger_path, index=False)
    experiment = "Logistic + patient-wise z-score" if args.model == "patient_z_logistic" else "RBF-SVM + patient-wise z-score"
    patients = _patient_metrics(test, experiment=experiment)
    metrics = compute_task1_metrics(test)
    patients.to_csv(output / "patient_metrics.csv", index=False)
    pd.DataFrame([{**metrics, "model": args.model, "seed": args.seed, "held_out_center": str(test["center"].iloc[0])}]).to_csv(output / "overall_metrics.csv", index=False)
    (output / "checkpoint").mkdir(exist_ok=True)
    joblib.dump({"feature_columns": columns, "preprocessor": preprocessor, "model": args.model, "seed": args.seed}, output / "checkpoint" / f"{args.model}.joblib")
    audit = {
        "status": "passed", "model": args.model, "seed": int(args.seed), "held_out_center": str(test["center"].iloc[0]),
        "n_fit_patients": int(fit["subject_id"].nunique()), "n_validation_patients": int(validation["subject_id"].nunique()),
        "n_test_patients": int(test["subject_id"].nunique()), "selected_threshold": float(threshold.threshold),
        "threshold_source": threshold.source, "patient_macro_f1": float(patients["patient_macro_f1"].mean()),
        "test_labels_used_for_selection": False, "feature_audit": feature_audit,
    }
    (output / "run_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    print(json.dumps(audit, indent=2, default=str))


if __name__ == "__main__":
    main()
