from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def safe_roc_auc(y, score):
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    if np.unique(y).size < 2:
        return 0.0
    return float(roc_auc_score(y, score))


def safe_auprc(y, score):
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    if np.unique(y).size < 2:
        return 0.0
    return float(average_precision_score(y, score))


def mrr_ez(y_ez, score_ez):
    y_ez = np.asarray(y_ez).astype(int)
    score_ez = np.asarray(score_ez).astype(float)
    order = np.argsort(score_ez)[::-1]
    hit = np.flatnonzero(y_ez[order] == 1)
    return float(1.0 / float(hit[0] + 1)) if hit.size else 0.0


def compute(input_path: Path, output_dir: Path):
    df = pd.read_csv(input_path)

    required = {"method", "fold_idx", "subject_id", "label_ez", "score_ez", "pred_ez_topk"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")

    if "score_nez" not in df.columns:
        df["score_nez"] = 1.0 - df["score_ez"].astype(float)

    patient_rows = []
    for (method, fold_idx, subject_id), g in df.groupby(["method", "fold_idx", "subject_id"], sort=False):
        y_ez = g["label_ez"].astype(int).to_numpy()
        pred_ez = g["pred_ez_topk"].astype(int).to_numpy()
        score_ez = g["score_ez"].astype(float).to_numpy()

        y_nez = 1 - y_ez
        pred_nez = 1 - pred_ez
        score_nez = 1.0 - score_ez

        # labels=[1,0] means first row = positive class, second row = negative class under that target.
        nez_p, nez_r, nez_f1, _ = precision_recall_fscore_support(
            y_nez, pred_nez, labels=[1], zero_division=0
        )
        ez_p, ez_r, ez_f1, _ = precision_recall_fscore_support(
            y_ez, pred_ez, labels=[1], zero_division=0
        )

        ez_count = int(y_ez.sum())
        tp_ez = int(((y_ez == 1) & (pred_ez == 1)).sum())
        recall_at_true_count = float(tp_ez / ez_count) if ez_count > 0 else 0.0

        row = {
            "method": str(method),
            "fold_idx": int(fold_idx),
            "subject_id": str(subject_id),
            "center": str(g["center"].iloc[0]) if "center" in g.columns else "",
            "valid_channel_count": int(len(g)),
            "ez_channel_count": ez_count,
            "patient_macro_accuracy": float(accuracy_score(y_ez, pred_ez)),
            "patient_macro_balanced_accuracy": float(balanced_accuracy_score(y_ez, pred_ez)) if np.unique(y_ez).size > 1 else 0.0,
            "patient_macro_f1": float(f1_score(y_ez, pred_ez, average="macro", zero_division=0)),
            "patient_weighted_f1": float(f1_score(y_ez, pred_ez, average="weighted", zero_division=0)),
            "patient_macro_nez_precision": float(nez_p[0]) if len(nez_p) else 0.0,
            "patient_macro_nez_recall": float(nez_r[0]) if len(nez_r) else 0.0,
            "patient_macro_nez_f1": float(nez_f1[0]) if len(nez_f1) else 0.0,
            "patient_macro_ez_precision": float(ez_p[0]) if len(ez_p) else 0.0,
            "patient_macro_ez_recall": float(ez_r[0]) if len(ez_r) else 0.0,
            "patient_macro_ez_f1": float(ez_f1[0]) if len(ez_f1) else 0.0,
            "patient_macro_auroc_nez": safe_roc_auc(y_nez, score_nez),
            "patient_macro_auprc_nez": safe_auprc(y_nez, score_nez),
            "patient_macro_auroc_ez": safe_roc_auc(y_ez, score_ez),
            "patient_macro_auprc_ez": safe_auprc(y_ez, score_ez),
            "patient_macro_ez_recall_at_true_count": recall_at_true_count,
            "patient_macro_ez_mrr": mrr_ez(y_ez, score_ez),
            "top1_is_ez": float(y_ez[int(np.argmax(score_ez))] == 1) if len(score_ez) else 0.0,
            "macro_topk_recall": recall_at_true_count,
            "ez_recall_at_true_count": recall_at_true_count,
        }
        patient_rows.append(row)

    patient_df = pd.DataFrame(patient_rows)

    metric_cols = [
        "patient_macro_accuracy",
        "patient_macro_balanced_accuracy",
        "patient_macro_f1",
        "patient_weighted_f1",
        "patient_macro_nez_precision",
        "patient_macro_nez_recall",
        "patient_macro_nez_f1",
        "patient_macro_ez_precision",
        "patient_macro_ez_recall",
        "patient_macro_ez_f1",
        "patient_macro_auroc_nez",
        "patient_macro_auprc_nez",
        "patient_macro_auroc_ez",
        "patient_macro_auprc_ez",
        "patient_macro_ez_recall_at_true_count",
        "patient_macro_ez_mrr",
        "top1_is_ez",
        "macro_topk_recall",
        "ez_recall_at_true_count",
    ]

    summary = patient_df.groupby("method", sort=True)[metric_cols].mean().reset_index()
    summary = summary.rename(columns={"top1_is_ez": "top1_is_ez_rate"})
    summary["n_patient_rows"] = patient_df.groupby("method", sort=True)["subject_id"].nunique().to_numpy().astype(float)

    output_dir.mkdir(parents=True, exist_ok=True)
    patient_df.to_csv(output_dir / "aligned_patient_rows.csv", index=False)
    summary.to_csv(output_dir / "aligned_summary.csv", index=False)

    # Also write JSON per method for easy comparison with old metric dumps.
    json_dir = output_dir / "aligned_summary_json"
    json_dir.mkdir(parents=True, exist_ok=True)
    for _, row in summary.iterrows():
        method = str(row["method"])
        payload = {k: float(row[k]) for k in summary.columns if k != "method"}
        (json_dir / f"{method}.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(summary.to_string(index=False))
    print("Wrote", output_dir / "aligned_summary.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_predictions", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    compute(Path(args.input_predictions), Path(args.output_dir))


if __name__ == "__main__":
    main()
