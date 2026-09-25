from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

KNOWN_CENTERS = {"hup", "lzu", "multicenter", "pediatric"}


def infer_center(subject_id: Any) -> str:
    sid = str(subject_id).strip().lower()
    if not sid:
        return "unknown"
    if ":" in sid:
        prefix = sid.split(":", 1)[0].strip().lower()
        if prefix:
            return prefix
    for center in KNOWN_CENTERS:
        if center in sid:
            return center
    if sid.startswith("sub-hup") or "hup" in sid:
        return "hup"
    if "lzu" in sid or sid.startswith("lzu"):
        return "lzu"
    if "ped" in sid or "child" in sid or "fudan" in sid:
        return "pediatric"
    return "unknown"


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def load_fold_csvs(pred_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(pred_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern!r} in {pred_dir}")
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["source_file"] = path.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def compute_patient_rank_metrics(channel_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    required = {"fold_idx", "subject_id", "true_ez", "score_ez_probability", "predicted_ez"}
    missing = required - set(channel_df.columns)
    if missing:
        raise ValueError(f"Channel prediction CSV is missing columns: {sorted(missing)}")

    for (fold_idx, subject_id), g in channel_df.groupby(["fold_idx", "subject_id"], sort=False):
        y = g["true_ez"].to_numpy(dtype=int)
        score = g["score_ez_probability"].to_numpy(dtype=float)
        pred = g["predicted_ez"].to_numpy(dtype=int)
        valid = y >= 0
        y = y[valid]
        score = score[valid]
        pred = pred[valid]
        true_count = int((y == 1).sum())
        if y.size == 0:
            continue
        order = np.argsort(score)[::-1]
        positive_ranks = np.where(y[order] == 1)[0]
        ez_mrr = float(1.0 / (positive_ranks[0] + 1)) if positive_ranks.size else 0.0
        rows.append(
            {
                "fold_idx": fold_idx,
                "subject_id": subject_id,
                "patient_channel_auroc_ez": safe_auc(y, score),
                "patient_channel_auprc_ez": safe_auprc(y, score),
                "patient_channel_macro_f1_ez_view": float(f1_score(y, pred, average="macro", zero_division=0)),
                "patient_ez_mrr_from_channels": ez_mrr,
                "true_ez_ratio": float(true_count / max(int(y.size), 1)),
                "n_channels_from_channels": int(y.size),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_center_patient(patient_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "patient_macro_f1",
        "patient_ez_f1",
        "patient_nez_f1",
        "patient_balanced_accuracy",
        "patient_weighted_f1",
        "ez_mrr",
        "ez_recall_at_true_count",
        "true_ez_count",
        "true_nez_count",
        "predicted_ez_count",
        "n_seizures",
        "patient_channel_auroc_ez",
        "patient_channel_auprc_ez",
        "true_ez_ratio",
    ]
    available = [c for c in metric_cols if c in patient_df.columns]
    summary = patient_df.groupby("center", dropna=False)[available].mean(numeric_only=True).reset_index()
    counts = patient_df.groupby("center", dropna=False).agg(n_patients=("subject_id", "nunique"), n_rows=("subject_id", "size")).reset_index()
    return counts.merge(summary, on="center", how="left").sort_values("patient_macro_f1", ascending=False)


def summarize_by_center_pooled_channels(channel_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for center, g in channel_df.groupby("center", sort=False):
        y = g["true_ez"].to_numpy(dtype=int)
        score = g["score_ez_probability"].to_numpy(dtype=float)
        pred = g["predicted_ez"].to_numpy(dtype=int)
        valid = y >= 0
        y = y[valid]
        score = score[valid]
        pred = pred[valid]
        rows.append(
            {
                "center": center,
                "n_channels": int(y.size),
                "pooled_ez_rate": float((y == 1).mean()) if y.size else float("nan"),
                "pooled_macro_f1_ez_view": float(f1_score(y, pred, average="macro", zero_division=0)) if y.size else float("nan"),
                "pooled_auroc_ez": safe_auc(y, score),
                "pooled_auprc_ez": safe_auprc(y, score),
            }
        )
    return pd.DataFrame(rows).sort_values("pooled_macro_f1_ez_view", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="A0 by-center and patient-level error audit for NeuroEZ-C outputs.")
    parser.add_argument("--pred_dir", required=True, type=Path, help="Directory containing test_*_predictions_neuroez_v2_fold_*.csv")
    parser.add_argument("--out_dir", required=True, type=Path, help="Directory to write A0 audit files")
    parser.add_argument("--high_ez_ratio", type=float, default=0.40)
    parser.add_argument("--low_auprc", type=float, default=0.20)
    parser.add_argument("--ok_auroc", type=float, default=0.65)
    args = parser.parse_args()

    pred_dir = args.pred_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    patient_df = load_fold_csvs(pred_dir, "test_patient_predictions_neuroez_v2_fold_*.csv")
    channel_df = load_fold_csvs(pred_dir, "test_channel_predictions_neuroez_v2_fold_*.csv")
    patient_df["center"] = patient_df["subject_id"].map(infer_center)
    channel_df["center"] = channel_df["subject_id"].map(infer_center)

    rank_df = compute_patient_rank_metrics(channel_df)
    patient_df = patient_df.merge(rank_df, on=["fold_idx", "subject_id"], how="left")
    if "true_ez_ratio" not in patient_df.columns:
        patient_df["true_ez_ratio"] = patient_df["true_ez_count"] / (patient_df["true_ez_count"] + patient_df["true_nez_count"]).replace(0, np.nan)
    patient_df["count_abs_error"] = (patient_df["predicted_ez_count"] - patient_df["true_ez_count"]).abs()
    patient_df["all_predicted_ez"] = patient_df["predicted_nez_count"].eq(0)
    patient_df["all_predicted_nez"] = patient_df["predicted_ez_count"].eq(0)

    center_patient = summarize_by_center_patient(patient_df)
    center_channel = summarize_by_center_pooled_channels(channel_df)
    fold_center_counts = patient_df.groupby(["fold_idx", "center"]).agg(n_patients=("subject_id", "nunique")).reset_index()

    worst = patient_df.sort_values(["patient_macro_f1", "patient_ez_f1", "patient_channel_auprc_ez"], ascending=[True, True, True]).head(30)
    ez_zero = patient_df[patient_df.get("patient_ez_f1", pd.Series(0, index=patient_df.index)).fillna(0).le(1e-12)].sort_values("patient_macro_f1")
    high_ez = patient_df[patient_df["true_ez_ratio"].ge(float(args.high_ez_ratio))].sort_values("true_ez_ratio", ascending=False)
    low_pr_ok_auc = patient_df[
        patient_df["patient_channel_auprc_ez"].le(float(args.low_auprc))
        & patient_df["patient_channel_auroc_ez"].ge(float(args.ok_auroc))
    ].sort_values(["center", "patient_channel_auprc_ez"])

    patient_df.to_csv(out_dir / "a0_patient_error_audit.csv", index=False)
    center_patient.to_csv(out_dir / "a0_by_center_patient_summary.csv", index=False)
    center_channel.to_csv(out_dir / "a0_by_center_pooled_channel_summary.csv", index=False)
    fold_center_counts.to_csv(out_dir / "a0_fold_center_counts.csv", index=False)
    worst.to_csv(out_dir / "a0_worst_patients_top30.csv", index=False)
    ez_zero.to_csv(out_dir / "a0_ez_f1_zero_patients.csv", index=False)
    high_ez.to_csv(out_dir / "a0_high_ez_ratio_patients.csv", index=False)
    low_pr_ok_auc.to_csv(out_dir / "a0_low_auprc_ok_auroc_patients.csv", index=False)

    summary = {
        "pred_dir": str(pred_dir),
        "n_patient_rows": int(len(patient_df)),
        "n_channel_rows": int(len(channel_df)),
        "n_centers": int(patient_df["center"].nunique()),
        "center_patient_summary_csv": str(out_dir / "a0_by_center_patient_summary.csv"),
        "center_channel_summary_csv": str(out_dir / "a0_by_center_pooled_channel_summary.csv"),
        "worst_patients_csv": str(out_dir / "a0_worst_patients_top30.csv"),
        "patient_macro_f1_mean": float(patient_df["patient_macro_f1"].mean()),
        "patient_ez_f1_mean": float(patient_df["patient_ez_f1"].mean()) if "patient_ez_f1" in patient_df else None,
    }
    with open(out_dir / "a0_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[A0] wrote", out_dir)
    print(center_patient.to_string(index=False))


if __name__ == "__main__":
    main()
