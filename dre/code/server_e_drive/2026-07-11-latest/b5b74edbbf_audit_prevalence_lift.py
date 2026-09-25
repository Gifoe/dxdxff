from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def _read_fold_csvs(pred_dir: Path, pattern: str) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(pred_dir.glob(pattern))]
    if not frames:
        raise FileNotFoundError(f"No files matched {pattern!r} under {pred_dir}.")
    return pd.concat(frames, ignore_index=True)


def _center_from_subject(subject_id: str) -> str:
    if ":" in subject_id:
        prefix = subject_id.split(":", 1)[0].strip().lower()
        if prefix in {"hup", "lzu", "multicenter", "pediatric"}:
            return prefix
    return "unknown"


def _safe_auc(y_true: np.ndarray, scores: np.ndarray, *, auprc: bool) -> float:
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return 0.0
    return float(average_precision_score(y_true, scores) if auprc else roc_auc_score(y_true, scores))


def _reciprocal_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
    if y_true.size == 0 or int((y_true == 1).sum()) == 0:
        return 0.0
    order = np.argsort(scores)[::-1]
    hits = np.where(y_true[order] == 1)[0]
    return float(1.0 / float(hits[0] + 1)) if hits.size else 0.0


def _recall_at_true_count(y_true: np.ndarray, scores: np.ndarray) -> float:
    true_count = int((y_true == 1).sum())
    if y_true.size == 0 or true_count <= 0:
        return 0.0
    top = np.argsort(scores)[::-1][:true_count]
    return float((y_true[top] == 1).sum() / max(true_count, 1))


def build_prevalence_table(pred_dir: Path, *, high_ez_threshold: float = 0.40) -> pd.DataFrame:
    patient_df = _read_fold_csvs(pred_dir, "test_patient_predictions_neuroez_v2_fold_*.csv")
    channel_df = _read_fold_csvs(pred_dir, "test_channel_predictions_neuroez_v2_fold_*.csv")
    rows: list[dict[str, Any]] = []
    patient_lookup = patient_df.groupby("subject_id", sort=False).first().to_dict("index")
    for subject_id, group in channel_df.groupby("subject_id", sort=False):
        y_ez = group["true_ez"].astype(float).to_numpy()
        score_col = "score_ez_final" if "score_ez_final" in group.columns else "score_ez_probability"
        scores = group[score_col].astype(float).to_numpy()
        n_channels = int(y_ez.size)
        n_ez = int((y_ez > 0.5).sum())
        ez_fraction = float(n_ez / max(n_channels, 1))
        patient_row = patient_lookup.get(subject_id, {})
        recall_at_true_count = float(patient_row.get("ez_recall_at_true_count", _recall_at_true_count(y_ez, scores)))
        auprc_ez = _safe_auc(y_ez.astype(int), scores, auprc=True)
        row = {
            "subject_id": subject_id,
            "center": str(patient_row.get("center", group.get("center", pd.Series([_center_from_subject(str(subject_id))])).iloc[0])),
            "n_channels": n_channels,
            "n_ez": n_ez,
            "ez_fraction": ez_fraction,
            "patient_ez_f1": float(patient_row.get("patient_ez_f1", 0.0)),
            "patient_macro_f1": float(patient_row.get("patient_macro_f1", 0.0)),
            "patient_auroc_ez": _safe_auc(y_ez.astype(int), scores, auprc=False),
            "patient_auprc_ez": auprc_ez,
            "ez_mrr": float(patient_row.get("ez_mrr", _reciprocal_rank(y_ez, scores))),
            "recall_at_true_count": recall_at_true_count,
            "recall_lift": recall_at_true_count - ez_fraction,
            "auprc_lift": auprc_ez - ez_fraction,
            "is_high_ez_fraction": bool(ez_fraction > float(high_ez_threshold)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _merge_baseline(candidate: pd.DataFrame, baseline_pred_dir: Path | None, high_ez_threshold: float) -> pd.DataFrame:
    if baseline_pred_dir is None:
        return candidate
    baseline = build_prevalence_table(baseline_pred_dir, high_ez_threshold=high_ez_threshold)
    merged = candidate.merge(
        baseline[
            [
                "subject_id",
                "patient_macro_f1",
                "patient_ez_f1",
                "recall_lift",
                "auprc_lift",
                "ez_mrr",
            ]
        ].rename(
            columns={
                "patient_macro_f1": "baseline_patient_macro_f1",
                "patient_ez_f1": "baseline_patient_ez_f1",
                "recall_lift": "baseline_recall_lift",
                "auprc_lift": "baseline_auprc_lift",
                "ez_mrr": "baseline_mrr",
            }
        ),
        on="subject_id",
        how="left",
    )
    merged["delta_patient_macro_f1"] = merged["patient_macro_f1"] - merged["baseline_patient_macro_f1"]
    merged["delta_patient_ez_f1"] = merged["patient_ez_f1"] - merged["baseline_patient_ez_f1"]
    merged["delta_recall_lift"] = merged["recall_lift"] - merged["baseline_recall_lift"]
    merged["delta_auprc_lift"] = merged["auprc_lift"] - merged["baseline_auprc_lift"]
    merged["delta_mrr"] = merged["ez_mrr"] - merged["baseline_mrr"]
    return merged


def run(pred_dir: Path, output_dir: Path, baseline_pred_dir: Path | None, high_ez_threshold: float) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    patient = _merge_baseline(build_prevalence_table(pred_dir, high_ez_threshold=high_ez_threshold), baseline_pred_dir, high_ez_threshold)
    patient.to_csv(output_dir / "prevalence_lift_patient.csv", index=False)
    by_center = patient.groupby("center", dropna=False).mean(numeric_only=True).reset_index()
    by_center.to_csv(output_dir / "prevalence_lift_by_center.csv", index=False)
    high_ez = patient.groupby("is_high_ez_fraction", dropna=False).mean(numeric_only=True).reset_index()
    high_ez.to_csv(output_dir / "prevalence_lift_high_ez_summary.csv", index=False)
    patient.sort_values("auprc_lift", ascending=True).head(30).to_csv(output_dir / "prevalence_lift_worst30.csv", index=False)
    summary = {
        "n_patients": int(patient["subject_id"].nunique()),
        "mean_recall_lift": float(patient["recall_lift"].mean()) if not patient.empty else 0.0,
        "mean_auprc_lift": float(patient["auprc_lift"].mean()) if not patient.empty else 0.0,
        "mean_mrr": float(patient["ez_mrr"].mean()) if not patient.empty else 0.0,
        "high_ez_threshold": float(high_ez_threshold),
        "baseline_pred_dir": str(baseline_pred_dir) if baseline_pred_dir is not None else "",
    }
    (output_dir / "prevalence_lift_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-patient prevalence-adjusted EZ localization audit metrics.")
    parser.add_argument("--pred_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--baseline_pred_dir", default=None, type=Path)
    parser.add_argument("--high_ez_threshold", default=0.40, type=float)
    args = parser.parse_args()
    summary = run(args.pred_dir, args.output_dir, args.baseline_pred_dir, args.high_ez_threshold)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
