"""Diagnostic-only reconstruction of P2 component scores from held-out ledgers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score


COMPONENTS = {
    "S0_DIRECT": ("direct_nez_logit",),
    "S1_DIRECT_ANCHOR": ("direct_nez_logit", "anchor_residual"),
    "S2_DIRECT_SEIZURE": ("direct_nez_logit", "seizure_residual"),
    "S3_DIRECT_CAUSAL": ("direct_nez_logit", "causal_residual"),
    "S4_DIRECT_ANCHOR_SEIZURE": ("direct_nez_logit", "anchor_residual", "seizure_residual"),
    "S5_FULL_CURRENT": ("final_nez_logit",),
}


def _mrr(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-(1.0 - scores))
    hits = np.flatnonzero((1 - labels)[order] > 0)
    return float(1.0 / (hits[0] + 1)) if hits.size else 0.0


def _oracle(labels: np.ndarray, scores: np.ndarray) -> float:
    values = np.unique(scores)
    candidates = np.r_[0.0, 1.0, (values[:-1] + values[1:]) / 2.0]
    return max(f1_score(labels, scores >= threshold, average="macro", zero_division=0) for threshold in candidates)


def _patient_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    patient_rows = []
    for subject_id, patient in frame.groupby("subject_id", sort=False):
        labels = patient.label_nez.to_numpy(dtype=int); scores = patient.score_nez.to_numpy(dtype=float)
        patient_rows.append({
            "subject_id": subject_id, "center": patient.center.iloc[0], "outer_fold": patient.outer_fold.iloc[0],
            "patient_oracle_macro_f1": _oracle(labels, scores),
            "ez_auprc": average_precision_score(1 - labels, 1 - scores) if np.unique(labels).size == 2 else np.nan,
            "nez_auprc": average_precision_score(labels, scores) if np.unique(labels).size == 2 else np.nan,
            "ez_mrr": _mrr(labels, scores),
            "recall_at_true_ez_count": float(((1 - labels)[np.argsort(scores)[:int((labels == 0).sum())]].sum()) / max(1, int((labels == 0).sum()))),
        })
    return pd.DataFrame(patient_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    ledgers = sorted(run_dir.glob("test_channel_predictions_neuroez_v2_fold_*.csv"))
    if not ledgers:
        raise FileNotFoundError(f"No P2 held-out channel ledgers under {run_dir}")
    frame = pd.concat((pd.read_csv(path) for path in ledgers), ignore_index=True)
    required = {"subject_id", "center", "outer_fold", "label_nez", "direct_nez_logit", "final_nez_logit", "anchor_residual", "seizure_residual", "causal_residual"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"P2 component audit cannot reconstruct missing fields: {missing}")
    output = Path(args.output_dir) if args.output_dir else run_dir / "p2_component_score_audit"
    output.mkdir(parents=True, exist_ok=True)
    overall, by_fold, by_center, by_patient = [], [], [], []
    for name, fields in COMPONENTS.items():
        logits = sum(frame[field].to_numpy(dtype=float) for field in fields)
        work = frame[["subject_id", "center", "outer_fold", "label_nez"]].copy()
        work["score_nez"] = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        patient_metrics = _patient_metrics(work)
        metric_columns = [column for column in patient_metrics.columns if column not in {"subject_id", "center", "outer_fold"}]
        overall.append({"component": name, **patient_metrics[metric_columns].mean(numeric_only=True).to_dict()})
        for fold, part in patient_metrics.groupby("outer_fold"):
            by_fold.append({"component": name, "outer_fold": fold, **part[metric_columns].mean(numeric_only=True).to_dict()})
        for center, part in patient_metrics.groupby("center"):
            by_center.append({"component": name, "center": center, **part[metric_columns].mean(numeric_only=True).to_dict()})
        by_patient.extend({"component": name, **row} for row in patient_metrics.to_dict(orient="records"))
    pd.DataFrame(overall).to_csv(output / "component_overall.csv", index=False)
    pd.DataFrame(by_fold).to_csv(output / "component_by_fold.csv", index=False)
    pd.DataFrame(by_center).to_csv(output / "component_by_center.csv", index=False)
    pd.DataFrame(by_patient).to_csv(output / "component_by_patient.csv", index=False)
    (output / "P2_COMPONENT_SCORE_AUDIT.md").write_text(
        "# P2 Component Score Audit\n\nDIAGNOSTIC_ONLY. NOT_DEPLOYABLE. NOT_USED_FOR_FORMAL_PREDICTION.\n"
        "Thresholds and patient-oracle scores use held-out labels only to diagnose ordering limits.\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
