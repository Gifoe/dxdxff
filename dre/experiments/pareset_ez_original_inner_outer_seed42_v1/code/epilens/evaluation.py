"""Patient-equal metrics and validation-only threshold selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    ndcg_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    patient_macro_f1: float
    patient_ez_f1: float
    patient_balanced_accuracy: float


def _safe_auc(target: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(target, score)) if len(np.unique(target)) == 2 else float("nan")


def patient_metrics(
    ledger: pd.DataFrame,
    threshold: float,
    score_column: str = "probability_nez",
) -> pd.DataFrame:
    """Compute every metric independently per patient."""
    required = {"patient_id", "label_nez", score_column}
    if not required.issubset(ledger.columns):
        raise ValueError(f"Ledger requires {sorted(required)}")
    rows = []
    for patient_id, group in ledger.groupby("patient_id", sort=False):
        label_nez = group["label_nez"].to_numpy(dtype=int)
        score_nez = group[score_column].to_numpy(dtype=float)
        prediction_nez = (score_nez >= threshold).astype(int)
        label_ez = 1 - label_nez
        score_ez = 1.0 - score_nez
        prediction_ez = 1 - prediction_nez
        true_k = int(label_ez.sum())
        order = np.argsort(-score_ez)
        first_ez = np.flatnonzero(label_ez[order])
        rows.append(
            {
                "patient_id": patient_id,
                "center": str(group["center"].iloc[0]) if "center" in group else "",
                "macro_f1": f1_score(label_nez, prediction_nez, average="macro", zero_division=0),
                "ez_f1": f1_score(label_ez, prediction_ez, zero_division=0),
                "nez_f1": f1_score(label_nez, prediction_nez, zero_division=0),
                "accuracy": accuracy_score(label_nez, prediction_nez),
                "balanced_accuracy": balanced_accuracy_score(label_nez, prediction_nez),
                "ez_auroc": _safe_auc(label_ez, score_ez),
                "ez_auprc": average_precision_score(label_ez, score_ez) if true_k else float("nan"),
                "ndcg_ez": ndcg_score(label_ez[None, :], score_ez[None, :]) if true_k else float("nan"),
                "recall_at_true_k": float(label_ez[order[:true_k]].mean()) if true_k else float("nan"),
                "mrr_ez": float(1.0 / (first_ez[0] + 1)) if len(first_ez) else float("nan"),
                "top1_ez_rate": float(label_ez[order[0]]) if true_k else float("nan"),
                "true_ez_fraction": float(label_ez.mean()),
                "predicted_ez_fraction": float(prediction_ez.mean()),
            }
        )
    return pd.DataFrame(rows)


def select_threshold(
    validation_ledger: pd.DataFrame,
    score_column: str = "probability_nez",
    step: float = 0.005,
) -> ThresholdSelection:
    """Maximize patient Macro-F1; tie-break by EZ-F1 and balanced accuracy."""
    candidates = []
    for threshold in np.arange(0.0, 1.0 + step / 2.0, step):
        patient = patient_metrics(validation_ledger, float(threshold), score_column)
        candidates.append(
            ThresholdSelection(
                threshold=float(threshold),
                patient_macro_f1=float(patient["macro_f1"].mean()),
                patient_ez_f1=float(patient["ez_f1"].mean()),
                patient_balanced_accuracy=float(patient["balanced_accuracy"].mean()),
            )
        )
    return max(
        candidates,
        key=lambda item: (
            item.patient_macro_f1,
            item.patient_ez_f1,
            item.patient_balanced_accuracy,
            -abs(item.threshold - 0.5),
        ),
    )


def summarize_patient_equal(
    ledger: pd.DataFrame,
    threshold: float,
    score_column: str = "probability_nez",
) -> dict[str, float]:
    patient = patient_metrics(ledger, threshold, score_column)
    result = {
        column: float(patient[column].mean())
        for column in (
            "macro_f1",
            "ez_f1",
            "nez_f1",
            "accuracy",
            "ez_auroc",
            "ez_auprc",
            "ndcg_ez",
            "recall_at_true_k",
            "mrr_ez",
            "top1_ez_rate",
        )
    }
    true_ez = 1 - ledger["label_nez"].to_numpy(dtype=int)
    predicted_ez = (
        ledger[score_column].to_numpy(dtype=float) < threshold
    ).astype(int)
    result["ez_fraction_bias"] = float(predicted_ez.mean() - true_ez.mean())
    return result
