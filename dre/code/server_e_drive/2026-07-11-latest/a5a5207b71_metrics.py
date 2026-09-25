from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _auc(target: np.ndarray, score: np.ndarray, *, pr: bool = False) -> float:
    if np.unique(target).size < 2:
        return float("nan")
    return float(average_precision_score(target, score) if pr else roc_auc_score(target, score))


def compute_task1_metrics(table: pd.DataFrame) -> dict[str, float | int]:
    y = table["label_nez"].to_numpy(dtype=int)
    p = table["score_nez_probability"].to_numpy(dtype=float)
    pred = table["predicted_nez"].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    values: dict[str, float | int] = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if np.unique(y).size == 2 else float("nan"),
        "precision_macro": float(precision_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "precision_nez": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall_nez": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "f1_nez": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "precision_ez": float(precision_score(y, pred, pos_label=0, zero_division=0)),
        "recall_ez": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "f1_ez": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "AUROC_NEZ": _auc(y, p),
        "AUPRC_NEZ": _auc(y, p, pr=True),
        "AUROC_EZ": _auc(1 - y, 1 - p),
        "AUPRC_EZ": _auc(1 - y, 1 - p, pr=True),
        "MCC": float(matthews_corrcoef(y, pred)) if np.unique(y).size == 2 else float("nan"),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "n_channels": int(len(table)),
        "n_patients": int(table["subject_id"].nunique()),
    }
    patient_rows = []
    for _, group in table.groupby("subject_id", sort=False):
        true = group["label_nez"].to_numpy(dtype=int)
        predicted = group["predicted_nez"].to_numpy(dtype=int)
        patient_rows.append(
            {
                "macro": f1_score(true, predicted, average="macro", labels=[0, 1], zero_division=0),
                "ez": f1_score(true, predicted, pos_label=0, zero_division=0),
                "nez": f1_score(true, predicted, pos_label=1, zero_division=0),
                "accuracy": accuracy_score(true, predicted),
            }
        )
    values.update(
        {
            "patient_macro_f1": float(np.mean([row["macro"] for row in patient_rows])),
            "patient_ez_f1": float(np.mean([row["ez"] for row in patient_rows])),
            "patient_nez_f1": float(np.mean([row["nez"] for row in patient_rows])),
            "patient_accuracy": float(np.mean([row["accuracy"] for row in patient_rows])),
        }
    )
    return values


__all__ = ["compute_task1_metrics"]
