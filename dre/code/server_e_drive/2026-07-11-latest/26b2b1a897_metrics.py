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


def _patient_ez_ranking_metrics(table: pd.DataFrame) -> dict[str, float]:
    """Patient-macro ranking metrics with EZ as the retrieval target."""
    ndcg: list[float] = []; reciprocal_rank: list[float] = []; recall_at_k: list[float] = []; precision_at_k: list[float] = []
    for _, group in table.groupby("subject_id", sort=False):
        relevance = (1 - group["label_nez"].to_numpy(dtype=int)).astype(float)
        scores = 1.0 - group["score_nez_probability"].to_numpy(dtype=float)
        ranked = relevance[np.argsort(-scores, kind="stable")]
        positives = int(relevance.sum())
        if positives == 0:
            ndcg.append(float("nan")); reciprocal_rank.append(float("nan")); recall_at_k.append(float("nan")); precision_at_k.append(float("nan")); continue
        discounts = 1.0 / np.log2(np.arange(2, len(ranked) + 2))
        ideal_dcg = float((np.sort(relevance)[::-1] * discounts).sum())
        ndcg.append(float((ranked * discounts).sum()) / ideal_dcg)
        reciprocal_rank.append(1.0 / (int(np.flatnonzero(ranked == 1)[0]) + 1))
        top = ranked[:positives]
        recall_at_k.append(float(top.sum() / positives)); precision_at_k.append(float(top.mean()))
    def mean_or_nan(values: list[float]) -> float:
        finite = np.asarray(values, dtype=float)
        return float(np.nanmean(finite)) if np.isfinite(finite).any() else float("nan")

    return {"NDCG_EZ": mean_or_nan(ndcg), "MRR_EZ": mean_or_nan(reciprocal_rank), "Recall_at_true_EZ_count": mean_or_nan(recall_at_k), "Precision_at_true_EZ_count": mean_or_nan(precision_at_k)}


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
    values.update(_patient_ez_ranking_metrics(table))
    values.update({"macro_f1": values["f1_macro"], "ez_f1": values["f1_ez"], "nez_f1": values["f1_nez"], "auroc": values["AUROC_EZ"], "auprc": values["AUPRC_EZ"], "ndcg": values["NDCG_EZ"], "mrr": values["MRR_EZ"], "recall_at_true_ez_count": values["Recall_at_true_EZ_count"]})
    return values


__all__ = ["compute_task1_metrics"]
