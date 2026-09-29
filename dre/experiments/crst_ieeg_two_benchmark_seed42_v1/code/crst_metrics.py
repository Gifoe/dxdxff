"""Patient-equal selection metrics and train-side F1 threshold choice."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def patient_metrics(labels, scores):
    y = np.asarray(labels, dtype=int)
    p = np.asarray(scores, dtype=float)
    keep = np.isin(y, (0, 1)) & np.isfinite(p)
    y, p = y[keep], p[keep]
    if len(y) == 0:
        return {"auroc": float("nan"), "ap": float("nan"),
                "mrr": float("nan"), "top1": float("nan")}
    has_both = len(np.unique(y)) == 2
    order = np.argsort(-p, kind="stable")
    positive_ranks = np.flatnonzero(y[order] == 1)
    return {"auroc": float(roc_auc_score(y, p)) if has_both else float("nan"),
            "ap": float(average_precision_score(y, p)) if positive_ranks.size else float("nan"),
            "mrr": float(1 / (1 + positive_ranks[0])) if positive_ranks.size else float("nan"),
            "top1": float(positive_ranks[0] == 0) if positive_ranks.size else float("nan")}


def aggregate_patients(rows):
    metrics = [patient_metrics(y, p) for y, p in rows]
    result = {}
    for name in ("auroc", "ap", "mrr", "top1"):
        values = [item[name] for item in metrics if math.isfinite(item[name])]
        result[name] = float(np.mean(values)) if values else float("nan")
        result[name + "_estimable_patients"] = len(values)
    return result


def frozen_threshold(validation_rows):
    y = np.concatenate([np.asarray(y, int) for y, _ in validation_rows])
    p = np.concatenate([np.asarray(p, float) for _, p in validation_rows])
    keep = np.isin(y, (0, 1)) & np.isfinite(p)
    y, p = y[keep], p[keep]
    if len(np.unique(y)) != 2:
        raise RuntimeError("Validation threshold needs both classes")
    candidates = np.unique(np.r_[0.0, 0.5, 1.0, p])
    best = None
    for tau in candidates:
        pred = p >= tau
        macro = f1_score(y, pred, average="macro", zero_division=0)
        pathological = f1_score(y, pred, pos_label=1, zero_division=0)
        sensitivity = np.mean(pred[y == 1])
        specificity = np.mean(~pred[y == 0])
        balanced = 0.5 * (sensitivity + specificity)
        rank = (macro, pathological, balanced, -abs(float(tau) - 0.5), -float(tau))
        if best is None or rank > best[0]:
            best = (rank, float(tau))
    return {"threshold": best[1], "validation_macro_f1": float(best[0][0]),
            "validation_pathological_f1": float(best[0][1]),
            "validation_balanced_accuracy": float(best[0][2]),
            "candidates": int(len(candidates))}
