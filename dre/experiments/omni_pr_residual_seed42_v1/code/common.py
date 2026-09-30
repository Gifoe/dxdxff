from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **values) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def table(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path.name}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    if any(list(row) != keys for row in rows):
        raise RuntimeError(f"Inconsistent table columns for {path.name}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp = np.exp(x[~positive])
    out[~positive] = exp / (1.0 + exp)
    return out


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks, zero based, without scipy dependency."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return result


def relative_features(embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    embedding = np.asarray(embedding, dtype=np.float32)
    if embedding.ndim != 2 or embedding.shape[1] != 32 or len(embedding) < 1:
        raise ValueError("Expected non-empty [channels,32] embedding")
    count = len(embedding)
    fallback = count < 3
    if fallback:
        reference = np.repeat(np.median(embedding, axis=0, keepdims=True), count, axis=0)
    else:
        reference = np.stack([
            np.median(np.delete(embedding, index, axis=0), axis=0)
            for index in range(count)
        ])
    difference = embedding - reference
    ranks = np.zeros_like(embedding, dtype=np.float32)
    if count > 1:
        for dimension in range(32):
            ranks[:, dimension] = (2.0 * average_ranks(embedding[:, dimension]) /
                                   float(count - 1) - 1.0)
    if not (np.isfinite(difference).all() and np.isfinite(ranks).all()):
        raise RuntimeError("Nonfinite patient-relative feature")
    return difference.astype(np.float32), ranks.astype(np.float32), fallback


def score_from_segment_logits(segment_logits: list[np.ndarray], delta: np.ndarray) -> np.ndarray:
    """Frozen official aggregation: sigmoid per segment, then mean."""
    return np.asarray([
        float(sigmoid(np.asarray(logits, dtype=np.float64) + float(shift)).mean())
        for logits, shift in zip(segment_logits, np.asarray(delta).reshape(-1))
    ], dtype=np.float64)


def mean_logit_score(segment_logits: list[np.ndarray], delta: np.ndarray) -> np.ndarray:
    return np.asarray([
        float(sigmoid(np.asarray(logits, dtype=np.float64).mean() + float(shift)))
        for logits, shift in zip(segment_logits, np.asarray(delta).reshape(-1))
    ], dtype=np.float64)


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    pred = (score >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        "auroc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
        "ap": float(average_precision_score(y, score)) if np.any(y == 1) else None,
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "pathological_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": None if sensitivity is None else float(sensitivity),
        "specificity": None if specificity is None else float(specificity),
        "accuracy": float(accuracy_score(y, pred)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def choose_threshold(y: np.ndarray, score: np.ndarray) -> dict:
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], np.asarray(score, dtype=float))))
    best = None
    for threshold in candidates:
        value = binary_metrics(y, score, float(threshold))
        key = (value["macro_f1"], -float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold), value)
    assert best is not None
    return {"threshold": best[1], "candidate_count": len(candidates), **best[2]}


def ranking_metrics(frame) -> dict:
    patient_ap, reciprocal, top1 = [], [], []
    for _, group in frame.groupby("patient", sort=False):
        by_channel = group.groupby("channel", sort=False).agg(
            y=("y", "first"), score=("score", "mean")).reset_index()
        y = by_channel.y.to_numpy(dtype=np.int8)
        s = by_channel.score.to_numpy(dtype=np.float64)
        if np.any(y == 1):
            patient_ap.append(float(average_precision_score(y, s)))
            order = np.argsort(-s, kind="stable")
            positives = np.flatnonzero(y[order] == 1)
            reciprocal.append(1.0 / float(positives[0] + 1))
            top1.append(float(y[order[0]] == 1))
    return {
        "patient_equal_ap": float(np.mean(patient_ap)) if patient_ap else None,
        "mrr": float(np.mean(reciprocal)) if reciprocal else None,
        "top1": float(np.mean(top1)) if top1 else None,
        "ranking_estimable_patients": len(patient_ap),
    }


def center_name(value: str) -> str:
    mapping = {"hup": "HUP", "open-ieeg": "Open-iEEG", "openieeg": "Open-iEEG",
               "sourcesink": "SourceSink", "source-sink": "SourceSink",
               "zurich": "Zurich"}
    key = str(value).strip().casefold()
    if key not in mapping:
        raise RuntimeError(f"Unexpected center {value!r}")
    return mapping[key]


def percentile_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "std": float(values.std()),
            **{f"p{p}": float(np.percentile(values, p)) for p in (5, 25, 50, 75, 95)}}
