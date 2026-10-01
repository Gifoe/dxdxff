from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp = np.exp(values[~positive])
    out[~positive] = exp / (1.0 + exp)
    return out


def center_from_patient(patient: str) -> str:
    low = str(patient).lower()
    if low.startswith("sub-hup"):
        return "HUP"
    if low.startswith("sub-openieeg"):
        return "Open-iEEG"
    if low.startswith("sub-sourcesink"):
        return "SourceSink"
    return "Other"


def patient_hash(patient: str) -> str:
    return hashlib.sha256(str(patient).encode("utf-8")).hexdigest()[:16]


def metrics(frame: pd.DataFrame, fixed_threshold: float = 0.5) -> dict:
    y = frame["y"].to_numpy(dtype=np.int8)
    score = frame["score"].to_numpy(dtype=np.float64)
    result = {
        "n_units": int(len(frame)),
        "auroc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
        "ap": float(average_precision_score(y, score)) if np.any(y == 1) else None,
        "macro_f1": float(f1_score(y, score >= fixed_threshold, average="macro", zero_division=0)),
    }
    patient_ap, reciprocal, top1, ndcg = [], [], [], []
    for _, group in frame.groupby("patient", sort=False):
        by_channel = group.groupby("channel", sort=False).agg(y=("y", "first"), score=("score", "mean")).reset_index()
        py = by_channel.y.to_numpy(dtype=np.int8)
        ps = by_channel.score.to_numpy(dtype=np.float64)
        if not np.any(py == 1):
            continue
        patient_ap.append(float(average_precision_score(py, ps)))
        order = np.argsort(-ps, kind="stable")
        positives = np.flatnonzero(py[order] == 1)
        reciprocal.append(1.0 / float(positives[0] + 1))
        top1.append(float(py[order[0]] == 1))
        discounts = 1.0 / np.log2(np.arange(2, len(py) + 2))
        dcg = float((py[order] * discounts).sum())
        ideal = float((np.sort(py)[::-1] * discounts).sum())
        ndcg.append(dcg / ideal if ideal else 0.0)
    result.update({
        "patient_equal_ap": float(np.mean(patient_ap)) if patient_ap else None,
        "mrr": float(np.mean(reciprocal)) if reciprocal else None,
        "top1": float(np.mean(top1)) if top1 else None,
        "ndcg": float(np.mean(ndcg)) if ndcg else None,
        "ranking_estimable_patients": int(len(patient_ap)),
    })
    return result


def edf_rng(seed: int, edf: str) -> np.random.Generator:
    key = int(hashlib.sha256(str(edf).encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(np.random.SeedSequence([int(seed), key]))


def select_indices(n: int, k: int | None, seed: int, edf: str) -> np.ndarray:
    if k is None or n <= k:
        return np.arange(n, dtype=np.int64)
    return edf_rng(seed, edf).permutation(n)[:k]


def describe(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(a)), "p5": float(np.percentile(a, 5)),
        "p25": float(np.percentile(a, 25)), "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)), "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)), "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=0)), "n": int(len(a)),
        "p_eq_5": float(np.mean(a == 5)),
    }
