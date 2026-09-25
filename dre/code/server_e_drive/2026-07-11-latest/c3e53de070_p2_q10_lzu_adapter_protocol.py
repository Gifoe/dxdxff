"""Protocol, frozen-feature construction, and metrics for P2-Q10 LZU adaptation."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score

REQUIRED_CHANNEL_FIELDS = {"subject_id", "center", "channel_name", "label_nez", "base_nez_logit", "contextual_channel_embedding", "temporal_delta_norm", "q10_nez_probability"}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonicalize_channel_frame(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {"true_nez": "label_nez", "final_nez_logit": "base_nez_logit", "seizure_nez_probability_q10": "q10_nez_probability", "negative_anchor_distance": "anchor_distance_z", "valid_seizure_count": "valid_seizure_count"}
    result = frame.rename(columns={source: target for source, target in aliases.items() if source in frame.columns and target not in frame.columns}).copy()
    if "label_nez" not in result and "true_ez" in result:
        result["label_nez"] = 1 - result["true_ez"].astype(int)
    if "label_ez" not in result and "label_nez" in result:
        result["label_ez"] = 1 - result["label_nez"].astype(int)
    if "outer_fold" not in result and "fold_idx" in result:
        result["outer_fold"] = result["fold_idx"]
    missing = REQUIRED_CHANNEL_FIELDS - set(result.columns)
    if missing:
        raise ValueError(f"Frozen P2-Q10 channel ledger missing required fields: {sorted(missing)}")
    if "anchor_distance_z" not in result.columns and "anchor_residual" in result.columns:
        result["anchor_distance_z"] = result["anchor_residual"]
    if "anchor_distance_z" not in result.columns:
        raise ValueError("Frozen P2-Q10 ledger lacks anchor_distance_z and permitted fallback anchor_residual")
    if "valid_seizure_count" not in result.columns:
        raise ValueError("Frozen P2-Q10 ledger lacks valid_seizure_count")
    result["center"] = result["center"].astype(str).str.lower()
    result["label_nez"] = result["label_nez"].astype(int)
    if not set(result.label_nez.unique()).issubset({0, 1}):
        raise ValueError("label_nez must use exactly NEZ=1, EZ=0 semantics")
    for field in ("base_nez_logit", "q10_nez_probability", "anchor_distance_z", "temporal_delta_norm", "valid_seizure_count"):
        values = pd.to_numeric(result[field], errors="coerce")
        if not np.isfinite(values).all():
            raise ValueError(f"Frozen P2-Q10 field {field} contains missing or non-finite values")
        result[field] = values.astype(float)
    if result.duplicated(["subject_id", "channel_name"]).any():
        raise ValueError("Frozen P2-Q10 ledger has duplicate subject/channel rows")
    return result


def _embedding(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value.astype(float)
    elif isinstance(value, (list, tuple)):
        array = np.asarray(value, dtype=float)
    elif isinstance(value, str) and value.strip():
        try:
            array = np.asarray(json.loads(value), dtype=float)
        except json.JSONDecodeError:
            array = np.asarray(ast.literal_eval(value), dtype=float)
    else:
        raise ValueError("contextual_channel_embedding is missing; rerun frozen P2-Q10 inference with embedding ledger export")
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("contextual_channel_embedding must be a finite non-empty vector")
    return array


def robust_patient_standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = np.median(values); scale = 1.4826 * np.median(np.abs(values - median))
    if scale < 1e-5:
        scale = float(np.std(values))
    if scale < 1e-5:
        return np.zeros_like(values)
    return np.clip((values - median) / scale, -4.0, 4.0)


def stable_ez_percentile_rank(base_nez_logit: np.ndarray) -> np.ndarray:
    """Largest EZ score (-NEZ logit) maps to 1; ties receive stable average rank."""
    score = -np.asarray(base_nez_logit, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    ranks = np.empty(score.size, dtype=float)
    index = 0
    while index < score.size:
        end = index + 1
        while end < score.size and score[order[end]] == score[order[index]]:
            end += 1
        # position 1 is strongest EZ. Scale to [0, 1], including singleton patients.
        value = 1.0 if score.size == 1 else 1.0 - (((index + 1 + end) / 2.0) - 1.0) / (score.size - 1.0)
        ranks[order[index:end]] = value
        index = end
    return ranks


def build_frozen_adapter_features(frame: pd.DataFrame) -> tuple[torch.Tensor, int]:
    frame = canonicalize_channel_frame(frame)
    vectors: list[np.ndarray] = []
    embedding_dim: int | None = None
    for _, group in frame.groupby("subject_id", sort=False):
        logits = group.base_nez_logit.to_numpy(float)
        z = robust_patient_standardize(logits)
        rank = stable_ez_percentile_rank(logits)
        q10 = group.q10_nez_probability.to_numpy(float)
        anchor = group.anchor_distance_z.to_numpy(float)
        temporal = group.temporal_delta_norm.to_numpy(float)
        seizures = np.clip(group.valid_seizure_count.to_numpy(float) / 3.0, 0.0, 1.0)
        for offset, (_, row) in enumerate(group.iterrows()):
            embedding = _embedding(row.contextual_channel_embedding)
            embedding_dim = embedding.size if embedding_dim is None else embedding_dim
            if embedding.size != embedding_dim:
                raise ValueError("contextual_channel_embedding dimension differs between channels")
            vectors.append(np.concatenate([embedding, [logits[offset], z[offset], rank[offset], q10[offset], anchor[offset], temporal[offset], seizures[offset]]]))
    if not vectors:
        raise ValueError("Cannot build adapter features from an empty ledger")
    return torch.tensor(np.stack(vectors), dtype=torch.float32), int(embedding_dim or 0)


def split_lzu_fit_validation(fit_frame: pd.DataFrame, validation_frame: pd.DataFrame, outer_fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = canonicalize_channel_frame(fit_frame)
    validation = canonicalize_channel_frame(validation_frame)
    train_lzu, val_lzu = train[train.center.eq("lzu")].copy(), validation[validation.center.eq("lzu")].copy()
    n_fit, n_validation = train_lzu.subject_id.nunique(), val_lzu.subject_id.nunique()
    if n_fit < 2 or n_validation < 1:
        raise RuntimeError(f"outer fold {outer_fold}: adapter requires >=2 fit LZU and >=1 validation LZU patients, got {n_fit}/{n_validation}")
    return train_lzu, val_lzu


def frozen_threshold(frame: pd.DataFrame, outer_fold: int) -> float:
    for field in ("selected_threshold", "predicted_threshold", "classification_threshold", "frozen_threshold"):
        if field in frame.columns:
            values = pd.to_numeric(frame[field], errors="coerce").dropna().unique()
            if values.size == 1 and 0.0 <= float(values[0]) <= 1.0:
                return float(values[0])
    raise RuntimeError(f"outer fold {outer_fold}: missing one frozen validation-only global threshold in its test ledger")


def _patient_metrics(group: pd.DataFrame, *, logit_column: str, threshold: float, truek: bool = False) -> dict[str, float]:
    y = group.label_nez.to_numpy(int); logit = group[logit_column].to_numpy(float); score_nez = 1.0 / (1.0 + np.exp(-logit)); score_ez = 1.0 - score_nez
    if truek:
        pred_ez = np.zeros(len(group), dtype=int); count = int((y == 0).sum())
        pred_ez[np.argsort(-score_ez, kind="mergesort")[:count]] = 1; pred = 1 - pred_ez
    else:
        pred = (score_nez >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=[1, 0], zero_division=0)
    ez_positions = np.flatnonzero((y == 0)[np.argsort(-score_ez, kind="mergesort")])
    return {"patient_macro_accuracy": float(accuracy_score(y, pred)), "patient_macro_balanced_accuracy": float(balanced_accuracy_score(y, pred)), "patient_macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)), "patient_weighted_f1": float(f1_score(y, pred, labels=[0, 1], average="weighted", zero_division=0)), "patient_macro_nez_precision": float(precision[0]), "patient_macro_nez_recall": float(recall[0]), "patient_macro_nez_f1": float(f1[0]), "patient_macro_ez_precision": float(precision[1]), "patient_macro_ez_recall": float(recall[1]), "patient_macro_ez_f1": float(f1[1]), "patient_macro_auroc_nez": float(roc_auc_score(y, score_nez)) if np.unique(y).size > 1 else 0.0, "patient_macro_auprc_nez": float(average_precision_score(y, score_nez)) if np.unique(y).size > 1 else float(y[0]), "patient_macro_auroc_ez": float(roc_auc_score(1-y, score_ez)) if np.unique(y).size > 1 else 0.0, "patient_macro_auprc_ez": float(average_precision_score(1-y, score_ez)) if np.unique(y).size > 1 else float(1-y[0]), "patient_macro_ez_mrr": float(1.0 / (ez_positions[0] + 1)) if ez_positions.size else 0.0, "top1_is_ez_rate": float(y[np.argmax(score_ez)] == 0), "predicted_nez_count_mae": float(abs(pred.sum() - y.sum())), "predicted_nez_fraction_mae": float(abs(pred.mean() - y.mean())), "predicted_ez_count_mae": float(abs((1-pred).sum() - (1-y).sum())), "predicted_ez_fraction_mae": float(abs((1-pred).mean() - (1-y).mean()))}


def evaluate_channel_ledger(frame: pd.DataFrame, *, logit_column: str, truek: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = canonicalize_channel_frame(frame)
    rows: list[dict] = []
    for subject, group in frame.groupby("subject_id", sort=True):
        threshold = frozen_threshold(group, int(group.outer_fold.iloc[0])) if not truek else float("nan")
        item = {"subject_id": str(subject), "center": str(group.center.iloc[0]), "outer_fold": int(group.outer_fold.iloc[0]), "n_channels": int(len(group)), "true_ez_count": int((group.label_nez == 0).sum()), "predicted_ez_count": int((1 - ((1/(1+np.exp(-group[logit_column].to_numpy(float)))) >= threshold).astype(int)).sum()) if not truek else int((group.label_nez == 0).sum()), "frozen_threshold": threshold, **_patient_metrics(group, logit_column=logit_column, threshold=threshold if not truek else .5, truek=truek)}
        rows.append(item)
    patients = pd.DataFrame(rows)
    metric_columns = [column for column in patients if column.startswith("patient_") or column in {"top1_is_ez_rate", "predicted_nez_count_mae", "predicted_nez_fraction_mae", "predicted_ez_count_mae", "predicted_ez_fraction_mae"}]
    def aggregate(source: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
        if by is None: return pd.DataFrame([{**{column: float(source[column].mean()) for column in metric_columns}, "n_patients": int(source.subject_id.nunique())}])
        return source.groupby(by, as_index=False).agg(n_patients=("subject_id", "nunique"), **{column: (column, "mean") for column in metric_columns})
    return patients, aggregate(patients), aggregate(patients, "outer_fold"), aggregate(patients, "center")


def non_lzu_invariance_audit(frame: pd.DataFrame, *, threshold: float | None = None) -> dict:
    non = frame[~frame.center.astype(str).str.lower().eq("lzu")]
    base_logit = non.base_nez_logit.to_numpy(float); adapted_logit = non.adapted_nez_logit.to_numpy(float)
    base_score, adapted_score = 1/(1+np.exp(-base_logit)), 1/(1+np.exp(-adapted_logit))
    if threshold is None: base_pred, adapted_pred = non.base_pred_nez.to_numpy(int), non.adapted_pred_nez.to_numpy(int)
    else: base_pred, adapted_pred = (base_score >= threshold), (adapted_score >= threshold)
    zero = float(np.max(np.abs(base_logit-adapted_logit))) if len(non) else 0.0
    return {"n_non_lzu_patients": int(non.subject_id.nunique()), "n_non_lzu_channels": int(len(non)), "max_abs_logit_delta": zero, "max_abs_score_nez_delta": float(np.max(np.abs(base_score-adapted_score))) if len(non) else 0.0, "max_abs_score_ez_delta": float(np.max(np.abs((1-base_score)-(1-adapted_score)))) if len(non) else 0.0, "n_prediction_changes": int(np.count_nonzero(base_pred != adapted_pred)), "bitwise_logit_equal": bool(np.array_equal(base_logit, adapted_logit)), "bitwise_score_nez_equal": bool(np.array_equal(base_score, adapted_score)), "bitwise_score_ez_equal": bool(np.array_equal(1-base_score, 1-adapted_score)), "bitwise_prediction_equal": bool(np.array_equal(base_pred, adapted_pred)), "passed": bool(np.array_equal(base_logit, adapted_logit) and np.array_equal(base_score, adapted_score) and np.array_equal(base_pred, adapted_pred))}


__all__ = ["REQUIRED_CHANNEL_FIELDS", "canonicalize_channel_frame", "sha256_file", "robust_patient_standardize", "stable_ez_percentile_rank", "build_frozen_adapter_features", "split_lzu_fit_validation", "frozen_threshold", "evaluate_channel_ledger", "non_lzu_invariance_audit"]
