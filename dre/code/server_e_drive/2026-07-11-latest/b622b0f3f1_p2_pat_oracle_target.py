"""Validation-only stable patient oracle targets for P2-Q10-PAT."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def _macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0))


def compute_stable_patient_oracle_threshold_target(
    base_nez_logit: np.ndarray,
    label_nez: np.ndarray,
    global_threshold_logit: float,
    *,
    subject_id: str = "",
    outer_fold: int = 0,
    split_role: str = "validation",
    epsilon_f1: float = 0.002,
    max_target_residual: float = 1.0,
) -> dict[str, float | int | str | bool]:
    """Create a stable oracle residual target using validation labels only."""
    if split_role != "validation":
        raise RuntimeError("Patient oracle threshold targets are validation-only")
    logits = np.asarray(base_nez_logit, dtype=float)
    labels = np.asarray(label_nez, dtype=int)
    if logits.ndim != 1 or logits.size < 2 or labels.shape != logits.shape:
        raise ValueError("Oracle target requires aligned one-dimensional arrays with >=2 channels")
    if not np.isfinite(logits).all() or not np.isfinite(global_threshold_logit):
        raise ValueError("Oracle target logits must be finite")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Oracle target labels must use NEZ=1, EZ=0")
    if epsilon_f1 < 0 or max_target_residual <= 0:
        raise ValueError("epsilon_f1 must be non-negative and max_target_residual positive")

    unique = np.unique(np.sort(logits))
    candidates = [float(unique[0] - 1e-6), float(unique[-1] + 1e-6), float(global_threshold_logit)]
    candidates.extend(float((left + right) / 2.0) for left, right in zip(unique[:-1], unique[1:]))
    candidates = sorted(set(candidates))
    scores = np.asarray([_macro_f1(labels, (logits >= threshold).astype(int)) for threshold in candidates])
    maximum = float(scores.max())
    near = np.flatnonzero(scores >= maximum - float(epsilon_f1) - 1e-15)
    selected_index = min(near, key=lambda index: (abs(candidates[index] - global_threshold_logit), candidates[index]))
    oracle_logit = float(candidates[selected_index])
    raw_residual = oracle_logit - float(global_threshold_logit)
    residual = float(np.clip(raw_residual, -max_target_residual, max_target_residual))
    base_f1 = _macro_f1(labels, (logits >= global_threshold_logit).astype(int))
    return {
        "subject_id": str(subject_id),
        "outer_fold": int(outer_fold),
        "global_threshold_logit": float(global_threshold_logit),
        "oracle_threshold_logit": oracle_logit,
        "oracle_threshold_probability": float(1.0 / (1.0 + np.exp(-oracle_logit))),
        "oracle_target_residual": residual,
        "oracle_patient_macro_f1": maximum,
        "base_global_patient_macro_f1": base_f1,
        "oracle_gain": maximum - base_f1,
        "n_candidate_thresholds": len(candidates),
        "n_near_optimal_thresholds": int(len(near)),
        "target_was_clipped": bool(abs(residual - raw_residual) > 1e-12),
    }


__all__ = ["compute_stable_patient_oracle_threshold_target"]
