from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


EPSILON = 1e-6
FEATURES = ("tail_excess", "q90_excess", "heterogeneity")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    result = np.empty_like(value)
    nonnegative = value >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-value[nonnegative]))
    exponent = np.exp(value[~nonnegative])
    result[~nonnegative] = exponent / (1.0 + exponent)
    return result


def logit(value):
    value = np.clip(np.asarray(value, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    return np.log(value) - np.log1p(-value)


def distribution_features(pathological_probabilities: np.ndarray) -> dict[str, float | int]:
    probability = np.asarray(pathological_probabilities, dtype=np.float64)
    if probability.ndim != 1 or len(probability) < 1 or not np.isfinite(probability).all():
        raise ValueError("Expected one finite, non-empty segment-probability vector")
    count = len(probability)
    top_count = max(1, int(math.ceil(0.2 * count)))
    mean = float(probability.mean())
    top = float(np.partition(probability, count - top_count)[-top_count:].mean())
    q90 = float(np.quantile(probability, 0.90, method="linear"))
    std = float(probability.std(ddof=0))
    return {
        "mean_score": mean,
        "top20_mean": top,
        "q90": q90,
        "tail_excess": top - mean,
        "q90_excess": q90 - mean,
        "heterogeneity": std,
        "segment_count": count,
    }


@dataclass(frozen=True)
class RobustScaler:
    median: np.ndarray
    iqr: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "RobustScaler":
        values = np.asarray(values, dtype=np.float64)
        median = np.median(values, axis=0)
        q25, q75 = np.quantile(values, [0.25, 0.75], axis=0, method="linear")
        return cls(median=median, iqr=q75 - q25)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return np.clip((values - self.median) / (self.iqr + EPSILON), -5.0, 5.0)

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {
            feature: {"median": float(self.median[index]), "iqr": float(self.iqr[index])}
            for index, feature in enumerate(FEATURES)
        }


VARIANTS = {
    "V0_MEAN": (),
    "V1_TAIL_EXCESS": (0,),
    "V2_Q90_EXCESS": (1,),
    "V3_HETEROGENEITY": (2,),
    "V4_FULL_DMIL": (0, 1, 2),
}


def patient_equal_loss_and_gradient(
    base_logit: np.ndarray,
    scaled_features: np.ndarray,
    labels: np.ndarray,
    patients: np.ndarray,
    beta: np.ndarray,
) -> tuple[float, np.ndarray]:
    _, patient_codes = np.unique(np.asarray(patients), return_inverse=True)
    return _coded_patient_equal_loss_and_gradient(
        base_logit, scaled_features, labels, patient_codes, beta
    )


def _coded_patient_equal_loss_and_gradient(
    base_logit: np.ndarray,
    scaled_features: np.ndarray,
    labels: np.ndarray,
    patient_codes: np.ndarray,
    beta: np.ndarray,
) -> tuple[float, np.ndarray]:
    score_logit = np.asarray(base_logit) + np.asarray(scaled_features) @ np.asarray(beta)
    probability = sigmoid(score_logit)
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.where(labels > 0.5, 2.0, 1.0)
    patient_codes = np.asarray(patient_codes, dtype=np.int64)
    patient_count = int(patient_codes.max()) + 1
    # Patient-equal weighted BCE: first apply the fixed 2:1 class weight to
    # each channel loss, then take the arithmetic mean over channels within
    # each patient. Dividing by the sum of class weights would partially
    # cancel the preregistered positive-class weight.
    denominator = np.bincount(patient_codes, minlength=patient_count)
    clipped = np.clip(probability, EPSILON, 1.0 - EPSILON)
    row_loss = -weights * (labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped))
    patient_loss = np.bincount(patient_codes, weights=row_loss, minlength=patient_count) / denominator
    residual = weights * (probability - labels)
    gradient_sum = np.stack(
        [
            np.bincount(
                patient_codes,
                weights=residual * scaled_features[:, column],
                minlength=patient_count,
            )
            for column in range(scaled_features.shape[1])
        ],
        axis=1,
    )
    patient_gradient = gradient_sum / denominator[:, None]
    return float(patient_loss.mean()), patient_gradient.mean(axis=0)


def fit_adapter(
    base_logit: np.ndarray,
    scaled_features: np.ndarray,
    labels: np.ndarray,
    patients: np.ndarray,
    active: tuple[int, ...],
    max_epochs: int = 500,
    patience: int = 50,
    learning_rate: float = 1e-2,
) -> dict[str, object]:
    beta = np.zeros(3, dtype=np.float64)
    _, patient_codes = np.unique(np.asarray(patients), return_inverse=True)
    if not active:
        loss, _ = _coded_patient_equal_loss_and_gradient(
            base_logit, scaled_features, labels, patient_codes, beta
        )
        return {"beta": beta, "selected_epoch": 0, "epochs_run": 0, "loss": loss}
    first_moment = np.zeros(3, dtype=np.float64)
    second_moment = np.zeros(3, dtype=np.float64)
    best_beta = beta.copy()
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    active_array = np.asarray(active, dtype=np.int64)
    for epoch in range(1, max_epochs + 1):
        _, gradient = _coded_patient_equal_loss_and_gradient(
            base_logit, scaled_features, labels, patient_codes, beta
        )
        inactive = np.ones(3, dtype=bool)
        inactive[active_array] = False
        gradient[inactive] = 0.0
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient * gradient
        corrected_first = first_moment / (1.0 - 0.9**epoch)
        corrected_second = second_moment / (1.0 - 0.999**epoch)
        beta[active_array] -= learning_rate * corrected_first[active_array] / (
            np.sqrt(corrected_second[active_array]) + 1e-8
        )
        beta[inactive] = 0.0
        loss, _ = _coded_patient_equal_loss_and_gradient(
            base_logit, scaled_features, labels, patient_codes, beta
        )
        if loss < best_loss - 1e-12:
            best_loss = loss
            best_beta = beta.copy()
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    return {
        "beta": best_beta,
        "selected_epoch": best_epoch,
        "epochs_run": epoch,
        "loss": best_loss,
    }


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))

    # Exact O(n log n) equivalent of evaluating score >= threshold for every
    # candidate. Candidates are ascending, so np.argmax also preserves the
    # preregistered tie break in favor of the smallest threshold.
    positive_scores = np.sort(scores[labels == 1])
    negative_scores = np.sort(scores[labels == 0])
    tp = len(positive_scores) - np.searchsorted(positive_scores, candidates, side="left")
    fp = len(negative_scores) - np.searchsorted(negative_scores, candidates, side="left")
    fn = len(positive_scores) - tp
    tn = len(negative_scores) - fp

    positive_denominator = 2 * tp + fp + fn
    negative_denominator = 2 * tn + fp + fn
    positive_f1 = np.divide(
        2 * tp,
        positive_denominator,
        out=np.zeros_like(candidates, dtype=np.float64),
        where=positive_denominator != 0,
    )
    negative_f1 = np.divide(
        2 * tn,
        negative_denominator,
        out=np.zeros_like(candidates, dtype=np.float64),
        where=negative_denominator != 0,
    )
    macro_f1 = 0.5 * (positive_f1 + negative_f1)
    best_index = int(np.argmax(macro_f1))
    threshold = float(candidates[best_index])
    metrics = binary_metrics(labels, scores, threshold)
    return {"threshold": threshold, "candidate_count": len(candidates), **metrics}


def binary_metrics(labels, scores, threshold: float | None = None, predictions=None) -> dict:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if predictions is None:
        if threshold is None:
            raise ValueError("threshold or predictions required")
        predictions = (scores >= threshold).astype(np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "ap": float(average_precision_score(labels, scores)) if np.any(labels == 1) else None,
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "pathological_f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def ranking_metrics(frame: pd.DataFrame) -> dict[str, float | int | None]:
    patient_ap: list[float] = []
    reciprocal: list[float] = []
    top1: list[float] = []
    ndcg: list[float] = []
    for _, group in frame.groupby("patient", sort=False):
        by_channel = group.groupby("channel", sort=False).agg(
            y=("y", "first"), score=("score", "mean")
        ).reset_index()
        y = by_channel.y.to_numpy(dtype=np.int8)
        score = by_channel.score.to_numpy(dtype=np.float64)
        if not np.any(y == 1):
            continue
        patient_ap.append(float(average_precision_score(y, score)))
        order = np.argsort(-score, kind="stable")
        positives = np.flatnonzero(y[order] == 1)
        reciprocal.append(1.0 / float(positives[0] + 1))
        top1.append(float(y[order[0]] == 1))
        discounts = 1.0 / np.log2(np.arange(2, len(y) + 2))
        dcg = float((y[order] * discounts).sum())
        ideal = float((np.sort(y)[::-1] * discounts).sum())
        ndcg.append(dcg / ideal if ideal else 0.0)
    return {
        "patient_equal_ap": float(np.mean(patient_ap)) if patient_ap else None,
        "mrr": float(np.mean(reciprocal)) if reciprocal else None,
        "top1": float(np.mean(top1)) if top1 else None,
        "ndcg": float(np.mean(ndcg)) if ndcg else None,
        "ranking_estimable_patients": len(patient_ap),
    }
