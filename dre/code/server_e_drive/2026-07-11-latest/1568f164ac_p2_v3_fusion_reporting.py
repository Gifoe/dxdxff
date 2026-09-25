"""Leak-free thresholding and patient-equal reporting for P2-Q10-CF."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score


THRESHOLD_CANDIDATES = np.arange(0.0, 1.000001, 0.005)


def _mrr(labels_ez: np.ndarray, score_ez: np.ndarray) -> float:
    order = np.argsort(-score_ez, kind="mergesort")
    hit = np.flatnonzero(labels_ez[order] == 1)
    return float(1.0 / (hit[0] + 1)) if len(hit) else 0.0


def _patient_row(group: pd.DataFrame, *, score_nez_column: str, prediction_nez: np.ndarray, threshold: float | None, truek: bool) -> dict:
    y = group.label_nez.to_numpy(int)
    pred = np.asarray(prediction_nez, dtype=int)
    score_nez = group[score_nez_column].to_numpy(float)
    score_ez = 1.0 - score_nez
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=[1, 0], zero_division=0)
    true_ez = (y == 0).astype(int)
    return {
        "subject_id": str(group.subject_id.iloc[0]), "center": str(group.center.iloc[0]), "outer_fold": int(group.outer_fold.iloc[0]),
        "n_channels": int(len(group)), "true_ez_count": int(true_ez.sum()), "predicted_ez_count": int((pred == 0).sum()),
        "selected_validation_threshold": np.nan if threshold is None else float(threshold),
        "patient_macro_accuracy": float(accuracy_score(y, pred)),
        "patient_macro_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "patient_macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
        "patient_weighted_f1": float(f1_score(y, pred, labels=[0, 1], average="weighted", zero_division=0)),
        "patient_macro_nez_precision": float(precision[0]), "patient_macro_nez_recall": float(recall[0]), "patient_macro_nez_f1": float(f1[0]),
        "patient_macro_ez_precision": float(precision[1]), "patient_macro_ez_recall": float(recall[1]), "patient_macro_ez_f1": float(f1[1]),
        "patient_macro_auroc_nez": float(roc_auc_score(y, score_nez)) if np.unique(y).size > 1 else 0.0,
        "patient_macro_auprc_nez": float(average_precision_score(y, score_nez)) if np.unique(y).size > 1 else float(y[0]),
        "patient_macro_auroc_ez": float(roc_auc_score(true_ez, score_ez)) if np.unique(true_ez).size > 1 else 0.0,
        "patient_macro_auprc_ez": float(average_precision_score(true_ez, score_ez)) if np.unique(true_ez).size > 1 else float(true_ez[0]),
        "patient_macro_ez_mrr": _mrr(true_ez, score_ez), "top1_is_ez_rate": float(true_ez[np.argmax(score_ez)] == 1),
        "patient_ez_recall_at_true_count": float(true_ez[np.argsort(-score_ez, kind="mergesort")[: int(true_ez.sum())]].mean()) if true_ez.sum() else 0.0,
        "predicted_nez_count_mae": float(abs(pred.sum() - y.sum())), "predicted_nez_fraction_mae": float(abs(pred.mean() - y.mean())),
        "predicted_ez_count_mae": float(abs((pred == 0).sum() - true_ez.sum())), "predicted_ez_fraction_mae": float(abs((pred == 0).mean() - true_ez.mean())),
        "true_count_used_for_prediction": bool(truek),
    }


def evaluate_probability_predictions(frame: pd.DataFrame, *, score_nez_column: str, threshold: float | None = None, truek: bool = False) -> pd.DataFrame:
    rows = []
    for _, group in frame.groupby("subject_id", sort=True):
        score_nez = group[score_nez_column].to_numpy(float)
        if truek:
            k = int((group.label_nez.to_numpy(int) == 0).sum())
            pred = np.ones(len(group), dtype=int)
            pred[np.argsort(-(1.0 - score_nez), kind="mergesort")[:k]] = 0
        else:
            if threshold is None:
                raise ValueError("Legal evaluation needs a validation-selected threshold")
            pred = (score_nez >= float(threshold)).astype(int)
        rows.append(_patient_row(group, score_nez_column=score_nez_column, prediction_nez=pred, threshold=threshold, truek=truek))
    return pd.DataFrame(rows)


def aggregate_patients(frame: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    metrics = [column for column in frame.columns if column.startswith("patient_") or column in {"top1_is_ez_rate", "predicted_nez_count_mae", "predicted_nez_fraction_mae", "predicted_ez_count_mae", "predicted_ez_fraction_mae"}]
    if by is None:
        return pd.DataFrame([{**{column: float(frame[column].mean()) for column in metrics}, "n_patients": int(frame.subject_id.nunique())}])
    return frame.groupby(by, as_index=False).agg(n_patients=("subject_id", "nunique"), **{column: (column, "mean") for column in metrics})


def center_gap(patient_frame: pd.DataFrame) -> float:
    values = patient_frame.groupby("center")["patient_macro_f1"].mean()
    return float(values.max() - values.min()) if len(values) else 0.0


def _threshold_patient_summary(validation: pd.DataFrame, *, score_nez_column: str, threshold: float) -> tuple[float, float, float, float, float, float]:
    """Fast threshold-only metrics; full sklearn metrics are computed only once after selection."""
    f1s: list[float] = []
    balanced: list[float] = []
    ez_f1s: list[float] = []
    nez_f1s: list[float] = []
    ez_mae: list[float] = []
    centers: dict[str, list[float]] = {}
    for _, group in validation.groupby("subject_id", sort=False):
        y = group.label_nez.to_numpy(int)
        pred = (group[score_nez_column].to_numpy(float) >= threshold).astype(int)
        def _f1(label: int) -> tuple[float, float]:
            tp = int(((y == label) & (pred == label)).sum())
            fp = int(((y != label) & (pred == label)).sum())
            fn = int(((y == label) & (pred != label)).sum())
            denominator = 2 * tp + fp + fn
            recall_denominator = tp + fn
            return (0.0 if denominator == 0 else 2 * tp / denominator, 0.0 if recall_denominator == 0 else tp / recall_denominator)
        nez_f1, nez_recall = _f1(1)
        ez_f1, ez_recall = _f1(0)
        macro = .5 * (nez_f1 + ez_f1)
        f1s.append(macro); balanced.append(.5 * (nez_recall + ez_recall)); ez_f1s.append(ez_f1); nez_f1s.append(nez_f1)
        ez_mae.append(abs(int((pred == 0).sum()) - int((y == 0).sum())))
        centers.setdefault(str(group.center.iloc[0]), []).append(macro)
    center_values = [float(np.mean(values)) for values in centers.values()]
    gap = float(max(center_values) - min(center_values)) if center_values else 0.0
    return float(np.mean(f1s)), float(np.mean(balanced)), float(np.mean(ez_f1s)), float(np.mean(nez_f1s)), float(np.mean(ez_mae)), gap


def select_validation_threshold(validation: pd.DataFrame, *, score_nez_column: str) -> tuple[float, pd.DataFrame]:
    rows = []
    best_key: tuple | None = None
    selected: float | None = None
    for threshold in THRESHOLD_CANDIDATES:
        macro_f1, balanced_accuracy, ez_f1, nez_f1, predicted_ez_count_mae, gap = _threshold_patient_summary(validation, score_nez_column=score_nez_column, threshold=float(threshold))
        row = {
            "threshold": float(threshold), "validation_patient_macro_f1": macro_f1,
            "validation_patient_balanced_accuracy": balanced_accuracy,
            "validation_patient_ez_f1": ez_f1, "validation_patient_nez_f1": nez_f1,
            "validation_predicted_ez_count_mae": predicted_ez_count_mae, "validation_center_gap": gap,
            "selected": False,
        }
        key = (row["validation_patient_macro_f1"], row["validation_patient_balanced_accuracy"], row["validation_patient_ez_f1"], -row["validation_predicted_ez_count_mae"], -row["validation_center_gap"], -abs(row["threshold"] - .5), -row["threshold"])
        if best_key is None or key > best_key:
            best_key, selected = key, float(threshold)
        rows.append(row)
    search = pd.DataFrame(rows)
    assert selected is not None
    search.loc[np.isclose(search.threshold, selected), "selected"] = True
    if int(search.selected.sum()) != 1:
        raise RuntimeError("Threshold selection must choose exactly one candidate")
    return selected, search


def oracle_patient_thresholds(frame: pd.DataFrame, *, score_nez_column: str) -> pd.DataFrame:
    """Test-label oracle diagnostic, intentionally isolated from formal prediction."""
    rows = []
    for subject_id, group in frame.groupby("subject_id", sort=True):
        threshold, _ = select_validation_threshold(group, score_nez_column=score_nez_column)
        patient = evaluate_probability_predictions(group, score_nez_column=score_nez_column, threshold=threshold).iloc[0].to_dict()
        patient.update({"subject_id": subject_id, "oracle_threshold": threshold, "analysis_status": "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE_TEST_LABEL_ORACLE", "true_test_labels_used": True, "formal_prediction": False})
        rows.append(patient)
    return pd.DataFrame(rows)


def paired_bootstrap(base: pd.DataFrame, candidate: pd.DataFrame, *, repeats: int, seed: int, metrics: Iterable[str]) -> pd.DataFrame:
    keys = ["subject_id", "center", "outer_fold"]
    paired = base.merge(candidate, on=keys, suffixes=("_p2", "_fusion"), validate="one_to_one")
    rng = np.random.default_rng(seed)
    rows = []
    worst_center = str(paired.groupby("center")["patient_macro_f1_p2"].mean().idxmin())
    scopes = [("overall", paired), ("lzu", paired[paired.center.eq("lzu")]), ("non_lzu", paired[~paired.center.eq("lzu")]), (f"worst_center:{worst_center}", paired[paired.center.eq(worst_center)])]
    for scope, data in scopes:
        for metric in metrics:
            delta = data[f"{metric}_fusion"].to_numpy(float) - data[f"{metric}_p2"].to_numpy(float)
            if not len(delta):
                continue
            samples = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(repeats)])
            rows.append({"scope": scope, "metric": metric, "mean_delta": float(delta.mean()), "CI_2.5": float(np.quantile(samples, .025)), "CI_97.5": float(np.quantile(samples, .975)), "probability_delta_gt_zero": float((samples > 0).mean()), "n_patients_improved": int((delta > 0).sum()), "n_patients_worsened": int((delta < 0).sum()), "n_patients_unchanged": int((delta == 0).sum()), "n_delta_gt_0.05": int((delta > .05).sum()), "n_delta_lt_minus_0.05": int((delta < -.05).sum()), "bootstrap_repeats": repeats, "seed": seed})
    return pd.DataFrame(rows)


__all__ = ["THRESHOLD_CANDIDATES", "evaluate_probability_predictions", "aggregate_patients", "center_gap", "select_validation_threshold", "oracle_patient_thresholds", "paired_bootstrap"]
