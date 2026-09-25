"""One authoritative patient-equal metric implementation for all ablations."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, ndcg_score, roc_auc_score


PATIENT_METRICS = [
    "patient_macro_f1", "patient_ez_f1", "patient_nez_f1", "patient_accuracy", "patient_balanced_accuracy",
    "patient_ez_auprc", "patient_ez_auroc", "patient_ez_mrr", "patient_ez_ndcg", "top1_is_ez_rate",
    "predicted_ez_count_mae", "predicted_ez_fraction_mae", "truek_patient_macro_f1",
]


def _patient_metrics(patient: pd.DataFrame, *, score_column: str, threshold: float) -> dict:
    label_nez = patient.label_nez.to_numpy(int)
    score_nez = patient[score_column].to_numpy(float)
    pred_nez = (score_nez >= threshold).astype(int)
    label_ez = 1 - label_nez
    score_ez = 1.0 - score_nez
    pred_ez = 1 - pred_nez
    order = np.argsort(-score_ez, kind="mergesort")
    hits = np.flatnonzero(label_ez[order] == 1)
    true_k = int(label_ez.sum())
    truek_pred_nez = np.ones(len(patient), dtype=int)
    truek_pred_nez[order[:true_k]] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        balanced = balanced_accuracy_score(label_nez, pred_nez)
    two_classes = np.unique(label_ez).size == 2
    return {
        "subject_id": str(patient.subject_id.iloc[0]), "center": str(patient.center.iloc[0]), "outer_fold": int(patient.outer_fold.iloc[0]),
        "n_channels": int(len(patient)), "true_ez_count": true_k, "predicted_ez_count": int(pred_ez.sum()),
        "true_ez_fraction": float(label_ez.mean()), "predicted_ez_fraction": float(pred_ez.mean()),
        "patient_macro_f1": float(f1_score(label_nez, pred_nez, labels=[0, 1], average="macro", zero_division=0)),
        "patient_ez_f1": float(f1_score(label_nez, pred_nez, labels=[0], average="macro", zero_division=0)),
        "patient_nez_f1": float(f1_score(label_nez, pred_nez, labels=[1], average="macro", zero_division=0)),
        "patient_accuracy": float((label_nez == pred_nez).mean()),
        "patient_balanced_accuracy": float(balanced),
        "patient_ez_auprc": float(average_precision_score(label_ez, score_ez)) if two_classes else np.nan,
        "patient_ez_auroc": float(roc_auc_score(label_ez, score_ez)) if two_classes else np.nan,
        "patient_ez_mrr": float(1.0 / (hits[0] + 1)) if len(hits) else np.nan,
        "patient_ez_ndcg": float(ndcg_score(label_ez[None, :], score_ez[None, :])) if true_k else np.nan,
        "top1_is_ez_rate": float(label_ez[order[0]] == 1) if len(order) else np.nan,
        "predicted_ez_count_mae": float(abs(pred_ez.sum() - label_ez.sum())),
        "predicted_ez_fraction_mae": float(abs(pred_ez.mean() - label_ez.mean())),
        "truek_patient_macro_f1": float(f1_score(label_nez, truek_pred_nez, labels=[0, 1], average="macro", zero_division=0)),
        "selected_validation_threshold": float(threshold),
    }


def evaluate(frame: pd.DataFrame, *, score_column: str, threshold: float, experiment: str, analysis_status: str) -> pd.DataFrame:
    rows = [_patient_metrics(patient, score_column=score_column, threshold=threshold) for _, patient in frame.groupby("subject_id", sort=True)]
    result = pd.DataFrame(rows)
    result["experiment"] = experiment
    result["analysis_status"] = analysis_status
    result["true_count_used_for_prediction"] = False
    return result


def aggregate(patient_rows: pd.DataFrame, *, by: str | None = None) -> pd.DataFrame:
    def row(group: pd.DataFrame) -> dict:
        result = {metric: float(np.nanmean(group[metric])) for metric in PATIENT_METRICS}
        result.update({
            "n_patients": int(group.subject_id.nunique()), "n_channels": int(group.n_channels.sum()),
            "true_ez_fraction": float(np.average(group.true_ez_fraction, weights=group.n_channels)),
            "predicted_ez_fraction": float(np.average(group.predicted_ez_fraction, weights=group.n_channels)),
            "ez_fraction_bias": float(np.average(group.predicted_ez_fraction - group.true_ez_fraction)),
            "valid_ez_auprc_patients": int(group.patient_ez_auprc.notna().sum()),
            "valid_ez_auroc_patients": int(group.patient_ez_auroc.notna().sum()),
        })
        return result
    if by is None:
        return pd.DataFrame([row(patient_rows)])
    return pd.DataFrame([{by: key, **row(group)} for key, group in patient_rows.groupby(by, sort=True)])


def summaries(patient_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    experiments = []
    folds = []
    centers = []
    for experiment, group in patient_rows.groupby("experiment", sort=False):
        common = {"experiment": experiment, "analysis_status": str(group.analysis_status.iloc[0])}
        experiments.append({**common, **aggregate(group).iloc[0].to_dict()})
        fold = aggregate(group, by="outer_fold"); fold["experiment"] = experiment; fold["analysis_status"] = common["analysis_status"]; folds.append(fold)
        center = aggregate(group, by="center"); center["experiment"] = experiment; center["analysis_status"] = common["analysis_status"]; centers.append(center)
    return pd.DataFrame(experiments), pd.concat(folds, ignore_index=True), pd.concat(centers, ignore_index=True)
