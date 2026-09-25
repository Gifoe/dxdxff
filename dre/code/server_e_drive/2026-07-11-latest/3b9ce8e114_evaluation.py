from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support


def _patient_metrics(group: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    y_ez = group["clinical_true_ez"].astype(int).to_numpy()
    pred_ez = group[prediction_column].astype(int).to_numpy()
    y_nez, pred_nez = 1 - y_ez, 1 - pred_ez
    _, _, class_f1, _ = precision_recall_fscore_support(y_nez, pred_nez, labels=[1, 0], zero_division=0)
    return {
        "subject_id": str(group["subject_id"].iloc[0]),
        "outer_fold": int(group["outer_fold"].iloc[0]),
        "center": str(group["center"].iloc[0]),
        "patient_macro_f1": float(f1_score(y_nez, pred_nez, average="macro", zero_division=0)),
        "patient_ez_f1": float(class_f1[1]),
        "patient_nez_f1": float(class_f1[0]),
        "patient_accuracy": float(accuracy_score(y_nez, pred_nez)),
        "patient_balanced_accuracy": float(balanced_accuracy_score(y_nez, pred_nez)),
        "n_channels": int(len(group)),
        "k": int(pred_ez.sum()),
    }


def evaluate_patient_predictions(ledger: pd.DataFrame, prediction_column: str = "old_v3_selected") -> dict[str, Any]:
    if prediction_column not in ledger:
        raise ValueError(f"missing prediction column {prediction_column}")
    rows = [_patient_metrics(group, prediction_column) for _, group in ledger.groupby("subject_id", sort=True)]
    patient = pd.DataFrame(rows)
    if patient.empty:
        return {"patient_macro_f1": 0.0, "patient_macro_ez_f1": 0.0, "patient_macro_nez_f1": 0.0, "patient_rows": patient}
    summary = {
        "patient_macro_f1": float(patient["patient_macro_f1"].mean()),
        "patient_macro_ez_f1": float(patient["patient_ez_f1"].mean()),
        "patient_macro_nez_f1": float(patient["patient_nez_f1"].mean()),
        "channel_micro_f1": float(f1_score(1 - ledger["clinical_true_ez"].astype(int), 1 - ledger[prediction_column].astype(int), average="micro", zero_division=0)),
        "patient_balanced_accuracy": float(patient["patient_balanced_accuracy"].mean()),
        "n_patients": int(len(patient)),
        "patient_rows": patient,
    }
    return summary


def paired_patient_delta(anchor: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    joined = anchor.merge(candidate, on="subject_id", suffixes=("_anchor", "_candidate"), validate="one_to_one")
    joined["patient_macro_f1_delta"] = joined["patient_macro_f1_candidate"] - joined["patient_macro_f1_anchor"]
    return joined
