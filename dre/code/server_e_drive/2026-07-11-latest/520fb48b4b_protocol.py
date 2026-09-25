from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from .evaluation import evaluate_patient_predictions
from .schemas import normalize_channel_name


class ProtocolError(ValueError):
    pass


FORBIDDEN_FEATURE_TOKENS = ("clinical_", "true_ez", "true_nez", "label", "oracle", "center", "subject_id", "path")


def assert_frozen_anchor(ledger: pd.DataFrame, *, expected_patients: int | None = 90, expected_folds: int = 5) -> dict[str, Any]:
    required = {"subject_id", "outer_fold", "channel_name_norm", "old_v3_selected", "clinical_true_ez"}
    missing = required - set(ledger.columns)
    if missing:
        raise ProtocolError(f"canonical ledger missing: {sorted(missing)}")
    subject_folds = ledger.groupby("subject_id")["outer_fold"].nunique()
    if (subject_folds != 1).any():
        raise ProtocolError("every subject must have exactly one source outer_fold")
    folds = sorted(int(item) for item in ledger["outer_fold"].unique())
    if expected_folds is not None and len(folds) != expected_folds:
        raise ProtocolError(f"expected {expected_folds} source folds, got {folds}")
    n_subjects = int(ledger["subject_id"].nunique())
    if expected_patients is not None and n_subjects != expected_patients:
        raise ProtocolError(f"expected {expected_patients} patients, got {n_subjects}")
    duplicates = ledger.duplicated(["subject_id", "channel_name_norm"], keep=False)
    if duplicates.any():
        raise ProtocolError("duplicate subject-channel rows are forbidden")
    return {"n_patients": n_subjects, "folds": folds, "n_channels": int(len(ledger)), "selected_total": int(ledger["old_v3_selected"].sum())}


def anchor_parity(ledger: pd.DataFrame, *, expected_macro_f1: float, tolerance: float = 1e-6) -> dict[str, Any]:
    summary = evaluate_patient_predictions(ledger, "old_v3_selected")
    delta = float(summary["patient_macro_f1"] - expected_macro_f1)
    return {"passed": abs(delta) <= tolerance, "expected_patient_macro_f1": float(expected_macro_f1), "recomputed_patient_macro_f1": float(summary["patient_macro_f1"]), "delta": delta, "tolerance": float(tolerance), "n_patients": int(summary["n_patients"])}


def assert_no_leakage(*, train_subjects: Iterable[str], test_subjects: Iterable[str], feature_names: Iterable[str],
                      calibration_subjects: Iterable[str] = (), gate_subjects: Iterable[str] = (),
                      scaler_subjects: Iterable[str] = (), gate_threshold_subjects: Iterable[str] = (),
                      hyperparameter_search_subjects: Iterable[str] = (), checkpoint_selection_subjects: Iterable[str] = (),
                      ensemble_selection_subjects: Iterable[str] = ()) -> dict[str, Any]:
    train, test = set(map(str, train_subjects)), set(map(str, test_subjects))
    overlap = sorted(train & test)
    if overlap:
        raise ProtocolError(f"train/test subject leakage: {overlap}")
    features = list(feature_names)
    forbidden = sorted(name for name in features if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS))
    if forbidden:
        raise ProtocolError(f"forbidden model features: {forbidden}")
    roles = {"scaler": scaler_subjects, "calibrator": calibration_subjects, "gate": gate_subjects,
             "gate_threshold": gate_threshold_subjects, "hyperparameter_search": hyperparameter_search_subjects,
             "checkpoint_selection": checkpoint_selection_subjects, "ensemble_selection": ensemble_selection_subjects}
    result: dict[str, Any] = {"train_test_subject_overlap": [], "forbidden_features": []}
    for role, subjects in roles.items():
        values = set(map(str, subjects))
        outside = sorted(values - train)
        overlap_test = sorted(values & test)
        if outside or overlap_test:
            raise ProtocolError(f"{role} subjects must be outer-train only; outside_train={outside}, outer_test_overlap={overlap_test}")
        result[f"{role}_fit_subjects"] = sorted(values)
        result[f"{role}_test_overlap"] = []
    return result


def audit_cache_subject_coverage(ledger: pd.DataFrame, channel_features: dict[str, dict[str, object]], *, strict: bool) -> dict[str, Any]:
    """Require all frozen V3 subjects while preserving per-run/channel missingness as an audit."""
    expected_subjects = set(ledger["subject_id"].astype(str))
    observed_subjects = set(map(str, channel_features.subjects() if hasattr(channel_features, "subjects") else channel_features))
    missing_subjects = sorted(expected_subjects - observed_subjects)
    missing_channels: dict[str, list[str]] = {}
    for subject_id, group in ledger.groupby("subject_id", sort=True):
        if hasattr(channel_features, "runs"):
            available = {normalize_channel_name(name) for run in channel_features.runs(str(subject_id)) for name in run.channel_names}
        else:
            available = {normalize_channel_name(name) for name in channel_features.get(str(subject_id), {})}
        absent = sorted(set(group["channel_name_norm"].astype(str)) - available)
        if absent:
            missing_channels[str(subject_id)] = absent
    audit = {"n_expected_subjects": len(expected_subjects), "n_cache_subjects": len(observed_subjects), "missing_subjects": missing_subjects, "missing_channel_count": int(sum(len(values) for values in missing_channels.values())), "missing_channels_by_subject": missing_channels}
    if strict and missing_subjects:
        raise ProtocolError(f"window cache missing frozen V3 subjects: {missing_subjects[:20]} (total={len(missing_subjects)})")
    return audit
