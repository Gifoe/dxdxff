from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from task1_baselines.models.feature_models import build_feature_estimator, feature_parameter_grid
from task1_baselines.thresholds import Task1Threshold, select_patient_macro_threshold


@dataclass
class FeatureOOFResult:
    oof: pd.DataFrame
    selected_parameters: pd.DataFrame
    scaler_fit_subjects: dict[int, tuple[str, ...]]


def _columns(table: pd.DataFrame) -> list[str]:
    excluded = {"subject_id", "center", "channel_name", "label_nez", "clinical_true_nez", "clinical_true_ez", "valid_seizure_count", "valid_window_count"}
    return [column for column in table if column not in excluded and pd.api.types.is_numeric_dtype(table[column])]


def _inner_predictions(train: pd.DataFrame, columns: list[str], model_name: str, params: dict[str, Any], seed: int, inner_folds: int, compact: bool) -> pd.DataFrame:
    subjects = train["subject_id"].to_numpy()
    unique_subjects = np.unique(subjects)
    splitter = GroupKFold(n_splits=min(inner_folds, len(unique_subjects)))
    rows = []
    for train_index, valid_index in splitter.split(train, train["label_nez"], groups=subjects):
        current = dict(params)
        if compact and model_name == "random_forest":
            current["_compact"] = True
        estimator = build_feature_estimator(model_name, current, seed=seed)
        estimator.fit(train.iloc[train_index][columns], train.iloc[train_index]["label_nez"])
        valid = train.iloc[valid_index][["subject_id", "label_nez"]].copy()
        valid["score_nez_probability"] = estimator.predict_proba(train.iloc[valid_index][columns])[:, 1]
        rows.append(valid)
    return pd.concat(rows, ignore_index=True)


def run_feature_oof(
    table: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    *,
    model_name: str,
    seed: int,
    inner_folds: int = 4,
    compact: bool = False,
    fixed_baseline: bool = False,
) -> FeatureOOFResult:
    merged = table.merge(fold_manifest[["subject_id", "outer_fold"]], on="subject_id", how="inner", validate="many_to_one")
    if len(merged) != len(table):
        raise ValueError("Task 1 feature table does not exactly match the frozen fold cohort.")
    columns = _columns(merged)
    oof_rows = []
    parameter_rows = []
    fit_subjects: dict[int, tuple[str, ...]] = {}
    for fold in sorted(merged["outer_fold"].unique()):
        train = merged[merged["outer_fold"] != fold].reset_index(drop=True)
        test = merged[merged["outer_fold"] == fold].copy()
        fit_subjects[int(fold)] = tuple(sorted(train["subject_id"].unique()))
        if fixed_baseline:
            # One conventional configuration per model; no inner tuning or outer-test selection.
            best_params = feature_parameter_grid(model_name, compact=True)[0]
            threshold = Task1Threshold(0.5, float("nan"), float("nan"), float("nan"), source="fixed_0.5")
        else:
            candidates = []
            for params in feature_parameter_grid(model_name, compact=compact):
                inner = _inner_predictions(train, columns, model_name, params, seed, inner_folds, compact)
                threshold = select_patient_macro_threshold(inner)
                candidates.append((threshold.patient_macro_f1, threshold.patient_ez_f1, params, threshold, inner))
            _, _, best_params, threshold, _ = max(candidates, key=lambda item: (item[0], item[1], str(sorted(item[2].items()))))
        fitted_params = dict(best_params)
        if (compact or fixed_baseline) and model_name == "random_forest":
            fitted_params["_compact"] = True
        estimator = build_feature_estimator(model_name, fitted_params, seed=seed).fit(train[columns], train["label_nez"])
        test["score_nez_probability"] = estimator.predict_proba(test[columns])[:, 1]
        test["score_ez_probability"] = 1.0 - test["score_nez_probability"]
        test["predicted_nez"] = (test["score_nez_probability"] >= threshold.threshold).astype(int)
        test["predicted_ez"] = 1 - test["predicted_nez"]
        test["selected_threshold"] = threshold.threshold
        test["threshold_source"] = threshold.source
        test["model"] = model_name
        test["seed"] = seed
        oof_rows.append(test)
        parameter_rows.append({"outer_fold": fold, "model": model_name, "seed": seed, "parameters": best_params, "threshold": threshold.threshold})
    oof = pd.concat(oof_rows, ignore_index=True).sort_values(["subject_id", "channel_name"], kind="stable").reset_index(drop=True)
    ledger_columns = [
        "model", "seed", "subject_id", "center", "outer_fold", "channel_name",
        "label_nez", "clinical_true_nez", "clinical_true_ez",
        "score_nez_probability", "score_ez_probability", "selected_threshold",
        "predicted_nez", "predicted_ez", "threshold_source",
        "valid_seizure_count", "valid_window_count",
    ]
    oof = oof[[column for column in ledger_columns if column in oof]]
    return FeatureOOFResult(oof, pd.DataFrame(parameter_rows), fit_subjects)


__all__ = ["FeatureOOFResult", "run_feature_oof"]
