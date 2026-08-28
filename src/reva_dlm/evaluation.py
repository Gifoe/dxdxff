"""Deterministic CPU-only out-of-fold evaluation for ReVA-DLM G1.

This module intentionally contains no threshold search or feature selection.
Every classifier uses a fixed 0.5 decision threshold, all preprocessing is fit
inside its training fold, and repeated out-of-fold predictions are averaged per
input row before metrics and case-clustered confidence intervals are computed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


DEFAULT_RANDOM_STATE = 20260827
DEFAULT_THRESHOLD = 0.5

__all__ = [
    "DEFAULT_RANDOM_STATE",
    "DEFAULT_THRESHOLD",
    "case_level_bootstrap_ci",
    "compute_binary_metrics",
    "evaluate_g1_models",
    "evaluate_models",
    "make_repeated_stratified_cv",
    "run_oof_evaluation",
    "to_serializable_records",
]


def _binary_target(y: Sequence[Any]) -> np.ndarray:
    values = np.asarray(y)
    if values.ndim != 1:
        values = values.reshape(-1)
    if values.size == 0:
        raise ValueError("y must contain at least one sample")
    if pd.isna(values).any():
        raise ValueError("y contains missing values")

    unique = np.unique(values)
    if unique.size != 2 or set(unique.tolist()) != {0, 1}:
        raise ValueError(f"y must contain both binary classes 0 and 1; got {unique!r}")
    return values.astype(np.int8, copy=False)


def _numeric_frame(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        frame = X.copy()
    else:
        array = np.asarray(X)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError("X must be a two-dimensional matrix")
        frame = pd.DataFrame(
            array,
            columns=[f"feature_{index:03d}" for index in range(array.shape[1])],
        )

    if frame.shape[1] == 0:
        raise ValueError("X must contain at least one feature")
    string_columns = [str(column) for column in frame.columns]
    if len(set(string_columns)) != len(string_columns):
        raise ValueError("feature names must be unique after string conversion")
    frame.columns = string_columns
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame.replace([np.inf, -np.inf], np.nan)


def make_repeated_stratified_cv(
    y: Sequence[Any],
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    max_splits: int = 5,
    n_repeats: int = 3,
) -> tuple[RepeatedStratifiedKFold, dict[str, Any]]:
    """Build deterministic stratified CV, capped at 5 folds x 3 repeats.

    The fold count is reduced to the minority-class count when necessary.  A
    minority class with fewer than two examples cannot support out-of-fold
    stratification and raises ``ValueError`` rather than silently fitting and
    evaluating on the same observations.
    """

    target = _binary_target(y)
    if int(max_splits) < 2:
        raise ValueError("max_splits must be at least 2")
    if int(n_repeats) < 1:
        raise ValueError("n_repeats must be at least 1")

    counts = np.bincount(target, minlength=2)
    minority_count = int(counts.min())
    if minority_count < 2:
        raise ValueError(
            "RepeatedStratifiedKFold requires at least two samples in each class; "
            f"class counts are {counts.tolist()}"
        )

    effective_splits = min(5, int(max_splits), minority_count)
    effective_repeats = min(3, int(n_repeats))
    reason = (
        "minority_class_count"
        if effective_splits < min(5, int(max_splits))
        else "requested_or_protocol_cap"
    )
    config: dict[str, Any] = {
        "splitter": "RepeatedStratifiedKFold",
        "requested_max_splits": int(max_splits),
        "requested_repeats": int(n_repeats),
        "effective_splits": int(effective_splits),
        "effective_repeats": int(effective_repeats),
        "total_folds": int(effective_splits * effective_repeats),
        "class_counts": {"0": int(counts[0]), "1": int(counts[1])},
        "minority_count": minority_count,
        "fold_adjustment_reason": reason,
        "random_state": int(random_state),
        "threshold": DEFAULT_THRESHOLD,
        "n_jobs": 1,
    }
    splitter = RepeatedStratifiedKFold(
        n_splits=effective_splits,
        n_repeats=effective_repeats,
        random_state=int(random_state),
    )
    return splitter, config


def compute_binary_metrics(
    y_true: Sequence[Any],
    y_probability: Sequence[float],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Compute the fixed-threshold G1 metric set.

    ``AUROC`` and ``AUPRC`` are returned as NaN for a one-class resample, which
    can occur in a case-level bootstrap draw.  Point estimates passed by the
    main evaluator always contain both classes.
    """

    target = np.asarray(y_true, dtype=np.int8).reshape(-1)
    probability = np.asarray(y_probability, dtype=float).reshape(-1)
    if target.size != probability.size:
        raise ValueError("y_true and y_probability must have equal length")
    if target.size == 0:
        raise ValueError("metrics require at least one sample")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    if not np.isfinite(probability).all():
        raise ValueError("predicted probabilities must be finite")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("predicted probabilities must lie in [0, 1]")

    prediction = (probability >= float(threshold)).astype(np.int8)
    has_both_classes = np.unique(target).size == 2
    return {
        "auroc": float(roc_auc_score(target, probability))
        if has_both_classes
        else float("nan"),
        "auprc": float(average_precision_score(target, probability))
        if has_both_classes
        else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction))
        if has_both_classes
        else float("nan"),
        "f1": float(f1_score(target, prediction, zero_division=0))
        if has_both_classes
        else float("nan"),
        "brier": float(brier_score_loss(target, probability)),
    }


def case_level_bootstrap_ci(
    y_true: Sequence[Any],
    y_probability: Sequence[float],
    case_ids: Sequence[Any],
    *,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = DEFAULT_RANDOM_STATE,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Return deterministic percentile CIs from a case-clustered bootstrap.

    Unique cases, not individual rows, are sampled with replacement.  If one
    case has several rows, all of its rows are carried together each time that
    case is drawn.  This prevents artificially narrow intervals for repeated
    observations of the same decoding case.
    """

    target = np.asarray(y_true, dtype=np.int8).reshape(-1)
    probability = np.asarray(y_probability, dtype=float).reshape(-1)
    cases = np.asarray(case_ids, dtype=object).reshape(-1)
    if not (target.size == probability.size == cases.size):
        raise ValueError("y_true, y_probability, and case_ids must have equal length")
    if int(n_bootstrap) < 1:
        raise ValueError("n_bootstrap must be at least 1")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    if pd.isna(cases).any():
        raise ValueError("case_ids contains missing values")

    unique_cases = pd.unique(cases)
    row_indices = {
        case: np.flatnonzero(cases == case).astype(np.int64) for case in unique_cases
    }
    rng = np.random.default_rng(int(random_state))
    names = ("auroc", "auprc", "balanced_accuracy", "f1", "brier")
    draws: dict[str, list[float]] = {name: [] for name in names}

    for _ in range(int(n_bootstrap)):
        sampled_cases = rng.choice(unique_cases, size=len(unique_cases), replace=True)
        sampled_rows = np.concatenate([row_indices[case] for case in sampled_cases])
        draw_metrics = compute_binary_metrics(
            target[sampled_rows], probability[sampled_rows], threshold=threshold
        )
        for name in names:
            value = draw_metrics[name]
            if np.isfinite(value):
                draws[name].append(float(value))

    point = compute_binary_metrics(target, probability, threshold=threshold)
    alpha = (1.0 - float(confidence_level)) / 2.0
    records: list[dict[str, Any]] = []
    for name in names:
        valid = np.asarray(draws[name], dtype=float)
        lower, upper = (
            np.quantile(valid, [alpha, 1.0 - alpha]).tolist()
            if valid.size
            else [float("nan"), float("nan")]
        )
        records.append(
            {
                "metric": name,
                "estimate": float(point[name]),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "confidence_level": float(confidence_level),
                "n_bootstrap": int(n_bootstrap),
                "n_valid_bootstrap": int(valid.size),
                "n_cases": int(len(unique_cases)),
            }
        )
    return pd.DataFrame.from_records(records)


def _model_specs(
    feature_names: Sequence[str],
    random_state: int,
    *,
    include_single_features: bool = True,
) -> list[dict[str, Any]]:
    logistic = LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=int(random_state),
        n_jobs=1,
    )

    specs: list[dict[str, Any]] = []
    if include_single_features:
        for feature in feature_names:
            specs.append(
                {
                    "model": f"single_logistic::{feature}",
                    "feature": feature,
                    "columns": [feature],
                    "estimator": Pipeline(
                        [
                            (
                                "impute",
                                SimpleImputer(strategy="median", keep_empty_features=True),
                            ),
                            ("scale", StandardScaler()),
                            ("classifier", clone(logistic)),
                        ]
                    ),
                }
            )

    specs.extend(
        [
            {
                "model": "logistic_all_features",
                "feature": "__all__",
                "columns": list(feature_names),
                "estimator": Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="median", keep_empty_features=True),
                        ),
                        ("scale", StandardScaler()),
                        ("classifier", clone(logistic)),
                    ]
                ),
            },
            {
                "model": "hist_gradient_boosting_balanced",
                "feature": "__all__",
                "columns": list(feature_names),
                "estimator": Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="median", keep_empty_features=True),
                        ),
                        (
                            "classifier",
                            HistGradientBoostingClassifier(
                                class_weight="balanced",
                                learning_rate=0.05,
                                max_iter=150,
                                max_leaf_nodes=15,
                                l2_regularization=1.0,
                                early_stopping=False,
                                random_state=int(random_state),
                            ),
                        ),
                    ]
                ),
            },
        ]
    )
    return specs


def evaluate_models(
    X: Any,
    y: Sequence[Any],
    case_ids: Sequence[Any] | None = None,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    max_splits: int = 5,
    n_repeats: int = 3,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    include_single_features: bool = True,
) -> dict[str, Any]:
    """Run all required G1 baselines with deterministic repeated OOF CV.

    Models comprise one balanced standardized logistic regression per feature
    when ``include_single_features`` is true,
    a balanced standardized logistic regression over all features, and a
    balanced ``HistGradientBoostingClassifier``.  Preprocessing is cloned and
    fit separately within each fold.  No threshold is tuned: all hard metrics
    use ``probability >= 0.5``.

    Parameters
    ----------
    X, y:
        Numeric causal feature matrix and binary ``CONTINUE_BENEFICIAL`` target.
    case_ids:
        Stable source-case identifiers used as bootstrap clusters.  If omitted,
        row numbers are used.  For strict case-independent CV, pass one row per
        case (the standard fixed-checkpoint G1 design).
    random_state:
        Seed shared by split generation, estimators, and bootstrap resampling.
    max_splits, n_repeats:
        Requested CV dimensions; protocol caps them at 5 and 3, respectively,
        and folds are reduced further when the minority class is small.
    n_bootstrap, confidence_level:
        Number of case-clustered percentile-bootstrap draws and CI level.

    Returns
    -------
    dict
        ``metrics``, ``bootstrap_ci``, ``oof_predictions``, and
        ``fold_predictions`` are tidy pandas DataFrames suitable for CSV or
        parquet output. ``cv_config`` and ``feature_names`` are JSON-serializable
        metadata.  OOF probabilities are averaged across repeats per input row
        before metrics and bootstrap intervals are calculated.
    """

    frame = _numeric_frame(X)
    target = _binary_target(y)
    if len(frame) != target.size:
        raise ValueError("X and y must contain the same number of rows")

    if case_ids is None:
        cases = np.asarray([str(index) for index in range(target.size)], dtype=object)
    else:
        cases = np.asarray(case_ids, dtype=object).reshape(-1)
        if cases.size != target.size:
            raise ValueError("case_ids and y must contain the same number of rows")
        if pd.isna(cases).any():
            raise ValueError("case_ids contains missing values")

    splitter, cv_config = make_repeated_stratified_cv(
        target,
        random_state=int(random_state),
        max_splits=int(max_splits),
        n_repeats=int(n_repeats),
    )
    splits = list(splitter.split(np.zeros(target.size, dtype=np.int8), target))
    n_splits = int(cv_config["effective_splits"])
    n_expected = int(cv_config["effective_repeats"])

    aggregate_records: list[pd.DataFrame] = []
    fold_records: list[pd.DataFrame] = []
    metric_records: list[dict[str, Any]] = []
    ci_frames: list[pd.DataFrame] = []

    for spec in _model_specs(
        frame.columns.tolist(),
        int(random_state),
        include_single_features=bool(include_single_features),
    ):
        columns = spec["columns"]
        matrix = frame.loc[:, columns]
        probability_sum = np.zeros(target.size, dtype=float)
        prediction_count = np.zeros(target.size, dtype=np.int16)

        for split_index, (train_index, test_index) in enumerate(splits):
            repeat = split_index // n_splits
            fold = split_index % n_splits
            estimator = clone(spec["estimator"])
            # Cap BLAS/OpenMP pools as well as estimator-level n_jobs.  This is
            # both reproducible and prevents accidental CPU oversubscription.
            with threadpool_limits(limits=1):
                estimator.fit(matrix.iloc[train_index], target[train_index])
                fold_probability = estimator.predict_proba(matrix.iloc[test_index])[:, 1]

            probability_sum[test_index] += fold_probability
            prediction_count[test_index] += 1
            fold_records.append(
                pd.DataFrame(
                    {
                        "row_id": test_index.astype(np.int64),
                        "case_id": cases[test_index],
                        "y_true": target[test_index].astype(np.int8),
                        "model": spec["model"],
                        "feature": spec["feature"],
                        "repeat": int(repeat),
                        "fold": int(fold),
                        "y_probability": fold_probability.astype(float),
                        "y_pred": (fold_probability >= DEFAULT_THRESHOLD).astype(np.int8),
                    }
                )
            )

        if not np.all(prediction_count == n_expected):
            raise RuntimeError(
                "OOF accounting failure: each row must be predicted once per repeat"
            )
        oof_probability = probability_sum / prediction_count
        oof = pd.DataFrame(
            {
                "row_id": np.arange(target.size, dtype=np.int64),
                "case_id": cases,
                "y_true": target.astype(np.int8),
                "model": spec["model"],
                "feature": spec["feature"],
                "y_probability": oof_probability.astype(float),
                "y_pred": (oof_probability >= DEFAULT_THRESHOLD).astype(np.int8),
                "n_oof_predictions": prediction_count.astype(np.int16),
            }
        )
        aggregate_records.append(oof)

        metrics = compute_binary_metrics(target, oof_probability)
        metric_records.append(
            {
                "model": spec["model"],
                "feature": spec["feature"],
                "n_samples": int(target.size),
                "n_cases": int(pd.unique(cases).size),
                "n_positive": int(target.sum()),
                "n_negative": int((1 - target).sum()),
                "threshold": DEFAULT_THRESHOLD,
                **metrics,
            }
        )
        ci = case_level_bootstrap_ci(
            target,
            oof_probability,
            cases,
            n_bootstrap=int(n_bootstrap),
            confidence_level=float(confidence_level),
            random_state=int(random_state),
            threshold=DEFAULT_THRESHOLD,
        )
        ci.insert(0, "feature", spec["feature"])
        ci.insert(0, "model", spec["model"])
        ci_frames.append(ci)

    result = {
        "metrics": pd.DataFrame.from_records(metric_records),
        "bootstrap_ci": pd.concat(ci_frames, ignore_index=True),
        "oof_predictions": pd.concat(aggregate_records, ignore_index=True),
        "fold_predictions": pd.concat(fold_records, ignore_index=True),
        "cv_config": cv_config,
        "feature_names": frame.columns.tolist(),
    }
    return result


def to_serializable_records(result: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an evaluation result to plain JSON-compatible records.

    Non-finite metric values (possible only for invalid one-class auxiliary
    inputs) become ``None``.  The main evaluator itself returns DataFrames so
    callers can write losslessly to parquet/CSV.
    """

    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return float(value) if np.isfinite(float(value)) else None
        return value

    converted: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, pd.DataFrame):
            converted[key] = clean(value.to_dict(orient="records"))
        else:
            converted[key] = clean(value)
    return converted


# Explicit, discoverable aliases for scripts that prefer domain-specific names.
evaluate_g1_models = evaluate_models
run_oof_evaluation = evaluate_models
