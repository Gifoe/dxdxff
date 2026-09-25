from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from outcome_hifos.calibration import ProtocolLeakageError


@dataclass(frozen=True)
class FusionResult:
    inner_oof: pd.DataFrame
    coefficients: dict[str, float]
    outer_predictions: pd.DataFrame
    input_representation: str


def _assert_inner_oof(table: pd.DataFrame, branch: str) -> None:
    required = {"subject_id", "role", "outcome", "raw_logit", "input_representation", "outer_fold_idx", "inner_fold_idx", "seed"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{branch} inner-OOF table is missing columns: {missing}")
    invalid = sorted(set(table["role"].astype(str)) - {"inner_oof"})
    if invalid:
        raise ProtocolLeakageError(f"{branch} stacker fit received non-inner roles: {invalid}")
    if table["subject_id"].astype(str).duplicated().any():
        raise ValueError(f"{branch} inner-OOF table has duplicate subjects.")
    if table["outer_fold_idx"].nunique() != 1:
        raise ValueError(f"{branch} inner-OOF table mixes outer fold values.")
    if table["seed"].nunique() != 1:
        raise ValueError(f"{branch} inner-OOF table mixes seed values.")
    representations = set(table["input_representation"].astype(str))
    if representations != {"raw_logit"}:
        raise ValueError(f"{branch} inner representation must be raw_logit, got {sorted(representations)}.")


def _safe_logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def fit_cross_fitted_stacker(
    feature_inner_oof: pd.DataFrame,
    fm_inner_oof: pd.DataFrame,
    outer_predictions: pd.DataFrame,
    *,
    regularization_c: float = 0.1,
) -> FusionResult:
    _assert_inner_oof(feature_inner_oof, "feature")
    _assert_inner_oof(fm_inner_oof, "fm")
    feature = feature_inner_oof.copy()
    fm = fm_inner_oof.copy()
    feature["subject_id"] = feature["subject_id"].astype(str)
    fm["subject_id"] = fm["subject_id"].astype(str)
    if int(feature["outer_fold_idx"].iloc[0]) != int(fm["outer_fold_idx"].iloc[0]):
        raise ValueError("Feature and FM inner-OOF tables use different outer fold values.")
    if int(feature["seed"].iloc[0]) != int(fm["seed"].iloc[0]):
        raise ValueError("Feature and FM inner-OOF tables use different seed values.")
    aligned = feature.merge(
        fm.loc[:, ["subject_id", "outcome", "raw_logit", "inner_fold_idx"]],
        on="subject_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_feature", "_fm"),
    ).sort_values("subject_id", kind="stable").reset_index(drop=True)
    if set(aligned["subject_id"]) != set(feature["subject_id"]) or set(aligned["subject_id"]) != set(fm["subject_id"]):
        raise ValueError("Feature and FM inner-OOF cohorts are not identical.")
    if not np.array_equal(aligned["outcome_feature"].to_numpy(), aligned["outcome_fm"].to_numpy()):
        raise ValueError("Feature and FM inner-OOF outcomes disagree.")
    if not np.array_equal(aligned["inner_fold_idx_feature"].to_numpy(), aligned["inner_fold_idx_fm"].to_numpy()):
        raise ValueError("Feature and FM inner-OOF rows use different inner fold assignments.")
    design = np.column_stack([aligned["raw_logit_feature"].to_numpy(dtype=float), aligned["raw_logit_fm"].to_numpy(dtype=float)])
    target = aligned["outcome_feature"].to_numpy(dtype=np.int64)
    estimator = LogisticRegression(C=float(regularization_c), penalty="l2", solver="lbfgs", max_iter=2000)
    estimator.fit(design, target)
    required_outer = {"subject_id", "feature_raw_logit", "fm_raw_logit", "feature_input_representation", "fm_input_representation", "role", "outer_fold_idx", "seed"}
    missing_outer = sorted(required_outer - set(outer_predictions.columns))
    if missing_outer:
        raise ValueError(f"Outer fusion table is missing columns: {missing_outer}")
    outer = outer_predictions.copy()
    if set(outer["role"].astype(str)) != {"outer_test"}:
        raise ProtocolLeakageError("Outer fusion predictions must contain only outer_test rows.")
    if outer["outer_fold_idx"].nunique() != 1 or int(outer["outer_fold_idx"].iloc[0]) != int(feature["outer_fold_idx"].iloc[0]):
        raise ValueError("Outer fusion rows do not match the inner-OOF outer fold.")
    if outer["seed"].nunique() != 1 or int(outer["seed"].iloc[0]) != int(feature["seed"].iloc[0]):
        raise ValueError("Outer fusion rows do not match the inner-OOF seed.")
    if set(outer["feature_input_representation"].astype(str)) != {"raw_logit"} or set(outer["fm_input_representation"].astype(str)) != {"raw_logit"}:
        raise ValueError("Outer branch representation must match inner raw_logit representation.")
    outer_design = np.column_stack([outer["feature_raw_logit"].to_numpy(dtype=float), outer["fm_raw_logit"].to_numpy(dtype=float)])
    outer["fusion_probability"] = estimator.predict_proba(outer_design)[:, 1]
    inner_export = aligned.rename(
        columns={
            "outcome_feature": "outcome",
            "raw_logit_feature": "feature_raw_logit",
            "raw_logit_fm": "fm_raw_logit",
        }
    ).drop(columns=["outcome_fm", "inner_fold_idx_fm"])
    inner_export = inner_export.rename(columns={"inner_fold_idx_feature": "inner_fold_idx"})
    inner_export["fusion_probability"] = estimator.predict_proba(design)[:, 1]
    return FusionResult(
        inner_oof=inner_export,
        coefficients={
            "intercept": float(estimator.intercept_[0]),
            "feature": float(estimator.coef_[0, 0]),
            "fm": float(estimator.coef_[0, 1]),
        },
        outer_predictions=outer,
        input_representation="raw_logit",
    )


__all__ = ["FusionResult", "fit_cross_fitted_stacker"]
