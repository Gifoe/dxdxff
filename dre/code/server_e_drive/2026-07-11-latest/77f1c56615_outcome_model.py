from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .preprocessing import NGBRPreprocessor


RIDGE_C = .3


@dataclass
class FittedNGBR:
    preprocessor: NGBRPreprocessor
    model: LogisticRegression
    input_features: list[str]
    hfo_enabled: bool

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.preprocessor.transform(frame))[:, 1]


def fit_outcome_model(frame: pd.DataFrame, y: Sequence[int], features: Sequence[str], *, outer_fold: int, seed: int, hfo_enabled: bool) -> FittedNGBR:
    preprocessor = NGBRPreprocessor.fit(frame, features, outer_fold)
    model = LogisticRegression(penalty="l2", solver="liblinear", C=RIDGE_C, class_weight=None, max_iter=5000, random_state=int(seed))
    model.fit(preprocessor.transform(frame), np.asarray(y, dtype=int))
    return FittedNGBR(preprocessor, model, list(features), bool(hfo_enabled))


def coefficient_rows(fitted: FittedNGBR, outer_fold: int) -> list[dict]:
    return [{"outer_fold": outer_fold, "feature": name, "coefficient": float(value), "standardized_coefficient": float(value), "absolute_coefficient": abs(float(value)), "intercept": float(fitted.model.intercept_[0]), "hfo_enabled_in_fold": fitted.hfo_enabled} for name, value in zip(fitted.preprocessor.retained_features, fitted.model.coef_[0])]


__all__ = ["FittedNGBR", "RIDGE_C", "coefficient_rows", "fit_outcome_model"]
