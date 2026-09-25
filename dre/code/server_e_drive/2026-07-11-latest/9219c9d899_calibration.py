from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    """Platt calibrator that accepts only supplied inner-OOF scores."""

    def __init__(self) -> None:
        self.model: LogisticRegression | None = None
        self.constant: float | None = None
        self.fit_rows = 0
        self.fit_subjects: set[str] = set()

    def fit(self, scores, labels, *, subject_ids=None) -> "ProbabilityCalibrator":
        x = np.asarray(scores, dtype=float).reshape(-1, 1)
        y = np.asarray(labels, dtype=int).reshape(-1)
        if len(x) != len(y) or len(x) == 0:
            raise ValueError("calibrator needs aligned nonempty inner-OOF scores and labels")
        self.fit_rows = len(y)
        self.fit_subjects = set(map(str, subject_ids)) if subject_ids is not None else set()
        if len(np.unique(y)) < 2:
            self.constant = float(y.mean())
        else:
            self.model = LogisticRegression(random_state=0).fit(x, y)
        return self

    def predict(self, scores) -> np.ndarray:
        x = np.asarray(scores, dtype=float).reshape(-1, 1)
        if self.model is not None:
            return self.model.predict_proba(x)[:, 1]
        return np.full(len(x), 0.5 if self.constant is None else self.constant, dtype=float)
