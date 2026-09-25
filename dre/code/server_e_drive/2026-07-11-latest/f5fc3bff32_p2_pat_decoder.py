"""Small linear patient-adaptive threshold decoder for frozen P2-Q10."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


RIDGE_ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)
SHRINKAGE_GRID = (0.25, 0.50, 0.75, 1.00)
MAX_PREDICTION_RESIDUAL_GRID = (0.25, 0.50, 0.75)


def probability_to_logit(probability: float) -> float:
    value = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    return float(np.log(value / (1.0 - value)))


def logit_to_probability(logit: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(logit))))


@dataclass
class P2PATDecoder:
    ridge_alpha: float
    shrinkage: float
    max_prediction_residual: float
    scaler: StandardScaler | None = None
    ridge: Ridge | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "P2PATDecoder":
        x = np.asarray(features, dtype=float)
        y = np.asarray(targets, dtype=float)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(y) < 2:
            raise ValueError("PAT decoder fit requires >=2 aligned patients")
        self.scaler = StandardScaler().fit(x)
        self.ridge = Ridge(alpha=float(self.ridge_alpha)).fit(self.scaler.transform(x), y)
        return self

    def predict_residual(self, features: np.ndarray) -> np.ndarray:
        if self.scaler is None or self.ridge is None:
            raise RuntimeError("PAT decoder is not fitted")
        raw = self.ridge.predict(self.scaler.transform(np.asarray(features, dtype=float)))
        return float(self.shrinkage) * np.clip(raw, -self.max_prediction_residual, self.max_prediction_residual)

    def predict_threshold_logit(self, features: np.ndarray, global_threshold_logit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        residual = self.predict_residual(features)
        return np.asarray(global_threshold_logit, dtype=float) + residual, residual


__all__ = [
    "P2PATDecoder", "RIDGE_ALPHA_GRID", "SHRINKAGE_GRID", "MAX_PREDICTION_RESIDUAL_GRID",
    "probability_to_logit", "logit_to_probability",
]
