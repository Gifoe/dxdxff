from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


def patient_summary_features(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    values = batch["feature_x"]
    valid = batch["window_channel_mask"]
    batch_size, _, _, _, dim = values.shape
    flat = values.reshape(batch_size, -1, dim)
    mask = valid.reshape(batch_size, -1)
    output: list[torch.Tensor] = []
    for patient in range(batch_size):
        selected = flat[patient][mask[patient]]
        if selected.numel() == 0:
            selected = torch.zeros((1, dim), device=values.device, dtype=values.dtype)
        output.append(torch.cat([selected.mean(0), selected.std(0, unbiased=False), selected.max(0).values, selected.min(0).values, torch.quantile(selected, 0.25, dim=0), torch.quantile(selected, 0.75, dim=0)]))
    return torch.stack(output)


class SummaryMLModel:
    def __init__(self, random_seed: int = 42) -> None:
        self.estimator = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            class_weight="balanced",
            random_state=int(random_seed),
            max_iter=5000,
        )

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "SummaryMLModel":
        self.estimator.fit(np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.int64))
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(np.asarray(features, dtype=np.float64))[:, 1]

    def get_params(self) -> dict[str, Any]:
        return self.estimator.get_params()


__all__ = ["SummaryMLModel", "patient_summary_features"]
