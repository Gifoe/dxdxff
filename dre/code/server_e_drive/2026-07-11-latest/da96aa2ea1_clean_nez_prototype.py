from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass
class PrototypeAudit:
    embedding_type: str
    n_patients: int
    n_nez_channels: int
    distance_mean: float
    distance_std: float
    distance_q05: float
    distance_q50: float
    distance_q95: float
    prototype_variance_min: float
    prototype_variance_max: float


class CleanNEZPrototype:
    """Outer-train-only, patient-balanced diagonal clean-NEZ prototype."""

    def __init__(self, embedding_type: str, variance_floor: float = 1e-4) -> None:
        self.embedding_type = str(embedding_type)
        self.variance_floor = float(variance_floor)
        self.fitted = False

    def fit(self, patient_embeddings: Sequence[torch.Tensor]) -> "CleanNEZPrototype":
        rows = [torch.as_tensor(value, dtype=torch.float32).reshape(-1, value.shape[-1]) for value in patient_embeddings if torch.as_tensor(value).numel()]
        if not rows:
            raise ValueError("CleanNEZPrototype requires at least one outer-train patient with a non-target channel")
        dimension = rows[0].shape[-1]
        if any(row.shape[-1] != dimension for row in rows):
            raise ValueError("Prototype embeddings have inconsistent dimensions")
        patient_means = torch.stack([row.mean(0) for row in rows])
        mu = patient_means.mean(0)
        # Each patient's average squared deviation receives equal weight,
        # regardless of its channel or seizure count.
        patient_second = torch.stack([(row - mu).square().mean(0) for row in rows])
        var = patient_second.mean(0).clamp_min(self.variance_floor)
        distances = torch.cat([self._distance_with(row, mu, var) for row in rows]).sort().values
        self.prototype_mu = mu
        self.prototype_var = var
        self.train_distances = distances
        self.train_distance_quantiles = torch.quantile(distances, torch.tensor([.05, .10, .25, .50, .75, .90, .95]))
        self.n_train_patients = len(rows)
        self.n_train_nez_channels = int(sum(row.shape[0] for row in rows))
        self.fitted = True
        return self

    @staticmethod
    def _distance_with(embedding: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        return ((embedding - mu) ** 2 / var).mean(dim=-1)

    def distance(self, embedding: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("Prototype must be fitted on outer-train before transform")
        return self._distance_with(torch.as_tensor(embedding, dtype=torch.float32), self.prototype_mu, self.prototype_var)

    def probability_nez(self, embedding: torch.Tensor) -> torch.Tensor:
        distance = self.distance(embedding)
        rank = torch.searchsorted(self.train_distances.to(distance.device), distance.contiguous(), right=True).to(distance.dtype)
        return (1.0 - rank / max(self.train_distances.numel(), 1)).clamp(.01, .99)

    def audit(self) -> PrototypeAudit:
        if not self.fitted:
            raise RuntimeError("Prototype is not fitted")
        d = self.train_distances
        return PrototypeAudit(
            self.embedding_type, self.n_train_patients, self.n_train_nez_channels,
            float(d.mean()), float(d.std(unbiased=False)), float(torch.quantile(d, .05)),
            float(torch.quantile(d, .50)), float(torch.quantile(d, .95)),
            float(self.prototype_var.min()), float(self.prototype_var.max()),
        )

    def state_dict(self) -> dict:
        if not self.fitted:
            raise RuntimeError("Prototype is not fitted")
        return {"embedding_type": self.embedding_type, "variance_floor": self.variance_floor, "prototype_mu": self.prototype_mu,
                "prototype_var": self.prototype_var, "train_distances": self.train_distances,
                "train_distance_quantiles": self.train_distance_quantiles, "n_train_patients": self.n_train_patients,
                "n_train_nez_channels": self.n_train_nez_channels}

    @classmethod
    def from_state_dict(cls, state: dict) -> "CleanNEZPrototype":
        value = cls(state["embedding_type"], state["variance_floor"])
        for key, item in state.items():
            if key not in {"embedding_type", "variance_floor"}: setattr(value, key, item)
        value.fitted = True
        return value


__all__ = ["CleanNEZPrototype", "PrototypeAudit"]
