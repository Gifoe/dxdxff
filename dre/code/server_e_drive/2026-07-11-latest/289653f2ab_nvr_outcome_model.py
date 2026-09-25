from __future__ import annotations

import torch
from torch import nn

from .monotone_outcome_head import MonotoneOutcomeHead
from .profiles import get_nvr_profile
from .residual_set_encoder import ResidualSetEncoder
from .nvr_features import scalar_features_for_profile


class NormalizedLinearHead(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__(); self.register_buffer("mean", torch.zeros(dimension)); self.register_buffer("std", torch.ones(dimension)); self.register_buffer("fitted", torch.tensor(False)); self.linear = nn.Linear(dimension, 1)
    def fit_normalizer(self, values: torch.Tensor) -> None:
        with torch.no_grad(): self.mean.copy_(values.mean(0)); self.std.copy_(values.std(0, unbiased=False).clamp_min(1e-6)); self.fitted.fill_(True)
    def forward(self, values: torch.Tensor) -> torch.Tensor: return self.linear((values-self.mean)/self.std).squeeze(-1)


class NVROutcomeModel(nn.Module):
    """Outcome model used only by the R0-R6 NVR mainline."""
    def __init__(self, patient_embedding_dim: int, profile: str) -> None:
        super().__init__(); self.profile = get_nvr_profile(profile); self.feature_names = scalar_features_for_profile(profile)
        self.set_encoder = ResidualSetEncoder(patient_embedding_dim) if self.profile.set_residual else None
        self.head = MonotoneOutcomeHead(self.feature_names, self.set_encoder.output_dim if self.set_encoder else 0) if self.profile.monotone_head else NormalizedLinearHead(len(self.feature_names))

    def fit_normalizer(self, scalar_values: torch.Tensor) -> None: self.head.fit_normalizer(scalar_values)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        values = batch["scalar_values"]
        set_output = None
        if self.set_encoder is not None:
            set_output = self.set_encoder(batch["patient_channel_embedding"], batch["channel_scalars"], batch["target"], batch["channel_mask"], batch["reliable_abnormality"])
        if isinstance(self.head, MonotoneOutcomeHead): output = self.head(values, None if set_output is None else set_output["set_embedding"])
        else:
            logit = self.head(values); probability = torch.sigmoid(logit)
            output = {"outcome_logit_success": logit, "outcome_probability_success": probability, "outcome_probability_failure": 1-probability}
        if set_output is not None: output.update(set_output)
        return output


__all__ = ["NVROutcomeModel", "NormalizedLinearHead"]
