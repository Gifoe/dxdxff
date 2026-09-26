"""Paper-aligned PRQ-Net, BCR-Net, and CDEL implementations."""

from __future__ import annotations

import math

import torch
from torch import nn


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights.unsqueeze(-1)).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1).unsqueeze(-1)


def _masked_mean_std(
    values: torch.Tensor, mask: torch.Tensor, dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = _masked_mean(values, mask, dim)
    centered = values - mean.unsqueeze(dim)
    variance = _masked_mean(centered.square(), mask, dim)
    return mean, variance.clamp_min(0).sqrt()


def patient_relative_z(values: torch.Tensor, valid_channels: torch.Tensor) -> torch.Tensor:
    """Feature-wise z-scoring across valid channels of one patient."""
    weights = valid_channels.to(values.dtype).unsqueeze(-1)
    mean = (values * weights).sum(0) / weights.sum(0).clamp_min(1)
    variance = ((values - mean).square() * weights).sum(0) / weights.sum(0).clamp_min(1)
    normalized = (values - mean) / variance.clamp_min(1e-6).sqrt()
    return normalized.masked_fill(~valid_channels.unsqueeze(-1), 0.0)


class ChannelContextBlock(nn.Module):
    def __init__(self, dimension: int = 32, heads: int = 2, dropout: float = 0.4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.normalization = nn.LayerNorm(dimension)

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(values)
        if valid.any():
            attended, _ = self.attention(
                values.unsqueeze(0),
                values.unsqueeze(0),
                values.unsqueeze(0),
                key_padding_mask=(~valid).unsqueeze(0),
                need_weights=False,
            )
            output = self.normalization(values + attended.squeeze(0))
            output = output.masked_fill(~valid.unsqueeze(-1), 0.0)
        return output


class EpiLENSEncoder(nn.Module):
    """Shared topology; PRQ-Net and BCR-Net instantiate independent copies."""

    def __init__(
        self, input_dimension: int = 36, dimension: int = 32, heads: int = 2, dropout: float = 0.4
    ):
        super().__init__()
        self.window_mlp = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, dimension),
        )
        self.window_context = ChannelContextBlock(dimension, heads, dropout)
        self.patient_projection = nn.Sequential(
            nn.LayerNorm(2 * dimension),
            nn.Linear(2 * dimension, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.patient_context = ChannelContextBlock(dimension, heads, dropout)

    def forward(
        self,
        features: torch.Tensor,
        valid: torch.Tensor,
        use_patient_relative: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode one patient with [seizure, window, channel, feature] input."""
        seizures, windows, channels, _ = features.shape
        encoded = self.window_mlp(features)
        contextual = torch.zeros_like(encoded)
        for seizure in range(seizures):
            for window in range(windows):
                contextual[seizure, window] = self.window_context(
                    encoded[seizure, window], valid[seizure, window]
                )
        seizure_valid = valid.any(dim=1)
        seizure_channel = _masked_mean(contextual, valid, dim=1)
        mean, std = _masked_mean_std(seizure_channel, seizure_valid, dim=0)
        channel_valid = seizure_valid.any(dim=0)
        patient = self.patient_projection(torch.cat((mean, std), dim=-1))
        if use_patient_relative:
            patient = patient_relative_z(patient, channel_valid)
        patient = self.patient_context(patient, channel_valid)
        return patient, seizure_channel, seizure_valid


def _robust_patient_z(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    selected = values[valid]
    median = selected.median()
    absolute = (selected - median).abs()
    mad = absolute.median()
    scale = 1.4826 * mad
    if float(scale.detach()) < 1e-6:
        scale = selected.std(unbiased=False).clamp_min(1e-6)
    result = ((values - median) / scale).clamp(-4.0, 4.0)
    return result.masked_fill(~valid, 0.0)


def _masked_quantile(
    values: torch.Tensor, valid: torch.Tensor, quantile: float
) -> torch.Tensor:
    channels = values.shape[1]
    output = values.new_zeros(channels)
    for channel in range(channels):
        selected = values[valid[:, channel], channel]
        if len(selected):
            output[channel] = torch.quantile(selected, quantile)
    return output


class PRQNet(nn.Module):
    """Patient-Relative Quantile Classification branch."""

    def __init__(
        self,
        input_dimension: int = 36,
        dimension: int = 32,
        heads: int = 2,
        dropout: float = 0.4,
        quantile: float = 0.10,
        use_temporal: bool = True,
        use_patient_relative: bool = True,
        temporal_aggregation: str = "quantile",
    ):
        super().__init__()
        self.encoder = EpiLENSEncoder(input_dimension, dimension, heads, dropout)
        self.base_head = nn.Linear(dimension, 1)
        self.seizure_head = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 1))
        self.residual_gate = nn.Parameter(torch.tensor(0.0))
        self.quantile = float(quantile)
        self.use_temporal = bool(use_temporal)
        self.use_patient_relative = bool(use_patient_relative)
        if temporal_aggregation not in {"mean", "quantile"}:
            raise ValueError("temporal_aggregation must be mean or quantile")
        self.temporal_aggregation = temporal_aggregation

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        patient, seizure_channel, seizure_valid = self.encoder(
            features, valid, self.use_patient_relative
        )
        channel_valid = seizure_valid.any(dim=0)
        base_logit = self.base_head(patient).squeeze(-1)
        if self.use_temporal:
            seizure_probability = torch.sigmoid(
                self.seizure_head(seizure_channel).squeeze(-1)
            )
            if self.temporal_aggregation == "mean":
                lower_tail = (
                    (seizure_probability * seizure_valid).sum(dim=0)
                    / seizure_valid.sum(dim=0).clamp_min(1)
                )
            else:
                lower_tail = _masked_quantile(
                    seizure_probability, seizure_valid, self.quantile
                )
            lower_tail = lower_tail.clamp(1e-5, 1 - 1e-5)
            tail_logit = torch.logit(lower_tail)
            relative = (
                _robust_patient_z(tail_logit, channel_valid)
                if self.use_patient_relative
                else tail_logit
            )
            alpha = 0.2 * torch.sigmoid(self.residual_gate)
            final_logit = base_logit + alpha * relative
        else:
            lower_tail = torch.sigmoid(base_logit)
            alpha = base_logit.new_tensor(0.0)
            final_logit = base_logit
        return {
            "probability_nez": torch.sigmoid(final_logit),
            "logit_nez": final_logit,
            "base_logit_nez": base_logit,
            "lower_tail_nez": lower_tail,
            "residual_weight": alpha,
            "channel_valid": channel_valid,
        }


class BCRNet(nn.Module):
    """Boundary- and Coverage-aware Ranking branch."""

    def __init__(
        self,
        input_dimension: int = 36,
        dimension: int = 32,
        heads: int = 2,
        dropout: float = 0.4,
        quantile_control: bool = False,
    ):
        super().__init__()
        self.encoder = EpiLENSEncoder(input_dimension, dimension, heads, dropout)
        self.ez_head = nn.Linear(dimension, 1)
        self.seizure_ez_head = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, 1)
        )
        self.quantile_control = quantile_control

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        patient, seizure_channel, seizure_valid = self.encoder(features, valid)
        channel_valid = seizure_valid.any(dim=0)
        logit_ez = self.ez_head(patient).squeeze(-1)
        if self.quantile_control:
            seizure_probability_nez = 1.0 - torch.sigmoid(
                self.seizure_ez_head(seizure_channel).squeeze(-1)
            )
            q10_nez = _masked_quantile(
                seizure_probability_nez, seizure_valid, 0.10
            ).clamp(1e-5, 1 - 1e-5)
            logit_ez = -torch.logit(q10_nez)
        return {
            "logit_ez": logit_ez,
            "probability_ez": torch.sigmoid(logit_ez),
            "probability_nez": 1.0 - torch.sigmoid(logit_ez),
            "channel_valid": channel_valid,
        }


def cdel_probability(
    prq_probability_nez: torch.Tensor,
    bcr_logit_ez: torch.Tensor,
    bcr_weight: float = 0.20,
) -> torch.Tensor:
    """Locked parameter-free fusion used for every primary CDEL result."""
    if not math.isclose(bcr_weight, 0.20, abs_tol=1e-12):
        raise ValueError("The primary CDEL rule requires BCR weight 0.20")
    bcr_probability_nez = 1.0 - torch.sigmoid(bcr_logit_ez)
    return (1.0 - bcr_weight) * prq_probability_nez + bcr_weight * bcr_probability_nez
