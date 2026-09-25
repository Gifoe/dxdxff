"""Minimal cross-seizure Q10 evidence for V3-QBC."""

from __future__ import annotations

import torch
from torch import nn


def robust_patient_zscore(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    mad_eps: float = 1e-5,
    clamp_value: float = 4.0,
) -> torch.Tensor:
    if values.shape != valid_mask.shape or values.ndim != 2:
        raise ValueError("values and valid_mask must have identical [B,C] shapes")
    output = torch.zeros_like(values)
    for patient_idx in range(values.shape[0]):
        mask = valid_mask[patient_idx].bool()
        patient_values = values[patient_idx, mask]
        if patient_values.numel() == 0:
            continue
        median = patient_values.median()
        mad = (patient_values - median).abs().median()
        scale = 1.4826 * mad
        if float(scale.detach()) < mad_eps:
            scale = patient_values.std(unbiased=False)
        if float(scale.detach()) < mad_eps:
            standardized = torch.zeros_like(patient_values)
        else:
            standardized = (patient_values - median) / scale
        output[patient_idx, mask] = standardized.clamp(-clamp_value, clamp_value)
    return output


class V3SimpleQ10Evidence(nn.Module):
    def __init__(
        self,
        model_dim: int,
        max_residual: float = 0.20,
        gate_init: float = -3.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.max_residual = float(max_residual)
        self.eps = float(eps)
        self.scorer = nn.Sequential(nn.LayerNorm(int(model_dim)), nn.Linear(int(model_dim), 1))
        nn.init.normal_(self.scorer[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.scorer[-1].bias)
        self.q10_gate_raw = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        seizure_channel_embedding: torch.Tensor,
        seizure_mask: torch.Tensor,
        seizure_channel_mask: torch.Tensor,
        channel_mask: torch.Tensor,
        base_nez_logit: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if seizure_channel_embedding.ndim != 4:
            raise ValueError("seizure_channel_embedding must be [B,S,C,D]")
        valid = (
            seizure_channel_mask.bool()
            & seizure_mask.bool().unsqueeze(-1)
            & channel_mask.bool().unsqueeze(1)
        )
        seizure_logit = self.scorer(seizure_channel_embedding).squeeze(-1)
        seizure_probability = torch.sigmoid(seizure_logit)
        bsz, _, channels = seizure_probability.shape
        q10 = torch.full(
            (bsz, channels), 0.5, dtype=seizure_probability.dtype, device=seizure_probability.device
        )
        probability_mean = torch.full_like(q10, 0.5)
        probability_std = torch.zeros_like(q10)
        valid_count = valid.sum(dim=1)
        for patient_idx in range(bsz):
            for channel_idx in range(channels):
                values = seizure_probability[patient_idx, :, channel_idx][valid[patient_idx, :, channel_idx]]
                if values.numel() == 0:
                    continue
                q10[patient_idx, channel_idx] = torch.quantile(values, 0.10)
                probability_mean[patient_idx, channel_idx] = values.mean()
                probability_std[patient_idx, channel_idx] = values.std(unbiased=False)
        q10_valid = (valid_count > 0) & channel_mask.bool()
        q10_logit = torch.logit(q10.clamp(1e-5, 1.0 - 1e-5))
        q10_z = robust_patient_zscore(q10_logit, q10_valid)
        gate = self.max_residual * torch.sigmoid(self.q10_gate_raw)
        residual = gate * q10_z
        residual = residual.masked_fill(~q10_valid, 0.0)
        final_nez_logit = base_nez_logit + residual
        final_nez_logit = final_nez_logit.masked_fill(~channel_mask.bool(), 0.0)
        score_nez = torch.sigmoid(final_nez_logit).masked_fill(~channel_mask.bool(), 0.0)
        score_ez = (1.0 - score_nez).masked_fill(~channel_mask.bool(), 0.0)
        return {
            "base_nez_logit": base_nez_logit,
            "final_nez_logit": final_nez_logit,
            "q10_nez_probability": q10,
            "q10_nez_logit": q10_logit,
            "q10_nez_z": q10_z,
            "q10_valid": q10_valid,
            "valid_seizure_count": valid_count,
            "seizure_nez_probability": seizure_probability,
            "seizure_nez_probability_mean": probability_mean,
            "seizure_nez_probability_std": probability_std,
            "q10_gate": gate,
            "q10_residual": residual,
            "score_nez": score_nez,
            "score_ez": score_ez,
        }


__all__ = ["V3SimpleQ10Evidence", "robust_patient_zscore"]
