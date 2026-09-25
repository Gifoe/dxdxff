"""Bounded simplex evidence fusion for P2.1."""

from __future__ import annotations

import math

import torch
from torch import nn

from .p21_v3_anchor import robust_patient_standardize_v3


def compute_causal_quality_gate(
    cp_valid_window_fraction: torch.Tensor,
    cp_valid_seizure_count: torch.Tensor,
    cp_mean_var_stability: torch.Tensor,
    cp_feature_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    coverage = cp_valid_window_fraction.clamp(0.0, 1.0)
    seizure = (cp_valid_seizure_count / 3.0).clamp(0.0, 1.0)
    stability = (cp_mean_var_stability / 0.50).clamp(0.0, 1.0)
    quality = cp_feature_valid.to(coverage.dtype) * coverage * seizure * stability
    return {"q_cp": quality, "q_coverage": coverage, "q_seizure": seizure, "q_stability": stability}


class SelectiveResidualFusion(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 16,
        max_total_correction: float = 0.20,
        priors: tuple[float, float, float, float] = (0.90, 0.04, 0.04, 0.02),
    ) -> None:
        super().__init__()
        if not 0.0 < float(max_total_correction) <= 0.5:
            raise ValueError("max_total_correction must be in (0,0.5]")
        if any(value <= 0 for value in priors) or not math.isclose(sum(priors), 1.0, abs_tol=1e-6):
            raise ValueError("Simplex gate priors must be positive and sum to one")
        self.max_total_correction = float(max_total_correction)
        self.embedding_norm = nn.LayerNorm(int(embedding_dim))
        self.gate = nn.Sequential(
            nn.Linear(int(embedding_dim) + 8, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(int(hidden_dim), 4),
        )
        nn.init.zeros_(self.gate[-1].weight)
        with torch.no_grad():
            self.gate[-1].bias.copy_(torch.log(torch.tensor(priors, dtype=self.gate[-1].bias.dtype)))

    def forward(
        self,
        contextual_channel_embedding: torch.Tensor,
        base_anchor_standardized: torch.Tensor,
        direct_nez_logit: torch.Tensor,
        u_anchor: torch.Tensor,
        u_seizure: torch.Tensor,
        u_causal: torch.Tensor,
        q_cp: torch.Tensor,
        seizure_agreement: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        valid = channel_mask.bool()
        direct_relative = robust_patient_standardize_v3(direct_nez_logit, valid, valid)["v3_anchor_standardized"]
        detached_direct = direct_relative.detach()
        scalars = torch.stack((
            base_anchor_standardized.clamp(-3.0, 3.0), detached_direct.clamp(-3.0, 3.0),
            u_anchor, u_seizure, u_causal, q_cp, seizure_agreement,
            (base_anchor_standardized - detached_direct).abs(),
        ), dim=-1)
        gate_input = torch.cat((self.embedding_norm(contextual_channel_embedding), scalars), dim=-1)
        weights = torch.softmax(self.gate(gate_input), dim=-1)
        weights = weights * valid.to(weights.dtype).unsqueeze(-1)
        correction_raw = weights[..., 1] * u_anchor + weights[..., 2] * u_seizure + weights[..., 3] * u_causal
        delta = (self.max_total_correction * correction_raw).masked_fill(~valid, 0.0)
        final = (base_anchor_standardized + delta).masked_fill(~valid, -1e9)
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1).masked_fill(~valid, 0.0)
        saturation = (delta.abs() >= 0.95 * self.max_total_correction) & valid
        return {
            "direct_standardized": direct_relative,
            "final_nez_logit": final,
            "delta": delta,
            "correction_raw": correction_raw.masked_fill(~valid, 0.0),
            "w_noop": weights[..., 0],
            "w_anchor": weights[..., 1],
            "w_seizure": weights[..., 2],
            "w_causal": weights[..., 3],
            "fusion_entropy": entropy,
            "correction_saturation": saturation,
        }


__all__ = ["SelectiveResidualFusion", "compute_causal_quality_gate"]
