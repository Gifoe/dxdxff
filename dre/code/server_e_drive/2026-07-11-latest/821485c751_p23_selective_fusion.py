"""Single bounded score-level fusion used by P23 profiles P3-P6."""

from __future__ import annotations

import math
import torch
from torch import nn


def patient_relative_logit(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    rows = []
    for index in range(logits.shape[0]):
        valid = mask[index].bool()
        values = logits[index, valid]
        if values.numel() == 0:
            rows.append(torch.zeros_like(logits[index]))
            continue
        median = values.median()
        q1, q3 = torch.quantile(values, torch.tensor([0.25, 0.75], device=values.device, dtype=values.dtype))
        row = ((logits[index] - median) / (q3 - q1).abs().clamp_min(1e-3)).clamp(-5.0, 5.0)
        rows.append(row.masked_fill(~valid, 0.0))
    return torch.stack(rows)


class P23SelectiveFusion(nn.Module):
    def __init__(self, embedding_dim: int, *, use_causal: bool = False, hidden_dim: int = 16, max_total_correction: float = 0.20, priors: tuple[float, ...] | None = None) -> None:
        super().__init__()
        self.use_causal = bool(use_causal)
        self.max_total_correction = float(max_total_correction)
        scalar_dim = 7 if self.use_causal else 5
        n_weights = 4 if self.use_causal else 3
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.gate = nn.Sequential(nn.Linear(embedding_dim + scalar_dim, hidden_dim), nn.GELU(), nn.Dropout(0.10), nn.Linear(hidden_dim, n_weights))
        default_priors = (0.88, 0.05, 0.05, 0.02) if self.use_causal else (0.90, 0.05, 0.05)
        priors = default_priors if priors is None else tuple(float(value) for value in priors)
        if len(priors) != n_weights or any(value <= 0.0 for value in priors) or abs(sum(priors) - 1.0) > 1e-6:
            raise ValueError("P23 fusion priors must be positive, match active heads, and sum to one.")
        nn.init.zeros_(self.gate[-1].weight)
        self.gate[-1].bias.data.copy_(torch.tensor([math.log(value) for value in priors], dtype=self.gate[-1].bias.dtype))

    def forward(self, embedding: torch.Tensor, direct_nez_logit: torch.Tensor, u_anchor: torch.Tensor, u_seizure: torch.Tensor, seizure_agreement: torch.Tensor, temporal_delta_norm: torch.Tensor, channel_mask: torch.Tensor, *, u_causal: torch.Tensor | None = None, q_cp: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        mask = channel_mask.bool()
        r_direct = patient_relative_logit(direct_nez_logit, mask)
        scalars = [r_direct.detach().clamp(-3.0, 3.0), u_anchor, u_seizure, seizure_agreement, temporal_delta_norm]
        if self.use_causal:
            if u_causal is None or q_cp is None:
                raise KeyError("P23 causal fusion requires u_causal and q_cp")
            scalars.extend((u_causal, q_cp))
        logits = self.gate(torch.cat((self.embedding_norm(embedding), *(value.unsqueeze(-1) for value in scalars)), dim=-1))
        weights = torch.softmax(logits, dim=-1).masked_fill(~mask.unsqueeze(-1), 0.0)
        w_noop, w_anchor, w_seizure = (weights[..., index] for index in range(3))
        w_causal = weights[..., 3] if self.use_causal else torch.zeros_like(w_noop)
        causal = u_causal if u_causal is not None else torch.zeros_like(u_anchor)
        correction = w_anchor * u_anchor + w_seizure * u_seizure + w_causal * causal
        delta = (self.max_total_correction * correction).masked_fill(~mask, 0.0)
        final = (direct_nez_logit + delta).masked_fill(~mask, -1e9)
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1).masked_fill(~mask, 0.0)
        return {
            "r_direct": r_direct, "w_noop": w_noop, "w_anchor": w_anchor, "w_seizure": w_seizure,
            "w_causal": w_causal, "delta": delta, "final_nez_logit": final,
            "final_score_nez": torch.sigmoid(final).masked_fill(~mask, 0.0),
            "final_score_ez": (1.0 - torch.sigmoid(final)).masked_fill(~mask, 0.0),
            "fusion_entropy": entropy, "correction_saturation": delta.abs() >= 0.95 * self.max_total_correction,
        }
