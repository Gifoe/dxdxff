from __future__ import annotations

import torch
from torch import nn

from .masks import masked_mean, masked_std


class ChannelOutcomePAM(nn.Module):
    def __init__(self, embedding_dim: int, *, burden_temperature: float = 0.5, eps: float = 1e-6) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.burden_temperature = float(burden_temperature)
        self.eps = float(eps)
        self.base_failure_evidence_head = nn.Sequential(nn.LayerNorm(self.embedding_dim), nn.Linear(self.embedding_dim, 1))
        self.beta_q_raw = nn.Parameter(torch.tensor(-3.0))
        self.tau_burden = nn.Parameter(torch.tensor(0.0))

    @property
    def beta_q(self) -> torch.Tensor:
        return 0.20 * torch.sigmoid(self.beta_q_raw)

    def forward(
        self,
        seizure_channel_embedding: torch.Tensor,
        simple_q10_nez_z: torch.Tensor,
        seizure_channel_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if seizure_channel_embedding.ndim != 4:
            raise ValueError("seizure_channel_embedding must be [B,S,C,D]")
        valid = seizure_channel_mask.bool()
        base_failure_evidence = self.base_failure_evidence_head(seizure_channel_embedding).squeeze(-1)
        q10_adjustment = -self.beta_q * simple_q10_nez_z.unsqueeze(1).expand_as(base_failure_evidence)
        base_failure_evidence = base_failure_evidence.masked_fill(~valid, 0.0)
        q10_adjustment = q10_adjustment.masked_fill(~valid, 0.0)
        failure_risk_logit = (base_failure_evidence + q10_adjustment).masked_fill(~valid, 0.0)
        membership = torch.sigmoid((failure_risk_logit - self.tau_burden) / self.burden_temperature).masked_fill(~valid, 0.0)
        count = valid.to(seizure_channel_embedding.dtype).sum(dim=-1).clamp_min(1.0)
        node_failure_burden = membership.sum(dim=-1) / count
        global_embedding = masked_mean(seizure_channel_embedding, valid, dim=2)
        channel_std_embedding = masked_std(seizure_channel_embedding, valid, dim=2)
        weighted = membership.unsqueeze(-1) * torch.nan_to_num(seizure_channel_embedding)
        risk_embedding = weighted.sum(dim=2) / membership.sum(dim=2).clamp_min(self.eps).unsqueeze(-1)
        risk_embedding = torch.where(valid.any(dim=2).unsqueeze(-1), risk_embedding, torch.zeros_like(risk_embedding))
        contrast = risk_embedding - global_embedding
        additive = (membership * failure_risk_logit).sum(dim=-1) / count
        return {
            "base_failure_evidence": base_failure_evidence,
            "q10_adjustment": q10_adjustment,
            "failure_risk_logit": failure_risk_logit,
            "risk_membership": membership,
            "node_failure_burden": node_failure_burden,
            "global_embedding": global_embedding,
            "channel_std_embedding": channel_std_embedding,
            "risk_embedding": risk_embedding,
            "risk_global_contrast": contrast,
            "additive_failure_evidence": additive,
            "beta_q": self.beta_q,
            "tau_burden": self.tau_burden,
        }


__all__ = ["ChannelOutcomePAM"]
