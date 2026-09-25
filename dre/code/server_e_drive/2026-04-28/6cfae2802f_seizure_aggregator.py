from __future__ import annotations

import torch
from torch import nn


class CrossSeizureMILAggregator(nn.Module):
    """
    Attention MIL over seizures for each patient channel.
    """

    def __init__(self, model_dim: int = 96, dropout: float = 0.25, pooling: str = "attention") -> None:
        super().__init__()
        self.pooling = str(pooling).lower()
        self.score_head = nn.Sequential(
            nn.Linear(int(model_dim), int(model_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(model_dim), 1),
        )

    def forward(
        self,
        seizure_channel_embedding: torch.Tensor,
        seizure_mask: torch.Tensor,
        seizure_channel_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, s, c, d = seizure_channel_embedding.shape
        valid = seizure_mask[:, :, None] & seizure_channel_mask
        scores = self.score_head(seizure_channel_embedding).squeeze(-1)
        if self.pooling in {"mean", "avg", "average"}:
            scores = torch.zeros_like(scores)
        scores = scores.masked_fill(~valid, -1e9)

        all_invalid = ~valid.any(dim=1)
        if torch.any(all_invalid):
            scores = scores.clone()
            scores = scores.permute(0, 2, 1)
            scores[all_invalid] = 0.0
            scores = scores.permute(0, 2, 1)

        attention_s = torch.softmax(scores, dim=1) * valid.float()
        attention_s = attention_s / attention_s.sum(dim=1, keepdim=True).clamp_min(1e-6)
        patient_channel_embedding = torch.sum(seizure_channel_embedding * attention_s.unsqueeze(-1), dim=1)
        patient_channel_embedding = patient_channel_embedding * valid.any(dim=1).float().unsqueeze(-1)
        seizure_attention = attention_s.permute(0, 2, 1).contiguous()
        return patient_channel_embedding, seizure_attention


__all__ = ["CrossSeizureMILAggregator"]
