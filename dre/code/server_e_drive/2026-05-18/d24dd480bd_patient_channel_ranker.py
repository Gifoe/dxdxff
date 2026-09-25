from __future__ import annotations

import torch
from torch import nn


class PatientChannelRanker(nn.Module):
    """
    Patient-level channel ranker over dynamic channel embeddings.

    V3-small interprets logits/scores as NEZ probability. EZ ranking therefore
    uses ``score_ez = 1 - scores``.
    """

    def __init__(
        self,
        model_dim: int = 96,
        num_heads: int = 4,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        dim = int(model_dim)
        self.local_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dim, 1),
        )
        self.channel_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(dim)
        self.context_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dim, 1),
        )
        self.count_head = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dim, 1),
        )

    def forward(
        self,
        patient_channel_embedding: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        h = _patient_relative_zscore(patient_channel_embedding, channel_mask)
        local_logits = self.local_head(h).squeeze(-1)

        key_padding_mask = ~channel_mask
        all_invalid = key_padding_mask.all(dim=1)
        if torch.any(all_invalid):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_invalid] = False
        context, _ = self.channel_attn(h, h, h, key_padding_mask=key_padding_mask)
        context = self.attn_norm(h + context)
        context_logits = self.context_head(context).squeeze(-1)
        logits = local_logits + context_logits
        logits = logits.masked_fill(~channel_mask, -1e9)

        mask_f = channel_mask.float().unsqueeze(-1)
        valid_count = mask_f.sum(dim=1).clamp_min(1.0)
        mean_pool = (context * mask_f).sum(dim=1) / valid_count
        max_pool = context.masked_fill(~channel_mask.unsqueeze(-1), -1e9).max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        predicted_count = torch.nn.functional.softplus(self.count_head(torch.cat([mean_pool, max_pool], dim=-1))).squeeze(-1)
        score_mass = torch.sigmoid(logits.masked_fill(~channel_mask, -20.0)).sum(dim=1)

        scores = torch.sigmoid(logits)
        return {
            "logits": logits,
            "scores": scores,
            "score_nez": scores,
            "score_ez": 1.0 - scores,
            "predicted_count": predicted_count,
            "score_mass": score_mass,
        }


def _patient_relative_zscore(x: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
    mask = channel_mask.float().unsqueeze(-1)
    count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (x * mask).sum(dim=1, keepdim=True) / count
    var = (((x - mean) * mask) ** 2).sum(dim=1, keepdim=True) / count
    z = (x - mean) / torch.sqrt(var + 1e-5)
    return z * mask


__all__ = ["PatientChannelRanker"]
