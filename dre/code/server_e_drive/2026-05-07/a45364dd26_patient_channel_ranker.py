from __future__ import annotations

import torch
from torch import nn


class PatientChannelRanker(nn.Module):
    """
    Patient-level channel ranker over dynamic channel embeddings.
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
        self.topology_proj = nn.Sequential(
            nn.LazyLinear(dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dim, dim),
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
        topology_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h_input = patient_channel_embedding
        if topology_features is not None:
            h_input = h_input + self.topology_proj(topology_features.to(device=h_input.device, dtype=h_input.dtype))
        h = _patient_relative_zscore(h_input, channel_mask)
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
        count_logit = self.count_head(torch.cat([mean_pool, max_pool], dim=-1)).squeeze(-1)
        predicted_count_ratio = torch.sigmoid(count_logit)
        valid_channel_count = channel_mask.float().sum(dim=1).clamp_min(1.0)
        predicted_count = predicted_count_ratio * valid_channel_count
        score_mass = torch.sigmoid(logits.masked_fill(~channel_mask, -20.0)).sum(dim=1)

        return {
            "logits": logits,
            "scores": torch.sigmoid(logits),
            "count_logit": count_logit,
            "predicted_count_ratio": predicted_count_ratio,
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
