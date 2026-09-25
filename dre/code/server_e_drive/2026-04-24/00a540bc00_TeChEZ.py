from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


def _get_arg(args: Any, name: str, default: Any) -> Any:
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


class ChannelAttentionBlock(nn.Module):
    def __init__(self, d_model: int, *, num_heads: int = 4, dropout: float = 0.25) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.ff_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, *, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.attn_norm(x + attn_out)
        x = self.ff_norm(x + self.ff(x))
        return x


class PatientChannelRanker(nn.Module):
    def __init__(self, args: Any) -> None:
        super().__init__()
        input_dim = int(_get_arg(args, "input_dim", _get_arg(args, "feature_dim", 128)))
        d_model = int(_get_arg(args, "d_model", 96))
        dropout = float(_get_arg(args, "dropout", 0.25))
        num_heads = int(_get_arg(args, "num_heads", 4))
        channel_layers = int(_get_arg(args, "channel_layers", 2))

        self.count_blend_weight = float(_get_arg(args, "count_blend_weight", 0.35))
        self.local_logit_weight = float(_get_arg(args, "local_logit_weight", 0.35))

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.local_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.blocks = nn.ModuleList(
            [ChannelAttentionBlock(d_model, num_heads=num_heads, dropout=dropout) for _ in range(channel_layers)]
        )
        self.context_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.count_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    @staticmethod
    def _relative_zscore(x: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        mask_f = channel_mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (x * mask_f).sum(dim=1, keepdim=True) / denom
        centered = x - mean
        var = (centered.pow(2) * mask_f).sum(dim=1, keepdim=True) / denom
        rel_x = centered / torch.sqrt(var + 1e-6)
        return rel_x * mask_f

    def forward(self, x: torch.Tensor, channel_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError("Expected x with shape [batch, channels, feature_dim].")
        if channel_mask.dim() != 2:
            raise ValueError("Expected channel_mask with shape [batch, channels].")

        rel_x = self._relative_zscore(x, channel_mask)
        tokens = self.input_proj(torch.cat([x, rel_x], dim=-1))
        tokens = tokens * channel_mask.float().unsqueeze(-1)

        local_logits = self.local_head(tokens).squeeze(-1)
        key_padding_mask = ~channel_mask
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=key_padding_mask)

        context_logits = self.context_head(tokens).squeeze(-1)
        logits = context_logits + self.local_logit_weight * local_logits
        logits = logits.masked_fill(~channel_mask, -12.0)
        scores = torch.sigmoid(logits) * channel_mask.float()

        mask_f = channel_mask.float()
        valid_counts = mask_f.sum(dim=1).clamp_min(1.0)
        pooled = (tokens * mask_f.unsqueeze(-1)).sum(dim=1) / valid_counts.unsqueeze(-1)
        count_fraction = torch.sigmoid(self.count_head(pooled).squeeze(-1))
        count_from_head = count_fraction * valid_counts
        score_mass = scores.sum(dim=1)
        predicted_count = (
            self.count_blend_weight * count_from_head
            + (1.0 - self.count_blend_weight) * score_mass
        )

        return {
            "logits": logits,
            "scores": scores,
            "count_fraction": count_fraction,
            "count_from_head": count_from_head,
            "score_mass": score_mass,
            "predicted_count": predicted_count,
        }


Model = PatientChannelRanker
TeChEZModel = PatientChannelRanker

__all__ = ["Model", "PatientChannelRanker", "TeChEZModel"]
