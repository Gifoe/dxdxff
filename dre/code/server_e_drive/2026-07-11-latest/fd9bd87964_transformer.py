from __future__ import annotations

import torch
from torch import nn


class MaskedTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, int(embed_dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout), nn.Linear(int(embed_dim * mlp_ratio), embed_dim), nn.Dropout(dropout))

    def forward(self, values: torch.Tensor, padding_mask: torch.Tensor, *, return_attention: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        attended, weights = self.attention(values, values, values, key_padding_mask=padding_mask, need_weights=return_attention, average_attn_weights=False)
        values = self.norm1(values + attended)
        values = self.norm2(values + self.mlp(values))
        return values.masked_fill(padding_mask.unsqueeze(-1), 0.0), weights if return_attention else None
