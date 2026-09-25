from __future__ import annotations

import torch
from torch import nn


class _ResidualTCNBlock(nn.Module):
    def __init__(self, dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + y)


class ChannelTemporalEncoder(nn.Module):
    """
    Channel-wise temporal encoder over peri-onset windows.

    Input:
        window_embeddings: [B, S, T, C, D]
        seizure_channel_mask: [B, S, C]
    Output:
        seizure_channel_embedding: [B, S, C, D]
        temporal_attention: [B, S, C, T]
    """

    def __init__(
        self,
        model_dim: int = 96,
        encoder_type: str = "tcn",
        num_layers: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.encoder_type = str(encoder_type).lower()
        if self.encoder_type in {"mean", "none", "identity"}:
            self.encoder = nn.Identity()
            self.out_proj = nn.Identity()
            self.attn = None
        elif self.encoder_type == "bigru":
            hidden = max(1, self.model_dim // 2)
            self.encoder = nn.GRU(
                input_size=self.model_dim,
                hidden_size=hidden,
                num_layers=max(1, int(num_layers)),
                batch_first=True,
                bidirectional=True,
                dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            )
            self.out_proj = nn.Linear(hidden * 2, self.model_dim)
            self.attn = nn.Sequential(
                nn.Linear(self.model_dim, self.model_dim),
                nn.Tanh(),
                nn.Linear(self.model_dim, 1),
            )
        else:
            self.encoder = nn.Sequential(
                *[_ResidualTCNBlock(self.model_dim, dilation=2**idx, dropout=float(dropout)) for idx in range(max(1, int(num_layers)))]
            )
            self.out_proj = nn.Identity()

            self.attn = nn.Sequential(
                nn.Linear(self.model_dim, self.model_dim),
                nn.Tanh(),
                nn.Linear(self.model_dim, 1),
            )

    def forward(
        self,
        window_embeddings: torch.Tensor,
        seizure_channel_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, s, t, c, d = window_embeddings.shape
        x = window_embeddings.permute(0, 1, 3, 2, 4).contiguous().reshape(b * s * c, t, d)

        if self.encoder_type in {"mean", "none", "identity"}:
            encoded = x
        elif self.encoder_type == "bigru":
            encoded, _ = self.encoder(x)
            encoded = self.out_proj(encoded)
        else:
            encoded = self.out_proj(self.encoder(x))

        if seizure_channel_mask is None:
            valid_seq = torch.ones((b * s * c,), dtype=torch.bool, device=window_embeddings.device)
        else:
            valid_seq = seizure_channel_mask.reshape(b * s * c)
        time_mask = valid_seq[:, None].expand(b * s * c, t)

        scores = torch.zeros((b * s * c, t), dtype=encoded.dtype, device=encoded.device) if self.attn is None else self.attn(encoded).squeeze(-1)
        scores = scores.masked_fill(~time_mask, -1e9)
        all_invalid = ~valid_seq
        if torch.any(all_invalid):
            scores = scores.clone()
            scores[all_invalid] = 0.0
        attention = torch.softmax(scores, dim=-1) * time_mask.float()
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.sum(encoded * attention.unsqueeze(-1), dim=1)
        pooled = pooled * valid_seq.float().unsqueeze(-1)

        seizure_channel_embedding = pooled.reshape(b, s, c, d)
        temporal_attention = attention.reshape(b, s, c, t)
        return seizure_channel_embedding, temporal_attention


__all__ = ["ChannelTemporalEncoder"]
