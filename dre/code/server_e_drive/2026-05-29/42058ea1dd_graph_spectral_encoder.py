from __future__ import annotations

import torch
from torch import nn


class WindowGraphSpectralEncoder(nn.Module):
    """
    Window-level feature encoder with optional adjacency-weighted message passing.

    Input:
        features: [B, S, T, C, F]
        adjacency: [B, S, T, C, C] or None
    Output:
        [B, S, T, C, D]
    """

    def __init__(
        self,
        model_dim: int = 96,
        num_heads: int = 4,
        dropout: float = 0.25,
        use_message_passing: bool = True,
        use_channel_attention: bool = True,
        use_graph_edge_dropout: bool = False,
        graph_edge_dropout: float = 0.0,
        use_learnable_edge_gate: bool = False,
        edge_gate_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.use_message_passing = bool(use_message_passing)
        self.use_channel_attention = bool(use_channel_attention)
        self.use_graph_edge_dropout = bool(use_graph_edge_dropout)
        self.graph_edge_dropout = float(graph_edge_dropout)
        self.use_learnable_edge_gate = bool(use_learnable_edge_gate)
        self.edge_gate_temperature = float(edge_gate_temperature)
        self._last_graph_sparsity_loss: torch.Tensor | None = None

        self.feature_mlp = nn.Sequential(
            nn.LazyLinear(self.model_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.model_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
        )
        self.message_proj = nn.Linear(self.model_dim, self.model_dim)
        self.message_update = nn.Sequential(
            nn.Linear(self.model_dim * 2, self.model_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.model_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
        )
        self.channel_attn = nn.MultiheadAttention(
            embed_dim=self.model_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.model_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.edge_gate_logit = nn.Parameter(torch.tensor(2.0)) if self.use_learnable_edge_gate else None

    @property
    def graph_sparsity_loss(self) -> torch.Tensor | None:
        return self._last_graph_sparsity_loss

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        seizure_channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._last_graph_sparsity_loss = None
        h = self.feature_mlp(features)
        if self.use_message_passing and adjacency is not None:
            adjacency_used = adjacency
            if self.use_learnable_edge_gate and self.edge_gate_logit is not None:
                temperature = max(float(self.edge_gate_temperature), 1e-3)
                gate = torch.sigmoid(self.edge_gate_logit / temperature)
                adjacency_used = adjacency_used * gate
                self._last_graph_sparsity_loss = gate
            else:
                self._last_graph_sparsity_loss = adjacency_used.abs().mean()

            if self.training and self.use_graph_edge_dropout and self.graph_edge_dropout > 0.0:
                keep_prob = 1.0 - min(max(float(self.graph_edge_dropout), 0.0), 0.95)
                keep_mask = torch.rand_like(adjacency_used) < keep_prob
                eye = torch.eye(adjacency_used.shape[-1], device=adjacency_used.device, dtype=torch.bool)
                while eye.ndim < adjacency_used.ndim:
                    eye = eye.unsqueeze(0)
                keep_mask = keep_mask | eye
                adjacency_used = adjacency_used * keep_mask.to(adjacency_used.dtype) / keep_prob

            weights = adjacency_used / adjacency_used.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
            message = torch.matmul(weights, h)
            message = self.message_proj(message)
            h = self.message_update(torch.cat([h, message], dim=-1))

        if self.use_channel_attention:
            b, s, t, c, d = h.shape
            flat_h = h.reshape(b * s * t, c, d)
            key_padding_mask = None
            if seizure_channel_mask is not None:
                invalid = ~seizure_channel_mask[:, :, None, :].expand(b, s, t, c)
                key_padding_mask = invalid.reshape(b * s * t, c)
                all_invalid = key_padding_mask.all(dim=1)
                if torch.any(all_invalid):
                    key_padding_mask = key_padding_mask.clone()
                    key_padding_mask[all_invalid] = False
            attn_h, _ = self.channel_attn(flat_h, flat_h, flat_h, key_padding_mask=key_padding_mask)
            flat_h = self.attn_norm(flat_h + self.dropout(attn_h))
            h = flat_h.reshape(b, s, t, c, d)
        return h


__all__ = ["WindowGraphSpectralEncoder"]
