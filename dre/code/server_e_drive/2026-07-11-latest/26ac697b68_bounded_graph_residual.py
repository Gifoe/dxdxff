from __future__ import annotations

import torch
from torch import nn

from .masks import masked_mean, masked_std


class BoundedPhaseGraphResidual(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.norm = nn.LayerNorm(self.embedding_dim)
        self.linear_self = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.linear_neighbor = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dropout = nn.Dropout(0.10)
        self.graph_gate_raw = nn.Parameter(torch.tensor(-3.0))
        self.n_phases = 3
        pooled_dim = self.n_phases * self.embedding_dim * 2 + (self.n_phases - 1) * self.embedding_dim + (2 * self.n_phases - 1)
        self.pool_projection = nn.Sequential(nn.Linear(pooled_dim, 64), nn.GELU(), nn.Dropout(0.10))

    @property
    def graph_gate(self) -> torch.Tensor:
        return 0.20 * torch.sigmoid(self.graph_gate_raw)

    def forward(
        self,
        phase_channel_embedding: torch.Tensor,
        adjacency: torch.Tensor,
        phase_channel_mask: torch.Tensor,
        phase_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if phase_channel_embedding.ndim != 5 or adjacency.ndim != 5:
            raise ValueError("phase embeddings/adjacency must be [B,S,P,C,D] and [B,S,P,C,C]")
        node_present = phase_channel_mask.bool()
        valid_node = node_present & phase_mask.bool().unsqueeze(-1)
        pair_mask = valid_node.unsqueeze(-1) & valid_node.unsqueeze(-2)
        matrix = torch.where(pair_mask, torch.nan_to_num(adjacency).clamp_min(0.0), torch.zeros_like(adjacency))
        identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
        matrix = matrix + identity * valid_node.to(matrix.dtype).unsqueeze(-1)
        degree = matrix.sum(dim=-1).clamp_min(1e-6)
        normalized = degree.rsqrt().unsqueeze(-1) * matrix * degree.rsqrt().unsqueeze(-2)
        base_h = torch.nan_to_num(phase_channel_embedding) * node_present.unsqueeze(-1).to(phase_channel_embedding.dtype)
        active_h = base_h * valid_node.unsqueeze(-1).to(base_h.dtype)
        message = normalized @ self.norm(active_h)
        update = self.linear_self(active_h) + self.linear_neighbor(message)
        update = self.dropout(torch.nn.functional.gelu(update)).masked_fill(~valid_node.unsqueeze(-1), 0.0)
        graph_has_edge = (matrix.sum(dim=(-2, -1)) - valid_node.sum(dim=-1).to(matrix.dtype)) > 1e-6
        active = phase_mask.bool() & graph_has_edge
        updated = base_h + self.graph_gate * update
        updated = torch.where(active.unsqueeze(-1).unsqueeze(-1), updated, base_h)
        graph_mean = masked_mean(updated, valid_node, dim=3)
        graph_std = masked_std(updated, valid_node, dim=3)
        if phase_channel_embedding.shape[2] != self.n_phases:
            raise ValueError(f"raw graph residual requires exactly {self.n_phases} network phases")
        delta_valid = active[..., :-1] & active[..., 1:]
        graph_delta = graph_mean[..., 1:, :] - graph_mean[..., :-1, :]
        graph_delta = graph_delta.masked_fill(~delta_valid.unsqueeze(-1), 0.0)
        flags = torch.cat((active.to(updated.dtype), delta_valid.to(updated.dtype)), dim=-1)
        pooled = torch.cat((graph_mean.flatten(-2), graph_std.flatten(-2), graph_delta.flatten(-2), flags), dim=-1)
        return {
            "phase_graph_embedding": updated,
            "graph_update": update.masked_fill(~active.unsqueeze(-1).unsqueeze(-1), 0.0),
            "graph_gate": self.graph_gate,
            "graph_embedding": self.pool_projection(pooled),
            "phase_graph_mean": graph_mean,
            "phase_graph_std": graph_std,
            "phase_graph_delta": graph_delta,
            "graph_phase_valid": active,
            "graph_delta_valid": delta_valid,
        }


__all__ = ["BoundedPhaseGraphResidual"]
