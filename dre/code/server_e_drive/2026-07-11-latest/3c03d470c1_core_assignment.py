from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn


@dataclass(frozen=True)
class CoreAssignmentOutput:
    responsibilities: torch.Tensor
    core_masses: torch.Tensor
    background_mass: torch.Tensor
    core_states: torch.Tensor


class CompetitiveCoreAssigner(nn.Module):
    def __init__(self, model_dim: int, num_cores: int, temperature: float = 0.2) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.num_cores = int(num_cores)
        self.temperature = float(temperature)
        self.null_core = nn.Parameter(torch.zeros(self.model_dim))
        self.core_bias = nn.Parameter(torch.zeros(self.num_cores + 1))
        self.value_projection = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.state_norm = nn.LayerNorm(self.model_dim)

    def forward(self, values: torch.Tensor, anchors: torch.Tensor, valid_mask: torch.Tensor) -> CoreAssignmentOutput:
        normalized_values = functional.normalize(values, dim=-1, eps=1e-8)
        all_anchors = torch.cat([anchors, self.null_core.view(1, 1, -1).expand(anchors.shape[0], 1, -1)], dim=1)
        normalized_anchors = functional.normalize(all_anchors, dim=-1, eps=1e-8)
        scores = torch.einsum("bswcd,bkd->bswck", normalized_values, normalized_anchors)
        scores = scores / max(self.temperature, 1e-6) + self.core_bias
        responsibilities = torch.softmax(scores, dim=-1) * valid_mask.unsqueeze(-1).to(values.dtype)
        core_responsibility = responsibilities[..., : self.num_cores]
        valid_count = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(values.dtype)
        core_masses = core_responsibility.sum(dim=-2) / valid_count
        background_mass = responsibilities[..., self.num_cores].sum(dim=-1) / valid_count.squeeze(-1)
        projected = self.value_projection(values)
        numerator = torch.einsum("bswck,bswcd->bswkd", core_responsibility, projected)
        denominator = core_responsibility.sum(dim=-2).unsqueeze(-1).clamp_min(1e-8)
        weighted = numerator / denominator
        states = self.state_norm(anchors[:, None, None, :, :] + weighted)
        window_valid = valid_mask.any(dim=-1)
        states = states * window_valid[..., None, None].to(states.dtype)
        core_masses = core_masses * window_valid.unsqueeze(-1).to(core_masses.dtype)
        background_mass = background_mass * window_valid.to(background_mass.dtype)
        return CoreAssignmentOutput(responsibilities, core_masses, background_mass, states)


__all__ = ["CompetitiveCoreAssigner", "CoreAssignmentOutput"]
