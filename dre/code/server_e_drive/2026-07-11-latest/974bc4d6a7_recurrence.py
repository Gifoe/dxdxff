from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from .seizure_encoder import SeizureTrajectorySummary
from .unbalanced_ot import UnbalancedSinkhornTransport


class CrossSeizureCoreRecurrence(nn.Module):
    def __init__(self, model_dim: int, solver: UnbalancedSinkhornTransport) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.solver = solver
        self.summary = SeizureTrajectorySummary(self.model_dim)
        self.projection = nn.Sequential(nn.Linear(self.model_dim * 4 + 4, self.model_dim * 4), nn.GELU(), nn.LayerNorm(self.model_dim * 4))

    def forward(
        self,
        seizure_embeddings: torch.Tensor,
        seizure_core_states: torch.Tensor,
        seizure_core_masses: torch.Tensor,
        seizure_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seizures, cores, _ = seizure_core_states.shape
        distances = seizure_embeddings.new_zeros((batch, seizures, seizures))
        for source_index in range(seizures):
            for target_index in range(source_index + 1, seizures):
                valid_pair = seizure_mask[:, source_index] & seizure_mask[:, target_index]
                source_state = functional.normalize(seizure_core_states[:, source_index], dim=-1, eps=1e-8)
                target_state = functional.normalize(seizure_core_states[:, target_index], dim=-1, eps=1e-8)
                cost = 1.0 - torch.einsum("bkd,bld->bkl", source_state, target_state)
                core_mask = valid_pair[:, None].expand(batch, cores).clone()
                invalid = ~valid_pair
                if invalid.any():
                    core_mask[invalid, 0] = True
                result = self.solver(seizure_core_masses[:, source_index], seizure_core_masses[:, target_index], cost, source_mask=core_mask, target_mask=core_mask)
                value = result.cost * valid_pair.to(result.cost.dtype)
                distances[:, source_index, target_index] = value
                distances[:, target_index, source_index] = value
        summary, attention = self.summary(seizure_embeddings, seizure_mask)
        pair_mask = seizure_mask[:, :, None] & seizure_mask[:, None, :]
        off_diagonal = ~torch.eye(seizures, device=distances.device, dtype=torch.bool).unsqueeze(0)
        valid = pair_mask & off_diagonal
        count = valid.sum(dim=(1, 2)).clamp_min(1).to(distances.dtype)
        mean = (distances * valid.to(distances.dtype)).sum(dim=(1, 2)) / count
        variance = ((distances - mean[:, None, None]).square() * valid.to(distances.dtype)).sum(dim=(1, 2)) / count
        maximum = distances.masked_fill(~valid, float("-inf")).amax(dim=(1, 2))
        maximum = torch.nan_to_num(maximum, neginf=0.0)
        minimum = distances.masked_fill(~valid, float("inf")).amin(dim=(1, 2))
        minimum = torch.nan_to_num(minimum, posinf=0.0)
        recurrence_stats = torch.stack([mean, torch.sqrt(variance.clamp_min(1e-8)), minimum, maximum], dim=-1)
        return self.projection(torch.cat([summary, recurrence_stats], dim=-1)), distances, attention


__all__ = ["CrossSeizureCoreRecurrence"]
