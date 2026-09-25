from __future__ import annotations

import torch
from torch import nn


class PatientConditionedCanonicalCoreBank(nn.Module):
    def __init__(self, num_cores: int, model_dim: int, *, conditioned: bool) -> None:
        super().__init__()
        self.num_cores = int(num_cores)
        self.model_dim = int(model_dim)
        self.conditioned = bool(conditioned)
        self.global_base_queries = nn.Parameter(torch.empty(self.num_cores, self.model_dim))
        nn.init.normal_(self.global_base_queries, std=0.1)
        self.adapter = nn.Linear(self.model_dim, self.num_cores * self.model_dim) if self.conditioned else None
        if self.adapter is not None:
            nn.init.normal_(self.adapter.weight, std=0.02)
            nn.init.zeros_(self.adapter.bias)
        self.norm = nn.LayerNorm(self.model_dim)

    def forward(self, patient_context: torch.Tensor) -> torch.Tensor:
        batch = patient_context.shape[0]
        base = self.global_base_queries.unsqueeze(0).expand(batch, -1, -1)
        delta = 0.0 if self.adapter is None else self.adapter(patient_context).reshape(batch, self.num_cores, self.model_dim)
        return self.norm(base + delta)


__all__ = ["PatientConditionedCanonicalCoreBank"]
