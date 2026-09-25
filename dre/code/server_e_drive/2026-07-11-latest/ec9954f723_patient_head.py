from __future__ import annotations

import torch
from torch import nn


class PatientOutcomeHead(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        hidden = min(64, max(16, int(input_dim) // 2))
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, 1),
        )

    def forward(self, patient_embedding: torch.Tensor) -> torch.Tensor:
        return self.network(patient_embedding).squeeze(-1)


__all__ = ["PatientOutcomeHead"]
