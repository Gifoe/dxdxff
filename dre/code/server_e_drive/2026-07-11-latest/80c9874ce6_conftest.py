from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


def synthetic_p2(batch: int = 2, seizures: int = 3, channels: int = 6, dim: int = 8):
    torch.manual_seed(7)
    seizure_mask = torch.ones(batch, seizures, dtype=torch.bool)
    seizure_channel_mask = torch.ones(batch, seizures, channels, dtype=torch.bool)
    channel_mask = torch.ones(batch, channels, dtype=torch.bool)
    phase_mask = seizure_channel_mask.unsqueeze(2).expand(batch, seizures, 4, channels).clone()
    p2 = {
        "seizure_channel_embedding": torch.randn(batch, seizures, channels, dim),
        "seizure_nez_probability": torch.sigmoid(torch.randn(batch, seizures, channels)),
        "phase_channel_embedding": torch.randn(batch, seizures, 4, channels, dim),
        "seizure_mask": seizure_mask,
        "seizure_channel_mask": seizure_channel_mask,
        "channel_mask": channel_mask,
        "phase_mask": phase_mask,
    }
    adjacency = torch.rand(batch, seizures, 3, channels, channels)
    adjacency = (adjacency + adjacency.transpose(-1, -2)) / 2.0
    adjacency.diagonal(dim1=-2, dim2=-1).zero_()
    graphs = {
        "adjacency": adjacency,
        "phase_channel_mask": phase_mask[:, :, :3].clone(),
        "graph_valid": torch.ones(batch, seizures, 3, dtype=torch.bool),
    }
    return p2, graphs
