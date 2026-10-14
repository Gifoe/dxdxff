"""Bounded multiplicative guidance; no physiology-to-output additive branch."""
import torch
from torch import nn

class PhysiologyGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(7,32), nn.GELU(), nn.Linear(32,24))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        assert sum(p.numel() for p in self.parameters()) == 1048

    def forward(self, u):
        return 1 + .25*torch.tanh(self.network(u))
