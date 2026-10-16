"""Matched nonlinear readout; the CNN remains frozen outside this module."""
import torch
from torch import nn


class Readout(nn.Module):
    def __init__(self, groups, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = nn.Sequential(nn.Linear(256,64), nn.GELU(), nn.Linear(64,1))
        mask=torch.zeros(256);mask[:192]=1
        for group in groups:mask[192+2*group:194+2*group]=1
        self.register_buffer('mask',mask)
        assert sum(p.numel() for p in self.parameters()) == 16513

    def forward(self, x):
        return self.network(x*self.mask).squeeze(-1)
