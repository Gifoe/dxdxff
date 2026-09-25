from __future__ import annotations
import torch
from torch import nn

def _masked_zscore(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    w = mask.to(x.dtype)
    if x.ndim == mask.ndim + 1: w = w.unsqueeze(-1)
    count = w.sum(dim=dim, keepdim=True)
    mean = (x * w).sum(dim=dim, keepdim=True) / count.clamp_min(1)
    var = ((x - mean).square() * w).sum(dim=dim, keepdim=True) / count.clamp_min(1)
    std = torch.sqrt(var.clamp_min(1e-12))
    usable = (count >= 2) & (std >= 1e-6)
    return torch.where(usable, (x - mean) / std, torch.zeros_like(x)) * w

def masked_patient_channel_zscore(x, mask): return _masked_zscore(x, mask, 1)
def masked_seizure_channel_zscore(x, mask): return _masked_zscore(x, mask, 2)

class CleanNEZAnchor(nn.Module):
    def __init__(self,input_dim,anchor_dim=16,dropout=.4):
        super().__init__(); self.project=nn.Sequential(nn.Linear(input_dim,input_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(input_dim,anchor_dim)); self.mu_nez=nn.Parameter(torch.zeros(anchor_dim))
    def forward(self,x,mask):
        z=self.project(x); distance=((z-self.mu_nez)**2).sum(-1).masked_fill(~mask,0.)
        return {"anchor_embedding":z*mask[...,None],"anchor_distance":distance,"anchor_distance_z":masked_patient_channel_zscore(distance,mask)}
__all__=["CleanNEZAnchor","masked_patient_channel_zscore","masked_seizure_channel_zscore"]
