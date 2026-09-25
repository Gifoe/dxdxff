from __future__ import annotations
import torch
from torch import nn
class ChannelPhasePool(nn.Module):
 def __init__(self): super().__init__();self.net=nn.Sequential(nn.Linear(195,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,32),nn.LayerNorm(32),nn.GELU())
 def forward(self,h,phase):
  parts=[]
  for p in range(3):
   x=h[phase==p];parts += [torch.cat([x.mean(0),x.amax(0),torch.ones(1,device=h.device)]) if len(x) else torch.zeros(65,device=h.device)]
  return self.net(torch.cat(parts))
