from __future__ import annotations
import torch
from torch import nn
class TBlock(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Conv1d(32,32,3,padding=1,groups=32),nn.Conv1d(32,32,1),nn.GroupNorm(4,32),nn.GELU(),nn.Dropout(.05))
 def forward(self,x):return self.net(x)
def mean(x,m):
 w=m.float().unsqueeze(-1);return (x*w).sum(1)/w.sum(1).clamp_min(1)
class PhaseChannelEncoder(nn.Module):
 def __init__(self):super().__init__();self.blocks=nn.ModuleList([TBlock(),TBlock()]);self.proj=nn.Sequential(nn.Linear(256,96),nn.LayerNorm(96),nn.GELU(),nn.Dropout(.1),nn.Linear(96,64),nn.LayerNorm(64),nn.GELU())
 def forward(self,h,mask,phase,strict=True):
  for p in range(3):
   if not ((phase==p)[None]&mask.bool()).any() and strict:raise ValueError(f'empty phase {p}')
  valid=mask.bool().unsqueeze(1);x=h.float().transpose(1,2)*valid
  for b in self.blocks:x=(x+b(x))*valid
  x=x.transpose(1,2);v=[]
  for p in range(3):v.append(mean(x,mask.bool()&((phase==p)[None])))
  g=mean(x,mask);w=mask.float().unsqueeze(-1);sd=torch.sqrt((((x-g[:,None])**2)*w).sum(1)/w.sum(1).clamp_min(1)).nan_to_num();a,b,c=v
  return self.proj(torch.cat([a,b,c,g,sd,b-a,c-b,c-a],1).float())
