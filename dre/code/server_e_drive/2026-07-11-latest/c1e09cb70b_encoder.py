from __future__ import annotations
import torch
from torch import nn

class DepthwiseSeparable(nn.Module):
 def __init__(self,c):
  super().__init__();self.net=nn.Sequential(nn.Conv1d(c,c,3,padding=1,groups=c),nn.Conv1d(c,c,1),nn.GroupNorm(2,c),nn.GELU())
 def forward(self,x):return self.net(x)

class SharedWindowEncoder(nn.Module):
 """Raw waveform and 12D side features only; emits one 32D window embedding."""
 def __init__(self,side_dim=12):
  super().__init__();self.branches=nn.ModuleList([nn.Sequential(nn.Conv1d(1,8,k,stride=2,padding=k//2),nn.GroupNorm(2,8),nn.GELU(),DepthwiseSeparable(8)) for k in (15,31,63)])
  self.raw=nn.Sequential(nn.Linear(48,32),nn.LayerNorm(32),nn.GELU())
  self.side=nn.Sequential(nn.Linear(side_dim,16),nn.LayerNorm(16),nn.GELU(),nn.Linear(16,16))
  self.fuse=nn.Sequential(nn.Linear(48,32),nn.LayerNorm(32),nn.GELU())
 def forward(self,x,side):
  if x.ndim!=3 or x.shape[1:]!=(1,500):raise ValueError(f'expected raw [N,1,500], got {tuple(x.shape)}')
  if side.ndim!=2 or side.shape[0]!=x.shape[0] or side.shape[1]!=12:raise ValueError(f'expected side [N,12], got {tuple(side.shape)}')
  summaries=[]
  for branch in self.branches:
   y=branch(x);summaries.append(torch.cat([y.mean(-1),y.amax(-1)],1))
  raw=torch.cat(summaries,1)
  return self.fuse(torch.cat([self.raw(raw),self.side(side.float())],1))
