from __future__ import annotations
import torch
from torch import nn
class Block(nn.Module):
 def __init__(self,c):super().__init__();self.net=nn.Sequential(nn.Conv1d(c,c,3,padding=1,groups=c),nn.Conv1d(c,c,1),nn.GroupNorm(2,c),nn.GELU())
 def forward(self,x):return self.net(x)
class WindowEncoder(nn.Module):
 """TRACE-compatible GroupNorm encoder: no BatchNorm/running patient statistics."""
 def __init__(self):
  super().__init__();self.branches=nn.ModuleList([nn.Sequential(nn.Conv1d(1,8,k,2,k//2),nn.GroupNorm(2,8),nn.GELU(),Block(8)) for k in (15,31,63)]);self.raw=nn.Sequential(nn.Linear(48,32),nn.LayerNorm(32),nn.GELU());self.side=nn.Sequential(nn.Linear(12,16),nn.LayerNorm(16),nn.GELU(),nn.Linear(16,16));self.fuse=nn.Sequential(nn.Linear(48,32),nn.LayerNorm(32),nn.GELU())
 def forward(self,x,side):
  z=[]
  for b in self.branches:
   y=b(x);z.append(torch.cat([y.mean(-1),y.amax(-1)],1))
  return self.fuse(torch.cat([self.raw(torch.cat(z,1)),self.side(side.float())],1))
