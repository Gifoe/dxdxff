from __future__ import annotations
import torch
from torch import nn
class RawWindowEncoder(nn.Module):
 def __init__(self,side_dim=12,use_side=True):
  super().__init__();self.use_side=use_side;self.branches=nn.ModuleList([nn.Sequential(nn.Conv1d(1,8,k,4,k//2),nn.GroupNorm(2,8),nn.GELU()) for k in (15,31,63)]);self.b1=nn.Sequential(nn.Conv1d(24,24,15,2,7,groups=24),nn.Conv1d(24,32,1),nn.GroupNorm(4,32),nn.GELU());self.b2=nn.Sequential(nn.Conv1d(32,32,9,2,4,groups=32),nn.Conv1d(32,48,1),nn.GroupNorm(6,48),nn.GELU());self.raw=nn.Sequential(nn.Linear(96,32),nn.LayerNorm(32),nn.GELU())
  if use_side:self.side=nn.Sequential(nn.Linear(side_dim,16),nn.LayerNorm(16),nn.GELU());self.fuse=nn.Linear(48,32)
 def forward(self,x,side=None):
  x=torch.cat([b(x) for b in self.branches],1);x=self.b2(self.b1(x));h=self.raw(torch.cat([x.mean(-1),x.amax(-1)],1));return self.fuse(torch.cat([h,self.side(side.float())],1)) if self.use_side else h
