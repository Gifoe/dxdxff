import torch
from torch import nn
class W(nn.Module):
 def __init__(self,c):super().__init__();self.n=nn.Sequential(nn.Conv1d(c,c,3,1,1,groups=c),nn.Conv1d(c,c,1),nn.GroupNorm(2,c),nn.GELU())
 def forward(self,x):return self.n(x)
class WindowEncoder(nn.Module):
 def __init__(self):super().__init__();self.b=nn.ModuleList([nn.Sequential(nn.Conv1d(1,8,k,2,k//2),nn.GroupNorm(2,8),nn.GELU(),W(8)) for k in (15,31,63)]);self.r=nn.Sequential(nn.Linear(48,32),nn.LayerNorm(32),nn.GELU());self.s=nn.Sequential(nn.Linear(12,16),nn.LayerNorm(16),nn.GELU(),nn.Linear(16,16));self.f=nn.Sequential(nn.Linear(48,48),nn.LayerNorm(48),nn.GELU())
 def forward(self,x,s):
  a=[]
  for b in self.b:
   y=b(x);a.append(torch.cat([y.mean(-1),y.amax(-1)],1))
  return self.f(torch.cat([self.r(torch.cat(a,1)),self.s(s.float())],1))
