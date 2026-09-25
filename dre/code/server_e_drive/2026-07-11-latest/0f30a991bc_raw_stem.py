from __future__ import annotations
import torch
from torch import nn

class _Separable(nn.Module):
    def __init__(self,cin,cout,k,stride,groups):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(cin,cin,k,stride,k//2,groups=cin),nn.Conv1d(cin,cout,1),nn.GroupNorm(groups,cout),nn.GELU(),nn.Dropout(.05))
    def forward(self,x):return self.net(x)
class RawWindowStem(nn.Module):
    def __init__(self):
        super().__init__();self.branches=nn.ModuleList([nn.Sequential(nn.Conv1d(1,8,k,4,k//2),nn.GroupNorm(2,8),nn.GELU()) for k in (15,31,63)]);self.block1=_Separable(24,32,15,2,4);self.block2=_Separable(32,48,9,2,6);self.proj=nn.Sequential(nn.Linear(96,64),nn.LayerNorm(64),nn.GELU())
    def forward(self,x):
        h=torch.cat([branch(x) for branch in self.branches],1);h=self.block2(self.block1(h));return self.proj(torch.cat([h.mean(-1),h.amax(-1)],-1))
    def stream(self,windows,batch_size=1024,min_batch_size=128,device=None):
        current=int(batch_size)
        device=device or next(self.parameters()).device
        while True:
            try:
                result=torch.cat([self(windows[i:i+current].to(device,non_blocking=True)) for i in range(0,len(windows),current)],0);self.last_window_batch_size=current;return result
            except RuntimeError as exc:
                if 'out of memory' not in str(exc).lower() or current//2<min_batch_size:raise
                current//=2
                if torch.device(device).type=='cuda':torch.cuda.empty_cache()
