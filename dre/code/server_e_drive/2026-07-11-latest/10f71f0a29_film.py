from __future__ import annotations
import torch
from torch import nn
class PhaseFiLM(nn.Module):
 def __init__(self): super().__init__(); self.gamma=nn.Embedding(3,32); self.beta=nn.Embedding(3,32); nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.beta.weight)
 def forward(self,h,phase): return (1+self.gamma(phase))*h+self.beta(phase)
