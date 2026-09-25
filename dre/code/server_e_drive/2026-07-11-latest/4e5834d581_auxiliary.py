from __future__ import annotations
import torch
from torch import nn
class ChannelAuxiliaryHead(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.LayerNorm(64),nn.Linear(64,1))
 def forward(self,h):return self.net(h.float()).squeeze(-1)
