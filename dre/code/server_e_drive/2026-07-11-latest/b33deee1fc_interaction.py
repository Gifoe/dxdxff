from __future__ import annotations
import torch
from torch import nn
class BidirectionalCrossSet(nn.Module):
 def __init__(self):
  super().__init__();self.ez_to_nez=nn.MultiheadAttention(48,1,dropout=.10,batch_first=True);self.nez_to_ez=nn.MultiheadAttention(48,1,dropout=.10,batch_first=True);self.ez_norm=nn.LayerNorm(48);self.nez_norm=nn.LayerNorm(48)
 def forward(self,ez,nez):
  if not len(ez) or not len(nez):raise ValueError('EZ and NEZ sets must both be non-empty')
  # Attention and residual normalization stay float32 under global CUDA autocast.
  with torch.autocast(device_type=ez.device.type,enabled=False):
   ez,nez=ez.float(),nez.float();a,_=self.ez_to_nez(ez[None],nez[None],nez[None],need_weights=False);b,_=self.nez_to_ez(nez[None],ez[None],ez[None],need_weights=False);a=self.ez_norm(ez+a[0]);b=self.nez_norm(nez+b[0])
  return a.mean(0),b.mean(0),a,b
