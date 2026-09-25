from __future__ import annotations
import torch
from torch import nn

class TemporalResidualBlock(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv1d(32,32,3,padding=1,groups=32),nn.Conv1d(32,32,1),nn.GroupNorm(4,32),nn.GELU())
 def forward(self,x):return self.net(x)

def masked_mean(x,mask):
 w=mask.float().unsqueeze(-1);return (x*w).sum(1)/w.sum(1).clamp_min(1.)
def masked_std(x,mask,mean):
 w=mask.float().unsqueeze(-1);return torch.sqrt((((x-mean[:,None])**2)*w).sum(1)/w.sum(1).clamp_min(1.)).nan_to_num()

class PhaseTemporalChannelEncoder(nn.Module):
 def __init__(self):
  super().__init__();self.blocks=nn.ModuleList([TemporalResidualBlock(),TemporalResidualBlock()]);self.project=nn.Sequential(nn.Linear(256,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(.10),nn.Linear(64,48),nn.LayerNorm(48),nn.GELU())
 def forward(self,h,window_mask,phase_ids,strict=True):
  if h.ndim!=3 or h.shape[-1]!=32:raise ValueError(f'expected channel windows [C,T,32], got {tuple(h.shape)}')
  if window_mask.shape!=h.shape[:2] or phase_ids.shape!=(h.shape[1],):raise ValueError('temporal mask/phase shape mismatch')
  phase_ids=phase_ids.to(h.device)
  for phase in range(3):
   if not ((phase_ids==phase)[None,:]&window_mask.bool()).any():
    if strict:raise ValueError(f'seizure requires a valid phase {phase} window')
  valid=window_mask.bool().unsqueeze(1)
  x=h.float().transpose(1,2)*valid
  # Mask before and after every temporal convolution: an invalid window must
  # not leak into adjacent valid positions through the convolution kernel.
  for block in self.blocks:x=(x+block(x))*valid
  x=x.transpose(1,2);means=[]
  for phase in range(3):means.append(masked_mean(x,window_mask.bool() & (phase_ids[None,:]==phase)))
  glob=masked_mean(x,window_mask);std=masked_std(x,window_mask,glob)
  pre,onset,spread=means
  return self.project(torch.cat([pre,onset,spread,glob,std,onset-pre,spread-onset,spread-pre],1).float())
