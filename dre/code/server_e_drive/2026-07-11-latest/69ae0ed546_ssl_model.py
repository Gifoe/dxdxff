import copy,torch
from torch import nn
from .window_encoder import WindowEncoder
from .temporal_encoder import Temporal
class NativeEncoder(nn.Module):
 def __init__(self):super().__init__();self.window=WindowEncoder();self.temporal=Temporal();self.mask_token=nn.Parameter(torch.zeros(48))
 def encode_seizure(self,s,mask_override=None):
  # Views are intentionally CPU-resident; the encoder owns the single
  # device-transfer boundary for both SSL and downstream outcome stages.
  d=next(self.parameters()).device;m=(mask_override if mask_override is not None else s['window_mask']).to(d).bool();w=s['windows'].to(d);side=s['side_features'].to(d);c,t=w.shape[:2];z=self.window(w.reshape(-1,1,500),side.reshape(-1,12)).reshape(c,t,48);z=torch.where(m[:,:,None],z,self.mask_token[None,None]);return self.temporal(z,m,s['phase_ids'].to(d),s['relative_times_sec'].to(d))
class SSLModel(nn.Module):
 def __init__(self):
  super().__init__();self.online=NativeEncoder();self.target=copy.deepcopy(self.online).eval();self.predictor=nn.Sequential(nn.Linear(48,96),nn.GELU(),nn.Linear(96,48));self.project=nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Linear(64,32));self.target_project=copy.deepcopy(self.project).eval();[p.requires_grad_(False) for p in list(self.target.parameters())+list(self.target_project.parameters())]
 @torch.no_grad()
 def ema(self,d=.996):
  for a,b in zip(self.target.parameters(),self.online.parameters()):a.mul_(d).add_(b.detach(),alpha=1-d)
  for a,b in zip(self.target_project.parameters(),self.project.parameters()):a.mul_(d).add_(b.detach(),alpha=1-d)
