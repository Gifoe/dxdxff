from __future__ import annotations
import torch
from torch import nn
from .encoder import RawEncoder
from .film import PhaseFiLM
from .pooling import ChannelPhasePool
from .sparsemax import sparsemax
class TRACERawMIL(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=RawEncoder();self.film=PhaseFiLM();self.channel=ChannelPhasePool();self.ezref=nn.Sequential(nn.Linear(64,32),nn.LayerNorm(32),nn.GELU());self.nezref=nn.Sequential(nn.Linear(64,32),nn.LayerNorm(32),nn.GELU());self.residual=nn.Sequential(nn.Linear(128,32),nn.LayerNorm(32),nn.Tanh(),nn.Linear(32,1));self.rel_content=nn.Linear(32,32);self.rel_gate=nn.Linear(32,32);self.seizure=nn.Sequential(nn.Linear(161,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(.2),nn.Linear(64,32),nn.LayerNorm(32),nn.GELU());self.seizure_head=nn.Linear(32,1);self.reliability=nn.Sequential(nn.Linear(32,16),nn.Tanh(),nn.Linear(16,1));self.head=nn.Sequential(nn.Linear(64,32),nn.LayerNorm(32),nn.GELU(),nn.Dropout(.25),nn.Linear(32,1));self.encoder_window_batch_size=1024;self.gradient_checkpoint_encoder=False;self.encoder_autocast=False;self.last_n_flat_windows=0;self.last_encoder_chunks=0
 def configure_encoder(self,window_batch_size,gradient_checkpoint_encoder=False,autocast_enabled=False):self.encoder_window_batch_size=int(window_batch_size);self.gradient_checkpoint_encoder=bool(gradient_checkpoint_encoder);self.encoder_autocast=bool(autocast_enabled)
 def _seizure(self,windows,phases,channels,labels):
  h=self.film(self.encoder.stream(windows,next(self.parameters()).device,self.encoder_window_batch_size,self.gradient_checkpoint_encoder,self.encoder_autocast),phases.to(next(self.parameters()).device));channels=channels.to(h.device);labels=labels.to(h.device);ids=torch.unique(channels);hc=torch.stack([self.channel(h[channels==c],phases.to(h.device)[channels==c]) for c in ids]);lab=torch.stack([labels[channels==c][0] for c in ids]);ez=lab==0;nez=lab==1
  if not ez.any() or not nez.any():return None
  qez=self.ezref(torch.cat([hc[ez].mean(0),hc[ez].amax(0)]));qnez=self.nezref(torch.cat([hc[nez].mean(0),hc[nez].amax(0)]));z=hc[nez];logits=self.residual(torch.cat([z,qez.expand_as(z),z-qez,z*qez],1)).squeeze(-1);alpha=sparsemax(logits);eres=(alpha[:,None]*z).sum(0);rho=(alpha*torch.sigmoid(logits)).sum();erel=torch.sigmoid(self.rel_gate(eres*qez))*torch.tanh(self.rel_content(eres-qez));u=self.seizure(torch.cat([qez,qnez,eres,qez-qnez,erel,rho[None]]));return u,rho,self.seizure_head(u).squeeze(),{'channel_ids':ids[nez],'logits':logits,'alpha':alpha}
 def forward(self,patients):
  outputs=[];self.last_n_flat_windows=0;self.last_encoder_chunks=0
  for patient in patients:
   seizures=[]
   for item in patient:
    x=self._seizure(*item);self.last_n_flat_windows+=self.encoder.last_n_flat_windows;self.last_encoder_chunks+=self.encoder.last_encoder_chunks
    if x is not None:seizures.append(x)
   if not seizures:raise ValueError('patient has no valid EZ/NEZ seizure')
   u=torch.stack([x[0] for x in seizures]);rho=torch.stack([x[1] for x in seizures]);sl=torch.stack([x[2] for x in seizures]);beta=torch.softmax(self.reliability(u).squeeze(-1),0);logit=self.head(torch.cat([(beta[:,None]*u).sum(0),u.amax(0)])).squeeze();outputs.append({'logit':logit,'rho':(beta*rho).sum(),'beta':beta,'seizure_logits':sl,'details':[x[3] for x in seizures]})
  return outputs
 def parameter_count(self):return sum(p.numel() for p in self.parameters())
