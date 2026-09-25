from __future__ import annotations
import torch
from torch import nn
from .encoder import SharedWindowEncoder
from .temporal import PhaseTemporalChannelEncoder
from .set_pooling import dual_statistics
from .interaction import BidirectionalCrossSet

class TrueEZNEZDualSet(nn.Module):
 def __init__(self,strict=True,channel_dropout=.10,seizure_dropout=.20):
  super().__init__();self.strict=strict;self.channel_dropout=channel_dropout;self.seizure_dropout=seizure_dropout;self.window_encoder=SharedWindowEncoder();self.temporal_encoder=PhaseTemporalChannelEncoder();self.cross_set=BidirectionalCrossSet();self.seizure_projection=nn.Sequential(nn.LayerNorm(528),nn.Linear(528,128),nn.GELU(),nn.Dropout(.20),nn.Linear(128,96),nn.LayerNorm(96),nn.GELU());self.patient_head=nn.Sequential(nn.LayerNorm(288),nn.Linear(288,64),nn.GELU(),nn.Dropout(.25),nn.Linear(64,16),nn.GELU(),nn.Dropout(.10),nn.Linear(16,1))
 def _drop(self,h):
  if not self.training or len(h)==1 or self.channel_dropout<=0:return h
  n=max(1,int(torch.ceil(torch.tensor(len(h)*(1-self.channel_dropout))).item()));return h[torch.randperm(len(h),device=h.device)[:n]]
 def _seizure(self,s):
  w=s['windows'];side=s['side_features'];labels=s['channel_labels_nez'].long();mask=s['window_mask'].bool();phases=s['phase_ids'].long();c,t=w.shape[:2]
  if w.shape[-1]!=500 or side.shape[:2]!=(c,t) or side.shape[-1]!=12 or mask.shape!=(c,t) or labels.shape!=(c,):raise ValueError(f"{s.get('seizure_id','?')}: invalid cache tensor alignment")
  if not torch.isfinite(w[mask]).all() or not torch.isfinite(side[mask]).all():raise ValueError(f"{s.get('seizure_id','?')}: non-finite valid window")
  if not torch.isin(labels,torch.tensor([0,1],device=labels.device)).all():raise ValueError(f"{s.get('seizure_id','?')}: labels must be EZ=0/NEZ=1")
  h=self.window_encoder(w.to(next(self.parameters()).device).reshape(-1,1,500),side.to(next(self.parameters()).device).reshape(-1,12)).reshape(c,t,32)
  channel=self.temporal_encoder(h,mask.to(h.device),phases.to(h.device),self.strict);ez_mask=labels.to(h.device)==0;nez_mask=labels.to(h.device)==1
  if not ez_mask.any() or not nez_mask.any():raise ValueError(f"{s.get('seizure_id','?')}: requires at least one true EZ and NEZ channel")
  ez,nez=self._drop(channel[ez_mask]),self._drop(channel[nez_mask]);z_ez,ez_parts=dual_statistics(ez);z_nez,nez_parts=dual_statistics(nez);x_ez,x_nez,_,_=self.cross_set(ez,nez);mu_ez,mu_nez=ez_parts['mean'],nez_parts['mean'];diff=mu_nez-mu_ez;raw=torch.cat([z_ez,z_nez,diff,diff.abs(),mu_nez*mu_ez,x_ez,x_nez]);u=self.seizure_projection(raw.float())
  audit={'seizure_id':s['seizure_id'],'n_ez':int(ez_mask.sum()),'n_nez':int(nez_mask.sum()),'ez_embedding_mean_norm':float(mu_ez.norm().detach()),'nez_embedding_mean_norm':float(mu_nez.norm().detach()),'ez_embedding_std_norm':float(ez_parts['std'].norm().detach()),'nez_embedding_std_norm':float(nez_parts['std'].norm().detach()),'ez_nez_difference_norm':float(diff.norm().detach()),'cross_ez_to_nez_norm':float(x_ez.norm().detach()),'cross_nez_to_ez_norm':float(x_nez.norm().detach())}
  return u,raw,audit,int(c*t)
 def forward(self,views):
  device=next(self.parameters()).device;out=[]
  for view in views:
   rows=[self._seizure(s) for s in view['seizures']]
   if self.training and len(rows)>1 and torch.rand((),device=device)<self.seizure_dropout:rows.pop(int(torch.randint(len(rows),(1,),device=device)))
   u=torch.stack([x[0] for x in rows]).float();mean=u.mean(0);std=u.std(0,unbiased=False) if len(u)>1 else torch.zeros_like(mean);patient=torch.cat([mean,std,u.max(0).values]);logit=self.patient_head(patient.float()).squeeze();
   if not torch.isfinite(logit):raise FloatingPointError('non-finite patient logit')
   out.append({'logit':logit,'patient_embedding':patient,'seizure_embeddings':u,'seizure_raw':[x[1] for x in rows],'seizure_audit':[x[2] for x in rows],'n_windows_encoded':sum(x[3] for x in rows)})
  return out
 def parameter_count(self):return sum(p.numel() for p in self.parameters())
 def module_parameter_counts(self):
  return {'raw_encoder_parameters':sum(p.numel() for p in self.window_encoder.parameters()),'temporal_encoder_parameters':sum(p.numel() for p in self.temporal_encoder.parameters()),'set_pooling_parameters':0,'cross_set_parameters':sum(p.numel() for p in self.cross_set.parameters()),'seizure_projection_parameters':sum(p.numel() for p in self.seizure_projection.parameters()),'patient_head_parameters':sum(p.numel() for p in self.patient_head.parameters())}
