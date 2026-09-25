from __future__ import annotations
import torch
from torch import nn
from .encoder import WindowEncoder
from .temporal import PhaseChannelEncoder
from .auxiliary import ChannelAuxiliaryHead
from .residual_pooling import ResidualRelation
class TraceAuxStable(nn.Module):
 def __init__(self,strict=True,channel_dropout=.1,seizure_dropout=.15):
  super().__init__();self.strict=strict;self.channel_dropout=channel_dropout;self.seizure_dropout=seizure_dropout;self.window_encoder=WindowEncoder();self.temporal_encoder=PhaseChannelEncoder();self.channel_aux=ChannelAuxiliaryHead();self.residual=ResidualRelation();self.seizure_proj=nn.Sequential(nn.LayerNorm(484),nn.Linear(484,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,96),nn.LayerNorm(96),nn.GELU());self.reliability=nn.Sequential(nn.LayerNorm(96),nn.Linear(96,1));self.gamma=nn.Parameter(torch.zeros(()));self.head=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,64),nn.GELU(),nn.Dropout(.25),nn.Linear(64,16),nn.GELU(),nn.Dropout(.1),nn.Linear(16,1))
 def _drop(self,h):
  if not self.training or len(h)<2 or self.channel_dropout<=0:return h
  keep=max(1,int(len(h)*(1-self.channel_dropout)));return h[torch.randperm(len(h),device=h.device)[:keep]]
 def seizure(self,s):
  d=next(self.parameters()).device;w=s['windows'];side=s['side_features'];mask=s['window_mask'].bool();lab=s['channel_labels_nez'].long();phase=s['phase_ids'].long();c,t=w.shape[:2]
  if w.shape[-1]!=500 or side.shape!=(c,t,12) or mask.shape!=(c,t) or lab.shape!=(c,):raise ValueError(f"{s.get('seizure_id','?')}: cache shape mismatch")
  if not torch.isfinite(w[mask]).all() or not torch.isfinite(side[mask]).all():raise ValueError(f"{s.get('seizure_id','?')}: nonfinite valid data")
  if not torch.isin(lab,torch.tensor([0,1])).all():raise ValueError(f"{s.get('seizure_id','?')}: invalid label direction")
  h=self.window_encoder(w.to(d).reshape(-1,1,500),side.to(d).reshape(-1,12)).reshape(c,t,32);ch=self.temporal_encoder(h,mask.to(d),phase.to(d),self.strict);lab=lab.to(d);logits=self.channel_aux(ch).float();ez,nez=ch[lab==0],ch[lab==1]
  if not len(ez) or not len(nez):raise ValueError(f"{s.get('seizure_id','?')}: missing EZ or NEZ")
  emb,score,weight,stat=self.residual(self._drop(ez),self._drop(nez));raw=torch.cat([stat['ez_mean'],stat['ez_max'],stat['ez_std'],stat['nez_mean'],stat['nez_std'],stat['nez_mean']-stat['ez_mean'],stat['nez_mean']*stat['ez_mean'],emb,stat['score_mean'].reshape(1),stat['score_std'].reshape(1),stat['score_max'].reshape(1),stat['burden'].reshape(1)]);u=self.seizure_proj(raw.float());audit={'seizure_id':s['seizure_id'],'n_ez':int((lab==0).sum()),'n_nez':int((lab==1).sum()),'ez_mean_norm':float(stat['ez_mean'].norm().detach()),'ez_std_norm':float(stat['ez_std'].norm().detach()),'nez_mean_norm':float(stat['nez_mean'].norm().detach()),'nez_std_norm':float(stat['nez_std'].norm().detach()),'residual_score_mean':float(stat['score_mean'].detach()),'residual_score_std':float(stat['score_std'].detach()),'residual_score_max':float(stat['score_max'].detach()),'residual_burden':float(stat['burden'].detach()),'sparsemax_nonzero_count':int((weight>0).sum()),'sparsemax_max_weight':float(weight.max().detach()),'seizure_representation_norm':float(u.norm().detach())}
  return u,(logits,lab),audit
 def forward(self,views):
  out=[]
  for v in views:
   rows=[self.seizure(s) for s in v['seizures']]
   if self.training and len(rows)>1 and torch.rand(())<self.seizure_dropout:rows.pop(int(torch.randint(len(rows),(1,))))
   u=torch.stack([x[0] for x in rows]).float();m=u.mean(0);sd=u.std(0,unbiased=False) if len(u)>1 else torch.zeros_like(m);mx=u.max(0).values;beta=torch.softmax(self.reliability(u).squeeze(-1).float(),0);rel=(beta[:,None]*u).sum(0);rep=torch.cat([m,sd,mx,.2*torch.tanh(self.gamma)*rel]);logit=self.head(rep.float()).squeeze();
   if not torch.isfinite(logit):raise FloatingPointError('nonfinite output')
   out.append({'logit':logit,'channel_rows':[x[1] for x in rows],'seizure_audit':[x[2] for x in rows],'patient_embedding':rep,'seizure_embeddings':u,'n_windows_encoded':sum(s['windows'].shape[0]*s['windows'].shape[1] for s in v['seizures'])})
  return out
 def set_stage(self,stage):
  for p in self.parameters():p.requires_grad_(False)
  groups={'a':[self.window_encoder,self.temporal_encoder,self.channel_aux],'b1':[self.residual,self.seizure_proj,self.reliability,self.head,self.gamma],'b2':[self.window_encoder,self.temporal_encoder,self.channel_aux,self.residual,self.seizure_proj,self.reliability,self.head,self.gamma]}[stage]
  for g in groups:
   if isinstance(g,nn.Parameter):g.requires_grad_(True)
   else:
    for p in g.parameters():p.requires_grad_(True)
 def parameter_count(self):return sum(p.numel() for p in self.parameters())
