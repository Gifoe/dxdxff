"""Raw-only RawTiny32 -> seizure mean/std64 -> Linear88 -> patientZ -> A0.

The unchanged original trainer indexes a label-free row-ID tensor. Row IDs
retrieve waveform/channel metadata only and never enter learned layers.
"""
import sys
import numpy as np
import torch
from dependencies import load


def encoder_class(source):
 sys.path.insert(0,str(source));sys.path.insert(0,str(source/'neuroez_c'))
 return load('compact_exact_rawtiny_frontend',source/'rawtiny_patient_relative_seed42_v1/code/models.py').RawTinyEncoder


def patient_z(x,mask,epsilon=1e-8):
 assert x.ndim==3 and mask.shape==x.shape[:2] and mask.any(1).all()
 v=mask.unsqueeze(-1).to(x.dtype);n=v.sum(1,keepdim=True);mu=(x*v).sum(1,keepdim=True)/n
 var=((x-mu).square()*v).sum(1,keepdim=True)/n
 sd=torch.where(var>0,torch.sqrt(torch.where(var>0,var,torch.ones_like(var))),torch.zeros_like(var))
 return ((x-mu)/sd.clamp_min(epsilon))*v


def pool(embedding,window_mask):
 assert embedding.shape[:-1]==window_mask.shape
 v=window_mask.unsqueeze(-1).to(embedding.dtype);wc=v.sum(3)
 seizure=(embedding*v).sum(3)/wc.clamp_min(1)
 available=window_mask.any(3);sv=available.unsqueeze(-1).to(embedding.dtype);count=sv.sum(1)
 mean=(seizure*sv).sum(1)/count.clamp_min(1)
 variance=((seizure-mean.unsqueeze(1)).square()*sv).sum(1)/count.clamp_min(1)
 sd=torch.where(variance>0,torch.sqrt(torch.where(variance>0,variance,torch.ones_like(variance))),torch.zeros_like(variance))
 return torch.cat([mean,sd],-1),available.any(1)


def constructor(original,source,store,meta,chunk_size=256):
 RawTinyEncoder=encoder_class(source)
 # Explicit allowlist: no x, y, scaler, physiological feature or source field.
 pid=np.asarray(meta['patient']).astype(str);channels=np.asarray(meta['channel']).astype(str)
 class RawOnly(torch.nn.Module):
  def __init__(self,seed=42):
   super().__init__();self.network=original.SourceA0(seed).network
   self.raw_encoder=RawTinyEncoder(chunk_size=chunk_size);self.raw_projection=torch.nn.Linear(64,88)
   assert sum(p.numel() for p in self.parameters())==18153

  def waveform_logits(self,raw,mask):
   embedding=self.raw_encoder(raw,mask);summary,cm=pool(embedding,mask.to(embedding.device))
   assert cm.all(),'Missing canonical channel must fail, not become a fake observation'
   latent=self.raw_projection(summary);z=patient_z(latent,cm)
   return self.network(z).squeeze(-1)[0]

  def forward(self,row_indices,g=None):
   ix=row_indices.detach().to('cpu').numpy().reshape(-1).astype(int);assert np.all((ix>=0)&(ix<len(pid)))
   assert len(ix)==len(set(ix.tolist()))
   out=torch.zeros(len(ix),device=next(self.parameters()).device)
   for p in sorted(set(pid[ix])):
    pos=np.flatnonzero(pid[ix]==p);ii=ix[pos]
    assert set(ii)==set(np.flatnonzero(pid==p)),'Patient Z requires complete canonical channel set'
    raw,mask=store.get(p,channels[ii].tolist());ll=self.waveform_logits(raw,mask)
    out=out.index_copy(0,torch.as_tensor(pos,device=out.device),ll)
   return out

  def regularizer(self,counts):return next(self.parameters()).new_zeros(())
 return RawOnly
