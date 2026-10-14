"""Frozen physiological pathway + explicitly pre-referenced forward GRU residual."""
import torch
from torch import nn

class RawDeltaEncoder(nn.Module):
 def __init__(self,base):
  super().__init__();self.base=base;self.base.eval()
  for p in self.base.parameters():p.requires_grad_(False)
  self.cnn=nn.Sequential(nn.Conv1d(1,12,9,4),nn.GELU(),nn.Conv1d(12,24,7,4),nn.GELU(),nn.AdaptiveAvgPool1d(1))
  self.gru=nn.GRU(25,24,batch_first=True);self.head=nn.Sequential(nn.Linear(96,64),nn.GELU(),nn.Linear(64,1));self.gamma=nn.Parameter(torch.zeros(()))
  assert sum(p.numel() for p in self.parameters())==13643
  assert sum(p.numel() for p in self.parameters() if p.requires_grad)==12106
 def train(self,mode=True):super().train(mode);self.base.eval();return self
 def features(self,raw,mask,times,perms,mode='true'):
  S,C,W,L=raw.shape;assert L==500 and mask.shape==(S,C,W) and times.shape==(S,W)
  flat=raw[mask];v=(flat-flat.mean(-1,keepdim=True))/flat.std(-1,unbiased=False,keepdim=True).clamp_min(1e-6);v=v.clamp(-8,8)
  parts=[self.cnn(q[:,None]).flatten(1) for q in v.split(4096)];emb=raw.new_zeros(S,C,W,24);emb[mask]=torch.cat(parts)
  summaries=[];available=[]
  for s in range(S):
   pre=mask[s]&(times[s][None]<0);np_=pre.sum(1);present=mask[s].any(1);assert (np_[present]>0).all()
   hpre=(emb[s]*pre[...,None]).sum(1)/np_.clamp_min(1)[:,None];delta=emb[s] if mode=='absolute' else emb[s]-hpre[:,None]
   post=torch.nonzero(mask[s].any(0)&(times[s]>=0)&(times[s]<20)).flatten();assert len(post)>0
   hh=delta[:,post]
   if mode=='shuffle':hh=torch.gather(hh,1,perms[s][...,None].expand(-1,-1,24))
   seq=torch.cat([hh,(times[s,post]/20)[None,:,None].expand(C,-1,-1)],-1)
   hidden,_=self.gru(seq[present]);vout=raw.new_zeros(C,len(post),24);vout[present]=hidden;st=[];av=[]
   for low,high in [(0,8),(8,20)]:
    select=mask[s].index_select(1,post)&((times[s,post]>=low)&(times[s,post]<high))[None];count=select.sum(1);st.append((vout*select[...,None]).sum(1)/count.clamp_min(1)[:,None]);av.append(count>0)
   summaries.append(torch.stack(st,1));available.append(torch.stack(av,1))
  summary=torch.stack(summaries);valid=torch.stack(available);v=valid[...,None].to(raw.dtype);n=v.sum(0);mu=(summary*v).sum(0)/n.clamp_min(1);variance=(((summary-mu[None]).square())*v).sum(0)/n.clamp_min(1)
  # sqrt at zero has undefined derivative; select exact zero for single-seizure/zero variance.
  sd=torch.where(variance>0,torch.sqrt(variance.clamp_min(1e-30)),torch.zeros_like(variance));assert (n>0).all(),'Missing temporal stages cannot be silently imputed'
  vector=torch.stack([mu,sd],2).reshape(C,96).double();mean=vector.mean(0);std=vector.std(0,unbiased=False);z=(vector-mean)/torch.where(std>1e-8,std,torch.ones_like(std));return z.to(raw.dtype)
 def forward(self,x,raw,mask,times,perms,mode='true',gamma_zero=False):
  with torch.no_grad():base=self.base(x)
  rawlogit=self.head(self.features(raw,mask,times,perms,mode)).squeeze(-1);residual=.5*torch.tanh(self.gamma)*torch.tanh(rawlogit)
  final=base if gamma_zero else base+residual
  return final,rawlogit,base,residual
