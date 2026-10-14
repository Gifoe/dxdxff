"""Small deterministic-stage and nonlinear-window residuals on exact frozen D1."""
import numpy as np
import torch
from torch import nn

def patient_z(vector):
    vector=vector.double();sd=vector.std(0,unbiased=False)
    return ((vector-vector.mean(0))/torch.where(sd>1e-8,sd,torch.ones_like(sd))).float()

def pool(embedding,mask,times):
    summaries=[];available=[]
    for s in range(len(embedding)):
        ss=[];aa=[]
        for low,high in [(0,8),(8,20)]:
            m=mask[s]&((times[s]>=low)&(times[s]<high))[None]
            n=m.sum(1);ss.append((embedding[s]*m[...,None]).sum(1)/n.clamp_min(1)[:,None]);aa.append(n>0)
        summaries.append(torch.stack(ss,1));available.append(torch.stack(aa,1))
    z=torch.stack(summaries);valid=torch.stack(available)[...,None].to(z.dtype);n=valid.sum(0)
    assert (n>0).all(), 'Missing genuine stage cannot be imputed'
    mu=(z*valid).sum(0)/n;var=((z-mu[None]).square()*valid).sum(0)/n
    sd=torch.where(var>0,torch.sqrt(var.clamp_min(1e-30)),torch.zeros_like(var))
    return patient_z(torch.stack([mu,sd],2).reshape(mu.shape[0],-1))

def condition(delta,mask,times,mode,perms):
    out=delta.clone();assert not delta.requires_grad
    if mode=='static':
        for s in range(len(out)):
            for low,high in [(0,8),(8,20)]:
                m=mask[s]&((times[s]>=low)&(times[s]<high))[None]
                mean=(delta[s]*m[...,None]).sum(1)/m.sum(1).clamp_min(1)[:,None]
                out[s]=torch.where(m[...,None],mean[:,None],out[s])
    elif mode=='shuffle':
        for s,p in enumerate(perms):
            p=torch.as_tensor(p,device=delta.device);assert torch.equal(torch.gather(mask[s],1,p),mask[s])
            out[s]=torch.gather(delta[s],1,p[...,None].expand(-1,-1,7))
    return out

class DeltaResidual(nn.Module):
    def __init__(self,base,arm):
        super().__init__();assert arm in ['M2_STAGE_DELTA','M3_DELTA_SET'];self.arm=arm;self.base=base.eval()
        for p in self.base.parameters():p.requires_grad_(False)
        if arm=='M2_STAGE_DELTA':
            self.encoder=nn.Identity();self.head=nn.Sequential(nn.Linear(28,24),nn.GELU(),nn.Linear(24,1));self.k=7;expected=722
        else:
            self.encoder=nn.Sequential(nn.Linear(7,16),nn.GELU(),nn.Linear(16,8));self.head=nn.Sequential(nn.Linear(32,32),nn.GELU(),nn.Linear(32,1));self.k=8;expected=1354
        self.alpha=nn.Parameter(torch.zeros(()))
        assert sum(p.numel() for p in self.parameters() if p.requires_grad)==expected
        assert sum(p.numel() for p in self.base.parameters())==1537

    def train(self,mode=True):super().train(mode);self.base.eval();return self

    def forward(self,x,raw,mask,times,delta,scale,mode='full',perms=None):
        assert delta.shape==(*mask.shape,7) and torch.isfinite(delta[mask]).all()
        u=(condition(delta,mask,times,mode,perms)/scale).clamp(-8,8)
        if mode.startswith('neutral_'):u[...,int(mode.split('_')[1])]=0
        emb=u.new_zeros((*mask.shape,self.k));emb[mask]=self.encoder(u[mask])
        features=pool(emb,mask,times);aux=self.head(features).squeeze(-1)
        with torch.no_grad():base=self.base(x)
        res=.5*torch.tanh(self.alpha)*torch.tanh(aux)
        final=base if mode=='residual0' else base+res
        return final,aux,base,res,u[mask]
