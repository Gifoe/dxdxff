"""Single-switch M8 mechanisms, preserving original arithmetic when enabled."""
import torch
from torch import nn
from direct_correction import MorphCorrection
from dgm_net import transform


class IdentityGate(nn.Module):
    def forward(self,u):
        return u.new_ones(*u.shape[:-1],24)


class M8Mechanism(MorphCorrection):
    def __init__(self,base,raw_pr=True,pre_reference=True,identity_gate=False):
        super().__init__(base)
        self.raw_pr=raw_pr;self.pre_reference=pre_reference
        if identity_gate:self.gate=IdentityGate()

    def features(self,raw,mask,times,delta,scale,mode='full',perms=None):
        S,C,W,L=raw.shape
        assert L==500 and mask.shape==(S,C,W) and times.shape==(S,W) and delta.shape==(S,C,W,7)
        assert not delta.requires_grad and torch.isfinite(delta[mask]).all()
        flat=raw[mask]
        v=((flat-flat.mean(-1,keepdim=True))/flat.std(-1,unbiased=False,keepdim=True).clamp_min(1e-6)).clamp(-8,8)
        emb=raw.new_zeros(S,C,W,24)
        emb[mask]=torch.cat([self.cnn(q[:,None]).flatten(1) for q in v.split(4096)])
        phys=transform(delta,mask,times,mode,perms);u=(phys/scale).clamp(-8,8)
        if mode.startswith('neutral_'):u[...,int(mode.split('_')[1])]=0
        summaries=[];available=[];activity=[]
        for s in range(S):
            pre=mask[s]&(times[s][None]<0);count=pre.sum(1);present=mask[s].any(1)
            assert (count[present]>0).all()
            if self.pre_reference:
                hpre=(emb[s]*pre[...,None]).sum(1)/count.clamp_min(1)[:,None]
                latent=emb[s]-hpre[:,None]
            else:latent=emb[s]
            post=mask[s]&((times[s]>=0)&(times[s]<20))[None]
            g=torch.ones_like(latent) if mode=='identity' else self.gate(u[s])
            activity.append(g[post]);guided=g*latent;st=[];av=[]
            for low,high in [(0,8),(8,20)]:
                select=mask[s]&((times[s]>=low)&(times[s]<high))[None];n=select.sum(1)
                st.append((guided*select[...,None]).sum(1)/n.clamp_min(1)[:,None]);av.append(n>0)
            summaries.append(torch.stack(st,1));available.append(torch.stack(av,1))
        summary=torch.stack(summaries);valid=torch.stack(available)[...,None].to(raw.dtype);n=valid.sum(0)
        assert (n>0).all()
        mu=(summary*valid).sum(0)/n;var=((summary-mu[None]).square()*valid).sum(0)/n
        sd=torch.where(var>0,torch.sqrt(var.clamp_min(1e-30)),torch.zeros_like(var))
        vector=torch.stack([mu,sd],2).reshape(C,96).double()
        if self.raw_pr:
            std=vector.std(0,unbiased=False)
            vector=(vector-vector.mean(0))/torch.where(std>1e-8,std,torch.ones_like(std))
        return vector.to(raw.dtype),torch.cat(activity)
