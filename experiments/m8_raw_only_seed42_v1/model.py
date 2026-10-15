"""M8 Raw morphology path alone: no physiology/base network or feature interface."""
import copy
import torch
from torch import nn


class M8RawOnly(nn.Module):
    def __init__(self, original_initial_model):
        super().__init__()
        self.cnn=copy.deepcopy(original_initial_model.cnn)
        self.head=copy.deepcopy(original_initial_model.head)
        assert sum(p.numel() for p in self.parameters())==8433
        assert all(p.requires_grad for p in self.parameters())

    def features(self,raw,mask,times):
        S,C,W,L=raw.shape
        assert L==500 and mask.shape==(S,C,W) and times.shape==(S,W)
        assert torch.isfinite(raw).all() and (mask.sum((0,2))>0).all()
        flat=raw[mask]
        v=((flat-flat.mean(-1,keepdim=True))/flat.std(-1,unbiased=False,keepdim=True).clamp_min(1e-6)).clamp(-8,8)
        emb=raw.new_zeros(S,C,W,24)
        emb[mask]=torch.cat([self.cnn(q[:,None]).flatten(1) for q in v.split(4096)])
        summaries=[];available=[]
        for s in range(S):
            pre=mask[s]&(times[s][None]<0);count=pre.sum(1);present=mask[s].any(1)
            assert (count[present]>0).all()
            hpre=(emb[s]*pre[...,None]).sum(1)/count.clamp_min(1)[:,None]
            latent=emb[s]-hpre[:,None]
            st=[];av=[]
            for low,high in [(0,8),(8,20)]:
                select=mask[s]&((times[s]>=low)&(times[s]<high))[None];n=select.sum(1)
                st.append((latent*select[...,None]).sum(1)/n.clamp_min(1)[:,None]);av.append(n>0)
            summaries.append(torch.stack(st,1));available.append(torch.stack(av,1))
        summary=torch.stack(summaries);valid=torch.stack(available)[...,None].to(raw.dtype);n=valid.sum(0)
        assert (n>0).all()
        mu=(summary*valid).sum(0)/n
        var=((summary-mu[None]).square()*valid).sum(0)/n
        sd=torch.where(var>0,torch.sqrt(var.clamp_min(1e-30)),torch.zeros_like(var))
        vector=torch.stack([mu,sd],2).reshape(C,96).double()
        std=vector.std(0,unbiased=False)
        z=(vector-vector.mean(0))/torch.where(std>1e-8,std,torch.ones_like(std))
        return z.to(raw.dtype)

    def forward(self,raw,mask,times):
        return torch.tanh(self.head(self.features(raw,mask,times)).squeeze(-1))
