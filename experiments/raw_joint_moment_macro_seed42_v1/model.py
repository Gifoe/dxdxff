"""Learned-response first/second moments; one Raw input and one classifier."""
import sys
import common as c
import torch
from torch import nn

pure=c.module('joint_moment_original_aggregate',c.parent.REFERENCE/'pure_model.py')
prior=sys.modules['common'];sys.modules['common']=c.wave
try:normalizer=c.module('joint_moment_exact_pre_normalizer',c.previous.WAVE_CODE/'model.py')
finally:sys.modules['common']=prior


class MomentPool(nn.Module):
    def __init__(self,arm):super().__init__();self.arm=arm
    def forward(self,h):
        mean=h.mean(-1,keepdim=True);energy=torch.log1p(h.square().mean(-1,keepdim=True))
        if self.arm=='PRE_MEAN_DUP':return torch.cat([mean,mean],1)
        if self.arm=='PRE_ENERGY_DUP':return torch.cat([energy,energy],1)
        assert self.arm=='PRE_JOINT';return torch.cat([mean,energy],1)


def aggregate(emb,mask,times):
    # Original exact arithmetic; only the feature dimension is generalized.
    summaries,available=[],[]
    for s in range(len(emb)):
        pre=mask[s]&(times[s][None]<0);count,present=pre.sum(1),mask[s].any(1)
        assert (count[present]>0).all()
        reference=(emb[s]*pre[...,None]).sum(1)/count.clamp_min(1)[:,None]
        latent=emb[s]-reference[:,None];stages,flags=[],[]
        for low,high in [(0,8),(8,20)]:
            use=mask[s]&((times[s]>=low)&(times[s]<high))[None];n=use.sum(1)
            stages.append((latent*use[...,None]).sum(1)/n.clamp_min(1)[:,None]);flags.append(n>0)
        summaries.append(torch.stack(stages,1));available.append(torch.stack(flags,1))
    summary=torch.stack(summaries);valid=torch.stack(available)[...,None].to(emb.dtype)
    n=valid.sum(0);assert (n>0).all();mean=(summary*valid).sum(0)/n
    variance=((summary-mean[None]).square()*valid).sum(0)/n
    sd=torch.where(variance>0,variance.clamp_min(1e-30).sqrt(),torch.zeros_like(variance))
    vector=torch.stack([mean,sd],2).reshape(emb.shape[1],4*emb.shape[-1]).double()
    scale=vector.std(0,unbiased=False)
    return ((vector-vector.mean(0))/torch.where(scale>1e-8,scale,torch.ones_like(scale))).to(emb.dtype)


class Model(nn.Module):
    def __init__(self,arm,fold):
        super().__init__();assert arm in c.TRAINED
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42+1009*fold)
            self.cnn=nn.Sequential(nn.Conv1d(1,12,9,4),nn.GELU(),nn.Conv1d(12,24,7,4),nn.GELU(),MomentPool(arm))
            self.head=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Linear(64,1))
        self.arm=arm
    def forward(self,raw,mask,times):
        flat=normalizer.reference_wave(raw,mask,times)[0]
        h=torch.cat([self.cnn(q[:,None]).flatten(1) for q in flat.split(4096)])
        emb=raw.new_zeros((*mask.shape,48));emb[mask]=h
        return self.head(aggregate(emb,mask,times)).squeeze(-1)


def objective(logit,y,pos):
    bce=nn.functional.binary_cross_entropy_with_logits(logit,y,pos_weight=pos);p=logit.sigmoid()
    a=2*(p*y).sum()/(p.sum()+y.sum()).clamp_min(1e-8)
    b=2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()).clamp_min(1e-8)
    return .5*bce+.5*(1-.5*(a+b))
