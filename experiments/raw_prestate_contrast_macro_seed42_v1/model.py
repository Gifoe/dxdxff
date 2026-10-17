"""One shared Raw encoder; keep the learned pre-state or discard it."""
import sys
import common as c
import torch
from torch import nn

pure=c.module('prestate_original_aggregate',c.parent.REFERENCE/'pure_model.py')
prior=sys.modules['common'];sys.modules['common']=c.wave
try:normalizer=c.module('prestate_exact_wave_normalizer',c.previous.WAVE_CODE/'model.py')
finally:sys.modules['common']=prior


def moments(values,valid):
    weight=valid[...,None].to(values.dtype);n=weight.sum(0);assert (n>0).all()
    mean=(values*weight).sum(0)/n
    variance=((values-mean[None]).square()*weight).sum(0)/n
    sd=torch.where(variance>0,variance.clamp_min(1e-30).sqrt(),torch.zeros_like(variance))
    return torch.stack([mean,sd],-2)


def views(emb,mask,times):
    delta,post,bases,phases,available=[],[],[],[],[]
    for s in range(len(emb)):
        pre=mask[s]&(times[s][None]<0);n=pre.sum(1);present=mask[s].any(1)
        assert (n[present]>0).all()
        baseline=(emb[s]*pre[...,None]).sum(1)/n.clamp_min(1)[:,None]
        latent=emb[s]-baseline[:,None];ds,ps,flags=[],[],[]
        for low,high in [(0,8),(8,20)]:
            use=mask[s]&((times[s]>=low)&(times[s]<high))[None];count=use.sum(1)
            ds.append((latent*use[...,None]).sum(1)/count.clamp_min(1)[:,None])
            ps.append((emb[s]*use[...,None]).sum(1)/count.clamp_min(1)[:,None]);flags.append(count>0)
        delta.append(torch.stack(ds,1));post.append(torch.stack(ps,1));phases.append(torch.stack(flags,1))
        bases.append(baseline);available.append(present)
    shape=(emb.shape[1],4*emb.shape[-1]);phase=torch.stack(phases)
    return (moments(torch.stack(delta),phase).reshape(shape),moments(torch.stack(post),phase).reshape(shape),
            moments(torch.stack(bases),torch.stack(available)).reshape(emb.shape[1],2*emb.shape[-1]))


def patient_z(vector):
    value=vector.double();scale=value.std(0,unbiased=False)
    return ((value-value.mean(0))/torch.where(scale>1e-8,scale,torch.ones_like(scale))).to(vector.dtype)


def aggregate(emb,mask,times,arm):
    delta,post,baseline=views(emb,mask,times)
    if arm=='DELTA_DUP':vector=torch.cat([delta,delta],-1)
    elif arm=='POST_DUP':vector=torch.cat([post,post],-1)
    else:
        assert arm=='PRE_DELTA';vector=torch.cat([delta,baseline,baseline],-1)
    return patient_z(vector)


class Model(nn.Module):
    def __init__(self,arm,fold):
        super().__init__();assert arm in c.TRAINED
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42+1009*fold)
            self.cnn=nn.Sequential(nn.Conv1d(1,12,9,4),nn.GELU(),nn.Conv1d(12,24,7,4),nn.GELU(),nn.AdaptiveAvgPool1d(1))
            self.head=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Linear(64,1))
        self.arm=arm
    def forward(self,raw,mask,times):
        flat=normalizer.reference_wave(raw,mask,times)[0]
        h=torch.cat([self.cnn(q[:,None]).flatten(1) for q in flat.split(4096)])
        emb=raw.new_zeros((*mask.shape,24));emb[mask]=h
        return self.head(aggregate(emb,mask,times,self.arm)).squeeze(-1)


def objective(logit,y,pos):
    bce=nn.functional.binary_cross_entropy_with_logits(logit,y,pos_weight=pos);p=logit.sigmoid()
    a=2*(p*y).sum()/(p.sum()+y.sum()).clamp_min(1e-8)
    b=2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()).clamp_min(1e-8)
    return .5*bce+.5*(1-.5*(a+b))
