"""Shared frozen Raw CNN, two waveform normalizations, one classifier."""
import math
import torch
from torch import nn
from common import module,parent

pure=module('precompand_pinned_pure_raw',parent.REFERENCE/'pure_model.py')
ARMS=['NATIVE','WINDOW_RESIDUAL','PRECOMPAND_RESIDUAL']
TRAINED=ARMS[1:]


def reference_wave(raw,mask,times):
    flat=raw[mask].double();centered=flat-flat.mean(-1,keepdim=True)
    variance=centered.square().mean(-1)
    dense=torch.zeros(mask.shape,device=raw.device,dtype=torch.float64);dense[mask]=variance
    pre=mask&(times[:,None,:]<0);count=pre.sum(-1);present=mask.any(-1)
    assert (count[present]>0).all(),'No genuine pre-onset reference for an available channel'
    rms=((dense*pre).sum(-1)/count.clamp_min(1)).sqrt()
    positive=rms[present&(rms>0)];assert len(positive)>0,'All pre windows constant; reference view unidentified'
    fallback=positive.median();scales=[];fallbacks=0
    for s in range(len(raw)):
        values=rms[s][present[s]&(rms[s]>0)]
        scales.append(values.median() if len(values) else fallback);fallbacks+=int(not len(values))
    scale=torch.stack(scales)
    divisor=scale[:,None,None].expand_as(mask)[mask]
    before=torch.asinh(centered/divisor[:,None])/math.asinh(1.)
    audit=dict(zero_pre_channels=int((present&(rms==0)).sum()),present_channel_seizure_units=int(present.sum()),
               fallback_seizures=fallbacks,encoded_samples=before.numel(),clipped_samples=int((before.abs()>8).sum()))
    assert torch.isfinite(before).all()
    return before.clamp(-8,8).float(),audit


def phase_summary(emb,mask,times):
    values=[];flags=[]
    for s in range(len(emb)):
        pre=mask[s]&(times[s][None]<0);count=pre.sum(1);present=mask[s].any(1)
        assert (count[present]>0).all()
        reference=(emb[s]*pre[...,None]).sum(1)/count.clamp_min(1)[:,None]
        difference=emb[s]-reference[:,None]
        stages=[];available=[]
        for low,high in [(0,8),(8,20)]:
            use=mask[s]&((times[s]>=low)&(times[s]<high))[None];n=use.sum(1)
            stages.append((difference*use[...,None]).sum(1)/n.clamp_min(1)[:,None]);available.append(n>0)
        values.append(torch.stack(stages,1));flags.append(torch.stack(available,1))
    return torch.stack(values),torch.stack(flags)


def aggregate(summary,available):
    valid=available[...,None].to(summary.dtype);n=valid.sum(0);assert (n>0).all()
    mean=(summary*valid).sum(0)/n
    variance=((summary-mean[None]).square()*valid).sum(0)/n
    sd=torch.where(variance>0,variance.clamp_min(1e-30).sqrt(),torch.zeros_like(variance))
    vector=torch.stack([mean,sd],2).reshape(summary.shape[1],96).double()
    scale=vector.std(0,unbiased=False)
    return ((vector-vector.mean(0))/torch.where(scale>1e-8,scale,torch.ones_like(scale))).float()


class Model(pure.PureRaw):
    def __init__(self,arm,fold,state):
        assert arm in ARMS
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42+1009*fold);super().__init__(compact=False);self.load_reference(state)
        self.requires_grad_(False);self.arm=arm
        if arm!='NATIVE':self.matrix=nn.Parameter(torch.zeros(24,24))

    def encode(self,raw,mask,times):
        with torch.no_grad():
            w=self.normalized_windows(raw,mask,times);a,audit=reference_wave(raw,mask,times)
            result={};flags=None
            for name,flat in [('window',w),('reference',a)]:
                emb=raw.new_zeros((*mask.shape,24))
                emb[mask]=torch.cat([self.cnn(q[:,None]).flatten(1) for q in flat.split(4096)])
                summary,valid=phase_summary(emb,mask,times);result[name]=summary
                if flags is not None:assert torch.equal(flags,valid)
                flags=valid
            result['available']=flags
        return result,audit

    def forward_stats(self,q):
        h=q['window']
        if self.arm!='NATIVE':
            other=q['window'] if self.arm=='WINDOW_RESIDUAL' else q['reference']
            h=h+nn.functional.linear(other,.25*self.matrix.tanh())
        return self.classify(aggregate(h,q['available']))

    def forward(self,raw,mask,times):
        q,_=self.encode(raw,mask,times);return self.forward_stats(q)


def objective(logit,y,pos):
    bce=nn.functional.binary_cross_entropy_with_logits(logit,y,pos_weight=pos);p=logit.sigmoid()
    f0=2*(p*y).sum()/(p.sum()+y.sum()).clamp_min(1e-8)
    f1=2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()).clamp_min(1e-8)
    return .75*bce+.25*(1-.5*(f0+f1))
