"""Exact frozen per-seizure sufficient statistics; never average probabilities."""
import torch

def own_pool(summary,valid,pr):
    valid=valid[...,None].to(summary.dtype);count=valid.sum(0);assert (count>0).all()
    mean=(summary*valid).sum(0)/count
    var=((summary-mean[None]).square()*valid).sum(0)/count
    sd=torch.where(var>0,var.clamp_min(1e-30).sqrt(),torch.zeros_like(var))
    v=torch.stack([mean,sd],2).reshape(summary.shape[1],96).double()
    if pr:
        sd=v.std(0,unbiased=False);v=(v-v.mean(0))/torch.where(sd>1e-8,sd,torch.ones_like(sd))
    return v.to(summary.dtype)

def time_pool(seizure,has):
    count=has.sum(0);assert (count>0).all()
    mean=(seizure*has).sum(0)/count
    var=((seizure-mean[None]).square()*has).sum(0)/count
    sd=torch.where(var>0,var.clamp_min(1e-30).sqrt(),torch.zeros_like(var))
    return torch.stack([mean,sd],-1).flatten(1)

def capture_own(model,raw,mask,times,delta,scale):
    original=torch.stack;captured={};S,C=raw.shape[:2]
    def stack(values,*args,**kwargs):
        out=original(values,*args,**kwargs)
        dim=args[0] if args else kwargs.get('dim',0)
        if dim==0 and tuple(out.shape)==(S,C,2,24):captured['summary']=out.detach().clone()
        if dim==0 and tuple(out.shape)==(S,C,2) and out.dtype==torch.bool:captured['valid']=out.detach().clone()
        return out
    torch.stack=stack
    try:
        with torch.inference_mode():direct,_=model.features(raw,mask,times,delta,scale)
    finally:torch.stack=original
    assert set(captured)=={'summary','valid'}
    cached=own_pool(captured['summary'],captured['valid'],model.raw_pr)
    assert torch.equal(direct,cached),'Own All cached/direct representation differs'
    return captured,direct

def capture_time(engine,model,raw,mask):
    original=engine.aggregate;captured={}
    def aggregate(windows,mask):
        v=windows.float();m=mask[...,None].to(v.dtype);n=m.sum(2)
        captured['seizure']=((v*m).sum(2)/n.clamp_min(1)).detach().clone();captured['has']=(n>0).detach().clone()
        return original(windows,mask)
    engine.aggregate=aggregate
    try:
        with torch.inference_mode():logit,direct=engine.forward(model,raw,mask,128,False)
    finally:engine.aggregate=original
    assert torch.equal(direct,time_pool(captured['seizure'],captured['has'])),'TimeConv All cached/direct representation differs'
    return captured,direct,logit
