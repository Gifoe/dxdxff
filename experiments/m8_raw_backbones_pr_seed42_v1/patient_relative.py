"""Exact M8 cross-channel population-Z convention, with an explicit mask."""
import torch


def patient_relative(x, valid=None):
    assert x.ndim==2 and torch.isfinite(x).all()
    valid=torch.ones(x.shape[0],dtype=torch.bool,device=x.device) if valid is None else valid
    assert valid.shape==(x.shape[0],) and valid.any()
    v=x[valid].double();sd=v.std(0,unbiased=False)
    z=(v-v.mean(0))/torch.where(sd>1e-8,sd,torch.ones_like(sd))
    out=torch.zeros_like(x);out[valid]=z.to(x.dtype)
    assert torch.isfinite(out).all()
    return out
