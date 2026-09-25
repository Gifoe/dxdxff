from __future__ import annotations
import torch

def _mask(mask,x):
    while mask.ndim<x.ndim:mask=mask.unsqueeze(-1)
    return mask.bool()
def masked_softmax(logits,mask,dim=1):
    m=_mask(mask,logits);values=logits.float().masked_fill(~m,-torch.inf)
    if (~m).all(dim=dim).any():raise ValueError("all-padding masked_softmax slice")
    weights=torch.softmax(values,dim=dim).masked_fill(~m,0.0)
    return weights.to(logits.dtype)
def masked_mean(x,mask,dim):
    m=_mask(mask,x);return (x.float()*m).sum(dim)/m.sum(dim).clamp_min(1)
def masked_max(x,mask,dim):
    m=_mask(mask,x);result=x.float().masked_fill(~m,-torch.inf).amax(dim)
    if torch.isinf(result).any():raise ValueError("all-padding masked_max slice")
    return result
def masked_std(x,mask,dim):
    m=_mask(mask,x);mean=masked_mean(x,mask,dim);count=m.sum(dim)
    expanded=mean.unsqueeze(dim);var=(((x.float()-expanded)**2)*m).sum(dim)/count.clamp_min(1)
    return torch.where(count>1,var.sqrt(),torch.zeros_like(var))
