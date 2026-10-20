"""Mask unsupported scorer inputs BEFORE matmul: 0*NaN must not poison dW."""
import torch

def attention_mean(latent,times,valid,weight):
    tau=times/20.
    temporal=torch.stack((tau,tau.square()),-1).expand(latent.shape[0],-1,-1)
    features=torch.cat((latent,temporal),-1)
    features=torch.where(valid[...,None],features,torch.zeros_like(features))
    score=features@weight
    present=valid.any(-1)
    score=torch.where(valid,score,torch.full_like(score,-torch.inf))
    score=torch.where(present[:,None],score,torch.zeros_like(score))
    weights=torch.softmax(score,-1)*valid.to(latent.dtype)
    result=(torch.where(valid[...,None],latent,torch.zeros_like(latent))*weights[...,None]).sum(-2)
    return result,present,weights
