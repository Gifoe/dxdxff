"""Complete-patient versus within-channel contexts; no labels or fitted updates."""
import torch


def patient_z(x):
    a=x.double();sd=a.std(0,unbiased=False)
    return ((a-a.mean(0))/torch.where(sd>1e-8,sd,torch.ones_like(sd))).to(x.dtype)


def channel_z(x):
    a=x.double();sd=a.std(-1,unbiased=False,keepdim=True)
    return ((a-a.mean(-1,keepdim=True))/torch.where(sd>1e-8,sd,torch.ones_like(sd))).to(x.dtype)


def views(vector,mean,scale,raw_pr=True,global_view=True):
    global_raw=((vector.double()-mean)/scale).to(vector.dtype)
    relative=patient_z(vector) if raw_pr else channel_z(global_raw)
    first=global_raw if global_view else channel_z(relative)
    return torch.cat([first,relative],dim=-1)
