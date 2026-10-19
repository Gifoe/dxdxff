"""Nonlinear residual comparisons; no labels, identifiers, or fitted feature source."""
import torch

def patient_z(x):
    """All patient channels, population SD; original constant-coordinate fallback."""
    a=x.double();sd=a.std(0,unbiased=False)
    return ((a-a.mean(0))/torch.where(sd>1e-8,sd,torch.ones_like(sd))).to(x.dtype)

def channel_z(x):
    """Parameter-free LayerNorm; only this channel's hidden coordinates are used."""
    a=x.double();sd=a.std(-1,unbiased=False,keepdim=True)
    return ((a-a.mean(-1,keepdim=True))/torch.where(sd>1e-8,sd,torch.ones_like(sd))).to(x.dtype)

def residual_hidden(head,a,b,gain_raw,gain_feature,raw_context=True,feature_context=True):
    """Preserve original hidden state exactly at zero gains; one final head only."""
    bias=head[0].bias;activation=head[1]
    h=activation(a+b+bias)
    ur=activation(a+bias/2);uf=activation(b+bias/2)
    qr=(patient_z if raw_context else channel_z)(ur)
    qf=(patient_z if feature_context else channel_z)(uf)
    return h+.25*torch.tanh(gain_raw)*qr+.25*torch.tanh(gain_feature)*qf
