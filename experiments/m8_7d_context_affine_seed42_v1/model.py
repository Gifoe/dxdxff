"""One scalar readout, context-dependent hidden gains/shift, same seven inputs."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS=['G0_GLOBAL','G1_PATIENT']


def hidden(body,z,x):
    first=body.head[0]
    a=F.linear(z*math.sqrt(110/192),first.weight[:,:96]);b=F.linear(x*math.sqrt(110/28),first.weight[:,96:])
    h=a+b+first.bias+.5*torch.tanh(body.interaction_gain)*a*torch.tanh(b)
    return body.head[1](h)


class Adjustment:
    def affine(self,h):
        if self.kind=='G0_GLOBAL':theta=self.offset[None]
        else:
            mean=h.double().mean(0);var=(h.double()-mean).square().mean(0)
            sd=torch.where(var>0,torch.sqrt(var.clamp_min(1e-30)),torch.zeros_like(var))
            stats=torch.cat([mean,sd]).to(h.dtype)[None];theta=self.context(stats)
        gain,shift=(.25*theta.tanh()).chunk(2,-1)
        return gain,shift

    def classify_hidden(self,h):
        assert h.ndim==2 and h.shape[1]==64 and len(h)>0
        gain,shift=self.affine(h)
        return self.readout(h*(1+gain)+shift).squeeze(-1)


class Decision(Adjustment,nn.Module):
    def __init__(self,kind,readout,seed):
        super().__init__();assert kind in ARMS;self.kind=kind;self.readout=copy.deepcopy(readout)
        for p in self.readout.parameters():p.requires_grad_(False)
        if kind=='G0_GLOBAL':self.offset=nn.Parameter(torch.zeros(128))
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.context=nn.Sequential(nn.Linear(128,8),nn.GELU(),nn.Linear(8,128))
                nn.init.zeros_(self.context[2].weight);nn.init.zeros_(self.context[2].bias)
        assert sum(p.numel() for p in self.parameters() if p.requires_grad)==(128 if kind=='G0_GLOBAL' else 2184)

    def forward(self,h):return self.classify_hidden(h)


class Fused(Adjustment,nn.Module):
    def __init__(self,body,decision):
        super().__init__();self.kind=decision.kind
        self.cnn=copy.deepcopy(body.cnn);self.gate=copy.deepcopy(body.gate);self.first=copy.deepcopy(body.head[0]);self.activation=copy.deepcopy(body.head[1])
        self.readout=copy.deepcopy(decision.readout);self.interaction_gain=nn.Parameter(body.interaction_gain.detach().clone(),requires_grad=False)
        self.register_buffer('keep7',body.keep7.detach().clone());self.raw_feature_function=body.raw_feature_function
        for p in self.parameters():p.requires_grad_(False)
        if self.kind=='G0_GLOBAL':self.offset=nn.Parameter(decision.offset.detach().clone())
        else:self.context=copy.deepcopy(decision.context)
        assert sum(p.numel() for p in self.parameters())==(10506 if self.kind=='G0_GLOBAL' else 12562)
        assert sum(isinstance(m,nn.Linear) and m.out_features==1 for m in self.modules())==1

    def features(self,raw,mask,times,delta,scale):
        return self.raw_feature_function(self,raw,mask,times,delta,scale,mode='full',perms=None)

    def forward(self,x,raw,mask,times,delta,scale):
        z,g=self.features(raw,mask,times,delta,scale)
        a=F.linear(z*math.sqrt(110/192),self.first.weight[:,:96]);b=F.linear(x*math.sqrt(110/28),self.first.weight[:,96:])
        h=self.activation(a+b+self.first.bias+.5*torch.tanh(self.interaction_gain)*a*torch.tanh(b))
        return dict(logit=self.classify_hidden(h),raw_summary=z,hidden=h,gate_activity=g)


def objective(logit,y,pos):
    p=logit.sigmoid();a=2*(p*y).sum()/(p.sum()+y.sum()).clamp_min(1e-8)
    b=2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()).clamp_min(1e-8)
    return .75*F.binary_cross_entropy_with_logits(logit,y,pos_weight=pos)+.25*(1-.5*(a+b))
