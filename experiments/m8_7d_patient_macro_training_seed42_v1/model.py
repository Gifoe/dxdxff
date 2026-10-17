"""Retain the designed C2 architecture; compare patient-Macro training objectives."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS=['C2_REFERENCE','T0_GLOBAL_NORMALIZED','T1_PATIENT_BALANCED']


class SingleHead(nn.Module):
    def __init__(self,arm,cnn,gate,raw_function,head_seed):
        super().__init__();assert arm in ARMS;self.variant=arm
        self.cnn=copy.deepcopy(cnn);self.gate=copy.deepcopy(gate);self.raw_feature_function=raw_function
        self.register_buffer('keep7',torch.ones(7,dtype=torch.float32))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(head_seed);self.head=nn.Sequential(nn.Linear(110,64),nn.GELU(),nn.Linear(64,1))
        self.interaction_gain=nn.Parameter(torch.zeros(()))
        assert sum(p.numel() for p in self.parameters())==10378

    def features(self,raw,mask,times,delta,scale):
        return self.raw_feature_function(self,raw,mask,times,delta,scale,mode='full',perms=None)

    def classify(self,z,x):
        assert z.shape==(len(x),96) and x.shape[1]==14
        first=self.head[0]
        a=F.linear(z*math.sqrt(110/192),first.weight[:,:96])
        b=F.linear(x*math.sqrt(110/28),first.weight[:,96:])
        h=a+b+first.bias+.5*torch.tanh(self.interaction_gain)*a*torch.tanh(b)
        return self.head[2](self.head[1](h)).squeeze(-1)

    def forward(self,x,raw,mask,times,delta,scale):
        z,g=self.features(raw,mask,times,delta,scale)
        return dict(logit=self.classify(z,x),raw_summary=z,gate_activity=g)


def soft_macro_loss(logit,y):
    p=logit.sigmoid();a=2*(p*y).sum()/(p.sum()+y.sum()).clamp_min(1e-8)
    b=2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()).clamp_min(1e-8)
    return 1-.5*(a+b)


def normalized_bce(logit,y,pos,local):
    losses=F.binary_cross_entropy_with_logits(logit,y,reduction='none')
    if local:
        assert (y==0).any() and (y==1).any(),'Patient-local balanced training requires both original classes'
        return .5*(losses[y==0].mean()+losses[y==1].mean())
    weights=(1-y)+pos*y
    assert pos>0
    return (losses*weights).sum()/weights.sum()


def objective(logit,y,pos,arm):
    if arm=='C2_REFERENCE':
        return .75*F.binary_cross_entropy_with_logits(logit,y,pos_weight=pos)+.25*soft_macro_loss(logit,y)
    assert arm in ARMS[1:]
    return .5*normalized_bce(logit,y,pos,arm=='T1_PATIENT_BALANCED')+.5*soft_macro_loss(logit,y)
