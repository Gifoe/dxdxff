"""Raw temporal dispersion versus matched mean-only capacity; one readout."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS=['C2_REFERENCE','R0_MEAN_CAPACITY','R1_RAW_DISPERSION']


class MomentCNN(nn.Module):
    def __init__(self,cnn,kind,seed):
        super().__init__();assert kind in ARMS[1:];self.kind=kind
        self.stem=copy.deepcopy(nn.Sequential(*list(cnn.children())[:-1]))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);self.projection=nn.Linear(24,24)
        nn.init.zeros_(self.projection.weight);nn.init.zeros_(self.projection.bias)

    def pool(self,h):
        mu=h.mean(-1)
        if self.kind=='R0_MEAN_CAPACITY':stat=mu
        else:
            var=(h-mu[...,None]).square().mean(-1)
            stat=torch.where(var>0,var.clamp_min(1e-30).sqrt(),torch.zeros_like(var))
        return mu+.25*torch.tanh(self.projection(stat))

    def forward(self,x):return self.pool(self.stem(x))[...,None]


class SingleHead(nn.Module):
    def __init__(self,arm,cnn,gate,raw_function,head_seed):
        super().__init__();assert arm in ARMS;self.variant=arm
        self.cnn=copy.deepcopy(cnn) if arm==ARMS[0] else MomentCNN(cnn,arm,head_seed+8011)
        self.gate=copy.deepcopy(gate);self.raw_feature_function=raw_function
        self.register_buffer('keep7',torch.ones(7,dtype=torch.float32))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(head_seed)
            self.head=nn.Sequential(nn.Linear(110,64),nn.GELU(),nn.Linear(64,1))
        self.interaction_gain=nn.Parameter(torch.zeros(()))
        assert sum(p.numel() for p in self.parameters())==10378+(600 if arm!=ARMS[0] else 0)

    def features(self,raw,mask,times,delta,scale):
        return self.raw_feature_function(self,raw,mask,times,delta,scale,mode='full',perms=None)

    def classify(self,z,x):
        assert z.shape[-1]==96 and x.shape[-1]==14
        first=self.head[0]
        a=F.linear(z*math.sqrt(110/192),first.weight[:,:96])
        b=F.linear(x*math.sqrt(110/28),first.weight[:,96:])
        h=a+b+first.bias+.5*torch.tanh(self.interaction_gain)*a*torch.tanh(b)
        return self.head[2](self.head[1](h)).squeeze(-1)

    def forward(self,x,raw,mask,times,delta,scale):
        z,g=self.features(raw,mask,times,delta,scale)
        return dict(logit=self.classify(z,x),raw_summary=z,gate_activity=g)


def objective(logit,y,pos):
    p=logit.sigmoid();a=2*(p*y).sum()/(p.sum()+y.sum()).clamp_min(1e-8)
    b=2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()).clamp_min(1e-8)
    return .75*F.binary_cross_entropy_with_logits(logit,y,pos_weight=pos)+.25*(1-.5*(a+b))
