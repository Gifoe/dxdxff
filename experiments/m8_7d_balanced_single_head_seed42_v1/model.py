"""Same seven measurements, one classifier, bounded hidden cross-interaction."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS=['C0_CONCAT','C1_BALANCED','C2_INTERACTION']


class SingleHead(nn.Module):
    def __init__(self,arm,cnn,gate,raw_function,head_seed):
        super().__init__();assert arm in ARMS;self.variant=arm
        self.cnn=copy.deepcopy(cnn);self.gate=copy.deepcopy(gate);self.raw_feature_function=raw_function
        self.register_buffer('keep7',torch.ones(7,dtype=torch.float32))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(head_seed)
            self.head=nn.Sequential(nn.Linear(110,64),nn.GELU(),nn.Linear(64,1))
        if arm=='C2_INTERACTION':self.interaction_gain=nn.Parameter(torch.zeros(()))
        assert sum(p.numel() for p in self.parameters())==10377+int(arm=='C2_INTERACTION')

    def features(self,raw,mask,times,delta,scale):
        return self.raw_feature_function(self,raw,mask,times,delta,scale,mode='full',perms=None)

    def classify(self,raw_summary,x):
        assert raw_summary.shape[-1]==96 and x.shape[-1]==14
        if self.variant=='C0_CONCAT':return self.head(torch.cat([raw_summary,x],-1)).squeeze(-1)
        # Fixed dimension balancing, not patient-specific recalibration or a new feature.
        first=self.head[0]
        a=F.linear(raw_summary*math.sqrt(110/192),first.weight[:,:96])
        b=F.linear(x*math.sqrt(110/28),first.weight[:,96:])
        hidden=a+b+first.bias
        if self.variant=='C2_INTERACTION':
            hidden=hidden+.5*torch.tanh(self.interaction_gain)*a*torch.tanh(b)
        return self.head[2](self.head[1](hidden)).squeeze(-1)

    def forward(self,x,raw,mask,times,delta,scale):
        z,g=self.features(raw,mask,times,delta,scale)
        return dict(logit=self.classify(z,x),raw_summary=z,gate_activity=g)


def objective(logit,y,pos):
    p=logit.sigmoid();a=2*(p*y).sum()/(p.sum()+y.sum()).clamp_min(1e-8)
    b=2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()).clamp_min(1e-8)
    return .75*F.binary_cross_entropy_with_logits(logit,y,pos_weight=pos)+.25*(1-.5*(a+b))
