"""Parameter-matched pointwise versus channel-relational hidden blocks; one head."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS=['C2_REFERENCE','P0_POINTWISE','P1_RELATIONAL']


class Pointwise(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Linear(64,24,bias=False);self.output=nn.Linear(24,64,bias=False)
        nn.init.zeros_(self.output.weight)
    def forward(self,h):return .25*torch.tanh(self.output(F.gelu(self.input(h))))


class Relational(nn.Module):
    def __init__(self):
        super().__init__();self.query=nn.Linear(64,8,bias=False);self.key=nn.Linear(64,8,bias=False)
        self.value=nn.Linear(64,16,bias=False);self.output=nn.Linear(16,64,bias=False)
        nn.init.zeros_(self.output.weight)

    def attention(self,h):
        n=len(h)
        if n==1:return h.new_zeros(1,1)
        score=self.query(h)@self.key(h).T/math.sqrt(8)
        score=score.masked_fill(torch.eye(n,device=h.device,dtype=torch.bool),float('-inf'))
        return score.softmax(-1)

    def forward(self,h):
        v=F.gelu(self.value(h));a=self.attention(h)
        # No neighbors is an identity fallback, not label-dependent exclusion.
        difference=v-a@v if len(h)>1 else v*0
        return .25*torch.tanh(self.output(difference))


class SingleHead(nn.Module):
    def __init__(self,arm,cnn,gate,raw_function,head_seed):
        super().__init__();assert arm in ARMS;self.variant=arm
        self.cnn=copy.deepcopy(cnn);self.gate=copy.deepcopy(gate);self.raw_feature_function=raw_function
        self.register_buffer('keep7',torch.ones(7,dtype=torch.float32))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(head_seed);self.head=nn.Sequential(nn.Linear(110,64),nn.GELU(),nn.Linear(64,1))
        self.interaction_gain=nn.Parameter(torch.zeros(()))
        if arm!=ARMS[0]:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(head_seed+8011)
                self.mixer=Pointwise() if arm=='P0_POINTWISE' else Relational()
            assert sum(p.numel() for p in self.mixer.parameters())==3072
        assert sum(p.numel() for p in self.parameters())==10378+(3072 if arm!=ARMS[0] else 0)

    def features(self,raw,mask,times,delta,scale):
        return self.raw_feature_function(self,raw,mask,times,delta,scale,mode='full',perms=None)

    def hidden(self,z,x):
        assert z.ndim==x.ndim==2 and z.shape==(len(x),96) and x.shape[1]==14
        first=self.head[0]
        a=F.linear(z*math.sqrt(110/192),first.weight[:,:96])
        b=F.linear(x*math.sqrt(110/28),first.weight[:,96:])
        return self.head[1](a+b+first.bias+.5*torch.tanh(self.interaction_gain)*a*torch.tanh(b))

    def classify(self,z,x):
        h=self.hidden(z,x)
        if self.variant!=ARMS[0]:h=h+self.mixer(h)
        return self.head[2](h).squeeze(-1)

    def forward(self,x,raw,mask,times,delta,scale):
        z,g=self.features(raw,mask,times,delta,scale)
        return dict(logit=self.classify(z,x),raw_summary=z,gate_activity=g)


def objective(logit,y,pos):
    p=logit.sigmoid();a=2*(p*y).sum()/(p.sum()+y.sum()).clamp_min(1e-8)
    b=2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()).clamp_min(1e-8)
    return .75*F.binary_cross_entropy_with_logits(logit,y,pos_weight=pos)+.25*(1-.5*(a+b))
