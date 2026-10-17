"""Original seven-coordinate physiology modulates Raw before/after first GELU."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS=['C2_REFERENCE','F0_AFTER_GELU','F1_BEFORE_GELU']


class ConditionedCNN(nn.Module):
    def __init__(self,cnn,kind,seed):
        super().__init__();assert kind in ARMS[1:];self.kind=kind
        self.stem=copy.deepcopy(nn.Sequential(*list(cnn.children())[:-1]))
        self.pool=copy.deepcopy(cnn[-1])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.conditioner=nn.Sequential(nn.Linear(7,32),nn.GELU(),nn.Linear(32,24))
        nn.init.zeros_(self.conditioner[-1].weight);nn.init.zeros_(self.conditioner[-1].bias)

    def forward(self,x,u):
        assert x.ndim==3 and x.shape[1:]==(1,500) and u.shape==(len(x),7)
        gamma,beta=self.conditioner(u).chunk(2,-1)
        gamma=1+.25*gamma.tanh()[...,None];beta=.25*beta.tanh()[...,None]
        h=self.stem[0](x)
        if self.kind=='F1_BEFORE_GELU':h=self.stem[1](gamma*h+beta)
        else:h=gamma*self.stem[1](h)+beta
        return self.pool(self.stem[3](self.stem[2](h)))


class SingleHead(nn.Module):
    def __init__(self,arm,cnn,gate,raw_function,head_seed):
        super().__init__();assert arm in ARMS;self.variant=arm
        self.cnn=copy.deepcopy(cnn) if arm==ARMS[0] else ConditionedCNN(cnn,arm,head_seed+8011)
        self.gate=copy.deepcopy(gate);self.raw_feature_function=raw_function
        self.register_buffer('keep7',torch.ones(7,dtype=torch.float32))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(head_seed)
            self.head=nn.Sequential(nn.Linear(110,64),nn.GELU(),nn.Linear(64,1))
        self.interaction_gain=nn.Parameter(torch.zeros(()))
        assert sum(p.numel() for p in self.parameters())==10378+(1048 if arm!=ARMS[0] else 0)

    def features(self,raw,mask,times,delta,scale):
        if self.variant==ARMS[0]:
            return self.raw_feature_function(self,raw,mask,times,delta,scale,mode='full',perms=None)
        S,C,W,L=raw.shape
        assert L==500 and mask.shape==(S,C,W) and times.shape==(S,W) and delta.shape==(S,C,W,7)
        assert not delta.requires_grad and torch.isfinite(delta[mask]).all()
        flat=raw[mask]
        v=((flat-flat.mean(-1,keepdim=True))/flat.std(-1,unbiased=False,keepdim=True).clamp_min(1e-6)).clamp(-8,8)
        transform=self.raw_feature_function.__globals__['transform']
        phys=transform(delta,mask,times,'full',None);u=(phys/scale).clamp(-8,8)
        emb=raw.new_zeros(S,C,W,24)
        emb[mask]=torch.cat([self.cnn(q[:,None],p).flatten(1) for q,p in zip(v.split(4096),u[mask].split(4096))])
        summaries=[];available=[];activity=[]
        for s in range(S):
            pre=mask[s]&(times[s][None]<0);count=pre.sum(1);present=mask[s].any(1)
            assert (count[present]>0).all(),'No genuine pre-reference'
            hpre=(emb[s]*pre[...,None]).sum(1)/count.clamp_min(1)[:,None]
            latent=emb[s]-hpre[:,None]
            post=mask[s]&((times[s]>=0)&(times[s]<20))[None]
            g=self.gate(u[s]);activity.append(g[post]);guided=g*latent;st=[];av=[]
            for low,high in [(0,8),(8,20)]:
                select=mask[s]&((times[s]>=low)&(times[s]<high))[None];n=select.sum(1)
                st.append((guided*select[...,None]).sum(1)/n.clamp_min(1)[:,None]);av.append(n>0)
            summaries.append(torch.stack(st,1));available.append(torch.stack(av,1))
        summary=torch.stack(summaries);valid=torch.stack(available)[...,None].to(raw.dtype);n=valid.sum(0)
        assert (n>0).all(),'Missing stage cannot be silently imputed'
        mu=(summary*valid).sum(0)/n
        var=((summary-mu[None]).square()*valid).sum(0)/n
        sd=torch.where(var>0,torch.sqrt(var.clamp_min(1e-30)),torch.zeros_like(var))
        vector=torch.stack([mu,sd],2).reshape(C,96).double()
        z=(vector-vector.mean(0))/torch.where(vector.std(0,unbiased=False)>1e-8,vector.std(0,unbiased=False),torch.ones_like(vector.std(0,unbiased=False)))
        return z.to(raw.dtype),torch.cat(activity)

    def classify(self,z,x):
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
