"""Exactly B0/B1/B2/B3; original network supplied by the hash-checked runtime."""
import numpy as np
import torch
from torch import nn


def magnitude(x, eps=1e-5):
    u=x.to(torch.float64)
    return torch.stack([u.mean(-1),.5*torch.log(u.var(-1,unbiased=False)+eps)],dim=-1)


def fit_statistics(x, patient, train):
    s=magnitude(torch.from_numpy(x)).numpy()
    assert np.isfinite(s).all()
    ids=np.unique(patient[train]);means=[];weights=np.zeros(len(x))
    for p in ids:
        ix=train[patient[train]==p];weights[ix]=1./(len(ids)*len(ix));means.append(s[ix].mean(0))
    mean=(s*weights[:,None]).sum(0)
    variance=((s-mean)**2*weights[:,None]).sum(0)
    assert (variance>1e-12).all(),'Constant eligible FIT magnitude statistics'
    scale=np.maximum(np.sqrt(variance),1e-8)
    within=np.mean([s[train[patient[train]==p]].var(0) for p in ids],axis=0)
    between=np.asarray(means).var(0)
    assert np.allclose(within+between,variance,atol=1e-12,rtol=1e-12)
    return {'mean':mean,'scale':scale,'variance':variance,'within_patient_variance':within,
        'between_patient_variance':between,'FIT_patients':len(ids),'FIT_channels':len(train)},s,weights


def constructor(original, arm, stats):
    assert arm in ['B0','B1','B2','B3']
    class Model(nn.Module):
        def __init__(self,seed=42):
            super().__init__();self.arm=arm
            self.network=original(seed).network
            if arm=='B1':self.network[0]=nn.Identity()
            self.eps=1e-5
            if arm in ['B2','B3']:
                self.hidden=nn.Linear(96,32)
                if arm=='B3':self.statistics=nn.Linear(2,32)
                self.output=nn.Linear(32,96)
                nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
                if arm=='B2':self.q=nn.Parameter(torch.zeros(96))
            if arm=='B3':
                self.register_buffer('stat_mean',torch.as_tensor(stats['mean'],dtype=torch.float64))
                self.register_buffer('stat_scale',torch.as_tensor(stats['scale'],dtype=torch.float64))
            assert sum(p.numel() for p in self.parameters())=={'B0':8817,'B1':8641,'B2':15185,'B3':15185}[arm]

        def normalized_stats(self,x):
            return ((magnitude(x,self.eps)-self.stat_mean)/self.stat_scale).to(x.dtype)

        def detail(self,x,stats_override=None,disable=False):
            h=self.network[:3](x)
            delta=torch.zeros_like(h)
            if not disable and self.arm in ['B2','B3']:
                a=torch.nn.functional.gelu(self.hidden(h))
                if self.arm=='B3':
                    s=self.normalized_stats(x) if stats_override is None else stats_override
                    delta=self.output(a*torch.tanh(self.statistics(s)))
                else:delta=(1+self.q)*self.output(a)
            logits=self.network[4](self.network[3](h+delta)).squeeze(-1)
            return logits,h,delta

        def forward(self,x,g=None):
            if self.arm in ['B0','B1']:return self.network(x).squeeze(-1)
            return self.detail(x)[0]

        def regularizer(self,counts):return next(self.parameters()).new_zeros(())
    return Model
