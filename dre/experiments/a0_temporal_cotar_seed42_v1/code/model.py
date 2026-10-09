"""Controlled local-core versus masked temporal core; common channel readout."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Core(nn.Module):
    def __init__(self, mode):
        super().__init__(); assert mode in ('T1','T2'); self.mode=mode
        self.lin1=nn.Linear(16,16); self.lin2=nn.Linear(16,4)
        self.lin3=nn.Linear(20,16); self.lin4=nn.Linear(16,16)

    def forward(self,x,mask):
        x=torch.where(mask[...,None],x,0.)
        v=self.lin2(F.gelu(self.lin1(x)))
        if self.mode=='T2':
            # All-missing rows have zero softmax weights and zero core, not NaN.
            logits=v.masked_fill(~mask[...,None],-torch.inf)
            logits=torch.where(mask.any(-1)[:,None,None],logits,0.)
            a=logits.softmax(dim=1)*mask[...,None]
            v=(a*v).sum(1,keepdim=True).expand_as(v)
        out=self.lin4(F.gelu(self.lin3(torch.cat([x,v],-1))))
        return torch.where(mask[...,None],out,0.)


class Temporal(nn.Module):
    def __init__(self,mode):
        super().__init__(); self.mode=mode
        self.input=nn.Linear(18,16); self.time=nn.Linear(16,16)
        self.register_buffer('frequencies',torch.exp(-math.log(10000.)*torch.arange(0,16,2)/16))
        self.core=Core(mode); self.norm1=nn.LayerNorm(16); self.norm2=nn.LayerNorm(16)
        self.ffn=nn.Sequential(nn.Linear(16,32),nn.GELU(),nn.Dropout(.1),nn.Linear(32,16),nn.Dropout(.1))
        self.head=nn.Sequential(nn.Linear(96,32),nn.GELU(),nn.Dropout(.1),nn.Linear(32,1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def encode(self,z,t,mask):
        z=torch.where(mask[...,None],z,0.); t=torch.where(mask,t,0.)
        phase=t[...,None]*self.frequencies
        fourier=torch.stack([phase.sin(),phase.cos()],-1).flatten(-2)
        e=self.input(z)+self.time(fourier)
        e=torch.where(mask[...,None],e,0.)
        h=self.norm1(e+self.core(e,mask)); h=self.norm2(h+self.ffn(h))
        return torch.where(mask[...,None],h,0.)

    def readout(self,h,mask,t):
        # [channel,seizure,time,16]; no electrode interaction or positional index.
        pooled=[]; present=[]
        for lo,hi in [(-10,0),(0,8),(8,20)]:
            m=mask & (t>=lo)&(t<hi); n=m.sum(-1)
            pooled.append((h*m[...,None]).sum(-2)/n.clamp_min(1)[...,None])
            present.append(n>0)
        p=torch.stack(pooled,-2); valid=torch.stack(present,-1)
        n=valid.sum(1).clamp_min(1)
        mean=(p*valid[...,None]).sum(1)/n[...,None]
        var=((p-mean[:,None])**2*valid[...,None]).sum(1)/n[...,None]
        # sqrt at exact zero otherwise has an infinite derivative. Safe branch
        # preserves exact population std, including the single-seizure zero.
        std=torch.where(var>0,torch.sqrt(var.clamp_min(1e-30)),torch.zeros_like(var))
        early=((mask&(t>=0)&(t<8)).sum(-1)>=2).any(1)
        return torch.cat([mean.flatten(1),std.flatten(1)],-1),early

    def forward(self,z,t,mask):
        c,s,w,d=z.shape
        h=self.encode(z.reshape(c*s,w,d),t.reshape(c*s,w),mask.reshape(c*s,w)).reshape(c,s,w,16)
        embedding,early=self.readout(h,mask,t)
        delta=.5*torch.tanh(self.head(embedding).squeeze(-1))
        return torch.where(early,delta,0.)
