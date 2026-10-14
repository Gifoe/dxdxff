"""Native-frequency morphology correction; no label/identity input."""
import os
import sys
import numpy as np
import torch
from torch import nn
sys.path.insert(0,os.environ.get('DGM_COMPLETED_CODE',r'D:\dgm_net_seed42_v1\code'))
from controls import transform
ARMS=['M16_SPECTRAL','M17_TIME_SPECTRAL','M18_TIME_CONTROL']

def spectrum(v):
    assert v.shape[-1]==500
    h=torch.hann_window(500,periodic=True,device=v.device,dtype=v.dtype)
    power=torch.fft.rfft(v*h,dim=-1).abs().square()[...,20:251]/h.square().sum()
    s=torch.log(power+1e-8)
    return (s-s.mean(-1,keepdim=True))/s.std(-1,unbiased=False,keepdim=True).clamp_min(1e-6)

class SpectralCorrection(nn.Module):
    def __init__(self,base,arm):
        super().__init__();assert arm in ARMS;self.arm=arm;self.base=base.eval()
        for p in self.base.parameters():p.requires_grad_(False)
        self.width={'M16_SPECTRAL':96,'M17_TIME_SPECTRAL':120,'M18_TIME_CONTROL':24}[arm]
        if arm!=ARMS[0]:self.time=nn.Sequential(nn.Conv1d(1,12,9,4),nn.GELU(),nn.Conv1d(12,24,7,4),nn.GELU(),nn.AdaptiveAvgPool1d(1))
        if arm!=ARMS[2]:self.frequency=nn.Sequential(nn.Conv1d(1,12,9,3),nn.GELU(),nn.Conv1d(12,24,5,3),nn.GELU(),nn.AdaptiveAvgPool1d(4))
        self.gate=nn.Sequential(nn.Linear(7,32),nn.GELU(),nn.Linear(32,self.width));nn.init.zeros_(self.gate[-1].weight);nn.init.zeros_(self.gate[-1].bias)
        self.head=nn.Sequential(nn.Linear(4*self.width+14,64),nn.GELU(),nn.Linear(64,1));nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def train(self,mode=True):super().train(mode);self.base.eval();return self
    def encode(self,v):
        fields=[]
        if hasattr(self,'time'):fields.append(self.time(v[:,None]).flatten(1))
        if hasattr(self,'frequency'):fields.append(self.frequency(spectrum(v)[:,None]).flatten(1))
        return torch.cat(fields,1)
    def forward(self,x,raw,mask,times,delta,scale,mode='full',perms=None):
        S,C,W,L=raw.shape;assert L==500 and mask.shape==(S,C,W) and times.shape==(S,W) and delta.shape==(S,C,W,7)
        assert torch.isfinite(delta[mask]).all() and not delta.requires_grad
        v=raw[mask];v=((v-v.mean(-1,keepdim=True))/v.std(-1,unbiased=False,keepdim=True).clamp_min(1e-6)).clamp(-8,8)
        emb=raw.new_zeros(S,C,W,self.width);emb[mask]=torch.cat([self.encode(q) for q in v.split(4096)])
        u=(transform(delta,mask,times,mode,perms)/scale).clamp(-8,8)
        if mode.startswith('neutral_'):u[...,int(mode.split('_')[1])]=0
        summaries=[];available=[];activity=[]
        for s in range(S):
            pre=mask[s]&(times[s][None]<0);n=pre.sum(1);present=mask[s].any(1);assert (n[present]>0).all()
            ref=(emb[s]*pre[...,None]).sum(1)/n.clamp_min(1)[:,None];latent=emb[s]-ref[:,None]
            g=torch.ones_like(latent) if mode=='identity' else 1+.25*torch.tanh(self.gate(u[s]));post=mask[s]&((times[s]>=0)&(times[s]<20))[None];activity.append(g[post]);guided=latent*g;ss=[];aa=[]
            for low,high in [(0,8),(8,20)]:
                take=mask[s]&((times[s]>=low)&(times[s]<high))[None];count=take.sum(1);ss.append((guided*take[...,None]).sum(1)/count.clamp_min(1)[:,None]);aa.append(count>0)
            summaries.append(torch.stack(ss,1));available.append(torch.stack(aa,1))
        summary=torch.stack(summaries);valid=torch.stack(available)[...,None].to(raw.dtype);n=valid.sum(0);assert (n>0).all();mu=(summary*valid).sum(0)/n;var=((summary-mu[None]).square()*valid).sum(0)/n;sd=torch.where(var>0,var.clamp_min(1e-30).sqrt(),torch.zeros_like(var));vec=torch.stack([mu,sd],2).reshape(C,4*self.width).double();den=vec.std(0,unbiased=False);z=((vec-vec.mean(0))/torch.where(den>1e-8,den,torch.ones_like(den))).to(raw.dtype)
        with torch.no_grad():base=self.base(x)
        aux=self.head(torch.cat([z,x],1)).squeeze(-1);res=1.5*torch.tanh(aux)
        return base if mode=='residual0' else base+res,aux,base,res,torch.cat(activity)

def soft_macro(logit,y):
    p=logit.sigmoid();return 1-.5*(2*(p*y).sum()/(p.sum()+y.sum()).clamp_min(1e-8)+2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()).clamp_min(1e-8))

def objective(z,y,bce):return .75*bce(z,y)+.25*soft_macro(z,y)
