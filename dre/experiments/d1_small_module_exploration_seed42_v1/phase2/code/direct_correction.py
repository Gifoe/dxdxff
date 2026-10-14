"""Zero-output-initialized bounded error correctors; no dead scalar/auxiliary loss."""
import torch
from torch import nn
from dgm_net import DGMNet
from exploration_models import pool,condition

class FeatureCorrection(nn.Module):
    def __init__(self,base,arm):
        super().__init__();self.base=base.eval();self.arm=arm
        for p in self.base.parameters():p.requires_grad_(False)
        assert arm in ['M4_FEATURE_CORRECTION','M5_STAGE_CORRECTION']
        width,hidden,expected=(14,16,257) if arm=='M4_FEATURE_CORRECTION' else (28,24,721)
        self.head=nn.Sequential(nn.Linear(width,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        assert sum(p.numel() for p in self.parameters() if p.requires_grad)==expected
    def train(self,mode=True):super().train(mode);self.base.eval();return self
    def forward(self,x,raw,mask,times,delta,scale,mode='full',perms=None):
        with torch.no_grad():base=self.base(x)
        if self.arm=='M4_FEATURE_CORRECTION':
            z=x.clone()
            if mode.startswith('neutral_'):
                k=int(mode.split('_')[1]);z[:,2*k:2*k+2]=0
            diagnostic=z
        else:
            u=(condition(delta,mask,times,mode,perms)/scale).clamp(-8,8)
            if mode.startswith('neutral_'):u[...,int(mode.split('_')[1])]=0
            z=pool(torch.where(mask[...,None],u,torch.zeros_like(u)),mask,times);diagnostic=u[mask]
        aux=self.head(z).squeeze(-1);res=torch.tanh(aux);final=base if mode=='residual0' else base+res
        return final,aux,base,res,diagnostic

class MorphCorrection(DGMNet):
    def __init__(self,base):
        super().__init__(base);del self.alpha
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        assert sum(p.numel() for p in self.parameters() if p.requires_grad)==9481
    def forward(self,x,raw,mask,times,delta,scale,mode='full',perms=None):
        with torch.no_grad():base=self.base(x)
        z,g=self.features(raw,mask,times,delta,scale,mode,perms);aux=self.head(z).squeeze(-1);res=torch.tanh(aux)
        return base if mode=='residual0' else base+res,aux,base,res,g
