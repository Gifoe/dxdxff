"""One110->64->1 classifier; no base,D1 predictor,scalar residual or output bound."""
import copy
import torch
from torch import nn

class SingleHeadConcat(nn.Module):
    def __init__(self,arm,cnn,gate,raw_feature_function,head_seed):
        super().__init__();assert arm in ['S0','S1','S2','S3','S4'];self.arm=arm
        self.cnn=copy.deepcopy(cnn);self.gate=copy.deepcopy(gate) if arm in ['S3','S4'] else nn.Identity()
        self.raw_feature_function=raw_feature_function
        mask=torch.ones(7,dtype=torch.float32)
        if arm=='S0':mask.zero_()
        if arm in ['S2','S4']:mask[4]=0
        self.register_buffer('keep7',mask)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(head_seed)
            self.head=nn.Sequential(nn.Linear(110,64),nn.GELU(),nn.Linear(64,1))
        assert sum(p.numel() for p in self.parameters())==(10377 if arm in ['S3','S4'] else 9329)
        assert all(p.requires_grad for p in self.parameters()) and not hasattr(self,'base')

    def features(self,raw,mask,times,delta,scale,identity_gate=False):
        mode='full' if self.arm in ['S3','S4'] and not identity_gate else 'identity'
        return self.raw_feature_function(self,raw,mask,times,delta*self.keep7,scale,mode=mode,perms=None)

    def classify(self,raw_summary,x,zero_phys=False):
        f=x*self.keep7.repeat_interleave(2)
        if zero_phys:f=torch.zeros_like(f)
        return self.head(torch.cat([raw_summary,f],dim=-1)).squeeze(-1)

    def forward(self,x,raw,mask,times,delta,scale,identity_gate=False):
        z,g=self.features(raw,mask,times,delta,scale,identity_gate)
        return dict(logit=self.classify(z,x),raw_summary=z,gate_activity=g)
