"""Only two scalar margin-gating parameters beyond the measured M6 backbone."""
import torch
from torch import nn
from direct_correction import MorphCorrection

class MarginGated(MorphCorrection):
    def __init__(self,base):
        super().__init__(base);self.margin_a=nn.Parameter(torch.zeros(()));self.margin_b=nn.Parameter(torch.zeros(()))
        assert sum(p.numel() for p in self.parameters() if p.requires_grad)==9483
    def forward(self,x,raw,mask,times,delta,scale,mode='full',perms=None):
        final,aux,base,res,g=super().forward(x,raw,mask,times,delta,scale,mode,perms)
        gain=torch.ones_like(base) if mode=='margin_identity' else 1+.5*torch.tanh(self.margin_a*torch.tanh(base.detach().abs())+self.margin_b)
        corrected=gain*res
        return base if mode=='residual0' else base+corrected,aux,base,corrected,g

def soft_macro_loss(logit,y):
    p=torch.sigmoid(logit)
    nez=2*(p*y).sum()/(p.sum()+y.sum()).clamp_min(1e-8)
    ez=2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()).clamp_min(1e-8)
    return 1-.5*(nez+ez)

def objective(logit,y,fn,arm):
    bce=fn(logit,y)
    if arm=='M8_MACRO_AWARE':return .75*bce+.25*soft_macro_loss(logit,y)
    return bce
