"""Small patient-observable decision adapters on frozen M8 evidence."""
import torch
from torch import nn

ARMS=['M9_CONTEXT_SHIFT','M10_CONTEXT_READOUT','M11_SHARP_READOUT']

class Adapter(nn.Module):
    def __init__(self,arm):
        super().__init__();assert arm in ARMS;self.arm=arm
        width,hidden=(16,16) if arm==ARMS[0] else (129,48)
        self.net=nn.Sequential(nn.Linear(width,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,base):
        v=x[:,-16:] if self.arm==ARMS[0] else x
        return base+torch.tanh(self.net(v).squeeze(-1))

def soft_macro(z,y):
    p=z.sigmoid();a=2*(p*y).sum()/(p.sum()+y.sum()).clamp_min(1e-8)
    b=2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()).clamp_min(1e-8)
    return 1-.5*(a+b)

def loss(z,y,bce,arm):
    w,temp=(.25,.25) if arm==ARMS[2] else (.5,.5)
    return w*bce(z,y)+(1-w)*soft_macro(z/temp,y)
