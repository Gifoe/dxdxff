"""Learn the extra Raw support before stride4, not after latent decimation."""
import sys
import common as c
import torch
from torch import nn
from torch.nn import functional as F

pure=c.module('early_fir_original_raw',c.parent.REFERENCE/'pure_model.py')
prior=sys.modules['common'];sys.modules['common']=c.wave
try:normalizer=c.module('early_fir_exact_normalizer',c.previous.WAVE_CODE/'model.py')
finally:sys.modules['common']=prior


class ExtendedFirstConv(nn.Module):
    def __init__(self,center):
        super().__init__();self.center=center;self.outer=nn.Parameter(torch.zeros(12,1,120))
    @property
    def weight(self):return self.center.weight
    def forward(self,x):
        kernel=torch.cat([self.outer[...,:60],self.outer.new_zeros(12,1,9),self.outer[...,60:]],-1)
        return self.center(x)+F.conv1d(x,kernel,stride=4,padding=60)


class LogEnergy(nn.Module):
    def forward(self,x):return torch.log1p(x.square().mean(-1,keepdim=True))


def collapsed_state(model):
    out={k:v for k,v in model.state_dict().items() if not k.startswith('cnn.0.')}
    first=model.cnn[0].center if isinstance(model.cnn[0],ExtendedFirstConv) else model.cnn[0]
    out.update({'cnn.0.weight':first.weight,'cnn.0.bias':first.bias})
    return out


def local_receptive_field(arm):return 153 if arm.startswith('LONG') else 33


class Model(pure.PureRaw):
    def __init__(self,arm,fold):
        assert arm in c.FAMILY
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42+1009*fold);super().__init__(compact=True)
            if arm.startswith('LONG'):self.cnn[0]=ExtendedFirstConv(self.cnn[0])
            if arm.endswith('ENERGY'):self.cnn[4]=LogEnergy()
        self.arm=arm
    def normalized_windows(self,raw,mask,times):return normalizer.reference_wave(raw,mask,times)[0]


def objective(logit,y,pos):
    bce=F.binary_cross_entropy_with_logits(logit,y,pos_weight=pos);p=logit.sigmoid()
    a=2*(p*y).sum()/(p.sum()+y.sum()).clamp_min(1e-8)
    b=2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()).clamp_min(1e-8)
    return .5*bce+.5*(1-.5*(a+b))
