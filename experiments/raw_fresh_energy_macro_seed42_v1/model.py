"""Four equal-capacity single-head Raw networks, all genuinely random-start."""
import sys
import torch
from torch import nn
import common as c

pure=c.module('fresh_energy_pure_raw',c.parent.REFERENCE/'pure_model.py')
prior=sys.modules['common'];sys.modules['common']=c.wave
try:normalizer=c.module('fresh_energy_fixed_waveform_normalizer',c.WAVE_CODE/'model.py')
finally:sys.modules['common']=prior


class LogEnergy(nn.Module):
    def forward(self,h):return torch.log1p(h.square().mean(-1,keepdim=True))


class Model(pure.PureRaw):
    def __init__(self,arm,fold):
        assert arm in c.TRAINED
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42+1009*fold);super().__init__(compact=False)
        self.arm=arm
        if arm.endswith('LOGENERGY'):self.cnn[4]=LogEnergy()

    def normalized_windows(self,raw,mask,times):
        if self.arm.startswith('WINDOW'):return super().normalized_windows(raw,mask,times)
        return normalizer.reference_wave(raw,mask,times)[0]


def objective(logit,y,pos):
    bce=nn.functional.binary_cross_entropy_with_logits(logit,y,pos_weight=pos);p=logit.sigmoid()
    a=2*(p*y).sum()/(p.sum()+y.sum()).clamp_min(1e-8)
    b=2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()).clamp_min(1e-8)
    return .5*bce+.5*(1-.5*(a+b))
