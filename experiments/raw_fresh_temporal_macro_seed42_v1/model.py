"""Fresh single-waveform CNNs; equal-size pointwise/temporal controls."""
import sys
import common as c
import torch
from torch import nn

pure=c.module('fresh_temporal_original_raw',c.parent.REFERENCE/'pure_model.py')
prior=sys.modules['common'];sys.modules['common']=c.wave
try:normalizer=c.module('fresh_temporal_exact_pre_normalizer',c.previous.WAVE_CODE/'model.py')
finally:sys.modules['common']=prior


class ResidualPool(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        if kind=='FRESH_POINTWISE':
            self.adapter=nn.Sequential(nn.Conv1d(24,20,1,bias=False),nn.GELU(),nn.Conv1d(20,24,1,bias=True))
            nn.init.zeros_(self.adapter[2].weight);nn.init.zeros_(self.adapter[2].bias)
        else:
            assert kind=='FRESH_TEMPORAL'
            self.adapter=nn.Sequential(nn.Conv1d(24,24,17,padding=8,groups=24,bias=False),nn.GELU(),nn.Conv1d(24,24,1,bias=False))
            nn.init.zeros_(self.adapter[0].weight)
            with torch.no_grad():self.adapter[0].weight[:,:,8]=1.
            nn.init.zeros_(self.adapter[2].weight)
        assert sum(p.numel() for p in self.parameters())==984
    def forward(self,x):
        return nn.functional.adaptive_avg_pool1d(x,1)+.25*self.adapter(x).tanh().mean(-1,keepdim=True)


def core_state_hash(model):
    return c.state_hash({k:v for k,v in model.state_dict().items() if not k.startswith('cnn.4.')})


def local_receptive_field(arm):
    assert arm in c.TRAINED
    return 33+16*16 if arm=='FRESH_TEMPORAL' else 33


class Model(pure.PureRaw):
    def __init__(self,arm,fold):
        assert arm in c.TRAINED;seed=42+1009*fold
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);super().__init__(compact=True)
            if arm!='FRESH_BASE':
                torch.manual_seed(seed+170003);self.cnn[4]=ResidualPool(arm)
        self.arm=arm
    def normalized_windows(self,raw,mask,times):return normalizer.reference_wave(raw,mask,times)[0]


def objective(logit,y,pos):
    bce=nn.functional.binary_cross_entropy_with_logits(logit,y,pos_weight=pos);p=logit.sigmoid()
    a=2*(p*y).sum()/(p.sum()+y.sum()).clamp_min(1e-8)
    b=2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()).clamp_min(1e-8)
    return .5*bce+.5*(1-.5*(a+b))
