"""Equal-size pointwise/temporal residuals inside one shared waveform encoder."""
import sys
import torch
from torch import nn
import common as c

prior=sys.modules['common'];sys.modules['common']=c.wave
try:old=c.module('temporal_frozen_precompand_model',c.WAVE_CODE/'model.py')
finally:sys.modules['common']=prior


class ResidualPool(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        if kind=='CHANNEL_MLP':
            self.adapter=nn.Sequential(nn.Conv1d(24,20,1,bias=False),nn.GELU(),nn.Conv1d(20,24,1,bias=True))
            nn.init.zeros_(self.adapter[2].weight);nn.init.zeros_(self.adapter[2].bias)
        else:
            assert kind=='TEMPORAL_CONTEXT'
            self.adapter=nn.Sequential(nn.Conv1d(24,24,17,padding=8,groups=24,bias=False),nn.GELU(),nn.Conv1d(24,24,1,bias=False))
            nn.init.zeros_(self.adapter[0].weight)
            with torch.no_grad():self.adapter[0].weight[:,:,8]=1.
            nn.init.zeros_(self.adapter[2].weight)
        assert sum(p.numel() for p in self.parameters())==984

    def forward(self,x):return nn.functional.adaptive_avg_pool1d(x,1)+.25*self.adapter(x).tanh().mean(-1,keepdim=True)


class Model(old.Model):
    def __init__(self,kind,fold,state,matrix):
        assert kind in c.TRAINED
        super().__init__('PRECOMPAND_RESIDUAL',fold,state)
        with torch.no_grad():self.matrix.copy_(matrix)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42+1009*fold)
            if kind!='PRE_TUNE':self.cnn[4]=ResidualPool(kind)
        self.kind=kind;self.requires_grad_(True)

    def encode(self,raw,mask,times):
        with torch.no_grad():
            window=self.normalized_windows(raw,mask,times);reference,audit=old.reference_wave(raw,mask,times)
        result={};flags=None
        for name,flat in [('window',window),('reference',reference)]:
            h=torch.cat([self.cnn(q[:,None]).flatten(1) for q in flat.split(4096)])
            dense=raw.new_zeros((*mask.shape,24));dense[mask]=h
            summary,valid=old.phase_summary(dense,mask,times);result[name]=summary
            if flags is not None:assert torch.equal(flags,valid)
            flags=valid
        result['available']=flags
        return result,audit


def objective(logit,y,pos):
    bce=nn.functional.binary_cross_entropy_with_logits(logit,y,pos_weight=pos);p=logit.sigmoid()
    a=2*(p*y).sum()/(p.sum()+y.sum()).clamp_min(1e-8)
    b=2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()).clamp_min(1e-8)
    return .5*bce+.5*(1-.5*(a+b))


def optimizer(model):
    base=[];adapter=[]
    for n,p in model.cnn.named_parameters():(adapter if n.startswith('4.') else base).append(p)
    groups=[dict(params=base,lr=3e-5),dict(params=model.head.parameters(),lr=1e-4),dict(params=[model.matrix],lr=3e-4)]
    if adapter:groups.append(dict(params=adapter,lr=3e-4))
    return torch.optim.AdamW(groups,weight_decay=.001)
