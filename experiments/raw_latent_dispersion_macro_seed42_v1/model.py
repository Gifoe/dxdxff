"""One learned Raw encoder/head; mean versus capacity-matched latent SD residual."""
import torch
from torch import nn
from common import module,parent

pure=module('dispersion_exact_raw_pool',parent.REFERENCE/'pure_model.py')
ARMS=['NATIVE','MEAN_TUNE','MEAN_RESIDUAL','DISPERSION']
TRAINED=ARMS[1:]


class ResidualPool(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        self.mix=nn.Conv1d(24,24,1,bias=False)
        nn.init.zeros_(self.mix.weight)

    def forward(self,x):
        mean=nn.functional.adaptive_avg_pool1d(x,1)
        if self.kind=='MEAN_RESIDUAL':a=mean
        else:
            assert self.kind=='DISPERSION'
            variance=(x-mean).square().mean(-1,keepdim=True)
            a=torch.where(variance>0,variance.clamp_min(1e-30).sqrt(),torch.zeros_like(variance))
        return mean+.25*self.mix(a).tanh()


class Model(pure.PureRaw):
    def __init__(self,arm,fold,state):
        assert arm in ARMS
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42+1009*fold)
            super().__init__(compact=False)
            self.load_reference(state)
            self.arm=arm
            if arm in ['MEAN_RESIDUAL','DISPERSION']:self.cnn[4]=ResidualPool(arm)


def objective(logit,y,pos):
    bce=nn.functional.binary_cross_entropy_with_logits(logit,y,pos_weight=pos)
    pn=logit.sigmoid()
    f0=2*(pn*y).sum()/(pn.sum()+y.sum()).clamp_min(1e-8)
    f1=2*((1-pn)*(1-y)).sum()/((1-pn).sum()+(1-y).sum()).clamp_min(1e-8)
    return .75*bce+.25*(1-.5*(f0+f1))


def optimizer(model):
    cnn=[];extra=[]
    for name,p in model.cnn.named_parameters():
        (extra if name.startswith('4.') else cnn).append(p)
    groups=[dict(params=cnn,lr=.0001),dict(params=list(model.head.parameters()),lr=.0003)]
    if extra:groups.append(dict(params=extra,lr=.0003))
    return torch.optim.AdamW(groups,weight_decay=.001)
