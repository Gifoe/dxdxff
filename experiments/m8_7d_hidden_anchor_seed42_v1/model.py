"""One shared nonlinear head; physiology-first training is not a second predictor."""
import copy
import torch
from torch import nn
from torch.nn import functional as F

ARMS=['H0_SCRATCH','H1_WARM_JOINT','H2_WARM_FIXED']


class Unified(nn.Module):
    def __init__(self,arm,source):
        super().__init__();assert arm in ARMS;self.variant=arm
        self.cnn=copy.deepcopy(source.cnn);self.gate=copy.deepcopy(source.gate)
        self.raw_feature_function=type(source).features
        self.phys_first=copy.deepcopy(source.base.first);self.readout=copy.deepcopy(source.base.last)
        self.dropout=nn.Dropout(.15)
        self.register_buffer('multiplicities',source.base.multiplicities.detach().clone());self.n0=source.base.n0
        self.raw_projection=nn.Linear(96,96,bias=False)
        nn.init.zeros_(self.raw_projection.weight)
        self.stage('joint')
        assert sum(p.numel() for p in self.parameters())==13961

    def normalized(self,x):
        assert x.shape[-1]==14
        m=self.multiplicities.to(x.dtype);mu=(x*m).sum(-1,keepdim=True)/88
        var=(((x-mu).square()*m).sum(-1,keepdim=True)+self.n0*mu.square())/88
        return (x-mu)/torch.sqrt(var+1e-5)

    def stage(self,name):
        assert name in ['phys','joint','fixed']
        for p in self.parameters():p.requires_grad_(True)
        modules=[self.cnn,self.gate,self.raw_projection] if name=='phys' else [self.phys_first,self.readout] if name=='fixed' else []
        for m in modules:
            for p in m.parameters():p.requires_grad_(False)
        self.stage_name=name

    def classify(self,raw_summary,x):
        assert raw_summary.shape==(len(x),96)
        hidden=self.phys_first(self.normalized(x))+torch.tanh(self.raw_projection(raw_summary))
        return self.readout(self.dropout(F.gelu(hidden))).squeeze(-1)

    def physiology_only(self,x):return self.classify(x.new_zeros(len(x),96),x)

    def features(self,raw,mask,times,delta,scale):
        return self.raw_feature_function(self,raw,mask,times,delta,scale,mode='full',perms=None)

    def forward(self,x,raw,mask,times,delta,scale):
        z,g=self.features(raw,mask,times,delta,scale)
        return dict(logit=self.classify(z,x),raw_summary=z,gate_activity=g)


def objective(logit,y,pos):
    p=logit.sigmoid();a=2*(p*y).sum()/(p.sum()+y.sum()).clamp_min(1e-8)
    b=2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()).clamp_min(1e-8)
    return .75*F.binary_cross_entropy_with_logits(logit,y,pos_weight=pos)+.25*(1-.5*(a+b))
