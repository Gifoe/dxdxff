"""One original classifier;128 matched zero-initialized contextual-residual gains."""
import importlib.util
import math
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
from relative_math import residual_hidden

PREVIOUS=Path(r'D:\drechan_pr_component_factorial_3seed_v1')
spec=importlib.util.spec_from_file_location('bpr_component_model',PREVIOUS/'model.py')
BASE=importlib.util.module_from_spec(spec);sys.modules[spec.name]=BASE;spec.loader.exec_module(BASE)
ARMS=['B0','FULL','WO_FEATURE_PR','WO_RAW_PR','WO_BOTH_PR','CHANNEL_CONTEXT']

class BranchResidual(BASE.PRFactorial):
    def __init__(self,source,arm):
        assert arm in ARMS[1:]
        super().__init__(source,'FULL')
        self.variant=arm
        self.feature_pr=arm not in ['WO_FEATURE_PR','WO_BOTH_PR']
        self.raw_pr=arm not in ['WO_RAW_PR','WO_BOTH_PR']
        self.raw_context=self.raw_pr and arm!='CHANNEL_CONTEXT'
        self.feature_context=self.feature_pr and arm!='CHANNEL_CONTEXT'
        self.gain_raw=nn.Parameter(torch.zeros(64))
        self.gain_feature=nn.Parameter(torch.zeros(64))
        self.register_buffer('anchor_mean',torch.zeros(96,dtype=torch.float64))
        self.register_buffer('anchor_scale',torch.ones(96,dtype=torch.float64))
        self.collect_anchor=False

    def features(self,*args,**kwargs):
        z,g=super().features(*args,**kwargs)
        if not self.raw_pr and not self.collect_anchor:
            z=((z.double()-self.anchor_mean)/self.anchor_scale).to(z.dtype)
        return z,g

    def classify(self,z,x):
        x=torch.where(self.keep7.repeat_interleave(2).bool(),x,torch.zeros_like(x))
        first=self.head[0]
        a=F.linear(z*math.sqrt(110/192),first.weight[:,:96])
        b=F.linear(x*math.sqrt(110/28),first.weight[:,96:])
        h=residual_hidden(self.head,a,b,self.gain_raw,self.gain_feature,self.raw_context,self.feature_context)
        return self.head[2](h).squeeze(-1)

    @torch.no_grad()
    def set_anchor(self,anchor):
        assert anchor['mean'].shape==anchor['scale'].shape==(96,)
        assert torch.isfinite(anchor['mean']).all() and torch.isfinite(anchor['scale']).all() and (anchor['scale']>0).all()
        self.anchor_mean.copy_(anchor['mean']);self.anchor_scale.copy_(anchor['scale'])

objective=BASE.objective
