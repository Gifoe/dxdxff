"""Global plus patient-relative Raw96views, original joint GELU and single output."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from view_math import views

PARENT=Path(r'D:\drechan_pruned_branch_fusion_3seed_v1')
spec=importlib.util.spec_from_file_location('rdv_frozen_pruned_model',PARENT/'model.py')
BASE=importlib.util.module_from_spec(spec);sys.modules[spec.name]=BASE;spec.loader.exec_module(BASE)
ARMS=['B0','PRUNED_REF','FULL','WO_FEATURE_PR','WO_RAW_PR','WO_BOTH_PR','WO_GLOBAL_VIEW','WO_PRE_REFERENCE']
REFERENCES={'B0':'B0','PRUNED_REF':'JOINT_PRUNED'}
PARAMETERS={a:11425 if a=='B0' else 9329 if a=='PRUNED_REF' else 15473 for a in ARMS}


class DualView(BASE.BASE.PRFactorial):
    def __init__(self,source,arm):
        assert arm in ARMS and arm not in REFERENCES
        super().__init__(source,'FULL');BASE.remove_guidance(self)
        self.variant=arm;self.feature_pr=arm not in ['WO_FEATURE_PR','WO_BOTH_PR']
        self.raw_pr=arm not in ['WO_RAW_PR','WO_BOTH_PR'];self.global_view=arm!='WO_GLOBAL_VIEW'
        self.pre_reference=arm!='WO_PRE_REFERENCE';self.collect_anchor=False
        old=self.head
        # Construct full original initializer first; discard random new Linear values
        # inside a CPU RNG fork, then copy only defined original weight slices.
        with torch.random.fork_rng(devices=[]):first=nn.Linear(206,64)
        first=first.to(old[0].weight.device)
        with torch.no_grad():
            first.weight[:,:96].copy_(old[0].weight[:,:96]/2)
            first.weight[:,96:192].copy_(old[0].weight[:,:96]/2)
            first.weight[:,192:].copy_(old[0].weight[:,96:])
            first.bias.copy_(old[0].bias)
        self.head=nn.Sequential(first,copy.deepcopy(old[1]),copy.deepcopy(old[2]))
        self.register_buffer('anchor_mean',torch.zeros(96,dtype=torch.float64))
        self.register_buffer('anchor_scale',torch.ones(96,dtype=torch.float64))
        assert not self.early and not self.late and not self.interaction
        assert not any('gain' in n or 'conditioner' in n or n.startswith('gate.') for n,_ in self.named_parameters())

    def pooled_vector(self,raw,mask,times,delta,scale,permutation=None):
        # Use the immutable original complete-patient kernel without changing this
        # model's PR flag or tensor math. The proxy owns no parameters/caches.
        proxy=SimpleNamespace(early=False,late=False,arm='FULL_F0',raw_pr=False,
                              pre_reference=self.pre_reference,cnn=self.cnn,
                              raw_feature_function=self.raw_feature_function,keep7=self.keep7)
        return BASE.BASE.PRFactorial.features(proxy,raw,mask,times,delta,scale,permutation)

    def features(self,*args,**kwargs):
        vector,activity=self.pooled_vector(*args,**kwargs)
        if self.collect_anchor:return vector,activity
        return views(vector,self.anchor_mean,self.anchor_scale,self.raw_pr,self.global_view),activity

    def classify(self,z,x):
        assert z.shape[-1]==192 and x.shape[-1]==14
        x=torch.where(self.keep7.repeat_interleave(2).bool(),x,torch.zeros_like(x));first=self.head[0]
        a=F.linear(z*math.sqrt(110/192),first.weight[:,:192])
        b=F.linear(x*math.sqrt(110/28),first.weight[:,192:])
        return self.head[2](self.head[1](a+b+first.bias)).squeeze(-1)

    @torch.no_grad()
    def set_anchor(self,anchor):
        assert anchor['mean'].shape==anchor['scale'].shape==(96,)
        assert torch.isfinite(anchor['mean']).all() and torch.isfinite(anchor['scale']).all() and (anchor['scale']>0).all()
        self.anchor_mean.copy_(anchor['mean']);self.anchor_scale.copy_(anchor['scale'])


objective=BASE.objective
