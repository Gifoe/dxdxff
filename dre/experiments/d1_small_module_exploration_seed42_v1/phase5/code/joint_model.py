"""Warm-start M8 with zero-initialized physiology/readout cross-interaction."""
import math
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).parents[2]
sys.path.insert(0,os.environ.get('DGM_COMPLETED_CODE',r'D:\dgm_net_seed42_v1\code'));sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'phase2/code'));sys.path.insert(0,str(ROOT/'phase3/code'))
from direct_correction import MorphCorrection
from decision_modules import soft_macro_loss

ARMS=['M12_JOINT_READOUT','M13_ADAPTIVE_CAP']
class Joint(MorphCorrection):
    def __init__(self,base,state,arm):
        super().__init__(base);assert arm in ARMS;self.load_state_dict(state,strict=True);self.arm=arm
        old=self.head[0];expanded=nn.Linear(110,64)
        with torch.no_grad():expanded.weight.zero_();expanded.weight[:,:96].copy_(old.weight);expanded.bias.copy_(old.bias)
        self.head[0]=expanded
        if arm==ARMS[1]:self.cap_parameter=nn.Parameter(torch.tensor(math.log(.5)))
    def forward(self,x,raw,mask,times,delta,scale,mode='full',perms=None):
        with torch.no_grad():base=self.base(x)
        z,g=self.features(raw,mask,times,delta,scale,mode,perms);aux=self.head(torch.cat([z,x],-1)).squeeze(-1)
        cap=3*torch.sigmoid(self.cap_parameter) if self.arm==ARMS[1] else 1.
        res=cap*torch.tanh(aux)
        return base if mode=='residual0' else base+res,aux,base,res,g

def objective(z,y,bce):return .5*bce(z,y)+.5*soft_macro_loss(z/.5,y)
