"""Independent masked centralized-core implementation; no TeCh code copied."""
from __future__ import annotations
import torch
from torch import nn
from .masked_ops import masked_softmax

class MaskedCoTAR(nn.Module):
    def __init__(self,d_model=64,d_core=16):
        super().__init__();self.core_hidden=nn.Linear(d_model,d_model);self.core_feature=nn.Linear(d_model,d_core);self.out=nn.Sequential(nn.Linear(d_model+d_core,d_model),nn.GELU(),nn.Linear(d_model,d_model));self.last_weights=None;self.last_core=None
    def forward(self,x,channel_mask):
        feature=self.core_feature(torch.nn.functional.gelu(self.core_hidden(x)));weights=masked_softmax(feature,channel_mask,1);core=(weights.float()*feature.float()).sum(1,keepdim=True);result=self.out(torch.cat([x,core.expand(-1,x.shape[1],-1).to(x.dtype)],-1));result=result*channel_mask[...,None];self.last_weights=weights.detach();self.last_core=core.detach();return result
class MaskedCoTAREncoderLayer(nn.Module):
    def __init__(self,d_model=64,d_core=16,dropout=.1):
        super().__init__();self.cotar=MaskedCoTAR(d_model,d_core);self.norm1=nn.LayerNorm(d_model);self.norm2=nn.LayerNorm(d_model);self.drop=nn.Dropout(dropout);self.mlp=nn.Sequential(nn.Linear(d_model,2*d_model),nn.GELU(),nn.Linear(2*d_model,d_model))
    def forward(self,x,mask):
        x=self.norm1(x+self.drop(self.cotar(x,mask)));x=self.norm2(x+self.drop(self.mlp(x)));return x*mask[...,None]
