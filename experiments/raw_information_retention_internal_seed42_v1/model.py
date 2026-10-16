"""Raw-only information-retention controls; one shared channel classifier."""
import importlib.util
from pathlib import Path
import math
import torch
from torch import nn
import torch.nn.functional as F

PARENT=Path(__file__).resolve().parents[1]/'raw_latent_contrast_optimization_seed42_v1/model.py'
if not PARENT.exists():PARENT=Path(r'D:\raw_latent_contrast_optimization_seed42_v1\model.py')
spec=importlib.util.spec_from_file_location('retention_pinned_pooling',PARENT)
pooling=importlib.util.module_from_spec(spec);spec.loader.exec_module(pooling)

CANDIDATES=['C0_WINDOW_SHORT','C1_PRE_SHORT','C2_WINDOW_LONG','C3_PRE_LONG','C4_PRE_LONG_CONTEXT']
ABLATIONS=['DROP_PRE_REFERENCE','DROP_TWO_PHASES','DROP_SEIZURE_SD','DROP_PATIENT_Z']
ARMS=CANDIDATES+ABLATIONS


def pre_scale(raw,mask,times):
    """Same-seizure/channel pre-window RMS is a scale, never a head input."""
    flat=raw[mask]
    variance=flat.double().var(-1,unbiased=False)
    dense=torch.zeros(mask.shape,device=raw.device,dtype=torch.float64);dense[mask]=variance
    pre=mask&(times[:,None,:]<0);n=pre.sum(-1);present=mask.any(-1)
    assert (n[present]>0).all(),'Missing genuine pre-onset waveform'
    scale=((dense*pre).sum(-1)/n.clamp_min(1)).sqrt()
    positive=scale[present&(scale>0)]
    assert len(positive)>0,'All pre-onset waveforms are constant; amplitude scale is unidentified'
    # Keep zero-baseline channels; use only other available pre-onset channels,
    # not post-onset samples, labels, learned features, or a unit-dependent floor.
    fallback=positive.median()
    scale=torch.where(scale>0,scale,fallback)
    return scale,dict(present=present,pre_windows=n,constant_pre_channels=present&((dense*pre).sum(-1)==0))


def normalize(raw,mask,times,kind):
    flat=raw[mask]
    if kind=='window':return pooling.normalize_windows(flat)
    assert kind=='pre'
    scale,_=pre_scale(raw,mask,times)
    divisor=scale[...,None].expand_as(mask)[mask]
    # Window-local centering removes DC; a common pre scale preserves the
    # within-record change in waveform amplitude without adding a feature slot.
    return ((flat-flat.mean(-1,keepdim=True)).double()/divisor[:,None]).float().clamp(-8,8)


class TemporalBlock(nn.Module):
    def __init__(self,dilation):
        super().__init__()
        self.depthwise=nn.Conv1d(24,24,7,padding=3*dilation,dilation=dilation,groups=24)
        self.mix=nn.Conv1d(24,24,1)

    def forward(self,x):
        return x+F.gelu(self.mix(F.gelu(self.depthwise(x))))/math.sqrt(3)


def long_cnn():
    return nn.Sequential(nn.Conv1d(1,24,15,2,7),nn.GELU(),nn.Conv1d(24,24,9,2,4),nn.GELU(),
                         TemporalBlock(1),TemporalBlock(4),TemporalBlock(16),nn.AdaptiveAvgPool1d(1))


class Model(nn.Module):
    def __init__(self,arm,fold=1):
        super().__init__();assert arm in ARMS;self.arm=arm
        self.normalization='pre' if arm in ['C1_PRE_SHORT','C3_PRE_LONG','C4_PRE_LONG_CONTEXT'] else 'window'
        self.long=arm in ['C2_WINDOW_LONG','C3_PRE_LONG','C4_PRE_LONG_CONTEXT']
        self.context=arm=='C4_PRE_LONG_CONTEXT';self.recipe=[1,1,1,1]
        if arm in ABLATIONS:self.recipe[ABLATIONS.index(arm)]=0
        seed=42+1009*fold
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);self.cnn=long_cnn() if self.long else pooling.cnn24()
            torch.manual_seed(seed+170003);self.head=nn.Sequential(nn.Linear(96,64),nn.GELU(),nn.Linear(64,1))
            if self.context:
                torch.manual_seed(seed+370003)
                self.context_encoder=nn.Sequential(nn.Linear(96,16),nn.GELU())
                self.context_conditioner=nn.Linear(16,128)
                nn.init.zeros_(self.context_conditioner.weight);nn.init.zeros_(self.context_conditioner.bias)

    def normalized_windows(self,raw,mask,times):return normalize(raw,mask,times,self.normalization)

    def aggregate(self,emb,mask,times):return pooling.pool_embeddings(emb,mask,times,self.recipe)

    def classify(self,z):
        h=self.head[1](self.head[0](z))
        if self.context:
            c=self.context_encoder(z).mean(0,keepdim=True)
            gamma,beta=self.context_conditioner(c).chunk(2,-1)
            h=h*(1+.25*gamma.tanh())+.25*beta.tanh()
        return self.head[2](h).squeeze(-1)


def local_receptive_field(long):
    size,jump=1,1
    layers=[(15,2,1),(9,2,1),(7,1,1),(7,1,4),(7,1,16)] if long else [(9,4,1),(7,4,1)]
    for kernel,stride,dilation in layers:size+=(kernel-1)*dilation*jump;jump*=stride
    return size
