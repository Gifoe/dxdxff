"""Three predeclared pooling/capacity alternatives to the immutable no-Late main."""
import copy
import importlib.util
from pathlib import Path
import sys
import torch
from torch import nn

PARENT=Path(r'D:\drechan_f0_no_lategate_ablation_seed42_v1')
if not PARENT.is_dir():PARENT=Path(__file__).resolve().parent.parent/PARENT.name
spec=importlib.util.spec_from_file_location('pooling_capacity_original_no_late',PARENT/'model.py')
BASE=importlib.util.module_from_spec(spec);sys.modules[spec.name]=BASE;spec.loader.exec_module(BASE)
ARMS=['FULL','WO_EARLY_FILM','P1_TEMPORAL_STATS','P2_TIME_ATTENTION','A2_CAPACITY_MATCHED']
REUSE={'FULL':'FULL','WO_EARLY_FILM':'WO_EARLY_FILM'}
PARAMETERS=dict(FULL=10377,WO_EARLY_FILM=9329,P1_TEMPORAL_STATS=10377,P2_TIME_ATTENTION=10403,A2_CAPACITY_MATCHED=10337)
DROP={};REFERENCES=set();ALIASES={}


def moments(value,valid):
    """Population mean/SD on complete valid support; no padded observations."""
    assert valid.shape==value.shape[:-1]
    count=valid.sum(-1);v=torch.where(valid[...,None],value,torch.zeros_like(value))
    mean=v.sum(-2)/count.clamp_min(1)[...,None]
    centered=torch.where(valid[...,None],value-mean[...,None,:],torch.zeros_like(value))
    variance=centered.square().sum(-2)/count.clamp_min(1)[...,None]
    sd=torch.where(variance>0,variance.clamp_min(1e-30).sqrt(),torch.zeros_like(variance))
    return mean,sd,count>0


def attention_mean(latent,times,valid,weight):
    """26 shared coefficients: 24 latent + normalized time + squared time.

    Each stage/channel softmax uses only its actual valid windows. Empty stages
    yield zero/absent, not NaN or an imputed observation. Zero coefficients give
    uniform valid-window attention. No labels/physiology enter the scorer.
    """
    tau=times/20.
    temporal=torch.stack((tau,tau.square()),-1).expand(latent.shape[0],-1,-1)
    score=torch.cat((latent,temporal),-1)@weight
    present=valid.any(-1)
    score=torch.where(valid,score,torch.full_like(score,-torch.inf))
    score=torch.where(present[:,None],score,torch.zeros_like(score))
    weights=torch.softmax(score,-1)*valid.to(latent.dtype)
    result=(torch.where(valid[...,None],latent,torch.zeros_like(latent))*weights[...,None]).sum(-2)
    return result,present,weights


class PoolingCapacity(BASE.NoLateF0):
    def __init__(self,source,arm,head_seed=0):
        assert arm in ARMS
        original='WO_EARLY_FILM' if arm in ['WO_EARLY_FILM','A2_CAPACITY_MATCHED'] else 'FULL'
        super().__init__(source,original)
        self.arm=arm
        if arm=='P2_TIME_ATTENTION':self.attention_weight=nn.Parameter(torch.zeros(26))
        if arm=='A2_CAPACITY_MATCHED':
            # Closest integer single-hidden width: 9329 + (73-64)*112 =10337.
            # Same original64 neuron slices; only9 additional ordinary neurons.
            previous=self.head
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(head_seed)+421001)
                first=nn.Linear(110,73);last=nn.Linear(73,1)
            with torch.no_grad():
                first.weight[:64].copy_(previous[0].weight);first.bias[:64].copy_(previous[0].bias)
                last.weight[:,:64].copy_(previous[2].weight);last.bias.copy_(previous[2].bias)
            self.head=nn.Sequential(first,copy.deepcopy(previous[1]),last)
        assert not self.late and not hasattr(self,'gate') and not self.interaction
        assert self.keep7[4]==0 and self.raw_pr and self.feature_pr and self.pre_reference
        assert sum(p.numel() for p in self.parameters())==PARAMETERS[arm]

    def features(self,raw,mask,times,delta,scale,permutation=None):
        if self.arm not in ['P1_TEMPORAL_STATS','P2_TIME_ATTENTION']:
            return super().features(raw,mask,times,delta,scale,permutation)
        assert permutation is None and raw.shape[-1]==500
        self.last_guidance_audit={}
        delta,self.last_guidance_audit=BASE.BASE.conditioning_delta(delta,mask,scale,self.keep7,self.collect_guidance_audit)
        # Exact original waveform normalization, physiology transformation,
        # EarlyFiLM, chunk size and own pre-ictal reference. Only pooling differs.
        S,C,W,L=raw.shape;flat=raw[mask]
        v=((flat-flat.mean(-1,keepdim=True))/flat.std(-1,unbiased=False,keepdim=True).clamp_min(1e-6)).clamp(-8,8)
        masked=torch.where(self.keep7.bool(),delta,torch.zeros_like(delta))
        transform=self.raw_feature_function.__globals__['transform']
        u=(transform(masked,mask,times,'full',None)/scale).clamp(-8,8)
        emb=raw.new_zeros(S,C,W,24)
        emb[mask]=torch.cat([self.cnn(q[:,None],p).flatten(1) for q,p in zip(v.split(4096),u[mask].split(4096))])
        summaries=[];available=[];activity=[]
        for s in range(S):
            pre=mask[s]&(times[s][None]<0);count=pre.sum(1);present=mask[s].any(1)
            assert (count[present]>0).all()
            hpre=(emb[s]*pre[...,None]).sum(1)/count.clamp_min(1)[:,None]
            latent=emb[s]-hpre[:,None]
            post=mask[s]&((times[s]>=0)&(times[s]<20))[None]
            activity.append(torch.ones_like(latent)[post])
            if self.arm=='P1_TEMPORAL_STATS':
                mean,sd,present=moments(latent,post)
                st=[mean,sd];av=[present,present]
            else:
                st=[];av=[]
                for low,high in [(0,8),(8,20)]:
                    select=mask[s]&((times[s]>=low)&(times[s]<high))[None]
                    mean,present,_=attention_mean(latent,times[s],select,self.attention_weight)
                    st.append(mean);av.append(present)
            summaries.append(torch.stack(st,1));available.append(torch.stack(av,1))
        summary=torch.stack(summaries);valid=torch.stack(available)[...,None].to(raw.dtype)
        n=valid.sum(0);assert (n>0).all(),'Original complete support contract required'
        mu=(summary*valid).sum(0)/n;var=((summary-mu[None]).square()*valid).sum(0)/n
        sd=torch.where(var>0,var.clamp_min(1e-30).sqrt(),torch.zeros_like(var))
        vector=torch.stack([mu,sd],2).reshape(C,96).double()
        spread=vector.std(0,unbiased=False)
        z=(vector-vector.mean(0))/torch.where(spread>1e-8,spread,torch.ones_like(spread))
        return z.to(raw.dtype),torch.cat(activity)


objective=BASE.objective
