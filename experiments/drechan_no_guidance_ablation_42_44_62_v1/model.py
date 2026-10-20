"""Fixed no-Early/no-Late main, meaningful deletions and labelled controls."""
import copy
import importlib.util
from pathlib import Path
import sys
import torch
from torch import nn
PARENT=Path(r'D:\drechan_f0_no_lategate_ablation_seed42_v1')
if not PARENT.is_dir():PARENT=Path(__file__).resolve().parent.parent/PARENT.name
sys.path.insert(0,str(PARENT))
spec=importlib.util.spec_from_file_location('nga_original_no_late_math',PARENT/'model.py')
BASE=importlib.util.module_from_spec(spec);sys.modules[spec.name]=BASE;spec.loader.exec_module(BASE)
ARMS=['FULL','EARLY_FILM_RESTORED','WO_FEATURE_PR','WO_RAW_PR','WO_BOTH_PR','WO_PRE_REFERENCE',
      'RAW_ONLY','PHYSIOLOGY_ONLY','DROP_BETA','DROP_LOW_GAMMA','DROP_PARTIAL_HIGH_GAMMA',
      'DROP_RMS','DROP_LINE_LENGTH','DROP_SPECTRAL_ENTROPY','P1_TEMPORAL_STATS','P2_TIME_ATTENTION','CAPACITY_MATCHED']
DROP=BASE.DROP
PARAMETERS={a:9329 for a in ARMS}
PARAMETERS.update(EARLY_FILM_RESTORED=10377,RAW_ONLY=8433,PHYSIOLOGY_ONLY=1025,P2_TIME_ATTENTION=9355,CAPACITY_MATCHED=10337)
ALIASES={'WO_EARLY_FILM':'FULL','WO_LATE_GATE':'FULL','WO_BOTH_GUIDANCE':'FULL','WO_WINDOW_PR':'FULL'}
REFERENCES={'EARLY_FILM_RESTORED'}
objective=BASE.objective

def moments(value,valid):
    assert valid.shape==value.shape[:-1]
    count=valid.sum(-1);v=torch.where(valid[...,None],value,torch.zeros_like(value))
    mean=v.sum(-2)/count.clamp_min(1)[...,None]
    centered=torch.where(valid[...,None],value-mean[...,None,:],torch.zeros_like(value))
    variance=centered.square().sum(-2)/count.clamp_min(1)[...,None]
    sd=torch.where(variance>0,variance.clamp_min(1e-30).sqrt(),torch.zeros_like(variance))
    return mean,sd,count>0

def attention_mean(latent,times,valid,weight):
    tau=times/20.
    temporal=torch.stack((tau,tau.square()),-1).expand(latent.shape[0],-1,-1)
    # Known engineering boundary: remove unsupported NaN timestamps BEFORE matmul.
    features=torch.where(valid[...,None],torch.cat((latent,temporal),-1),torch.zeros((*valid.shape,26),device=latent.device,dtype=latent.dtype))
    score=features@weight;present=valid.any(-1)
    score=torch.where(valid,score,torch.full_like(score,-torch.inf))
    score=torch.where(present[:,None],score,torch.zeros_like(score))
    weights=torch.softmax(score,-1)*valid.to(latent.dtype)
    result=(torch.where(valid[...,None],latent,torch.zeros_like(latent))*weights[...,None]).sum(-2)
    return result,present,weights

class NoGuidanceF0(BASE.NoLateF0):
    def __init__(self,source,arm,head_seed=0):
        assert arm in ARMS
        original='FULL' if arm=='EARLY_FILM_RESTORED' else 'WO_EARLY_FILM' if arm in ['FULL','P1_TEMPORAL_STATS','P2_TIME_ATTENTION','CAPACITY_MATCHED'] else arm
        super().__init__(source,original)
        if arm!='EARLY_FILM_RESTORED':
            if self.branch!='phys' and self.early:
                self.cnn=nn.Sequential(*copy.deepcopy(list(self.cnn.stem.children())),copy.deepcopy(self.cnn.pool))
            self.early=False;self.guidance_pr=False
        self.arm=arm
        if arm=='P2_TIME_ATTENTION':self.attention_weight=nn.Parameter(torch.zeros(26))
        if arm=='CAPACITY_MATCHED':
            previous=self.head
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(head_seed)+421001)
                first=nn.Linear(110,73);last=nn.Linear(73,1)
            with torch.no_grad():
                first.weight[:64].copy_(previous[0].weight);first.bias[:64].copy_(previous[0].bias)
                last.weight[:,:64].copy_(previous[2].weight);last.bias.copy_(previous[2].bias)
            self.head=nn.Sequential(first,copy.deepcopy(previous[1]),last)
        assert not self.late and not hasattr(self,'gate') and not self.interaction
        assert self.early==(arm=='EARLY_FILM_RESTORED')
        assert self.guidance_pr==(arm=='EARLY_FILM_RESTORED')
        assert sum(p.numel() for p in self.parameters())==PARAMETERS[arm]
        assert self.head[0].out_features==(73 if arm=='CAPACITY_MATCHED' else 64)
    def features(self,raw,mask,times,delta,scale,permutation=None):
        if self.arm in ['P1_TEMPORAL_STATS','P2_TIME_ATTENTION']:
            return self.pooled_features(raw,mask,times,delta,scale,permutation)
        return super().features(raw,mask,times,delta,scale,permutation)
    def pooled_features(self,raw,mask,times,delta,scale,permutation=None):
        assert permutation is None and raw.shape[-1]==500
        self.last_guidance_audit={}
        # No physiological conditioning in either alternative; keep full raw support.
        S,C,W,L=raw.shape;flat=raw[mask]
        v=((flat-flat.mean(-1,keepdim=True))/flat.std(-1,unbiased=False,keepdim=True).clamp_min(1e-6)).clamp(-8,8)
        emb=raw.new_zeros(S,C,W,24)
        emb[mask]=torch.cat([self.cnn(q[:,None]).flatten(1) for q in v.split(4096)])
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
