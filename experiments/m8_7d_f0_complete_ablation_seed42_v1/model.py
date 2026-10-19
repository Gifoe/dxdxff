"""Single-switch F0 ablations: same one-head initializer and complete patient pooling."""
import copy
import hashlib
import json
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARMS=['FULL_F0','WO_EARLY_FILM','WO_LATE_GATE','WO_BOTH_GUIDANCE','WO_FEATURE_PR','WO_RAW_PR','WO_RAW_PRE_REFERENCE','WO_HEAD_INTERACTION','EARLY_FILM_MISMATCH','RAW_ONLY','PHYSIOLOGY_ONLY',
      'DROP_BETA','DROP_LOW_GAMMA','DROP_PARTIAL_HIGH_GAMMA','DROP_RMS','DROP_VARIANCE','DROP_LINE_LENGTH','DROP_SPECTRAL_ENTROPY']

def channel_mapping(mask,channels,patient,fold):
    """Record-wise whole-trajectory derangement, only within identical support."""
    support=mask.detach().cpu().numpy().astype(bool);S,C,W=support.shape
    assert len(channels)==C and len(set(channels))==C
    mapping=np.tile(np.arange(C),(S,1));changed=eligible=singleton=0
    for s in range(S):
        groups={}
        for c in range(C):groups.setdefault(support[s,c].tobytes(),[]).append(c)
        for signature,group in groups.items():
            group=sorted(group,key=lambda c:str(channels[c]));n=len(group);valid=int(support[s,group[0]].sum())
            if not valid:continue
            if n<2:singleton+=valid;continue
            seed=hashlib.sha256(json.dumps([42,int(fold),str(patient),s,hashlib.sha256(signature).hexdigest()],sort_keys=True).encode()).digest()
            shift=1+int.from_bytes(seed[:8],'little')%(n-1);donors=np.roll(group,shift)
            mapping[s,group]=donors;eligible+=n*valid;changed+=n*valid
            assert np.array_equal(support[s,group],support[s,donors]) and np.all(mapping[s,group]!=group)
    return torch.as_tensor(mapping,device=mask.device),dict(changed_valid_windows=changed,eligible_valid_windows=eligible,singleton_valid_windows=singleton,total_valid_windows=int(support.sum()))

def subset_head(source,width,indices):
    with torch.random.fork_rng(devices=[]):first=nn.Linear(width,64)
    with torch.no_grad():first.weight.copy_(source.head[0].weight[:,indices]);first.bias.copy_(source.head[0].bias)
    return nn.Sequential(first,copy.deepcopy(source.head[1]),copy.deepcopy(source.head[2]))

class Ablation(nn.Module):
    def __init__(self,source,arm):
        super().__init__();assert arm in ARMS;self.arm=arm
        self.branch='raw' if arm=='RAW_ONLY' else 'phys' if arm=='PHYSIOLOGY_ONLY' else 'joint'
        self.early=arm not in ['WO_EARLY_FILM','WO_BOTH_GUIDANCE','RAW_ONLY','PHYSIOLOGY_ONLY']
        self.late=arm not in ['WO_LATE_GATE','WO_BOTH_GUIDANCE','RAW_ONLY','PHYSIOLOGY_ONLY']
        self.raw_pr=arm!='WO_RAW_PR';self.pre_reference=arm!='WO_RAW_PRE_REFERENCE'
        self.interaction=self.branch=='joint' and arm!='WO_HEAD_INTERACTION'
        keep=torch.ones(7);drop=ARMS.index(arm)-11
        if drop>=0:keep[drop]=0
        self.register_buffer('keep7',keep);self.raw_feature_function=source.raw_feature_function
        if self.branch!='phys':
            self.cnn=copy.deepcopy(source.cnn) if self.early else nn.Sequential(*copy.deepcopy(list(source.cnn.stem.children())),copy.deepcopy(source.cnn.pool))
            if self.late:self.gate=copy.deepcopy(source.gate)
        self.head=copy.deepcopy(source.head) if self.branch=='joint' else subset_head(source,96 if self.branch=='raw' else 14,slice(0,96) if self.branch=='raw' else slice(96,110))
        if self.interaction:self.interaction_gain=nn.Parameter(source.interaction_gain.detach().clone())
        assert sum(isinstance(m,nn.Linear) and m.out_features==1 for m in self.modules())==1

    def features(self,raw,mask,times,delta,scale,permutation=None):
        S,C,W,L=raw.shape;assert L==500 and mask.shape==(S,C,W)
        flat=raw[mask];v=((flat-flat.mean(-1,keepdim=True))/flat.std(-1,unbiased=False,keepdim=True).clamp_min(1e-6)).clamp(-8,8)
        if self.early or self.late:
            assert delta.shape==(S,C,W,7) and not delta.requires_grad
            masked=torch.where(self.keep7.bool(),delta,torch.zeros_like(delta));assert torch.isfinite(masked[mask]).all()
            transform=self.raw_feature_function.__globals__['transform'];phys=transform(masked,mask,times,'full',None);u=(phys/scale).clamp(-8,8)
        else:u=None
        if self.arm=='EARLY_FILM_MISMATCH':
            assert permutation is not None and permutation.shape==(S,C)
            early_u=torch.gather(u,1,permutation[...,None,None].expand(-1,-1,W,7))
            assert torch.equal(torch.gather(mask,1,permutation[...,None].expand(-1,-1,W)),mask)
        else:early_u=u
        emb=raw.new_zeros(S,C,W,24)
        if self.early:emb[mask]=torch.cat([self.cnn(q[:,None],p).flatten(1) for q,p in zip(v.split(4096),early_u[mask].split(4096))])
        else:emb[mask]=torch.cat([self.cnn(q[:,None]).flatten(1) for q in v.split(4096)])
        summaries=[];available=[];activity=[]
        for s in range(S):
            pre=mask[s]&(times[s][None]<0);count=pre.sum(1);present=mask[s].any(1)
            assert (count[present]>0).all()
            hpre=(emb[s]*pre[...,None]).sum(1)/count.clamp_min(1)[:,None]
            latent=emb[s]-hpre[:,None] if self.pre_reference else emb[s]
            post=mask[s]&((times[s]>=0)&(times[s]<20))[None]
            g=self.gate(u[s]) if self.late else torch.ones_like(latent)
            activity.append(g[post]);guided=g*latent;st=[];av=[]
            for low,high in [(0,8),(8,20)]:
                select=mask[s]&((times[s]>=low)&(times[s]<high))[None];n=select.sum(1)
                st.append((guided*select[...,None]).sum(1)/n.clamp_min(1)[:,None]);av.append(n>0)
            summaries.append(torch.stack(st,1));available.append(torch.stack(av,1))
        summary=torch.stack(summaries);valid=torch.stack(available)[...,None].to(raw.dtype);n=valid.sum(0);assert (n>0).all()
        mu=(summary*valid).sum(0)/n;var=((summary-mu[None]).square()*valid).sum(0)/n
        sd=torch.where(var>0,torch.sqrt(var.clamp_min(1e-30)),torch.zeros_like(var));vector=torch.stack([mu,sd],2).reshape(C,96).double()
        z=(vector-vector.mean(0))/torch.where(vector.std(0,unbiased=False)>1e-8,vector.std(0,unbiased=False),torch.ones_like(vector.std(0,unbiased=False))) if self.raw_pr else vector
        return z.to(raw.dtype),torch.cat(activity)

    def classify(self,z,x):
        if self.branch=='raw':return self.head(z*math.sqrt(110/192)).squeeze(-1)
        x=torch.where(self.keep7.repeat_interleave(2).bool(),x,torch.zeros_like(x))
        if self.branch=='phys':return self.head(x*math.sqrt(110/28)).squeeze(-1)
        first=self.head[0];a=F.linear(z*math.sqrt(110/192),first.weight[:,:96]);b=F.linear(x*math.sqrt(110/28),first.weight[:,96:])
        h=a+b+first.bias
        if self.interaction:h=h+.5*torch.tanh(self.interaction_gain)*a*torch.tanh(b)
        return self.head[2](self.head[1](h)).squeeze(-1)

    def forward(self,x,raw=None,mask=None,times=None,delta=None,scale=None,permutation=None):
        if self.branch=='phys':return dict(logit=self.classify(None,x))
        z,g=self.features(raw,mask,times,delta,scale,permutation)
        return dict(logit=self.classify(z,x),raw_summary=z,gate_activity=g)

def objective(logit,y,pos):
    p=logit.sigmoid();a=2*(p*y).sum()/(p.sum()+y.sum()).clamp_min(1e-8);b=2*((1-p)*(1-y)).sum()/((1-p).sum()+(1-y).sum()).clamp_min(1e-8)
    return .75*F.binary_cross_entropy_with_logits(logit,y,pos_weight=pos)+.25*(1-.5*(a+b))
