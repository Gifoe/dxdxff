"""Strict Raw-only shared encoder, factorial pooling, and latent-only attention."""
import itertools
import torch
from torch import nn
import torch.nn.functional as F

FACTORS = list(itertools.product([False, True], repeat=4))


def recipe_key(recipe):
    return ''.join(str(int(x)) for x in recipe)


def cnn24():
    return nn.Sequential(nn.Conv1d(1,12,9,4),nn.GELU(),nn.Conv1d(12,24,7,4),
                         nn.GELU(),nn.AdaptiveAvgPool1d(1))


def normalize_windows(x):
    return ((x-x.mean(-1,keepdim=True))/x.std(-1,unbiased=False,keepdim=True).clamp_min(1e-6)).clamp(-8,8)


def patient_z(x):
    x=x.double()
    sd=x.std(0,unbiased=False)
    z=(x-x.mean(0))/torch.where(sd>1e-8,sd,torch.ones_like(sd))
    constant=x.max(0).values==x.min(0).values
    return torch.where(constant[None],torch.zeros_like(z),z).float()


def weighted_mean(value, valid, scorer=None, dim=0):
    """Value[...,D], validity without D; unavailable entries cannot get weight."""
    assert valid.dtype==torch.bool
    assert valid.any(dim=dim).all(), 'Missing genuine phase/seizure support'
    if scorer is None:
        return (value*valid[...,None]).sum(dim)/valid.sum(dim)[...,None]
    score=scorer(F.layer_norm(value,(24,),eps=1e-5)).squeeze(-1)
    weight=score.masked_fill(~valid,-torch.inf).softmax(dim)
    return (value*weight[...,None]).sum(dim)


def pool_embeddings(emb,mask,times,recipe,window_attention=None,seizure_attention=None):
    reference,stages,spread,pr=map(bool,recipe)
    S,C,W,D=emb.shape
    assert D==24 and mask.shape==(S,C,W) and times.shape==(S,W)
    summaries=[];available=[]
    for s in range(S):
        present=mask[s].any(1)
        pre=mask[s]&(times[s][None]<0)
        if reference:
            n=pre.sum(1)
            assert (n[present]>0).all(), 'Missing genuine pre-onset reference'
            hpre=(emb[s]*pre[...,None]).sum(1)/n.clamp_min(1)[:,None]
            h=emb[s]-hpre[:,None]
        else:
            h=emb[s]
        values=[];flags=[]
        for low,high in ([(0,8),(8,20)] if stages else [(0,20),(0,20)]):
            use=mask[s]&((times[s]>=low)&(times[s]<high))[None]
            n=use.sum(1)
            assert (n[present]>0).all(), 'Missing stage is not silently imputed'
            # Absent channels have zero contribution and remain unavailable.
            if window_attention is None:
                mean=(h*use[...,None]).sum(1)/n.clamp_min(1)[:,None]
            else:
                score=window_attention(F.layer_norm(h,(24,),eps=1e-5)).squeeze(-1)
                score=score.masked_fill(~use,-torch.inf)
                score=torch.where(present[:,None],score,torch.zeros_like(score))
                weight=score.softmax(1)*use
                mean=(h*weight[...,None]).sum(1)
            values.append(mean);flags.append(n>0)
        summaries.append(torch.stack(values,1));available.append(torch.stack(flags,1))
    value=torch.stack(summaries);valid=torch.stack(available)
    mean=weighted_mean(value,valid,seizure_attention,0)
    if seizure_attention is None:
        var=((value-mean[None]).square()*valid[...,None]).sum(0)/valid.sum(0)[...,None]
    else:
        score=seizure_attention(F.layer_norm(value,(24,),eps=1e-5)).squeeze(-1)
        weight=score.masked_fill(~valid,-torch.inf).softmax(0)
        var=((value-mean[None]).square()*weight[...,None]).sum(0)
    sd=torch.where(var>0,torch.sqrt(var.clamp_min(1e-30)),torch.zeros_like(var)) if spread else torch.zeros_like(mean)
    vector=torch.stack([mean,sd],2).reshape(C,96)
    return patient_z(vector) if pr else vector


class RawModel(nn.Module):
    def __init__(self,fold=1,recipe=(1,1,1,1),attention=False):
        super().__init__();self.recipe=tuple(recipe);self.attention=bool(attention)
        seed=42+1009*fold
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);self.cnn=cnn24()
            torch.manual_seed(seed+170003)
            self.head=nn.Sequential(nn.Linear(96,64),nn.GELU(),nn.Linear(64,1))
            if attention:
                torch.manual_seed(seed+270003)
                self.window_attention=nn.Sequential(nn.Linear(24,8),nn.GELU(),nn.Linear(8,1))
                self.seizure_attention=nn.Sequential(nn.Linear(24,8),nn.GELU(),nn.Linear(8,1))
                for module in [self.window_attention,self.seizure_attention]:
                    nn.init.zeros_(module[-1].weight);nn.init.zeros_(module[-1].bias)

    def aggregate(self,emb,mask,times):
        return pool_embeddings(emb,mask,times,self.recipe,
            self.window_attention if self.attention else None,
            self.seizure_attention if self.attention else None)

    def classify(self,x):
        return self.head(x).squeeze(-1)


def objective(logit_nez,y_nez,pos_weight,kind):
    bce=F.binary_cross_entropy_with_logits(logit_nez,y_nez,pos_weight=pos_weight)
    p=logit_nez.sigmoid();ez=1-y_nez
    soft_ez=2*((1-p)*ez).sum()/((1-p).sum()+ez.sum()).clamp_min(1e-8)
    if kind=='macro':
        soft_nez=2*(p*y_nez).sum()/(p.sum()+y_nez.sum()).clamp_min(1e-8)
        return .75*bce+.25*(1-.5*(soft_ez+soft_nez))
    assert kind=='ez_rank'
    a=logit_nez[y_nez==0];b=logit_nez[y_nez==1]
    rank=F.softplus(a[:,None]-b[None]).mean() if a.numel() and b.numel() else logit_nez.sum()*0
    return .5*bce+.25*(1-soft_ez)+.25*rank
