"""Exact masked leave-one-channel-out robust references, not an all-pairs tensor.

Reference statistics are stop-gradient background in ALL three controls. The
recipient r and q-to-fusion paths remain differentiable. Four channel-axis sorts
compute exact even/odd LOO median/MAD using at most three possible LOO centers.
"""
import hashlib
import torch
from torch import nn

MIN_OTHERS=3
EPSILON=1e-4
MAD_FLOOR=1e-6
CLIP=8.

def sorted_rank(x,valid):
    values,order=torch.sort(torch.where(valid,x,torch.full_like(x,float('inf'))),dim=0,stable=True)
    ranks=torch.empty_like(order)
    ranks.scatter_(0,order,torch.arange(x.shape[0],device=x.device)[:,None].expand_as(order))
    return values,ranks

def omit_quantile(values,ranks,n):
    low=((n-2)//2).clamp(0,values.shape[0]-1)
    high=((n-1)//2).clamp(0,values.shape[0]-1)
    li=(low[None]+(ranks<=low[None])).clamp_max(values.shape[0]-1)
    hi=(high[None]+(ranks<=high[None])).clamp_max(values.shape[0]-1)
    return .5*(values.gather(0,li)+values.gather(0,hi))

@torch.no_grad()
def loo_reference(r,mask):
    """r[C,W,D], mask[C,W]; no channel or time with invalid support contributes."""
    C,W,D=r.shape;assert C>0 and mask.shape==(C,W)
    x=r.reshape(C,W*D);valid=mask[:,:,None].expand(-1,-1,D).reshape_as(x)
    n=valid.sum(0);eligible=n>=MIN_OTHERS+1
    v,rank=sorted_rank(x,valid)
    low=((n-2)//2).clamp(0,C-1);high=((n-1)//2).clamp(0,C-1)
    centers=[.5*(v.gather(0,low[None]).squeeze(0)+v.gather(0,high[None]).squeeze(0)),
        .5*(v.gather(0,low[None]).squeeze(0)+v.gather(0,(high+1).clamp_max(C-1)[None]).squeeze(0)),
        .5*(v.gather(0,(low+1).clamp_max(C-1)[None]).squeeze(0)+v.gather(0,(high+1).clamp_max(C-1)[None]).squeeze(0))]
    kind=torch.where(rank<=low[None],2,torch.where(rank>high[None],0,1))
    median=torch.zeros_like(x);mad=torch.zeros_like(x)
    for k,center in enumerate(centers):
        center=torch.where(eligible,center,torch.zeros_like(center))
        dv,drank=sorted_rank((x-center[None]).abs(),valid)
        dm=omit_quantile(dv,drank,n)
        chosen=(kind==k)&valid&eligible[None]
        median=torch.where(chosen,center[None],median)
        mad=torch.where(chosen,dm,mad)
    ok=valid&eligible[None]&(mad>MAD_FLOOR)
    return median.reshape_as(r),mad.reshape_as(r),ok.reshape_as(r)

def relative(r,median,mad,ok):
    # Avoid NaN/inf even in a fully masked or constant-coordinate fallback.
    numerator=r-torch.where(ok,median,torch.zeros_like(median))
    denominator=torch.where(ok,mad+EPSILON,torch.ones_like(mad))
    return torch.where(ok,(numerator/denominator).clamp(-CLIP,CLIP),torch.zeros_like(r))

@torch.no_grad()
def shifted_reference(median,mad,ok,mask,times,fold,seizure):
    """Wrong-time control: identical FULL channel support, within one seizure.

    Same fixed nonzero cyclic shift for all latent dimensions/candidates in a
    support class. Singleton time groups retain correct references and are counted.
    No patient from a different case or held-out cohort is ever read here.
    """
    C,W,D=median.shape;mapping=torch.arange(W,device=times.device);groups={}
    support=mask.detach().cpu().numpy();t=times.detach().cpu().numpy()
    for w in range(W):
        if 0<=float(t[w])<20 and support[:,w].any():groups.setdefault(support[:,w].tobytes(),[]).append(w)
    movable=total=0
    for signature,members in groups.items():
        members=sorted(members,key=lambda w:float(t[w]));total+=int(mask[:,members].sum())
        if len(members)<2:continue
        key=f'42|{fold}|{seizure}|'.encode()+signature
        shift=1+int.from_bytes(hashlib.sha256(key).digest()[:8],'little')%(len(members)-1)
        donor=members[-shift:]+members[:-shift]
        mapping[torch.tensor(members,device=times.device)]=torch.tensor(donor,device=times.device)
        movable+=int(mask[:,members].sum())
    assert torch.equal(mask.index_select(1,mapping),mask)
    return median.index_select(1,mapping),mad.index_select(1,mapping),ok.index_select(1,mapping),dict(time_shifted_valid_windows=movable,time_singleton_valid_windows=total-movable,post_valid_windows=total)

class Fusion(nn.Module):
    def __init__(self,seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
            self.network=nn.Sequential(nn.Linear(48,24),nn.GELU(),nn.Linear(24,24))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        assert sum(p.numel() for p in self.parameters())==1776
    def forward(self,r,q):return r+.1*torch.tanh(self.network(torch.cat([r,q],-1)))
