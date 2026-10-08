"""Onset-conditioned iEEG SSL with controlled patient-relative / relation ablations.

Inputs are one patient: pair [S,C,2,2500] (pre,ictal), present [S,C].
The recording onset is NEVER inferred from raw waveform shape here: that must be
verified upstream by the audit/export command.
"""
from __future__ import annotations
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def valid_patient_z(u: torch.Tensor, mask: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Z-score only across *this patient's full unlabeled available channel set*."""
    if u.ndim != 2 or mask.shape != (u.shape[0],):
        raise ValueError('expected channel embeddings [C,D] and valid mask [C]')
    weights = mask.to(u.dtype)[:, None]
    count = weights.sum().clamp_min(1.0)
    mu = (u * weights).sum(0, keepdim=True) / count
    var = (((u - mu) * weights).square()).sum(0, keepdim=True) / count
    # No cross-channel covariance whitening; near-constant dims become 0.
    return ((u - mu) / torch.sqrt(var + eps)) * weights


class SmallSpecEncoder(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        self.register_buffer('hann', torch.hann_window(128), persistent=False)
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False), nn.GroupNorm(4,16), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False), nn.GroupNorm(4,32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8,64), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64,width), nn.LayerNorm(width),
        )

    def forward(self, waves: torch.Tensor) -> torch.Tensor:
        # Each pre+ictal pair is normalized using ONLY its preictal baseline.
        # Per-window standardization is forbidden: it would discard transition amplitude.
        if waves.ndim != 3 or waves.shape[1:] != (2,2500):
            raise ValueError('expected [N,2,2500] waves')
        pre = waves[:,0,:]
        mu = pre.mean(-1,keepdim=True)
        sigma = pre.std(-1,unbiased=False,keepdim=True).clamp_min(1e-5)
        x = ((waves - mu.unsqueeze(1)) / sigma.unsqueeze(1)).clamp(-12,12)
        # Keep both segments separately (not a two-channel learned filter).
        a = x.reshape(-1,2500)
        stft = torch.stft(a, n_fft=128, hop_length=64, win_length=128,
                          window=self.hann.to(a.device), center=False, return_complex=True)
        power = stft.abs().square().clamp_min(1e-8)
        log_power = torch.log1p(power)
        return self.net(log_power.unsqueeze(1)).reshape(waves.shape[0],2,-1)


class OnsetEncoder(nn.Module):
    def __init__(self, width: int=64, chunk: int=48):
        super().__init__()
        self.spec = SmallSpecEncoder(width)
        self.transition = nn.Sequential(nn.Linear(width*2,width), nn.GELU(), nn.LayerNorm(width))
        self.chunk = int(chunk)

    def forward(self, pair: torch.Tensor, present: torch.Tensor, augment: bool=False):
        if pair.ndim != 4 or pair.shape[2:] != (2,2500) or present.shape != pair.shape[:2]:
            raise ValueError('expected [S,C,2,2500] and [S,C]')
        flat = pair.reshape(-1,2,2500)
        ids = torch.nonzero(present.reshape(-1), as_tuple=False).flatten()
        all_z = []
        for chunk_ids in ids.split(self.chunk):
            x = flat.index_select(0,chunk_ids)
            if augment:
                # One gain for BOTH phases preserves the onset transition ratio.
                amp = torch.exp(.08*torch.randn((len(x),1,1),device=x.device))
                x = amp*x + .015*torch.randn_like(x)*x[:,0,:].std(-1,keepdim=True).unsqueeze(1).clamp_min(1e-5)
                # Short mask, independent of EZ labels and phase order.
                if x.shape[0]:
                    start = torch.randint(100,2250,(len(x),),device=x.device)
                    times = torch.arange(2500,device=x.device)[None,:]
                    cut = ((times >= start[:,None]) & (times < start[:,None]+50))
                    x = x.masked_fill(cut[:,None,:],0)
            h = self.spec(x)
            out = self.transition(torch.cat([h[:,1],h[:,1]-h[:,0]], dim=-1))
            all_z.append(out)
        z = torch.zeros((flat.shape[0],self.transition[0].out_features),device=pair.device,dtype=pair.dtype)
        if ids.numel():
            z = z.index_copy(0,ids,torch.cat(all_z,dim=0))
        return z.reshape(*present.shape,-1)


def masked_seizure_mean(z:torch.Tensor, present:torch.Tensor):
    weights=present.float().unsqueeze(-1)
    mu=(z*weights).sum(0) / weights.sum(0).clamp_min(1)
    return mu, present.any(0)


class RelativeDifferenceAttention(nn.Module):
    """Optional E4; patient-wise permutation equivariant with explicit pair differences."""
    def __init__(self, d:int=64):
        super().__init__()
        self.q=nn.Linear(d,16,bias=False)
        self.k=nn.Linear(d,16,bias=False)
        self.v=nn.Linear(d,16,bias=False)
        self.pair=nn.Sequential(nn.Linear(d,16),nn.GELU(),nn.Linear(16,1))
        self.out=nn.Sequential(nn.Linear(d*2+16,32),nn.GELU(),nn.Linear(32,1))
        self.alpha=nn.Parameter(torch.tensor(0.0))

    def forward(self,u:torch.Tensor,r:torch.Tensor,mask:torch.Tensor):
        # For large channel sets compute only active pairs, preserving exact permutation equivariance.
        active=torch.nonzero(mask,as_tuple=False).flatten()
        correction=torch.zeros(u.shape[0],device=u.device,dtype=u.dtype)
        if not len(active): return correction
        ua=u[active]; ra=r[active]
        score=(self.q(ra)@self.k(ra).T)/4.0
        difference=ra[:,None,:]-ra[None,:,:]
        score=score+self.pair(difference).squeeze(-1)
        attn=F.softmax(score,dim=-1)
        context=attn@self.v(ua)
        v=self.out(torch.cat([ua,ra,context],-1)).squeeze(-1)
        correction=correction.index_copy(0,active,v)
        return self.alpha*correction


class IctalLocalization(nn.Module):
    def __init__(self, use_pr:bool, use_attention:bool=False, width:int=64):
        super().__init__()
        self.encoder=OnsetEncoder(width)
        self.use_pr=bool(use_pr)
        self.head=nn.Sequential(nn.Linear(width*2,64),nn.GELU(),nn.Dropout(.10),nn.Linear(64,1))
        self.attention=RelativeDifferenceAttention(width) if use_attention else None

    def representation(self, pair:torch.Tensor, present:torch.Tensor):
        z=self.encoder(pair,present,augment=False)
        return masked_seizure_mean(z,present)

    def forward(self,pair:torch.Tensor,present:torch.Tensor):
        u,mask=self.representation(pair,present)
        if not mask.any(): raise RuntimeError('patient has no usable pre/post pairs')
        r=valid_patient_z(u,mask)
        # Same head capacity with PR on or off.  Logits are NEZ positive.
        h=torch.cat([u,r if self.use_pr else torch.zeros_like(r)],-1)
        logits=self.head(h).squeeze(-1)
        if self.attention is not None:
            logits=logits+self.attention(u,r,mask)
        return logits,mask


def vicreg(z1:torch.Tensor,z2:torch.Tensor,sim:float=25.,var:float=25.,cov:float=1.):
    if z1.shape != z2.shape or z1.ndim!=2:raise ValueError('inconsistent SSL sample shape')
    if len(z1)<4:raise ValueError('VICReg requires >=4 valid channel-seizure pairs')
    inv=F.mse_loss(z1,z2)
    std1=torch.sqrt(z1.var(0,unbiased=True)+1e-4)
    std2=torch.sqrt(z2.var(0,unbiased=True)+1e-4)
    variance=(F.relu(1-std1).mean()+F.relu(1-std2).mean())/2
    def decorrelate(z):
        z=z-z.mean(0)
        c=(z.T@z)/(len(z)-1)
        off=c-torch.diag(torch.diagonal(c))
        return off.square().sum()/z.shape[-1]
    covariance=decorrelate(z1)+decorrelate(z2)
    return sim*inv+var*variance+cov*covariance, (float(inv.detach()),float(variance.detach()),float(covariance.detach()))


def patient_bce(logits:torch.Tensor, labels_ez:torch.Tensor, mask:torch.Tensor):
    # Reuse exact A1 objective semantics: EZ=0, NEZ=1; EZ weight 2, NEZ weight 1.
    y_nez=1-labels_ez
    valid=mask & (labels_ez>=0)
    weights=torch.where(labels_ez>0.5,2.0,1.0)*valid.float()
    return (F.binary_cross_entropy_with_logits(logits,y_nez.clamp(0,1),reduction='none')*weights).sum()/weights.sum().clamp_min(1)


def reset_same_supervised_init(seed:int,with_pr:bool,attention:bool=False):
    torch.manual_seed(seed)
    return IctalLocalization(use_pr=with_pr,use_attention=attention)
