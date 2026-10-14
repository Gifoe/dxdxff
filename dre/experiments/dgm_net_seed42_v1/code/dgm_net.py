"""DGM A2: frozen D1 + morphology/physiology interaction, no recurrent module."""
import torch
from torch import nn
from physiology_gate import PhysiologyGate
from controls import transform

class DGMNet(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base=base.eval()
        for p in base.parameters():p.requires_grad_(False)
        # Shared morphology/head created before the conditioner consumes RNG.
        self.cnn=nn.Sequential(nn.Conv1d(1,12,9,4),nn.GELU(),nn.Conv1d(12,24,7,4),nn.GELU(),nn.AdaptiveAvgPool1d(1))
        self.head=nn.Sequential(nn.Linear(96,64),nn.GELU(),nn.Linear(64,1))
        self.alpha=nn.Parameter(torch.zeros(()))
        self.gate=PhysiologyGate()
        assert sum(p.numel() for p in self.parameters())==11019
        assert sum(p.numel() for p in self.parameters() if p.requires_grad)==9482

    def train(self,mode=True):
        super().train(mode); self.base.eval(); return self

    def features(self, raw, mask, times, delta, scale, mode='full', perms=None):
        S,C,W,L=raw.shape
        assert L==500 and mask.shape==(S,C,W) and times.shape==(S,W) and delta.shape==(S,C,W,7)
        assert not delta.requires_grad and torch.isfinite(delta[mask]).all()
        flat=raw[mask]
        v=((flat-flat.mean(-1,keepdim=True))/flat.std(-1,unbiased=False,keepdim=True).clamp_min(1e-6)).clamp(-8,8)
        emb=raw.new_zeros(S,C,W,24)
        emb[mask]=torch.cat([self.cnn(q[:,None]).flatten(1) for q in v.split(4096)])
        phys=transform(delta,mask,times,mode,perms)
        u=(phys/scale).clamp(-8,8)
        if mode.startswith('neutral_'):
            u[...,int(mode.split('_')[1])]=0
        summaries=[]; available=[]; activity=[]
        for s in range(S):
            pre=mask[s]&(times[s][None]<0); count=pre.sum(1); present=mask[s].any(1)
            assert (count[present]>0).all(), 'No genuine pre-reference'
            hpre=(emb[s]*pre[...,None]).sum(1)/count.clamp_min(1)[:,None]
            latent=emb[s]-hpre[:,None]
            post=mask[s]&((times[s]>=0)&(times[s]<20))[None]
            g=torch.ones_like(latent) if mode=='identity' else self.gate(u[s])
            activity.append(g[post])
            guided=g*latent
            st=[]; av=[]
            for low,high in [(0,8),(8,20)]:
                select=mask[s]&((times[s]>=low)&(times[s]<high))[None]
                n=select.sum(1)
                st.append((guided*select[...,None]).sum(1)/n.clamp_min(1)[:,None]); av.append(n>0)
            summaries.append(torch.stack(st,1)); available.append(torch.stack(av,1))
        summary=torch.stack(summaries); valid=torch.stack(available)[...,None].to(raw.dtype); n=valid.sum(0)
        assert (n>0).all(), 'Missing stage cannot be silently imputed'
        mu=(summary*valid).sum(0)/n
        var=((summary-mu[None]).square()*valid).sum(0)/n
        sd=torch.where(var>0,torch.sqrt(var.clamp_min(1e-30)),torch.zeros_like(var))
        vector=torch.stack([mu,sd],2).reshape(C,96).double()
        z=(vector-vector.mean(0))/torch.where(vector.std(0,unbiased=False)>1e-8,vector.std(0,unbiased=False),torch.ones_like(vector.std(0,unbiased=False)))
        return z.to(raw.dtype), torch.cat(activity)

    def forward(self,x,raw,mask,times,delta,scale,mode='full',perms=None):
        with torch.no_grad():base=self.base(x)
        z,g=self.features(raw,mask,times,delta,scale,mode,perms)
        aux=self.head(z).squeeze(-1)
        residual=.5*torch.tanh(self.alpha)*torch.tanh(aux)
        final=base if mode=='residual0' else base+residual
        return final,aux,base,residual,g
