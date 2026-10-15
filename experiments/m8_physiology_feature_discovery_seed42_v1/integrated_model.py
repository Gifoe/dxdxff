"""Fixed-width matched feature integration with explicit94-position normalization."""
import copy
import torch
from torch import nn
from compact_delta_classifier import CompactDelta


class SlotBase(nn.Module):
    def __init__(self,network,records):
        super().__init__()
        legacy=CompactDelta(network,records)  # Exact legacy constructor RNG consumption.
        self.register_buffer('multiplicities',torch.cat([legacy.multiplicities,torch.ones(6,dtype=torch.float64)]))
        self.n0=legacy.n0;self.positions=94
        assert self.n0==66 and int(self.multiplicities.sum())==28
        with torch.random.fork_rng(devices=[]):
            self.first=nn.Linear(20,96,dtype=network[1].weight.dtype)
        w=network[1].weight.detach().double();a=w*network[0].weight.detach().double()
        active=[i for r in records for i in r['copies']];zero=[i for i in range(88) if i not in active]
        correction=-a[:,zero].sum(1)/self.n0
        with torch.no_grad():
            self.first.weight[:,:14].copy_(legacy.first.weight)
            self.first.weight[:,14:].copy_(correction[:,None].expand(-1,6))
            self.first.bias.copy_(legacy.first.bias)
        self.last=copy.deepcopy(legacy.last);self.dropout=nn.Dropout(.15)
        assert sum(p.numel() for p in self.parameters())==2113

    def normalized(self,x):
        assert x.shape[-1]==20
        m=self.multiplicities.to(x.dtype);mu=(x*m).sum(-1,keepdim=True)/self.positions
        var=(((x-mu).square()*m).sum(-1,keepdim=True)+self.n0*mu.square())/self.positions
        return (x-mu)/torch.sqrt(var+1e-5)

    def forward(self,x,g=None):
        return self.last(self.dropout(torch.nn.functional.gelu(self.first(self.normalized(x))))).squeeze(-1)

    def regularizer(self,counts):return self.first.weight.new_zeros(())


class SlotGate(nn.Module):
    def __init__(self,original):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.network=nn.Sequential(nn.Linear(10,32),nn.GELU(),nn.Linear(32,24))
        with torch.no_grad():
            self.network[0].weight.zero_();self.network[0].weight[:,:7].copy_(original.network[0].weight)
            self.network[0].bias.copy_(original.network[0].bias)
            self.network[2].weight.copy_(original.network[2].weight);self.network[2].bias.copy_(original.network[2].bias)
        assert sum(p.numel() for p in self.parameters())==1144

    def forward(self,u):return 1+.25*torch.tanh(self.network(u))


class IntegratedM8(nn.Module):
    def __init__(self,base,original_fresh_M8):
        super().__init__();self.base=base.eval()
        for p in self.base.parameters():p.requires_grad_(False)
        self.cnn=copy.deepcopy(original_fresh_M8.cnn);self.head=copy.deepcopy(original_fresh_M8.head)
        self.gate=SlotGate(original_fresh_M8.gate)
        assert sum(p.numel() for p in self.parameters() if p.requires_grad)==9577
        assert sum(p.numel() for p in self.parameters())==11690

    def train(self,mode=True):super().train(mode);self.base.eval();return self

    def features(self,raw,mask,times,delta,scale,mode='full',perms=None):
        S,C,W,L=raw.shape
        assert L==500 and delta.shape==(S,C,W,10) and scale.shape==(10,)
        assert not delta.requires_grad and torch.isfinite(delta[mask]).all()
        flat=raw[mask];v=((flat-flat.mean(-1,keepdim=True))/flat.std(-1,unbiased=False,keepdim=True).clamp_min(1e-6)).clamp(-8,8)
        emb=raw.new_zeros(S,C,W,24);emb[mask]=torch.cat([self.cnn(q[:,None]).flatten(1) for q in v.split(4096)])
        u=(delta.clone()/scale).clamp(-8,8)
        if mode.startswith('neutral_'):u[...,int(mode.split('_')[1])]=0
        summaries=[];available=[];activity=[]
        for s in range(S):
            pre=mask[s]&(times[s][None]<0);count=pre.sum(1);present=mask[s].any(1);assert (count[present]>0).all()
            hpre=(emb[s]*pre[...,None]).sum(1)/count.clamp_min(1)[:,None];latent=emb[s]-hpre[:,None]
            post=mask[s]&((times[s]>=0)&(times[s]<20))[None]
            g=torch.ones_like(latent) if mode=='identity' else self.gate(u[s]);activity.append(g[post]);guided=g*latent;st=[];av=[]
            for low,high in [(0,8),(8,20)]:
                select=mask[s]&((times[s]>=low)&(times[s]<high))[None];n=select.sum(1)
                st.append((guided*select[...,None]).sum(1)/n.clamp_min(1)[:,None]);av.append(n>0)
            summaries.append(torch.stack(st,1));available.append(torch.stack(av,1))
        summary=torch.stack(summaries);valid=torch.stack(available)[...,None].to(raw.dtype);n=valid.sum(0);assert (n>0).all()
        mu=(summary*valid).sum(0)/n;var=((summary-mu[None]).square()*valid).sum(0)/n
        sd=torch.where(var>0,torch.sqrt(var.clamp_min(1e-30)),torch.zeros_like(var))
        vector=torch.stack([mu,sd],2).reshape(C,96).double();std=vector.std(0,unbiased=False)
        z=(vector-vector.mean(0))/torch.where(std>1e-8,std,torch.ones_like(std))
        return z.to(raw.dtype),torch.cat(activity)

    def forward(self,x,raw,mask,times,delta,scale,mode='full',perms=None):
        with torch.no_grad():base=self.base(x)
        z,g=self.features(raw,mask,times,delta,scale,mode,perms);aux=self.head(z).squeeze(-1);res=torch.tanh(aux)
        return base if mode=='residual0' else base+res,aux,base,res,g
