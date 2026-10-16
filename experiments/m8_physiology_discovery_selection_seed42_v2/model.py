"""Fixed global feature slots; D1, single-head concat and bounded M8 correction."""
import copy
import torch
from torch import nn


class FeatureHead(nn.Module):
    def __init__(self, seed=42):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.first=nn.Linear(64,96);self.dropout=nn.Dropout(.15);self.last=nn.Linear(96,1)
        assert sum(p.numel() for p in self.parameters())==6337

    def forward(self,x,g=None):
        z=torch.nn.functional.layer_norm(x,(64,),eps=1e-5)
        return self.last(self.dropout(torch.nn.functional.gelu(self.first(z)))).squeeze(-1)

    def regularizer(self,counts):return self.first.weight.new_zeros(())


class Gate(nn.Module):
    def __init__(self,original,seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network=nn.Sequential(nn.Linear(32,32),nn.GELU(),nn.Linear(32,24))
            nn.init.uniform_(self.network[0].weight,-7**-.5,7**-.5)
        with torch.no_grad():
            self.network[0].weight[:,:7].copy_(original.network[0].weight)
            self.network[0].bias.copy_(original.network[0].bias)
            self.network[2].weight.copy_(original.network[2].weight)
            self.network[2].bias.copy_(original.network[2].bias)
        assert sum(p.numel() for p in self.parameters())==1848

    def forward(self,x):return 1+.25*torch.tanh(self.network(x))


class RawFeatures:
    def features(self,raw,mask,times,delta,scale,mode='full',perms=None):
        S,C,W,L=raw.shape
        assert L==500 and delta.shape==(S,C,W,32) and scale.shape==(32,)
        assert not delta.requires_grad and torch.isfinite(delta).all()
        flat=raw[mask]
        v=((flat-flat.mean(-1,keepdim=True))/flat.std(-1,unbiased=False,keepdim=True).clamp_min(1e-6)).clamp(-8,8)
        emb=raw.new_zeros(S,C,W,24)
        emb[mask]=torch.cat([self.cnn(q[:,None]).flatten(1) for q in v.split(4096)])
        u=((delta/scale).clamp(-8,8))*self.keep32
        summaries=[];available=[];activity=[]
        for s in range(S):
            pre=mask[s]&(times[s][None]<0);count=pre.sum(1);present=mask[s].any(1)
            assert (count[present]>0).all()
            reference=(emb[s]*pre[...,None]).sum(1)/count.clamp_min(1)[:,None]
            latent=emb[s]-reference[:,None]
            post=mask[s]&((times[s]>=0)&(times[s]<20))[None]
            g=torch.ones_like(latent) if mode=='identity' else self.gate(u[s])
            activity.append(g[post]);guided=g*latent;st=[];av=[]
            for low,high in [(0,8),(8,20)]:
                select=mask[s]&((times[s]>=low)&(times[s]<high))[None];n=select.sum(1)
                st.append((guided*select[...,None]).sum(1)/n.clamp_min(1)[:,None]);av.append(n>0)
            summaries.append(torch.stack(st,1));available.append(torch.stack(av,1))
        summary=torch.stack(summaries);valid=torch.stack(available)[...,None].to(raw.dtype);n=valid.sum(0)
        assert (n>0).all()
        mu=(summary*valid).sum(0)/n;var=((summary-mu[None]).square()*valid).sum(0)/n
        sd=torch.where(var>0,torch.sqrt(var.clamp_min(1e-30)),torch.zeros_like(var))
        vector=torch.stack([mu,sd],2).reshape(C,96).double();std=vector.std(0,unbiased=False)
        z=(vector-vector.mean(0))/torch.where(std>1e-8,std,torch.ones_like(std))
        return z.to(raw.dtype),torch.cat(activity)


class M8(RawFeatures,nn.Module):
    def __init__(self,base,original,groups,seed):
        super().__init__();self.base=base.eval()
        for p in self.base.parameters():p.requires_grad_(False)
        self.cnn=copy.deepcopy(original.cnn);self.head=copy.deepcopy(original.head)
        self.gate=Gate(original.gate,seed+310003)
        keep=torch.zeros(32);keep[groups]=1;self.register_buffer('keep32',keep)
        assert sum(p.numel() for p in self.parameters())==16618
        assert sum(p.numel() for p in self.parameters() if p.requires_grad)==10281

    def train(self,mode=True):super().train(mode);self.base.eval();return self

    def forward(self,x,raw,mask,times,delta,scale,mode='full',perms=None):
        with torch.no_grad():base=self.base(x*self.keep32.repeat_interleave(2))
        z,g=self.features(raw,mask,times,delta,scale,mode,perms);aux=self.head(z).squeeze(-1);res=torch.tanh(aux)
        return base if mode=='residual0' else base+res,aux,base,res,g


class Concat(RawFeatures,nn.Module):
    def __init__(self,original,groups,seed):
        super().__init__();self.cnn=copy.deepcopy(original.cnn)
        keep=torch.zeros(32);keep[groups]=1;self.register_buffer('keep32',keep)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+170003)
            self.head=nn.Sequential(nn.Linear(160,64),nn.GELU(),nn.Linear(64,1))
        assert sum(p.numel() for p in self.parameters())==12529 and not hasattr(self,'base')

    def forward(self,x,raw,mask,times,delta,scale,mode='full',perms=None):
        z,g=self.features(raw,mask,times,delta,scale,mode='identity')
        logit=self.head(torch.cat([z,x*self.keep32.repeat_interleave(2)],-1)).squeeze(-1)
        zero=torch.zeros_like(logit)
        return logit,logit,zero,logit,g
