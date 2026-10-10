"""Rank-configurable original D2, state-dict-compatible R4; R0 bias only."""
import numpy as np
import torch

SOURCES=['HUP','LZU','multicenter','pediatric']
def source_index(center):
    lookup={s.lower():i for i,s in enumerate(SOURCES)}
    return np.asarray([lookup.get(str(s).lower(),-1) for s in center],dtype=np.int64)

class SourceA0(torch.nn.Module):
    def __init__(self,seed=42,rank=4):
        super().__init__(); assert rank in [0,1,2,4]; self.rank=rank
        self.network=torch.nn.Sequential(torch.nn.LayerNorm(88),torch.nn.Linear(88,96),torch.nn.GELU(),torch.nn.Dropout(.15),torch.nn.Linear(96,1))
        if rank:
            gen=torch.Generator(device='cpu').manual_seed(42+7000+seed)
            self.V=torch.nn.Parameter(.01*torch.randn(rank,96,generator=gen))
            self.u=torch.nn.Parameter(torch.zeros(4,rank))
        self.b=torch.nn.Parameter(torch.zeros(4))
        assert sum(p.numel() for p in self.parameters())==8821+100*rank

    def components(self,x,g):
        h=self.network[:4](x); base=self.network[4](h).squeeze(-1); idx=g.clamp(0,3)
        direction=((h@self.V.T)*self.u[idx]).sum(-1) if self.rank else torch.zeros_like(base)
        known=(g>=0)&(g<4)
        return base,torch.where(known,direction,torch.zeros_like(base)),torch.where(known,self.b[idx],torch.zeros_like(base))

    def parts(self,x,g):
        base,direction,bias=self.components(x,g); return base,direction+bias

    def forward(self,x,g):
        base,delta=self.parts(x,g); return base+delta

    def regularizer(self,counts):
        weights=(counts.sum()/4)/counts.clamp(min=1)
        if not self.rank:return (self.b.square()*weights).mean()
        return self.V.square().mean()+((self.u.square().mean(1)+self.b.square())*weights).mean()

    def effective_matrix(self):
        if not self.rank:return np.zeros((4,96),dtype=np.float64)
        return self.u.detach().cpu().numpy().astype('float64')@self.V.detach().cpu().numpy().astype('float64')
