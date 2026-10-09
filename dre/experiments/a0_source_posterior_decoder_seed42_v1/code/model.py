"""One shared A0 plus jointly trained rank-four operational-source correction."""
import torch
import numpy as np

SOURCES = ['HUP', 'LZU', 'multicenter', 'pediatric']

def source_index(center):
    lookup={v.lower():i for i,v in enumerate(SOURCES)}
    return np.asarray([lookup.get(str(v).lower(),-1) for v in center], dtype=np.int64)

class SourceA0(torch.nn.Module):
    def __init__(self, seed=42):
        super().__init__()
        self.network=torch.nn.Sequential(torch.nn.LayerNorm(88),torch.nn.Linear(88,96),
            torch.nn.GELU(),torch.nn.Dropout(.15),torch.nn.Linear(96,1))
        gen=torch.Generator(device='cpu').manual_seed(42+7000+seed)
        self.V=torch.nn.Parameter(.01*torch.randn(4,96,generator=gen))
        self.u=torch.nn.Parameter(torch.zeros(4,4))
        self.b=torch.nn.Parameter(torch.zeros(4))
        assert sum(p.numel() for p in self.parameters())==9221

    def parts(self,x,g):
        h=self.network[:4](x)
        base=self.network[4](h).squeeze(-1)
        idx=g.clamp(0,3)
        delta=((h@self.V.T)*self.u[idx]).sum(-1)+self.b[idx]
        delta=torch.where((g>=0)&(g<4),delta,torch.zeros_like(delta))
        return base,delta

    def forward(self,x,g):
        base,delta=self.parts(x,g)
        return base+delta

    def regularizer(self,counts):
        weights=(counts.sum()/4)/counts.clamp(min=1)
        return self.V.square().mean()+((self.u.square().mean(1)+self.b.square())*weights).mean()
