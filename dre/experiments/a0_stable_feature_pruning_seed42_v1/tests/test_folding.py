"""Synthetic double/float parity, arbitrary learned affine, pruning independence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
import numpy as np
import torch
from folding import Compact,inflate


def test_fold():
    torch.set_num_threads(2);torch.manual_seed(42)
    records=[dict(index=j,copies=[j]+([42+j] if j<8 else []),multiplicity=2 if j<8 else 1) for j in range(42)]
    for dtype,tol in [(torch.float64,1e-10),(torch.float32,3e-6)]:
        net=torch.nn.Sequential(torch.nn.LayerNorm(88),torch.nn.Linear(88,96),torch.nn.GELU(),torch.nn.Dropout(.15),torch.nn.Linear(96,1)).to(dtype).eval()
        with torch.no_grad():net[0].weight.uniform_(.5,1.5);net[0].bias.uniform_(-.2,.2)
        for subset in [records,records[6:],records[::2]]:
            m=Compact(net,subset).eval().to(dtype)
            for scale in [0,1e-8,1,1e4]:
                x=torch.randn(17,len(subset),dtype=dtype)*scale
                assert (m(x)-net(inflate(x,subset)).squeeze(-1)).abs().max()<tol
            full=inflate(x,subset);other=full.clone();deleted=[i for i in range(88) if i not in [q['index'] for q in subset]]
            other[:,deleted]=torch.randn(17,len(deleted),dtype=dtype)
            assert torch.equal(m(full),m(other))
            assert sum(p.numel() for p in m.parameters())==96*len(subset)+193
