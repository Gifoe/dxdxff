import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from exploration_models import DeltaResidual,pool
torch.set_num_threads(2)

class D1(nn.Module):
    def __init__(self):super().__init__();self.first=nn.Linear(14,96);self.last=nn.Linear(96,1)
    def forward(self,x):return self.last(torch.nn.functional.gelu(self.first(x))).squeeze(-1)

def fixture():
    torch.manual_seed(42);t=torch.tensor([[-4,-2,0,2,8,10,12],[-4,-2,0,2,8,10,12]],dtype=torch.float32);m=torch.ones(2,9,7,dtype=torch.bool);m[1,0]=False
    return torch.randn(9,14),torch.empty(0),m,t,torch.randn(2,9,7,7),torch.ones(7)

def test_parameter_budgets_and_initial_D1_identity():
    args=fixture()
    for arm,n in [('M2_STAGE_DELTA',722),('M3_DELTA_SET',1354)]:
        model=DeltaResidual(D1(),arm).eval();q=model(*args)
        assert sum(p.numel() for p in model.parameters())==1537+n and torch.equal(q[0],q[2])
        assert not any(isinstance(m,(nn.GRU,nn.LSTM,nn.MultiheadAttention)) for m in model.modules())

def test_mask_and_channel_equivariance_and_no_raw_use():
    args=list(fixture())
    for arm in ['M2_STAGE_DELTA','M3_DELTA_SET']:
        model=DeltaResidual(D1(),arm).eval();model.alpha.data.fill_(.1);q=model(*args)[0]
        p=torch.randperm(9);other=args.copy();other[0]=args[0][p];other[2]=args[2][:,p];other[4]=args[4][:,p]
        assert torch.allclose(q[p],model(*other)[0],atol=1e-6,rtol=0)
        other=args.copy();other[4]=args[4].clone();other[4][~args[2]]=1e20;other[1]=torch.randn(2,9,7,500)
        assert torch.equal(q,model(*other)[0])

def test_window_permutation_invariance_and_finite_backward():
    args=list(fixture());p=torch.tensor([0,1,3,2,6,4,5]);other=args.copy();other[4]=args[4][:,:,p]
    for arm in ['M2_STAGE_DELTA','M3_DELTA_SET']:
        model=DeltaResidual(D1(),arm);q=model(*args);qq=model(*other)
        assert torch.allclose(q[1],qq[1],atol=1e-6,rtol=0)
        (q[0].sum()+.25*q[1].square().mean()).backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert all(p.grad is None for p in model.base.parameters())

def test_relative_scale_preserves_native_units():
    d=np.random.default_rng(42).normal(size=(4,71,7))*np.array([1e-10,1e-11,1e-12,1e-5,1e-9,1e-4,.02])
    s=np.maximum(np.median(np.median(np.abs(d),axis=1),axis=0),1e-30)
    convert=np.array([1e12,1e12,1e12,1e6,1e12,1e6,1.]);ss=np.maximum(np.median(np.median(np.abs(d*convert),axis=1),axis=0),1e-30)
    assert np.allclose((d/s).clip(-8,8),(d*convert/ss).clip(-8,8),atol=1e-12,rtol=0)
