import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from dgm_net import DGMNet
from physiology_gate import PhysiologyGate
from controls import permutations,transform
torch.set_num_threads(2)

class DummyD1(nn.Module):
    def __init__(self):
        super().__init__(); self.first=nn.Linear(14,96); self.last=nn.Linear(96,1)
    def forward(self,x):return self.last(torch.nn.functional.gelu(self.first(x))).squeeze(-1)

def sample():
    torch.manual_seed(7);raw=torch.randn(2,8,7,500);mask=torch.ones(2,8,7,dtype=torch.bool)
    times=torch.tensor([[-4,-2,0,2,8,10,12],[-4,-2,0,2,8,10,12]],dtype=torch.float32)
    delta=torch.randn(2,8,7,7);scale=torch.ones(7);x=torch.randn(8,14)
    return x,raw,mask,times,delta,scale

def test_parameters_identity():
    m=DGMNet(DummyD1()).eval();args=sample()
    with torch.no_grad():a=m(*args);b=m(*args,mode='identity')
    assert sum(p.numel() for p in m.parameters())==11019
    assert torch.equal(a[0],a[2]) and torch.equal(a[1],b[1]) and torch.equal(a[4],torch.ones_like(a[4]))
    assert not any(isinstance(q,(nn.GRU,nn.LSTM,nn.MultiheadAttention)) for q in m.modules())

def test_permutation_static_mask():
    args=sample();_,raw,mask,times,delta,scale=args
    plans=[dict(times=t.numpy(),compound_key=('x','run',str(i))) for i,t in enumerate(times)]
    p,a=permutations('x',[str(i) for i in range(8)],plans,mask.numpy());p2,a2=permutations('x',[str(i) for i in range(8)],plans,mask.numpy())
    assert a==a2 and a['eligible']==a['changed'] and all(np.array_equal(x,y) for x,y in zip(p,p2))
    moved=transform(delta,mask,times,'shuffle',p);assert torch.equal(delta[:,:,:2],moved[:,:,:2])
    for s in range(2):
        for c in range(8):
            for ix in [[2,3],[4,5,6]]:
                assert sorted(map(tuple,delta[s,c,ix].numpy()))==sorted(map(tuple,moved[s,c,ix].numpy()))
    static=transform(delta,mask,times,'static');assert torch.equal(static[:,:,2],static[:,:,3])
    assert torch.allclose(static[:,:,2],delta[:,:,2:4].mean(2))

def test_joint_stage_permutation_invariance():
    m=DGMNet(DummyD1()).eval();args=list(sample())
    with torch.no_grad():m.gate.network[-1].weight.normal_(0,.04)
    original=m(*args)[1];perm=torch.tensor([0,1,3,2,6,4,5]);other=args.copy()
    other[1]=args[1][:,:,perm];other[4]=args[4][:,:,perm]
    assert torch.allclose(original,m(*other)[1],atol=2e-6,rtol=0)

def test_gate_gradients_and_no_physiology_gradient():
    m=DGMNet(DummyD1());args=sample();opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001)
    for step in range(2):
        opt.zero_grad();a=m(*args);loss=nn.functional.binary_cross_entropy_with_logits(a[0],torch.arange(8).float()%2)+.25*nn.functional.binary_cross_entropy_with_logits(a[1],torch.arange(8).float()%2)
        loss.backward();assert m.gate.network[-1].weight.grad.norm()>0
        if step:assert m.gate.network[0].weight.grad.norm()>0
        assert all(p.grad is None for p in m.base.parameters());opt.step()
