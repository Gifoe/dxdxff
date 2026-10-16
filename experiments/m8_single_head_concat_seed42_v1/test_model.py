import pytest
torch=pytest.importorskip('torch')
from torch import nn
from model import SingleHeadConcat

def fixture(arm):
    torch.manual_seed(51)
    cnn=nn.Sequential(nn.Conv1d(1,12,9,4),nn.GELU(),nn.Conv1d(12,24,7,4),nn.GELU(),nn.AdaptiveAvgPool1d(1))
    class Gate(nn.Module):
        def __init__(self):super().__init__();self.network=nn.Sequential(nn.Linear(7,32),nn.GELU(),nn.Linear(32,24))
        def forward(self,x):return 1+.25*torch.tanh(self.network(x))
    def features(self,raw,mask,times,delta,scale,mode='full',perms=None):
        z=self.cnn(raw[0,:,0,None,:]).flatten(1)
        if mode=='full':z=z*self.gate(delta[0,:,0]/scale)
        return torch.cat([z,z*2,z*3,z*4],1),z
    return SingleHeadConcat(arm,cnn,Gate(),features,111)

def test_parameter_counts_single_output_and_masking():
    raw=torch.randn(1,6,1,500);mask=torch.ones(1,6,1,dtype=torch.bool);times=torch.zeros(1,1);delta=torch.randn(1,6,1,7);x=torch.randn(6,14);scale=torch.ones(7)
    models=[fixture(f'S{i}') for i in range(5)]
    for i,m in enumerate(models):
        assert not hasattr(m,'base') and sum(p.numel() for p in m.parameters())==(9329 if i<3 else 10377)
        out=m(x,raw,mask,times,delta,scale);assert out['logit'].shape==(6,) and out['raw_summary'].shape==(6,96)
        if i in [0,2,4]:
            xx=x.clone();dd=delta.clone()
            if i==0:xx+=123;dd+=321
            else:xx[:,8:10]+=123;dd[...,4]+=321
            assert torch.equal(out['logit'],m(xx,raw,mask,times,dd,scale)['logit'])
    assert all(torch.equal(models[0].head[0].weight,m.head[0].weight) for m in models)

def test_direct_head_has_raw_and_physiology_gradient():
    m=fixture('S1');z=torch.randn(6,96,requires_grad=True);x=torch.randn(6,14,requires_grad=True)
    m.classify(z,x).square().mean().backward()
    assert z.grad.norm()>0 and x.grad.norm()>0 and m.head[0].weight.grad.norm()>0
