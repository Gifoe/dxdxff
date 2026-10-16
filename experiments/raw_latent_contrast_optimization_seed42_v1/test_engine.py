import copy
import torch
from model import RawModel
from engine import cached,direct


def test_cpu_two_pass_gradients_and_all_raw_inputs():
    torch.manual_seed(37);raw=torch.randn(2,6,7,500)
    mask=torch.ones(2,6,7,dtype=torch.bool);mask[1,4:]=False
    times=torch.tensor([[-3.,-1,1,5,9,15,19]]).repeat(2,1)
    y=torch.tensor([0.,1.,1.,0.,1.,1.]);pos=torch.tensor(.5)
    for attention in [False,True]:
        a=RawModel(1,attention=attention);b=copy.deepcopy(a)
        loss,logit,z=direct(a,raw,mask,times,y,pos,'ez_rank',chunk=11);loss.backward()
        replay,other,zz=cached(b,raw,mask,times,y,pos,'ez_rank',chunk=11)
        assert torch.equal(logit,other) and torch.equal(z,zz) and replay==float(loss.detach())
        for (name,p),(nn,q) in zip(a.named_parameters(),b.named_parameters()):
            assert name==nn and (p.grad is None)==(q.grad is None)
            if p.grad is not None:assert torch.allclose(p.grad,q.grad,atol=1e-6,rtol=1e-5),name
