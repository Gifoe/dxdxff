import numpy as np
import torch
from torch import nn
from spectral_model import ARMS,SpectralCorrection,spectrum,objective

class Base(nn.Module):
    def __init__(self):super().__init__();self.weights=nn.Parameter(torch.zeros(1537))
    def forward(self,x):return x[:,0]

def args():
    torch.manual_seed(42);mask=torch.ones(2,6,4,dtype=torch.bool);mask[0,5]=False
    return torch.randn(6,14),torch.randn(2,6,4,500),mask,torch.tensor([[-3.,-1.,3.,10.],[-3.,-1.,3.,10.]]),torch.randn(2,6,4,7),torch.ones(7)

def test_native_fft_against_independent_numpy():
    torch.manual_seed(42);v=torch.randn(4,500);h=np.hanning(501)[:-1];p=np.abs(np.fft.rfft(v.numpy()*h,axis=-1))**2/np.sum(h*h);s=np.log(p[:,20:251]+1e-8);reference=(s-s.mean(-1,keepdims=True))/s.std(-1,keepdims=True)
    assert np.allclose(spectrum(v).numpy(),reference,atol=1e-5,rtol=1e-5)
    assert spectrum(v).shape==(4,231) and torch.isfinite(spectrum(torch.zeros(2,500))).all()

def test_physical_frequency_coordinates():
    t=torch.arange(500)/250.;v=torch.sin(2*torch.pi*30*t);s=spectrum(v[None]);assert int(s.argmax())==40
    assert np.array_equal(np.fft.rfftfreq(500,1/250)[20:251],np.arange(10,125.5,.5))

def test_identity_masking_and_permutation():
    x,r,mask,t,d,s=args()
    for arm in ARMS:
        m=SpectralCorrection(Base(),arm).eval()
        with torch.no_grad():assert torch.equal(m(x,r,mask,t,d,s)[0],m.base(x))
        nn.init.normal_(m.head[-1].weight,std=.02);q=m(x,r,mask,t,d,s)[0];rr=r.clone();rr[~mask]=1000
        assert torch.allclose(q,m(x,rr,mask,t,d,s)[0],rtol=0,atol=1e-6)
        p=torch.tensor([5,2,0,4,1,3]);assert torch.allclose(q[p],m(x[p],r[:,p],mask[:,p],t,d[:,p],s)[0],rtol=0,atol=1e-6)
        assert (q-m.base(x)).abs().max()<=1.5+1e-6

def test_all_branches_have_gradients_and_D1_frozen():
    inputs=args();y=torch.arange(6).remainder(2).float()
    for arm in ARMS:
        m=SpectralCorrection(Base(),arm);opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.003)
        for step in range(3):
            opt.zero_grad();loss=objective(m(*inputs)[0],y,nn.BCEWithLogitsLoss());loss.backward();assert torch.isfinite(loss) and all(p.grad is None for p in m.base.parameters())
            assert m.head[-1].weight.grad.norm()>0
            if step:
                assert m.head[0].weight.grad.norm()>0 and m.gate[-1].weight.grad.norm()>0
                for name in ['time','frequency']:
                    if hasattr(m,name):assert getattr(m,name)[0].weight.grad.norm()>0
            opt.step()
