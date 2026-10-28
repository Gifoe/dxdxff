import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from reference import fit_global,percentile_rank
from pr2 import PR2
from losses import pairwise,objective

def fixture(d=12):
    torch.manual_seed(173);x=torch.randn(19,d);b=torch.randn(19);m=PR2(d,fit_global(x));return x,b,m

def activate(m):
    with torch.no_grad():m.ranking[-1].weight.fill_(.07);m.calibration.weight.fill_(.03)

def test_identity_and_parameter_counts():
    for d,count in [(12,213),(32,533),(108,1749)]:
        x,b,m=fixture(d);o=m(x,b)
        assert torch.equal(o['final'],b) and torch.equal(o['anchor'],b)
        assert sum(p.numel() for p in m.parameters())-(d+1)==count
        a=PR2(d,fit_global(x),mode='abs')
        assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in m.parameters())
        assert torch.equal(a.ranking[0].weight,m.ranking[0].weight)

def test_equivariance_and_isolation():
    x,b,m=fixture();activate(m);g=torch.arange(len(x))//7;o=m(x,b,g)
    order=torch.randperm(len(x));r=m(x[order],b[order],g[order])
    assert torch.allclose(o['final'][order],r['final'],atol=2e-6)
    altered=x.clone();altered[g==2]+=1000
    assert torch.equal(o['final'][g==0],m(altered,b,g)['final'][g==0])

def test_constant_singleton_and_ties():
    x=torch.ones(1,12);m=PR2(12,fit_global(x));activate(m)
    assert torch.equal(percentile_rank(x),torch.zeros_like(x))
    assert torch.isfinite(m(x,torch.zeros(1))['final']).all()
    q=percentile_rank(torch.tensor([[1.],[1.],[2.],[3.]]))
    assert torch.equal(q[:,0],torch.tensor([-.5,-.5,.25,.75]))

def test_mask():
    x,b,m=fixture();activate(m);valid=torch.arange(len(x))%3!=0
    x[~valid]=float('nan');b[~valid]=0
    a=m(x,b,valid=valid);c=m(x[valid],b[valid])
    assert torch.equal(a['final'][valid],c['final'])

def test_gradient_and_calibration():
    x,b,m=fixture();activate(m);x.requires_grad_();o=m(x,b)
    assert o['delta_rank'].std()>0 and torch.unique(o['delta_cal']).numel()==1
    loss=objective(o,(torch.arange(len(x))%3==0).float(),.3);loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    no_cal=m(x.detach(),b,calibration=False)
    assert torch.equal(o['rank'],no_cal['final'])
    y=(np.arange(len(x))%3==0).astype(int)
    for metric in [roc_auc_score,average_precision_score]:
        assert metric(y,no_cal['final'].detach().double())==metric(y,no_cal['final'].detach().double()+4.)

def test_pairwise_exact_and_single_class():
    x=torch.tensor([-.1,.2,.8,-.3],requires_grad=True);y=torch.tensor([1,0,1,0])
    exact=torch.nn.functional.softplus(-(x[y==1,None]-x[y==0][None,:])/.25).mean()
    assert torch.allclose(pairwise(x,y,1),exact)
    assert pairwise(x,torch.zeros(4)).item()==0

def test_checkpoint_optimizer_replay(tmp_path):
    x,b,m=fixture();opt=torch.optim.AdamW(m.parameters(),lr=.001)
    loss=objective(m(x,b),(torch.arange(len(x))%2).float(),.3);loss.backward();opt.step()
    path=tmp_path/'checkpoint.pt';torch.save({'model':m.state_dict(),'optimizer':opt.state_dict()},path)
    restored=PR2(12,fit_global(x));q=torch.load(path,weights_only=False);restored.load_state_dict(q['model'])
    ropt=torch.optim.AdamW(restored.parameters());ropt.load_state_dict(q['optimizer'])
    assert torch.equal(m(x,b)['final'],restored(x,b)['final'])
    assert len(opt.state)==len(ropt.state)

if __name__=='__main__':
    import tempfile
    tests=[v for k,v in list(globals().items()) if k.startswith('test_')]
    for test in tests:
        if test.__code__.co_argcount:
            with tempfile.TemporaryDirectory() as directory:test(Path(directory))
        else:test()
    print('PR2_SYNTHETIC_PASS',len(tests),flush=True)
