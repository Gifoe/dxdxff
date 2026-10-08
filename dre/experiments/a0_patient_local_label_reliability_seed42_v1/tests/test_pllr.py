import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from baseline import PRMLP, seed_all, state_hash, train_model
from pllr import (leave_one_out, build_pllr, class_normalize, raw_weights,
                  permutation, stable_seed, rank_correlation)
from train import train_weighted


def fixture():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(60,88)).astype(np.float32)
    y = np.tile(np.repeat([0,1],10),3)
    patient = np.repeat(['a','b','c'],20)
    ch = np.tile(np.asarray([f'c{i:02d}' for i in range(20)]),3)
    return x,y,patient,ch,np.repeat('synthetic',60)


def test_architecture_orientation_init_and_bce():
    seed_all(42); a = PRMLP(); seed_all(42); b = PRMLP()
    assert sum(p.numel() for p in a.parameters()) == 8817
    assert state_hash(a.state_dict()) == state_hash(b.state_dict())
    z = torch.tensor([-.7,.2,.6],requires_grad=True); y = torch.tensor([0.,1.,1.])
    old = F.binary_cross_entropy_with_logits(z,y,pos_weight=torch.tensor(.5))
    new = F.binary_cross_entropy_with_logits(z,y,pos_weight=torch.tensor(.5),weight=torch.ones_like(y))
    assert torch.equal(old,new)
    assert torch.equal(torch.autograd.grad(old,z,retain_graph=True)[0],torch.autograd.grad(new,z)[0])
    assert torch.sigmoid(-z)[0] > .5 and torch.sigmoid(z)[1] > .5


def test_leave_one_out_brute_and_support():
    rng = np.random.default_rng(8); z = rng.normal(size=(17,8)); y = np.r_[np.zeros(6,int),np.ones(11,int)]
    m, valid = leave_one_out(z,y); assert valid
    for c in range(len(y)):
        same = (y==y[c]) & (np.arange(len(y))!=c); other = y!=y[c]
        expected = np.mean((z[c]-z[same].mean(0))**2)-np.mean((z[c]-z[other].mean(0))**2)
        assert abs(m[c]-expected)<1e-12
    assert not leave_one_out(z[:15],np.r_[np.zeros(4,int),np.ones(11,int)])[1]
    assert leave_one_out(z[:10],np.r_[np.zeros(5,int),np.ones(5,int)])[1]
    # Modify only query feature: its same/other prototypes must stay unchanged.
    c=0; same=(y==0)&(np.arange(len(y))!=c); other=y==1
    zs=z.copy(); zs[c]+=1000
    assert np.array_equal(z[same].mean(0),zs[same].mean(0))
    assert np.array_equal(z[other].mean(0),zs[other].mean(0))


def test_fit_only_order_equivariance_constant_safety():
    x,y,p,ch,ce=fixture()
    original,*_=build_pllr(x[:40],y[:40],p[:40],ch[:40],ce[:40],1,draws=4)
    order=np.random.default_rng(9).permutation(40)
    changed,*_=build_pllr(x[:40][order],y[:40][order],p[:40][order],ch[:40][order],ce[:40][order],1,draws=4)
    assert np.allclose(original['B2'][order],changed['B2'],atol=1e-12,rtol=0)
    assert np.array_equal(original['B1'][order],changed['B1'])
    # Held-out data do not occur in the API call or PCA fit.
    x[40:]=1e20; y[40:]=1-y[40:]
    repeat,*_=build_pllr(x[:40],y[:40],p[:40],ch[:40],ce[:40],1,draws=4)
    assert np.array_equal(original['B2'],repeat['B2'])
    constant,*_=build_pllr(np.zeros((40,88)),y[:40],p[:40],ch[:40],ce[:40],1,draws=4)
    assert constant['tau'] is None and np.array_equal(constant['B2'],np.ones(40))


def test_weights_permutation_mass_and_finite_gradient():
    x,y,p,ch,ce=fixture(); a,*_=build_pllr(x,y,p,ch,ce,2,draws=4)
    assert a['raw'].min()>=.5 and a['raw'].max()<=1
    for pid in np.unique(p):
        ix=np.flatnonzero(p==pid)
        perm,_=permutation(a['B2'][ix],y[ix],ch[ix],2,pid)
        assert np.array_equal(perm,a['B1'][ix])
        for label in [0,1]:
            take=ix[y[ix]==label]
            assert abs(a['B2'][take].sum()-len(take))<1e-12
            assert np.array_equal(np.sort(a['B1'][take]),np.sort(a['B2'][take]))
    model=PRMLP(); loss=(torch.tensor(a['B2'])*F.binary_cross_entropy_with_logits(
        model(torch.from_numpy(x)),torch.tensor(y,dtype=torch.float32),reduction='none')).mean()
    loss.backward(); assert all(torch.isfinite(v.grad).all() for v in model.parameters())


def test_invalid_support_is_one():
    x,y,p,ch,ce=fixture(); y[:20]=np.r_[np.zeros(3,int),np.ones(17,int)]
    a,_,_,_=build_pllr(x,y,p,ch,ce,1,draws=4)
    assert np.array_equal(a['B2'][:20],np.ones(20)) and not a['eligible'][:20].any()


def test_epoch_one_identity_resume_and_binding(tmp_path):
    torch.set_num_threads(2)
    x,y,p,ch,ce=fixture(); tr=np.arange(40); va=np.arange(40,60)
    seed_all(1051); initial=copy.deepcopy(PRMLP().state_dict())
    base,_,_=train_model(x,y,p,tr,va,tmp_path/'old',1051,initial,'synthetic','cpu',max_epochs=2)
    whole,_,_=train_weighted(x,y,p,tr,va,tmp_path/'whole',1051,initial,'synthetic','cpu','A0',np.ones(60),max_epochs=2)
    assert state_hash(base.state_dict())==state_hash(whole.state_dict())
    train_weighted(x,y,p,tr,va,tmp_path/'resume',1051,initial,'synthetic','cpu','A0',np.ones(60),max_epochs=2,stop_after=1)
    resumed,_,_=train_weighted(x,y,p,tr,va,tmp_path/'resume',1051,initial,'synthetic','cpu','A0',np.ones(60),max_epochs=2)
    for path in ['whole','resume']:
        state=torch.load(tmp_path/path/'LAST_PRIVATE.pt',weights_only=False)
        if path=='whole': expected=state
        else:
            assert state_hash(state['model'])==state_hash(expected['model'])
            assert state['history']==expected['history']
            assert torch.equal(state['rng']['torch'],expected['rng']['torch'])
    assert state_hash(resumed.state_dict())==state_hash(whole.state_dict())
    weights=np.ones(60); weights[:20]=np.tile([.7,1.3],10)
    weighted,_,history=train_weighted(x,y,p,tr,va,tmp_path/'weighted',1051,initial,'weighted','cpu','B2',weights,max_epochs=1)
    one,_,_=train_weighted(x,y,p,tr,va,tmp_path/'one',1051,initial,'one','cpu','A0',np.ones(60),max_epochs=1)
    assert state_hash(weighted.state_dict())!=state_hash(one.state_dict()) and history[0]['weights_active']
    with pytest.raises(AssertionError):
        train_weighted(x,y,p,tr,va,tmp_path/'resume',1051,initial,'changed','cpu','A0',np.ones(60),max_epochs=2)
