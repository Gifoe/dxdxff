import itertools
import numpy as np
import pytest
import torch
from model import RawModel,cnn24,pool_embeddings,objective,patient_z
from metrics import patient_rows,threshold


def example():
    torch.manual_seed(9)
    h=torch.randn(3,6,7,24)
    mask=torch.ones(3,6,7,dtype=torch.bool);mask[1,4:]=False
    times=torch.tensor([[-3,-1,1,5,9,15,19.]]).repeat(3,1)
    return h,mask,times


def test_all_factor_levels_padding_order_and_gradient():
    h,m,t=example();permutation=torch.tensor([3,0,5,2,1,4])
    for recipe in itertools.product([False,True],repeat=4):
        z=pool_embeddings(h,m,t,recipe);padded=h.clone();padded[~m]=1234
        assert torch.equal(z,pool_embeddings(padded,m,t,recipe))
        assert torch.allclose(z[permutation],pool_embeddings(h[:,permutation],m[:,permutation],t,recipe),atol=2e-6,rtol=1e-6)
        x=h.clone().requires_grad_(True);loss=pool_embeddings(x,m,t,recipe).square().mean();loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all() and (x.grad[~m]==0).all()
        assert z.shape==(6,96) and torch.isfinite(z).all()


def test_ablation_slot_semantics():
    h,m,t=example()
    a=pool_embeddings(h,m,t,(1,0,0,0)).reshape(6,2,2,24)
    assert torch.equal(a[:,0],a[:,1]) and (a[:,:,1]==0).all()
    assert torch.equal(patient_z(torch.ones(6,96)),torch.zeros(6,96))


def test_uniform_attention_and_matched_initialization():
    h,m,t=example();base=RawModel(1);att=RawModel(1,attention=True)
    for k,v in base.state_dict().items():assert torch.equal(v,att.state_dict()[k])
    assert torch.allclose(base.aggregate(h,m,t),att.aggregate(h,m,t),atol=3e-6,rtol=1e-6)
    loss=att.classify(att.aggregate(h,m,t)).square().mean();loss.backward()
    assert att.window_attention[-1].weight.grad.abs().sum()>0
    assert att.seizure_attention[-1].weight.grad.abs().sum()>0


def test_loss_sign_and_metric_definition():
    y=torch.tensor([0.,0.,1.,1.]);good=torch.tensor([-3.,-2.,2.,3.],requires_grad=True)
    bad=-good.detach()
    for kind in ['macro','ez_rank']:
        assert objective(good,y,torch.tensor(1.),kind)<objective(bad,y,torch.tensor(1.),kind)
    rows=patient_rows(np.array([1,0,1,0]),np.array([.9,.1,.8,.2]),np.array(['a','a','b','b']),
        np.array(['1','2','1','2']),np.array(['x','x','x','x']),.5)
    assert all(r['f1']==r['macro_f1']==r['auroc']==r['mrr']==r['hit_at_1']==1 for r in rows)


def test_missing_pre_and_stage_are_errors():
    h,m,t=example();t=t+4
    with pytest.raises(AssertionError):pool_embeddings(h,m,t,(1,1,1,1))


def test_threshold_objective_is_not_swapped():
    y=np.array([1,0,0,0,1,0]);p=np.array([.45,.4,.35,.3,.25,.2]);pid=np.array(['a']*6)
    ez=threshold(y,p,pid,'ez');macro=threshold(y,p,pid,'macro')
    rr=patient_rows(y,p,pid,np.arange(6).astype(str),np.array(['x']*6),ez['threshold'])[0]
    assert rr['f1']==ez['f1'] and rr['macro_f1']==ez['macro_f1']
    assert ez['f1']>=macro['f1']
