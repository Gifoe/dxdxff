import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
from model import valid_patient_z, IctalLocalization, vicreg, patient_bce, RelativeDifferenceAttention, OnsetEncoder
from run_experiment import metric_one, fit_threshold, gated_attention_allowed
from audit_export import frozen_folds, normalize_channel_name


def torch_seed():torch.manual_seed(123)


def test_normalize_and_missing_channel():
    assert normalize_channel_name('EEG LA-001')=='LA1'
    u=torch.tensor([[1.,1.],[2.,1.],[100.,100.]])
    mask=torch.tensor([True,True,False])
    z=valid_patient_z(u,mask)
    assert abs(float(z[0,0]+z[1,0]))<1e-5
    assert float(z[2].abs().sum())==0
    assert abs(float(z[0,1]))<1e-5


def test_permutation_equivariance_pr_and_attention():
    torch_seed()
    u=torch.randn(8,64);m=torch.tensor([1,1,1,0,1,1,1,1],dtype=torch.bool)
    perm=torch.tensor([5,1,6,2,0,4,7,3]); z=valid_patient_z(u,m)
    assert torch.allclose(valid_patient_z(u[perm],m[perm]),z[perm],atol=1e-6)
    a=RelativeDifferenceAttention();a.alpha.data.fill_(1.)
    a.eval(); s=a(u,z,m)
    assert torch.allclose(a(u[perm],z[perm],m[perm]),s[perm],atol=1e-5)


def test_ssl_gradient_and_patient_mask():
    torch_seed()
    pair=torch.randn(1,4,2,2500)
    pair[:,:,1,:]*=1.8
    present=torch.tensor([[1,1,1,0]],dtype=torch.bool)
    enc=OnsetEncoder(width=64,chunk=8)
    z1=enc(pair,present,augment=True)
    z2=enc(pair,present,augment=True)
    assert z1.shape==(1,4,64)
    assert torch.all(z1[~present]==0)
    # VICReg needs >=4 valid pairs
    present[0,3]=True
    z1=enc(pair,present,augment=True)[present];z2=enc(pair,present,augment=True)[present]
    loss,_=vicreg(z1,z2);loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in enc.parameters())


def test_f1_orientation_and_weighted_bce():
    labels=torch.tensor([1.,1.,0.,0.])  # EZ-positive
    nez_logits=torch.tensor([-4.,-3.,3.,4.],requires_grad=True)
    mask=torch.tensor([1,1,1,1],dtype=torch.bool)
    loss=patient_bce(nez_logits,labels,mask)
    wrong=patient_bce(-nez_logits,labels,mask)
    assert loss<wrong
    result=metric_one(labels.detach().numpy(),torch.sigmoid(-nez_logits).detach().numpy(),.5)
    assert result['macro_f1']==1.0 and result['ez_f1']==1.0
    loss.backward();assert torch.isfinite(nez_logits.grad).all()


def test_equal_head_capacity():
    a=IctalLocalization(use_pr=False);b=IctalLocalization(use_pr=True)
    assert sum(x.numel() for x in a.parameters())==sum(x.numel() for x in b.parameters())
    assert a.head[0].weight.shape==b.head[0].weight.shape


def test_threshold_selection_training_only():
    rows=[{'labels_ez':np.array([1,0,1,0]),'scores_ez':np.array([.95,.05,.8,.2])},
          {'labels_ez':np.array([1,0]),'scores_ez':np.array([.75,.1])}]
    t=fit_threshold(rows)
    assert 0.2<t<=.75


def test_fold_overlap_rejected(tmp_path):
    import json
    ids=[str(i) for i in range(80)]; folds=[]
    for f in range(5):
        test=ids[16*f:16*(f+1)]
        train=[x for x in ids if x not in test]
        folds.append({'fold_idx':f+1,'fit_subjects':train[:51],
                      'validation_subjects':train[51:],'test_subjects':test})
    path=tmp_path/'folds.json';path.write_text(json.dumps({'folds':folds}))
    with pytest.raises(ValueError,match='original frozen'):
        frozen_folds(path,ids)  # historical split is 16/16/17/15/16, not 16*5
    folds[0]['fit_subjects'].append(folds[0]['test_subjects'][0])
    path.write_text(json.dumps({'folds':folds}))
    with pytest.raises(ValueError,match='leaks'):
        frozen_folds(path,ids)


def test_gate_requires_both_metrics_and_fold_consistency():
    rows=[]
    for fold in range(1,6):
        rows += [{'fold':fold,'variant':'E1','val_metrics':{'macro_f1':.61,'ez_ap':.52}},
                 {'fold':fold,'variant':'E3','val_metrics':{'macro_f1':.64,'ez_ap':.53}}]
    assert gated_attention_allowed(rows)['passed']
    rows[0]['val_metrics']['macro_f1']=.69
    assert not gated_attention_allowed(rows)['passed']
