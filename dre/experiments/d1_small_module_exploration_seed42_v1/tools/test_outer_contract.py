"""No data required: original frozen preprocessing and resume guards."""
import numpy as np
import pytest
from evaluate_outer_m8 import transform,roles,cached,save

def test_preprocessor_no_refit():
    class Core:
        @staticmethod
        def patient_z(x,p):return x
    x=np.array([[np.nan,3],[2,7]],np.float32)
    pre=dict(imputer_statistics=np.array([4.,8.]),mean=np.array([1.,2.]),scale=np.array([2.,5.]))
    original=x.copy();v=transform(x,np.array(['x','x']),pre,Core)
    assert np.array_equal(v,np.array([[1.5,.2],[.5,1.]],np.float32))
    assert np.array_equal(x,original,equal_nan=True)

def test_own_fold_isolation():
    meta=dict(split_patient=np.array(['a','b','c']),split_fold=np.array([1,1,1]),split_role=np.array(['fit','validation','test']))
    assert roles(meta,1)['test']=={'c'}
    meta['split_patient'][2]='a'
    with pytest.raises(AssertionError):roles(meta,1)

def test_private_resume_binding(tmp_path):
    p=tmp_path/'scores.pkl';ix=np.array([1,3]);q=dict(logit=np.array([.1,.2]),indices=ix,freeze_sha256='f',checkpoint_sha256='c');save(p,q)
    assert np.array_equal(cached(p,'f','c',ix)['logit'],q['logit'])
    with pytest.raises(AssertionError):cached(p,'other','c',ix)
    with pytest.raises(AssertionError):cached(p,'f','c',ix[::-1])
    q['y']=np.array([0,1]);save(p,q)
    with pytest.raises(AssertionError):cached(p,'f','c',ix)
