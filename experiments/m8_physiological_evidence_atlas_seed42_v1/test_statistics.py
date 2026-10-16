import numpy as np
from statistics_core import effect,bh,pvalue


def test_patient_effect_ties_missing_and_sign():
    y=np.array([1,1,0,0]);x=np.array([4,4,1,1],float)
    assert effect(x,y)==1 and effect(-x,y)==-1 and effect(np.ones(4),y)==0
    x[:2]=np.nan;assert np.isnan(effect(x,y))
    assert pvalue(np.zeros(7))==1


def test_joint_bh_family_not_per_source():
    p=np.array([.01,.04,.03,.002]);np.testing.assert_allclose(bh(p),[.02,.04,.04,.008])
