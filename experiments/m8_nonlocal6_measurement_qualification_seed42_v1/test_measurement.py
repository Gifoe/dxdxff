import numpy as np
from measurement import crossing,aggregate,patient_z,prepare


def test_censoring_and_absent_channel():
    t=np.array([-2,-1,0,1,2],np.float32)
    a=np.array([[0,0,1],[1,1,1],[4,0,2],[0,0,2],[0,0,2]],np.float32)
    e,o,f=crossing(a,t)
    assert e.tolist()==[True,True,False] and o.tolist()==[True,False,False]
    assert f[0]==0 and np.isnan(f[1:]).all()
    s=np.ones((2,3,6));p=np.array([[1,1,1],[1,0,0]],bool)
    E=np.stack([np.stack([e,e],-1)]*2);O=np.stack([np.stack([o,o],-1)]*2)
    F=np.stack([np.stack([f,f],-1)]*2)
    out=aggregate(s,p,E,O,F)
    assert out[0,4]==0 and out[0,5]==1
    assert np.isnan(out[1,4]) and out[1,5]==0
    assert np.isnan(out[2,4:8]).all()
    assert out[1,0]==1 and out[1,1]==0


def test_unit_and_train_only_preprocessing():
    x=np.array([[1e-18,1],[2e-18,np.nan],[4e-18,3],[8e-18,4]],float)
    p=np.array(['a','a','b','b'])
    z=patient_z(x,p)
    for k in [1e-12,1e12]:np.testing.assert_allclose(z,patient_z(x*k,p),atol=1e-9,equal_nan=True)
    _,prep=prepare(z,np.array([0,1]))
    zz=z.copy();zz[2:]=1e9
    _,changed=prepare(zz,np.array([0,1]))
    for key in prep:np.testing.assert_array_equal(prep[key],changed[key])
    assert np.max(np.abs(patient_z(x,p,True)[:,0]-z[:,0]))>.1
