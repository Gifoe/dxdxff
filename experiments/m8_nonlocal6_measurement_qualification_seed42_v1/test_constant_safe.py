import numpy as np
from measurement import patient_z as frozen
from constant_safe import patient_z


def test_exact_constant_guard_and_nonconstant_preservation():
    x=np.stack([np.full(77,1/3),np.linspace(1e-18,7e-18,77)],1)
    p=np.full(77,'a');z=patient_z(x,p)
    assert (z[:,0]==0).all()
    np.testing.assert_array_equal(z[:,1],frozen(x,p)[:,1])
    for scale in [1e-12,1e12]:np.testing.assert_allclose(z,patient_z(x*scale,p),atol=1e-9)
    np.testing.assert_array_equal(patient_z(x,p,True),frozen(x,p,True))
    missing=np.full_like(x,np.nan);assert np.isnan(patient_z(missing,p)).all()
