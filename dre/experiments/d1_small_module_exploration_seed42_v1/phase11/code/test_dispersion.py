import numpy as np
from dispersion import transform_scaled,cutoff

def test_inverse_and_ranking_without_label_input():
    p=np.array([.1,.2,.7,.9,.01,.2,.3,.8]);patient=np.array(['a']*4+['b']*4)
    for coefficient in [0.,.25,.5,1.]:
        for exponent in [.5,1.]:
            q,m,s=transform_scaled(p,patient,coefficient,exponent);tau=np.array([.33]*4+[.66]*4);t=cutoff(tau,m,s,coefficient);assert np.array_equal(p>=t,q>=tau);assert np.array_equal(np.argsort(p[:4]),np.argsort(q[:4])) and (s>0).all()

def test_constant_patient_fallback_and_endpoint_cutoffs():
    p=np.array([.4,.4,.4]);q,m,s=transform_scaled(p,np.array(['a']*3),.5,1.);assert np.array_equal(s,np.ones(3)) and np.isfinite(q).all()
    assert np.array_equal(cutoff(np.array([0.,1.]),np.array([5.,-5.]),np.array([.5,2]),1.),[0.,1.])
