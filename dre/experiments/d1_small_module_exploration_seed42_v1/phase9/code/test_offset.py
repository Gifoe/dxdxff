import numpy as np
from offset import transform,effective

def test_invertible_decision_without_labels():
    p=np.array([.1,.2,.7,.9,.01,.2,.3,.8]);patient=np.array(['a']*4+['b']*4)
    for coefficient in [.25,.5,.75,1.]:
        q,m=transform(p,patient,coefficient);tau=np.array([.33]*4+[.66]*4);t=effective(tau,m,coefficient);assert np.array_equal(p>=t,q>=tau)
        assert np.all(m[:4]==m[0]) and np.all(m[4:]==m[4]);assert np.array_equal(np.argsort(p[:4]),np.argsort(q[:4]))

def test_zero_coefficient_and_boundaries():
    p=np.array([.1,.4,.8]);patient=np.array(['a']*3);q,m=transform(p,patient,0);assert np.allclose(p,q)
    assert np.array_equal(effective(np.array([0.,1.]),np.array([5.,-5.]),1.),[0.,1.])
