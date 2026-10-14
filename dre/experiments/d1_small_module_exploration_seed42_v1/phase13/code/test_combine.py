import numpy as np
from combine import combine_logits

def test_equal_fixed_logits_and_single_shared_D1():
    base=np.array([-.5,.2,1.]);a=base+np.array([.1,.2,-.1]);b=base+np.array([-.1,.4,.3]);z,p=combine_logits([a,b]);assert np.allclose(z,base+np.array([0,.3,.1]));assert ((p>0)&(p<1)).all();zz,pp=combine_logits([b,a]);assert np.array_equal(z,zz) and np.array_equal(p,pp)
