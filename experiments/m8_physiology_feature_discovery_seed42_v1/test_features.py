"""Outcome-free formula and mask tests before any patient extraction."""
import numpy as np
from features import extract,delta_summary,NAMES

rng=np.random.default_rng(42);x=rng.normal(size=(8,500))
a,q=extract(x);b,_=extract(x*3)
assert np.allclose(a[:,0]+2*np.log(3),b[:,0],atol=1e-6)
assert np.allclose(a[:,1:],b[:,1:],atol=1e-5,rtol=1e-6)
zero,_=extract(np.zeros((1,500)));assert np.isfinite(zero).all() and (zero[:,1:]==0).all()
t=np.arange(500)/250;sine=np.sin(2*np.pi*30*t)[None]
s,_=extract(sine);assert abs(s[0,NAMES.index('spectral_centroid_13_95')]-30)<1e-4
assert abs(s[0,NAMES.index('hjorth_mobility_per_second')]-500*np.sin(np.pi*30/250))<1
v=np.arange(2*3*5*11,dtype=np.float32).reshape(2,3,5,11)
mask=np.ones((2,3,5),bool);mask[1,1,:]=False;mask[0,0,-1]=False
times=np.array([[-4,-2,2,6,12]]*2)
d,summary=delta_summary(v,mask,times);vv=v.copy();vv[~mask]=1e20
dd,ss=delta_summary(vv,mask,times)
assert np.array_equal(d,dd) and np.array_equal(summary,ss)
assert np.array_equal(summary[1,1::2],np.zeros(11))
print('FEATURE_UNIT_SCALE_SINE_CONSTANT_MASK_SINGLETON_TESTS_PASS')
