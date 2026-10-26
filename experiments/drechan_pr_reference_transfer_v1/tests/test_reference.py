import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import math_core as m
from independent import crps_other

class ReferenceTests(unittest.TestCase):
    def test_crps_quadrature(self):
        for x,mu,sd in [(0,0,1),(3,-1,.4),(-2,1,2),(0,0,0),(2,-1,0),(.1,.3,1e-7),(100,0,1)]:
            if sd:
                z=(x-mu)/sd
                expected=sd*(quad(lambda t:ndtr(t)**2,-np.inf,z)[0]+quad(lambda t:(1-ndtr(t))**2,z,np.inf)[0]) if abs(z)<30 else abs(x-mu)-sd/np.sqrt(np.pi)
            else:expected=abs(x-mu)
            self.assertAlmostEqual(float(m.crps(x,mu,sd)),expected,places=9)
            self.assertAlmostEqual(float(crps_other(np.array([x]),np.array([mu]),np.array([sd]))[0]),expected,places=9)
    def test_crps_extremes_and_broadcast(self):
        x=np.array([0.,1.,1e150]);s=np.array([0.,1e-300,1e-300]);v=m.crps(x,0,s)
        self.assertTrue(np.isfinite(v).all());np.testing.assert_allclose(v,x)
        self.assertGreater(float(m.crps(0,0,10)),float(m.crps(0,0,1)))
    def test_channel_split_isolation_and_order(self):
        channels=np.array(['a','b','c','d','e','f','g']);a=m.split_masks('synthetic',channels)
        self.assertTrue((a.sum(1)==3).all());self.assertEqual(a.shape,(100,7))
        order=np.array([5,1,6,3,0,4,2]);b=m.split_masks('synthetic',channels[order]);np.testing.assert_array_equal(a[:,order],b)
        x=np.arange(84,dtype=float).reshape(7,12);mu,sd=m.moments(x[a[0]])
        changed=x.copy();changed[~a[0]]+=100000
        np.testing.assert_array_equal(m.moments(changed[a[0]]),[mu,sd])
    def test_no_label_inputs(self):
        import inspect
        for f in [m.split_masks,m.reference_parameters,m.patient_endpoints]:self.assertNotIn('y',inspect.signature(f).parameters)
    def test_constants_and_support(self):
        x=np.ones((7,12))*.1;mu,sd=m.moments(x);self.assertTrue((sd==0).all())
        np.testing.assert_array_equal(m.relative_query(x,mu,sd),np.zeros_like(x))
        self.assertTrue(np.isfinite(m.crps(x,mu,sd)).all())
        s=np.ones(12);s[0]=1e-9;z=m.relative_query(x,mu+1,s);self.assertTrue((z[:,0]==0).all())
    def test_positive_affine_crps_units(self):
        x=np.array([-.5,2.,5.]);loc=np.zeros(3);s=np.array([.3,1.,0.])
        np.testing.assert_allclose(m.crps(x*3+7,loc*3+7,s*3),3*m.crps(x,loc,s),rtol=1e-13)
    def test_weighting_odd_query_sizes(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(9,12));masks=m.split_masks('p',[str(i) for i in range(9)])
        p=dict(x=x,masks=masks,references=m.reference_parameters(x,masks),source_stats=m.moments(x+1),**{'global':(np.zeros(12),np.ones(12))})
        e=m.patient_endpoints(p);np.testing.assert_allclose(e['crps'][0],m.crps(x,0,1).mean(0),rtol=1e-12)
        self.assertEqual(e['deviation'].shape,(9,12))
    def test_independent_full_replay(self):
        from independent import replay
        rng=np.random.default_rng(13);x=rng.normal(size=(11,12));x[:,1]=0
        channels=np.array([f'c{i}' for i in range(11)]);masks=m.split_masks('p',channels)
        p=dict(patient='p',channels=channels,x=x,masks=masks,references=m.reference_parameters(x,masks),source_stats=m.moments(x+1),**{'global':(np.zeros(12),np.ones(12))})
        y=np.array([1]*3+[0]*8);a=m.patient_endpoints(p);b=replay(p,y)
        for k in ['crps','deviation','unsupported','unsupported_query_nonconstant']:np.testing.assert_allclose(a[k],b[k],rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(m.permuted_gaps(a['deviation'],y,'p'),b['permutation'],rtol=1e-10,atol=1e-12)
    def test_permutation_and_bootstrap(self):
        dev=np.arange(96,dtype=float).reshape(8,12);y=np.array([1,1,0,0,0,0,0,0]);before=y.copy()
        a=m.permuted_gaps(dev,y,'s');np.testing.assert_array_equal(a,m.permuted_gaps(dev,y,'s'));np.testing.assert_array_equal(y,before);self.assertEqual(a.shape,(1000,))
        src=np.array(['hup']*36+['lzu']*21+['multicenter']*15+['pediatric']*8);draw=m.bootstrap_indices(src)
        self.assertEqual(draw.shape,(10000,80))
        for s,n in [('hup',36),('lzu',21),('multicenter',15),('pediatric',8)]:self.assertTrue(((src[draw]==s).sum(1)==n).all())

if __name__=='__main__':unittest.main()
