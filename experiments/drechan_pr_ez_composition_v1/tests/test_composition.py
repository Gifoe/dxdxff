import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core
import independent
import stats_report

def fixture(seed=4,n=7):
    rng=np.random.default_rng(seed);x=rng.normal(size=(n,12));x[:,1]=0
    channels=np.array([f'synthetic_c{i}' for i in range(n)]);order=core.rankings('synthetic',channels);masks=np.zeros((100,n),bool);np.put_along_axis(masks,order[:,:n//2],True,axis=1)
    refs=np.array([[core.moments(x[m]),core.moments(x[~m])] for m in masks])
    return dict(patient='synthetic',x=x,channels=channels,masks=masks,references=refs,source_stats=core.moments(x+1),**{'global':(np.zeros(12),np.ones(12))})

class CompositionTests(unittest.TestCase):
    def test_crps_quadrature_and_zero(self):
        for x,mu,s in [(0,0,1),(2,-1,.5),(-2,0,2),(1,1,0),(5,2,0)]:
            t=(x-mu)/s if s else 0
            expected=s*(quad(lambda z:ndtr(z)**2,-np.inf,t)[0]+quad(lambda z:(1-ndtr(z))**2,t,np.inf)[0]) if s else abs(x-mu)
            self.assertAlmostEqual(float(core.crps(x,mu,s)),expected,places=10)
            self.assertAlmostEqual(float(independent.crps_other(np.array([x]),mu,s)[0]),expected,places=10)
        self.assertTrue(np.isfinite(core.crps(np.array([1e150,0]),0,np.array([1e-300,0]))).all())
    def test_nested_sets_exclude_query(self):
        p=fixture(n=9)
        for order in core.rankings(p['patient'],p['channels']):
            sets=core.nested_indices(order)
            for q in range(9):
                for a in sets:self.assertNotIn(q,a[q])
                for i in range(3):self.assertTrue(set(sets[i][q])<=set(sets[i+1][q]))
                self.assertEqual(set(sets[3][q]),set(range(9))-{q})
    def test_parent_half_and_order_invariance(self):
        p=fixture();o=core.rankings(p['patient'],p['channels']);np.testing.assert_array_equal(np.sort(o[:,:3],axis=1),np.array([np.flatnonzero(m) for m in p['masks']]))
        permutation=np.array([5,2,3,1,6,0,4]);other=core.rankings(p['patient'],p['channels'][permutation]);np.testing.assert_array_equal(permutation[other],o)
    def test_query_parameter_isolation(self):
        p=fixture();f=core.freeze_patient(p);changed=dict(p);changed['x']=p['x'].copy();changed['x'][2]+=1000;g=core.freeze_patient(changed)
        np.testing.assert_array_equal(f['parameters'][:,:,:,2],g['parameters'][:,:,:,2])
        np.testing.assert_array_equal(f['parameters'][:,3],np.broadcast_to(f['parameters'][0,3],f['parameters'][:,3].shape))
    def test_full_independent_replay_and_parent_reconstruction(self):
        p=fixture();y=np.array([1,1,0,0,0,0,0]);f=core.freeze_patient(p);a=core.evaluate_patient(p,f,y);b=independent.replay(p,y,f)
        for k in b:np.testing.assert_allclose(a[k],b[k],rtol=1e-9,atol=1e-10,equal_nan=True,err_msg=k)
        weighted=(a['e1_class']*np.array([2,5])[:,None,None]).sum(0)/7;np.testing.assert_allclose(weighted,a['e1_all'])
        scores,_=core.e1_scores(p);np.testing.assert_allclose(scores.mean(0),a['e1_all'])
    def test_label_swap_antisymmetry(self):
        p=fixture();y=np.array([1,1,0,0,0,0,0]);s,_=core.e1_scores(p)
        a=core.h_from_class(core.class_mean(s,y));b=core.h_from_class(core.class_mean(s,1-y));np.testing.assert_array_equal(a,-b);np.testing.assert_array_equal(a[0],np.zeros(12))
    def test_global_source_and_full_reference_invariance(self):
        p=fixture();y=np.array([1,1,0,0,0,0,0]);a=core.evaluate_patient(p,core.freeze_patient(p),y)
        for arm in [0,1]:np.testing.assert_array_equal(a['e2_all'][:,arm],np.broadcast_to(a['e2_all'][0,arm],a['e2_all'][:,arm].shape))
        self.assertTrue((a['reference_counts']==[2,3,5,6]).all())
    def test_composition_within_query_centering(self):
        rng=np.random.default_rng(51);u=rng.random((100,5));offset=np.array([1e3,-1e3,5,100,20]);v=offset+3*u
        slope,den=core.composition_slope(u,v);self.assertAlmostEqual(slope,.3,places=12);self.assertGreater(den,0)
        slope,den=core.composition_slope(np.full((100,5),.25),v);self.assertTrue(np.isnan(slope));self.assertEqual(den,0)
    def test_patient_bootstrap_and_nonestimable_reporting(self):
        src=['hup']*4+['lzu']*3+['multicenter']*2+['pediatric']*2;rows=[dict(source=s) for s in src];inf=stats_report.Inference(rows)
        for s in core.SOURCES:self.assertTrue(((np.array(src)[inf.draw]==s).sum(1)==src.count(s)).all())
        r=inf.summary(np.arange(11.),family=2);v=np.arange(11.)[inf.draw].mean(1);np.testing.assert_allclose([r['family_low'],r['family_high']],np.quantile(v,[.0125,.9875]))
        r=inf.summary(np.full(11,np.nan));self.assertEqual(r['status'],'not_estimable');self.assertEqual(r['mean'],'')
        a=np.full(11,np.nan);a[0]=1;r=inf.summary(a);self.assertEqual(r['status'],'single_patient_numerics_withheld');self.assertEqual(r['mean'],'')
        self.assertEqual(len(stats_report.PRIMARY),2);self.assertEqual(len(stats_report.SECONDARY),11)

if __name__=='__main__':unittest.main()
