import unittest
import warnings
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
import core as c
import verify as v


class Tests(unittest.TestCase):
    def test_patient_class_equal_weights_and_permutation(self):
        pid=np.array(['a']*8+['b']*4);y=np.array([1,0,0,0,0,0,0,0,1,1,1,0])
        w=c.weights(y,pid);np.testing.assert_array_equal(w,v.independent_weight(y,pid))
        for p in ['a','b']:
            for label in [0,1]:self.assertAlmostEqual(w[(pid==p)&(y==label)].sum(),3.)
        for b in range(10):
            yp=c.permuted_labels(y,pid,1,b);np.testing.assert_array_equal(yp,v.independent_permutation(y,pid,1,b))
            for p in ['a','b']:self.assertEqual(yp[pid==p].sum(),y[pid==p].sum())
            wp=c.weights(yp,pid)
            for p in ['a','b']:
                for label in [0,1]:self.assertAlmostEqual(wp[(pid==p)&(yp==label)].sum(),3.)

    def test_FIT_only_preprocessing_query_independence_and_constants(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(48,12))*np.logspace(-12,3,12);pid=np.repeat(['a','b','c','d'],12)
        x[:12,1]=0;fit=np.arange(24)
        for pr in [False,True]:
            pre=c.fit_pre(x[fit],pid[fit],pr);z=c.transform(x,pid,pre)
            altered=x.copy();altered[24:]*=1e7
            self.assertEqual(c.digest(pre),c.digest(c.fit_pre(altered[fit],pid[fit],pr)))
            for p in np.unique(pid):np.testing.assert_array_equal(z[pid==p],c.transform(x[pid==p],pid[pid==p],pre))
            np.testing.assert_array_equal(z,v.independent_transform(x,pid,pre))
        pre=c.fit_pre(x[fit],pid[fit],True)
        self.assertTrue((c.relative(x,pid,pre['units'])[pid=='a',1]==0).all())
        a=np.ones((6,12));a[:,3]=np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');pre=c.fit_pre(a,np.repeat(['a','b'],3),True)
            self.assertTrue(np.isfinite(c.transform(a,np.repeat(['a','b'],3),pre)).all())

    def test_inner_patient_isolation_C_ties_and_inactive_inputs(self):
        pid=np.repeat([f'p{i}' for i in range(17)],3);splits=c.inner_splits(pid,2)
        self.assertEqual(sorted(np.concatenate([iv for _,iv in splits]).tolist()),list(range(len(pid))))
        for it,iv in splits:self.assertFalse(set(pid[it])&set(pid[iv]))
        self.assertEqual(c.choose_c({.01:.7,.1:.7+1e-13,1.:.6,10.:.5}),.01)
        rng=np.random.default_rng(1);x=rng.normal(size=(len(pid),12));pre=c.fit_pre(x,pid,True)
        for f in c.FEATURES:
            cols=c.columns(f+'_PR');a=x.copy();a[:,np.setdiff1d(np.arange(12),cols)]+=10
            np.testing.assert_array_equal(c.transform(x,pid,pre)[:,cols],c.transform(a,pid,c.fit_pre(a,pid,True))[:,cols])

    def test_probability_EZ_direction_and_manual_logit(self):
        x=np.array([[-3.],[-2.],[-1.],[1.],[2.],[3.]]);y=np.array([0,0,0,1,1,1]);pid=np.array(['a']*6)
        model,state=c.fit_lr(x,y,pid,1.);p=c.predict(model,x)
        self.assertGreater(p[-1],p[0]);self.assertGreater(roc_auc_score(y,p),.99)
        self.assertIsNone(model.class_weight);self.assertTrue(state['converged'])

    def test_tied_metrics_and_independent_threshold(self):
        y=np.array([1,0,1,0,0,1]);p=np.array([.8,.8,.5,.2,.2,.1]);names=np.array(['b','a','c','f','e','d'])
        for t in [0,.5,1]:
            a=c.endpoints(y,p,p>=t,names);b=v.independent_metrics(y,p,p>=t,names)
            for key in c.METRICS:self.assertAlmostEqual(a[key],b[key],places=13)
            self.assertAlmostEqual(a['AUROC'],roc_auc_score(y,p));self.assertAlmostEqual(a['AP'],average_precision_score(y,p))
        pid=np.repeat(['a','b'],6);yy=np.tile(y,2);pp=np.tile(p,2)
        q=c.threshold(yy,pp,pid);expected=v.independent_threshold(yy,pp,pid)
        np.testing.assert_allclose([q['threshold'],q['CAL_Macro_F1'],q['CAL_EZ_F1']],expected,atol=1e-13,rtol=0)

    def test_patient_bootstrap_and_family_intervals(self):
        x=np.random.default_rng(4).normal(size=80)
        for adjusted in [False,True]:
            a=c.bootstrap(x,adjusted);np.testing.assert_allclose([a['mean'],a['ci_low'],a['ci_high']],v.interval(x,adjusted),atol=1e-12,rtol=0)
        a=c.bootstrap(x,True);b=c.bootstrap(x)
        self.assertLessEqual(a['ci_low'],b['ci_low']);self.assertGreaterEqual(a['ci_high'],b['ci_high'])
        self.assertEqual(c.bootstrap(np.zeros(80))['ci_high'],0.)


if __name__=='__main__':unittest.main()
