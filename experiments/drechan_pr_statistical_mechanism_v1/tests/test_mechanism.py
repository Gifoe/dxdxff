"""Synthetic mathematical/isolation checks, never substitutes for real-cohort evidence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from sklearn.metrics import roc_auc_score
import mechanism_math as m
import trusted_io as io


class Tests(unittest.TestCase):
    def setUp(self):
        r=np.random.default_rng(711);self.x=r.normal(size=(24,12));self.y=np.array([0]*12+[1]*12)
        self.x[:,3]=.7;self.x[:,5]=np.arange(24)*1e-12

    def test_affine_rank_and_full_invariance(self):
        x=self.x;st=m.patient_stats(x);reps=m.represent(x,st)
        for j in range(12):
            for arm in m.FACTOR: self.assertEqual(roc_auc_score(self.y,x[:,j]),roc_auc_score(self.y,reps[arm][:,j]))
        a=np.linspace(.8,1.3,12);b=np.linspace(-2,2,12);z=x*a+b;zs=m.patient_stats(z)
        good=st['support']&zs['support']
        np.testing.assert_allclose(reps['G3_FULL'][:,good],m.represent(z,zs)['G3_FULL'][:,good],atol=1e-12)

    def test_centering_delta_direction_T_identities(self):
        reps=m.represent(self.x,m.patient_stats(self.x))
        for a,b in [('G0_GLOBAL','G1_CENTER'),('G2_SCALE','G3_FULL')]:
            q=m.moments(reps[a],self.y);z=m.moments(reps[b],self.y)
            np.testing.assert_allclose(q['delta'],z['delta'],atol=1e-12)
            for est in m.ESTIMATORS:
                model=m.direction([q],est);other=m.direction([z],est)
                np.testing.assert_allclose(model['v'],other['v'],atol=1e-11)
                for key in ['T','A']:self.assertAlmostEqual(m.endpoints(q,model)[key],m.endpoints(z,other)[key],places=10)

    def test_constant_support_missing_and_zero_direction(self):
        a=self.x.copy();a[:,7]=np.nan;pre=m.fit_global(a);z=m.global_query(a,pre)
        self.assertTrue(np.isfinite(z).all());self.assertTrue((z[:,7]==0).all())
        st=m.patient_stats(z);self.assertFalse(st['support'][3])
        self.assertFalse(m.patient_stats(self.x)['support'][5])
        self.assertTrue(st['support'][5]) # nonconstant TRAIN-scale normalization is unit consistent
        q=m.moments(np.zeros((24,12)),self.y);model=m.direction([q],'DIRECT');out=m.endpoints(q,model)
        self.assertEqual(out['T'],0);self.assertEqual(out['A'],0);self.assertTrue(out['zero_direction'])

    def test_query_and_labels_cannot_fit_preprocessor_or_donors(self):
        train=self.x[:12];pre=m.fit_global(train);altered=self.x.copy();altered[12:]*=1e9
        self.assertEqual(io.digest(pre),io.digest(m.fit_global(altered[:12])))
        source={'a':'hup','b':'hup','c':'lzu'};counts={'a':64,'b':65,'c':128}
        donors=m.donor_map(['a','b','c'],['a','b'],source,counts,1)
        self.assertEqual(donors,{'a':'b','b':'a','c':donors['c']});self.assertIn(donors['c'],['a','b'])

    def test_permutation_counts_and_independent_bootstrap(self):
        ids=np.array(['a']*12+['b']*12);y=np.tile([0,1],12);out=m.permute(y,ids,2,3)
        for p in set(ids):self.assertEqual(y[ids==p].sum(),out[ids==p].sum())
        v=np.array([-2.,0.,1.,3.]);b=m.boot_ids(4);s=m.summary(v,b)
        counts=np.zeros((10000,4));np.add.at(counts,(np.arange(10000)[:,None],b),1)
        np.testing.assert_allclose([s['ci_low'],s['ci_high']],np.quantile(counts@v/4,[.025,.975]))

    def test_fixed_T_and_midpoint_formula(self):
        q=m.moments(self.x,self.y);model=m.direction([q],'DIRECT');v=model['v'];score=self.x@v
        eps=1e-8*max(model['noise'],1e-12)
        expected=(score[self.y==1].mean()-score[self.y==0].mean())/np.sqrt(.5*(score[self.y==1].var()+score[self.y==0].var())+eps)
        self.assertAlmostEqual(m.endpoints(q,model)['T'],expected,places=11)
        midpoint=.5*(score[self.y==1].mean()+score[self.y==0].mean())
        self.assertAlmostEqual(midpoint,score.mean()+(.5-self.y.mean())*(q['delta']@v),places=12)

    def test_all80_metadata_only_assignment_is_legal_and_complete(self):
        ids=np.array([f'synthetic{i:03d}' for i in range(80)])
        centers=np.array(['hup']*36+['lzu']*21+['multicenter']*15+['pediatric']*8)
        patients=[];folds=[];roles=[]
        for f in range(1,6):
            patients.extend(ids);folds.extend([f]*80);roles.extend(['test' if i%5==f-1 else 'fit' for i in range(80)])
        meta=dict(patient=ids,center=centers,split_patient=np.array(patients),split_fold=np.array(folds),split_role=np.array(roles))
        assigned=io.full80_assignment(meta)
        self.assertEqual(set.union(*assigned.values()),set(ids));self.assertEqual(sum(map(len,assigned.values())),80)
        for f,a in assigned.items():
            self.assertEqual(len(a),16)
            legal=set(meta['split_patient'][(meta['split_fold']==f)&(meta['split_role']=='fit')])
            self.assertTrue(a<=legal)

    def test_independent_scalar_statistic_reconstruction(self):
        import verify_mechanism as v
        for estimator in m.ESTIMATORS:
            primary=m.direction([m.moments(self.x,self.y)],estimator)
            independent=v.independent_model({'p':{'x':self.x}},{'p':self.y},['p'],'x',estimator)
            result=v.endpoint(self.x,self.y,independent,1e-8)
            for key in ['T','A','z','MAE']:
                self.assertAlmostEqual(result[key],m.endpoints(m.moments(self.x,self.y),primary)[key],places=9)


if __name__=='__main__':unittest.main()
