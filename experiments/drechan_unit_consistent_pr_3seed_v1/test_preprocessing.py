"""Four changed-path checks; no surrogate prediction-performance evidence."""
import unittest
import numpy as np
from unit_pr import EPS,relative,prepare,fit_units


def legacy_relative(raw,patient):
    out=np.asarray(raw,np.float64).copy()
    for p in np.unique(patient):
        ix=patient==p;a=out[ix];sd=np.nanstd(a,axis=0)
        out[ix]=(a-np.nanmean(a,axis=0))/np.where(sd>EPS,sd,1.)
    return out


class Tests(unittest.TestCase):
    def sample(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(40,14));p=np.repeat(np.arange(5),8).astype(str)
        return x,p,np.arange(24)

    def test_units_rescaling_and_nondegenerate_legacy_parity(self):
        x,p,tr=self.sample();units,_=fit_units(x,tr);q=relative(x,p,units)
        self.assertTrue(np.array_equal(q,legacy_relative(x,p)))
        factor=np.logspace(-12,12,14);scaled=relative(x*factor,p,units*factor)
        np.testing.assert_allclose(q,scaled,rtol=1e-12,atol=1e-12)

    def test_degenerate_zero_and_missing_preserved(self):
        x,p,tr=self.sample();x[:,3]=1e-10;x[2,5]=np.nan
        q=relative(x,p,np.ones(14));self.assertTrue(np.array_equal(q[:,3],np.zeros(len(x))))
        self.assertTrue(np.isnan(q[2,5]));self.assertTrue(np.isfinite(q[:,3]).all())
        x[:,3]=1e-10+np.tile(np.arange(8)*1e-14,5)
        q=relative(x,p,np.ones(14));self.assertTrue(np.array_equal(q[:,3],np.zeros(len(x))))

    def test_TRAIN_fit_and_other_patient_isolation(self):
        x,p,tr=self.sample();a,pre=prepare(x,p,tr);changed=x.copy();changed[24:]=changed[24:]*8+3
        b,new=prepare(changed,p,tr)
        self.assertTrue(np.array_equal(a[tr],b[tr]))
        for k in pre:
            self.assertTrue(pre[k]==new[k] if k=='fit_patient_ids' else np.array_equal(pre[k],new[k]))
        self.assertTrue(np.array_equal(a[24:32],b[24:32]))

    def test_old_mixed_unit_discontinuity_without_toy_accuracy(self):
        p=np.repeat(np.arange(3),8).astype(str);base=np.tile(np.arange(8)-3.5,3)[:,None]
        raw=np.tile(base,(1,14))*np.repeat([1e-10,2e-10,2e-8],8)[:,None];tr=np.arange(16)
        old=legacy_relative(raw,p);oldsd=old[tr].std(0,ddof=0)
        oldscaled=(old-old[tr].mean(0))/oldsd
        new,pre=prepare(raw,p,tr)
        self.assertGreater(float(abs(oldscaled[16:]).max()),1e8)
        self.assertLess(float(abs(new).max()),3.)
        self.assertEqual(pre['fit_patient_ids'],['0','1'])


if __name__=='__main__':unittest.main(verbosity=2)
