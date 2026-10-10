"""Changed preprocessing path only: synthetic FIT/held-out data, no model training."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


spec=importlib.util.spec_from_file_location('no_pr_runner',Path(__file__).parents[1]/'code/run_ablation.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class ChangedPath(unittest.TestCase):
    def test_fit_only_global_transform_sklearn_parity(self):
        rng=np.random.default_rng(42); x=rng.normal(size=(40,88)); x[2,5]=np.nan
        train=np.arange(24);result,pre=runner.prepare_absolute(x,train)
        im=SimpleImputer().fit(x[train]);sc=StandardScaler().fit(im.transform(x[train]))
        self.assertTrue(np.array_equal(result,sc.transform(im.transform(x)).astype(np.float32)))
        self.assertTrue(np.array_equal(result,runner.frozen_absolute(x,pre)))
        changed=x.copy();changed[24:]=changed[24:]*100+300
        _,second=runner.prepare_absolute(changed,train)
        for k in pre:self.assertTrue(np.array_equal(pre[k],second[k]))

    def test_no_patient_context_or_local_scaling_at_inference(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(40,88))
        _,pre=runner.prepare_absolute(x,np.arange(24))
        single=runner.frozen_absolute(x[24:25],pre)
        together=runner.frozen_absolute(x[24:],pre)
        self.assertTrue(np.array_equal(single[0],together[0]))
        offset=x.copy();offset[24:]+=20
        self.assertFalse(np.array_equal(runner.frozen_absolute(offset[24:],pre),together))


if __name__=='__main__':unittest.main()
