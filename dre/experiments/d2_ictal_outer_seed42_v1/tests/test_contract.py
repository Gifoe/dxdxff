"""Synthetic contracts only: no patient data, model training, or test labels."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


root = Path(__file__).resolve().parents[1]
audit = load('outer_audit', root / 'code/evaluate_outer.py')
historical = root.parent / 'a0_source_posterior_decoder_seed42_v1/code/common.py'
core_path = root.parent / 'pr_uncertainty_aware_supervision_seed42_v1/code/uas_core.py'
# Deployed experiment is a direct child of the original source directory too.
common = load('outer_metric_contract', historical)
core = load('outer_pre_contract', core_path)


class Contract(unittest.TestCase):
    def test_frozen_preprocessor_sklearn_parity_and_patient_independence(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(32,88)); x[2,5] = np.nan
        patient = np.repeat(['fit1','fit2','held1','held2'], 8)
        train = np.arange(16)
        result, pre = core.prepare(x, patient, train)
        replay = audit.frozen_transform(x, patient, pre, core)
        self.assertTrue(np.array_equal(result, replay))
        self.assertTrue(np.array_equal(replay[16:], audit.frozen_transform(x[16:],patient[16:],pre,core)))
        self.assertEqual(set(pre['fit_patient_ids']), {'fit1','fit2'})
        changed = x.copy(); changed[16:] *= 7
        _, pre2 = core.prepare(changed, patient, train)
        for k in ['mean','scale','var','imputer_statistics']:
            self.assertTrue(np.array_equal(pre[k], pre2[k]))

    def test_label_orientation_confusion_and_ranking_ties(self):
        y = np.array([0,0,1,1,0,1]); scores = np.array([.1,.4,.4,.8,.7,.2])
        m = common.metric(y,scores,.5)
        self.assertEqual([m[k] for k in ['TP','FP','TN','FN']], [2,2,1,1])
        self.assertAlmostEqual(m['ez_auroc'], roc_auc_score(1-y,1-scores))
        self.assertAlmostEqual(m['ez_auprc'], average_precision_score(1-y,1-scores))
        self.assertEqual(common.metric(np.array([0,1]),np.array([.5,.5]),.5)['top1_is_ez'],1)

    def test_single_class_not_estimable_and_no_patient_removal(self):
        m = common.metric(np.ones(5,dtype=int),np.linspace(.1,.9,5),.5)
        self.assertTrue(np.isnan(m['ez_auroc']))
        self.assertTrue(np.isnan(m['sensitivity']))
        self.assertEqual(m['ez_auprc'],0)
        self.assertEqual(m['ez_mrr'],0)
        self.assertEqual(m['top1_is_ez'],0)

    def test_bootstrap_equal_patient_seed_and_nan_policy(self):
        patient=np.array(['a','b','c'])
        values=np.array([[.1,np.nan],[.4,.2],[.7,.9]])
        first=common.cluster_bootstrap(patient,values,10000)
        second=common.cluster_bootstrap(patient,values,10000)
        self.assertTrue(np.array_equal(first,second,equal_nan=True))
        self.assertEqual(first.shape,(10000,2))
        self.assertTrue(np.isfinite(first[:,0]).all())


if __name__ == '__main__':
    unittest.main()
