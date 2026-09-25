import unittest
import numpy as np
from neuroez_c.p2_pat_oracle_target import compute_stable_patient_oracle_threshold_target


class OracleTargetTests(unittest.TestCase):
    def test_candidates_include_global(self):
        result=compute_stable_patient_oracle_threshold_target(np.array([-1.,0.,1.]),np.array([0,0,1]),.123)
        self.assertGreaterEqual(result["n_candidate_thresholds"],4)
    def test_perfect_threshold(self):
        result=compute_stable_patient_oracle_threshold_target(np.array([-2.,-1.,1.,2.]),np.array([0,0,1,1]),0.)
        self.assertEqual(result["oracle_patient_macro_f1"],1.)
    def test_near_optimal_prefers_global(self):
        result=compute_stable_patient_oracle_threshold_target(np.array([-2.,-1.,1.,2.]),np.array([0,0,1,1]),0.,epsilon_f1=.01)
        self.assertEqual(result["oracle_threshold_logit"],0.)
    def test_residual_definition(self):
        result=compute_stable_patient_oracle_threshold_target(np.array([-2.,-.1,.2,2.]),np.array([0,0,1,1]),-.5)
        self.assertAlmostEqual(result["oracle_target_residual"],result["oracle_threshold_logit"]+.5)
    def test_residual_clipping(self):
        result=compute_stable_patient_oracle_threshold_target(np.array([0.,1.,2.,3.]),np.array([0,0,0,1]),-3.,max_target_residual=.25)
        self.assertLessEqual(abs(result["oracle_target_residual"]),.25)
        self.assertTrue(result["target_was_clipped"])
    def test_rejects_test_role(self):
        with self.assertRaises(RuntimeError): compute_stable_patient_oracle_threshold_target(np.array([0.,1.]),np.array([0,1]),0.,split_role="outer_test")
    def test_macro_f1_fixed_labels(self):
        result=compute_stable_patient_oracle_threshold_target(np.array([0.,1.]),np.array([1,1]),0.)
        self.assertLessEqual(result["oracle_patient_macro_f1"],.5)
