import unittest

import pandas as pd

from a12_vcsn.models.learned_gate import LearnedGateModel
from a12_vcsn.patient_gate import RuleGate


class A12GateTests(unittest.TestCase):
    def test_rule_gate_has_explicit_no_action_path(self):
        gate = RuleGate(tau_benefit=0.7, tau_harm=0.2, tau_utility=0.01, tau_delta=0.01, tau_positive_seed_fraction=1.0)
        self.assertFalse(gate.accept({"p_benefit": 0.9, "p_harm": 0.3, "utility": 0.2, "pred_delta": 0.1, "seed_agreement": 1.0}))
        self.assertTrue(gate.accept({"p_benefit": 0.9, "p_harm": 0.1, "utility": 0.2, "pred_delta": 0.1, "seed_positive_fraction": 1.0}))

    def test_learned_gate_trains_only_on_cross_fitted_patient_actions(self):
        frame = pd.DataFrame({"subject_id": ["p1", "p2", "p3", "p4"], "utility": [0.2, -0.1, 0.3, -0.2], "p_benefit": [0.9, 0.1, 0.8, 0.2], "p_harm": [0.1, 0.8, 0.2, 0.9], "pred_delta": [0.2, -0.1, 0.1, -0.2], "net_beneficial": [1, 0, 1, 0]})
        gate = LearnedGateModel().fit(frame, ["utility", "p_benefit", "p_harm", "pred_delta"])
        self.assertEqual(gate.fit_subjects, {"p1", "p2", "p3", "p4"})
        self.assertTrue((gate.predict_proba(frame) >= 0).all())
