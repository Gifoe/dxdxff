from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from a12_vcsn.action_simulation import simulate_patient_policy
from a12_vcsn.models.trajectory_tcn import TrajectoryTCNEncoder
from a12_vcsn.trajectory_store import PatientTrajectoryStore, RunTrajectory
from a12_vcsn.anchor_features import build_patient_nez_anchor_features


class A12FinalP0Tests(unittest.TestCase):
    def test_invalid_values_are_masked_not_valid_zero_observations(self):
        store = PatientTrajectoryStore(feature_names=("f0",))
        store.add_run("p", RunTrajectory("r", ("A1",), np.array([[[np.nan]], [[1.0]], [[np.inf]]])))
        values, window_mask, seizure_mask = store.get_channel_trajectory("p", "A1")
        self.assertEqual(seizure_mask.tolist(), [True])
        self.assertEqual(window_mask.tolist(), [[False, True, False]])
        self.assertTrue(np.isnan(values[0, 0, 0]))

    def test_tcn_keeps_seizure_axis_and_does_not_flatten(self):
        encoder = TrajectoryTCNEncoder(feature_dim=1, hidden_dim=4)
        values = np.array([[[[1.0], [2.0]], [[100.0], [200.0]]]], dtype=np.float32)
        window_mask = np.array([[[True, True], [True, True]]])
        seizure_mask = np.array([[True, True]])
        embedding = encoder.forward(values, window_mask, seizure_mask)
        self.assertEqual(embedding.shape, (1, 4 * 24 + 2))

    def test_policy_uses_authoritative_evaluator(self):
        ledger = pd.DataFrame({"subject_id": ["p"] * 2, "outer_fold": [1, 1], "center": ["x", "x"], "channel_name_original": ["A1", "A2"], "clinical_true_ez": [1, 0], "old_v3_selected": [1, 0]})
        pairs = pd.DataFrame({"subject_id": ["p"], "eject_channel": ["A1"], "add_channel": ["A2"], "p_benefit": [.9], "p_harm": [.1], "pred_delta": [.1], "utility": [.1], "seed_positive_fraction": [1.0]})
        result = simulate_patient_policy(ledger, pairs, tau_benefit=.5, tau_harm=.2, tau_utility=0., tau_delta=0., tau_positive_seed_fraction=1., max_swaps=1)
        self.assertLess(result.delta_patient_macro_f1, 0.)

    def test_anchor_uses_normalized_channel_key(self):
        ledger = pd.DataFrame({"subject_id": ["p", "p"], "channel_name_original": ["LA01", "LA02"], "channel_name_norm": ["LA1", "LA2"], "old_v3_selected": [0, 1], "old_v3_score_ez": [.1, .9]})
        result, _ = build_patient_nez_anchor_features(ledger, {"p": {"LA1": [1., 2.], "LA2": [3., 4.]}}, quantile=.5, min_channels=1)
        self.assertTrue(np.isfinite(result.loc[0, "distance_to_patient_nez_anchor_l2"]))


if __name__ == "__main__":
    unittest.main()
