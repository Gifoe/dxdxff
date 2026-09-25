from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from a12_vcsn.action_simulation import recompute_utility, simulate_patient_policy
from a12_vcsn.audit import audit_anchor
from a12_vcsn.feature_blocks import PairFeatureBlocks
from a12_vcsn.suite import make_synthetic_ledger
from a12_vcsn.suite import run_a12_suite
from a12_vcsn.config import A12Config
from a12_vcsn.protocol import ProtocolError
from a12_vcsn.trajectory_store import PatientTrajectoryStore, RunTrajectory
from a12_vcsn.models.trajectory_tcn import TrajectoryTCNUtilityModel


class A12RepairRegressionTests(unittest.TestCase):
    def test_trajectory_store_preserves_runs_and_masks_missing_channels(self):
        store = PatientTrajectoryStore(feature_names=("f0", "f1"))
        store.add_run("P1", RunTrajectory("run-1", ("A1", "A2"), np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])))
        store.add_run("P1", RunTrajectory("run-2", ("A1",), np.array([[[9., 10.]], [[11., 12.]], [[13., 14.]]])))
        values, mask, seizure_mask = store.get_channel_trajectory("P1", "A2")
        self.assertEqual(values.shape, (2, 3, 2))
        self.assertEqual(mask.tolist(), [[True, True, False], [False, False, False]])
        self.assertEqual(seizure_mask.tolist(), [True, False])
        self.assertEqual(values[0, 0, 0], 3.0)

    def test_pair_feature_blocks_are_named_not_midpoint_split(self):
        frame = pd.DataFrame({"eject_score_ez": [0.8], "add_score_ez": [0.2], "score_delta": [-0.6], "patient_k": [2]})
        blocks = PairFeatureBlocks.infer(frame, ["eject_score_ez", "add_score_ez", "score_delta", "patient_k"])
        self.assertEqual(blocks.eject_columns, ("eject_score_ez",))
        self.assertEqual(blocks.add_columns, ("add_score_ez",))
        self.assertEqual(blocks.pair_columns, ("score_delta",))
        self.assertEqual(blocks.patient_columns, ("patient_k",))
        self.assertNotEqual(blocks.matrix(frame, "eject").tolist(), blocks.matrix(frame, "add").tolist())

    def test_calibration_utility_is_recomputed(self):
        frame = pd.DataFrame({"p_benefit": [0.2], "p_harm": [0.5], "pred_delta": [0.4], "utility": [99.0], "pred_delta_std": [0.1]})
        got = recompute_utility(frame, risk_lambda=0.1, uncertainty_lambda=0.2)
        self.assertAlmostEqual(float(got.loc[0, "utility"]), 0.2 * 0.4 - 0.1 * 0.5 - 0.2 * 0.1)

    def test_policy_simulates_matching_not_pair_mean(self):
        ledger = pd.DataFrame({
            "subject_id": ["P"] * 4, "outer_fold": [1] * 4, "center": ["x"] * 4,
            "channel_name_original": ["A1", "A2", "A3", "A4"],
            "clinical_true_ez": [1, 0, 0, 0], "old_v3_selected": [1, 1, 0, 0],
        })
        pairs = pd.DataFrame({
            "subject_id": ["P", "P"], "eject_channel": ["A1", "A2"], "add_channel": ["A3", "A4"],
            "p_benefit": [.9, .9], "p_harm": [.1, .1], "pred_delta": [.2, .2], "utility": [.2, .2], "seed_agreement": [1., 1.],
        })
        result = simulate_patient_policy(ledger, pairs, tau_benefit=.5, tau_harm=.2, tau_utility=0., tau_delta=0., tau_positive_seed_fraction=1., max_swaps=2)
        self.assertEqual(result.action_count, 2)
        self.assertLess(result.delta_patient_macro_f1, 0.0)

    def test_anchor_metric_parity_refuses_self_validation(self):
        ledger = make_synthetic_ledger(n_subjects=4, n_folds=2)
        audit = audit_anchor(ledger, expected_patients=4, expected_folds=2)
        self.assertTrue(audit["passed"])
        self.assertFalse(audit["metric_parity"]["passed"])
        self.assertEqual(audit["metric_parity"]["status"], "not_claimed_without_independent_expected_metric")

    def test_tcn_has_temporal_parameters_and_is_not_siamese_alias(self):
        trajectory = [np.ones((2, 3, 2), dtype=np.float32) * value for value in (1, 2, 3, 4)]
        pairs = pd.DataFrame({"subject_id": ["a", "a", "b", "b"], "eject_score_ez": [.9, .8, .2, .1], "add_score_ez": [.1, .2, .8, .9], "beneficial_label": [0, 0, 1, 1], "harmful_label": [1, 1, 0, 0], "delta_patient_macro_f1": [-.2, -.1, .1, .2], "patient_pair_weight": [.5] * 4, "eject_trajectory_values": trajectory, "add_trajectory_values": trajectory, "eject_trajectory_mask": [np.ones((2, 3), bool)] * 4, "add_trajectory_mask": [np.ones((2, 3), bool)] * 4})
        model = TrajectoryTCNUtilityModel(epochs=1, hidden_dim=8).fit(pairs, ["eject_score_ez", "add_score_ez"])
        names = [name for name, _ in model.network.module.named_parameters()]
        self.assertTrue(any("temporal_block" in name for name in names))

    def test_resume_rejects_changed_frozen_input_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = make_synthetic_ledger(n_subjects=4, n_folds=2)
            run_a12_suite(first, output_dir=tmp, config=A12Config(variants=("A12-V0",), strict=False), synthetic=True)
            changed = first.copy(); changed.loc[0, "old_v3_score_ez"] += .001
            with self.assertRaises(ProtocolError):
                run_a12_suite(changed, output_dir=tmp, config=A12Config(variants=("A12-V0",), strict=False, resume=True), synthetic=True)


if __name__ == "__main__":
    unittest.main()
