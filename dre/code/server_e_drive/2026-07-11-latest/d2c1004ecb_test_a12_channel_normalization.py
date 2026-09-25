from __future__ import annotations

import unittest

import pandas as pd

from a12_vcsn.action_simulation import simulate_patient_policy
from a12_vcsn.schemas import LedgerSchemaError, build_canonical_ledger, normalize_channel_name


class A12ChannelNormalizationTests(unittest.TestCase):
    def test_LA01_equals_LA1(self):
        self.assertEqual(normalize_channel_name("LA01"), normalize_channel_name("LA1"))

    def test_case_and_space_normalization(self):
        self.assertEqual(normalize_channel_name(" la 01 "), normalize_channel_name("LA1"))

    def test_prime_contact_preserved(self):
        self.assertNotEqual(normalize_channel_name("LA1'"), normalize_channel_name("LA1"))

    def test_duplicate_normalized_channel_detected(self):
        raw = pd.DataFrame({
            "fold_idx": [1, 1], "subject_id": ["P", "P"], "center": ["x", "x"],
            "channel_name": ["LA01", "la1"], "true_ez": [1, 0], "true_nez": [0, 1],
            "score_ez_probability": [.9, .1], "predicted_ez": [1, 0],
        })
        with self.assertRaises(LedgerSchemaError):
            build_canonical_ledger(raw)

    def test_actions_apply_by_normalized_key(self):
        ledger = pd.DataFrame({
            "subject_id": ["P", "P"], "outer_fold": [1, 1], "center": ["x", "x"],
            "channel_name_original": ["LA01", "LB01"], "channel_name_norm": ["LA1", "LB1"],
            "clinical_true_ez": [0, 1], "old_v3_selected": [1, 0],
        })
        pairs = pd.DataFrame({
            "subject_id": ["P"], "eject_channel": ["la1"], "add_channel": ["lb1"],
            "p_benefit": [.9], "p_harm": [.1], "pred_delta": [.2], "utility": [.2],
            "seed_positive_fraction": [1.],
        })
        result = simulate_patient_policy(ledger, pairs, tau_benefit=.5, tau_harm=.2,
                                         tau_utility=0., tau_delta=0.,
                                         tau_positive_seed_fraction=1., max_swaps=1)
        self.assertEqual(result.patient_macro_f1, 1.0)


if __name__ == "__main__":
    unittest.main()
