from __future__ import annotations

import unittest

import pandas as pd

from a12_vcsn.patient_gate import (apply_patient_gate, build_patient_action_summary,
                                    cross_fit_patient_gate)


class A12GateCrossfitTests(unittest.TestCase):
    def test_summary_candidate_eligible_matched_counts(self):
        ledger = pd.DataFrame({"subject_id": ["P", "P", "P"], "old_v3_selected": [1, 0, 0],
                               "old_v3_score_ez": [.8, .7, .2]})
        scored = pd.DataFrame({"subject_id": ["P"] * 3, "eject_channel": ["A"] * 3,
                               "add_channel": ["B", "C", "D"], "utility": [.3, .2, -.1],
                               "p_benefit": [.9, .8, .1], "p_harm": [.1, .2, .9],
                               "pred_delta": [.2, .1, -.1], "seed_positive_fraction": [1., 1., 0.],
                               "seed_consensus": [1., .8, 1.], "eject_source_anchor": [1, 0, 0]})
        eligible, matched = scored.iloc[:2], scored.iloc[:1]
        summary = build_patient_action_summary(ledger, scored, eligible, matched, {"max_swaps": 1})
        self.assertEqual((summary.loc[0, "candidate_count"], summary.loc[0, "eligible_candidate_count"], summary.loc[0, "matched_action_count"]), (3, 2, 1))
        self.assertEqual(summary.loc[0, "candidate_source_count_eject_source_anchor"], 1)

    def test_probabilities_are_cross_fitted_and_test_subjects_not_fit(self):
        frame = pd.DataFrame({
            "subject_id": [f"P{i}" for i in range(10)], "top1_utility": [i / 10 for i in range(10)],
            "matched_action_count": [1] * 10, "anchor_patient_macro_f1": [.5] * 10,
            "net_delta_patient_macro_f1": [-.1] * 5 + [.1] * 5,
            "net_beneficial": [0] * 5 + [1] * 5,
        })
        result = cross_fit_patient_gate(frame, ["top1_utility", "matched_action_count"], n_folds=5, seed=3)
        self.assertEqual(result["subject_id"].nunique(), 10)
        self.assertFalse(result["cross_fitted_gate_probability"].isna().any())
        self.assertEqual(result["gate_fold"].nunique(), 5)
        self.assertTrue(all(subject not in set(fit_subjects)
                            for subject, fit_subjects in zip(result["subject_id"], result["gate_fit_subjects"])))

    def test_gate_fail_removes_complete_action_set_and_pass_preserves_it(self):
        actions = pd.DataFrame({"subject_id": ["A", "A", "B"], "eject_channel": ["x", "y", "z"]})
        probabilities = pd.DataFrame({"subject_id": ["A", "B"], "gate_probability": [.4, .8]})
        kept = apply_patient_gate(actions, probabilities, threshold=.5)
        self.assertEqual(kept["subject_id"].tolist(), ["B"])


if __name__ == "__main__":
    unittest.main()
