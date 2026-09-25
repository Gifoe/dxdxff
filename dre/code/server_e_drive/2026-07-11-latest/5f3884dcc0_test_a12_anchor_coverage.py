from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from a12_vcsn.anchor_features import build_patient_nez_anchor_features


class A12AnchorCoverageTests(unittest.TestCase):
    def setUp(self):
        self.ledger = pd.DataFrame({
            "subject_id": ["P"] * 3, "channel_name_original": ["A01", "A2", "A3"],
            "channel_name_norm": ["A1", "A2", "A3"], "old_v3_selected": [0, 0, 1],
            "old_v3_score_ez": [.1, .2, .9],
        })

    def test_anchor_coverage_report(self):
        augmented, coverage = build_patient_nez_anchor_features(
            self.ledger, {"P": {"a1": [1., 2.], "A2": [2., 3.], "A3": [4., 5.]}},
            quantile=1., min_channels=1, minimum_channel_coverage=.9,
            minimum_feature_finite_ratio=.9, strict=True)
        self.assertEqual(len(augmented), 3)
        self.assertEqual(coverage.loc[0, "status"], "available")
        self.assertEqual(coverage.loc[0, "n_matched_channels"], 3)
        self.assertNotIn("P", coverage.loc[0, "subject_id_hash"])

    def test_anchor_low_coverage_fails_strict(self):
        with self.assertRaises(ValueError):
            build_patient_nez_anchor_features(self.ledger, {"P": {"A1": [1., 2.]}},
                                              min_channels=1, minimum_channel_coverage=.8,
                                              minimum_feature_finite_ratio=.5, strict=True)

    def test_anchor_low_coverage_marks_unavailable_non_strict(self):
        augmented, coverage = build_patient_nez_anchor_features(
            self.ledger, {"P": {"A1": [1., 2.]}}, min_channels=1,
            minimum_channel_coverage=.8, minimum_feature_finite_ratio=.5, strict=False)
        self.assertEqual(coverage.loc[0, "status"], "unavailable_low_channel_coverage")
        self.assertTrue(augmented.filter(like="distance_to_patient").isna().all().all())

    def test_anchor_nonfinite_ratio_fails(self):
        with self.assertRaises(ValueError):
            build_patient_nez_anchor_features(
                self.ledger, {"P": {"A1": [np.nan, 2.], "A2": [np.nan, 3.], "A3": [np.nan, 4.]}},
                min_channels=1, minimum_channel_coverage=1.,
                minimum_feature_finite_ratio=.75, strict=True)


if __name__ == "__main__":
    unittest.main()
