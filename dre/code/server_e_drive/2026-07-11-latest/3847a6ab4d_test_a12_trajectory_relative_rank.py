from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from a12_vcsn.trajectory_features import build_relative_trajectory_features
from a12_vcsn.trajectory_store import PatientTrajectoryStore, RunTrajectory


class A12RelativeTrajectoryTests(unittest.TestCase):
    def test_relative_rank_same_patient_same_seizure_and_multiple_features(self):
        ledger = pd.DataFrame({"subject_id": ["P", "P"],
                               "channel_name_original": ["A1", "A2"],
                               "channel_name_norm": ["A1", "A2"]})
        store = PatientTrajectoryStore(feature_names=("power", "line_length"))
        store.add_run("P", RunTrajectory("s1", ("A1", "A2"), np.array([
            [[10., 1.], [1., 10.]], [[8., 2.], [2., 8.]],
        ])))
        store.add_run("P", RunTrajectory("s2", ("A1", "A2"), np.array([
            [[9., 2.], [2., 9.]], [[7., 3.], [3., 7.]],
        ])))
        result, mapping = build_relative_trajectory_features(ledger, store, top_q=.5)
        required = {"trajectory_mean", "trajectory_rank_median", "trajectory_topq_entry_rate",
                    "trajectory_rank_stability", "trajectory_leave_one_seizure_out_variance",
                    "trajectory_seizure_coverage"}
        self.assertTrue(required.issubset(result.columns))
        self.assertEqual(mapping["feature_names"], ["power", "line_length"])
        self.assertGreater(result["trajectory_topq_entry_rate"].min(), 0.)

    def test_padding_and_missing_seizure_invariance(self):
        ledger = pd.DataFrame({"subject_id": ["P"], "channel_name_original": ["A1"],
                               "channel_name_norm": ["A1"]})
        first = PatientTrajectoryStore(feature_names=("f0", "f1"))
        first.add_run("P", RunTrajectory("s1", ("A1",), np.array([[[1., 2.]], [[3., 4.]]])))
        second = PatientTrajectoryStore(feature_names=("f0", "f1"))
        second.add_run("P", RunTrajectory("s1", ("A1",), np.array([[[1., 2.]], [[3., 4.]], [[999., 999.]]]),
                                            window_mask=np.array([1, 1, 0], dtype=bool)))
        second.add_run("P", RunTrajectory("missing", ("A2",), np.ones((2, 1, 2))))
        a, _ = build_relative_trajectory_features(ledger, first)
        b, _ = build_relative_trajectory_features(ledger, second)
        columns = [c for c in a if c.startswith("trajectory_")]
        np.testing.assert_allclose(a[columns], b[columns], equal_nan=True)


if __name__ == "__main__":
    unittest.main()
