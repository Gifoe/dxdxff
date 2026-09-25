import unittest

import numpy as np

from a12_vcsn.trajectory_features import summarize_channel_trajectory


class A12TrajectoryTests(unittest.TestCase):
    def test_masked_trajectory_ignores_nan_and_padding(self):
        values = np.array([[[1.0], [3.0], [np.nan], [99.0]]], dtype=float)
        mask = np.array([[True, True, False, False]])
        features, audit = summarize_channel_trajectory(values, mask)
        self.assertAlmostEqual(features["trajectory_mean_f0"], 2.0)
        self.assertEqual(audit["valid_windows"], 2)

