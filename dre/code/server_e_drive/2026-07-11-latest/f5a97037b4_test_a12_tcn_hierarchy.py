from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from a12_vcsn.models.trajectory_tcn import TrajectoryTCNEncoder, TrajectoryTCNUtilityModel


class A12TCNHierarchyTests(unittest.TestCase):
    def test_tcn_padding_invariant_and_last_valid_timestep(self):
        encoder = TrajectoryTCNEncoder(feature_dim=2, hidden_dim=4, dropout=0.)
        values = np.array([[[[1., 2.], [3., 4.]]]], dtype=np.float32)
        mask = np.array([[[1, 1]]], dtype=bool)
        padded = np.pad(values, ((0, 0), (0, 0), (0, 2), (0, 0)), constant_values=999.)
        padded_mask = np.pad(mask, ((0, 0), (0, 0), (0, 2)), constant_values=False)
        a = encoder.forward(values, mask).detach().numpy()
        b = encoder.forward(padded, padded_mask).detach().numpy()
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_no_static_fallback_success(self):
        pairs = pd.DataFrame({"subject_id": ["a", "b"], "eject_score_ez": [.9, .1],
                              "add_score_ez": [.1, .9], "beneficial_label": [0, 1],
                              "harmful_label": [1, 0], "delta_patient_macro_f1": [-.1, .1]})
        with self.assertRaises(RuntimeError):
            TrajectoryTCNUtilityModel(epochs=1).fit(pairs, ["eject_score_ez", "add_score_ez"])


if __name__ == "__main__":
    unittest.main()
