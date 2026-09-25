from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from neuroez_c.evidence_views import physics_state_features


def _args(**overrides):
    values = {
        "physics_state_features": "log_bp_high_gamma,line_length_per_sec,rms,variance",
        "physics_feature_parts": "zdelta,delta",
        "self_compare_eps": 1e-5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class PhysicsViewTests(unittest.TestCase):
    def test_physics_state_features_builds_finite_self_reference_trajectory(self):
        window_features = np.arange(4 * 3 * 20, dtype=np.float32).reshape(4, 3, 20) + 1.0
        window_centers = np.asarray([-30.0, -10.0, 10.0, 30.0], dtype=np.float32)

        out = physics_state_features(window_features, window_centers, _args())

        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape[:2], (4, 3))
        self.assertGreater(out.shape[-1], 0)
        self.assertTrue(np.isfinite(out).all())

    def test_physics_state_features_default_feature_selection_works(self):
        window_features = np.ones((4, 3, 20), dtype=np.float32)
        window_centers = np.asarray([-30.0, -10.0, 10.0, 30.0], dtype=np.float32)

        out = physics_state_features(window_features, window_centers, _args())

        self.assertEqual(out.shape, (4, 3, 8))

    def test_physics_state_features_rejects_unknown_feature_name(self):
        window_features = np.ones((4, 3, 20), dtype=np.float32)
        window_centers = np.asarray([-30.0, -10.0, 10.0, 30.0], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "Unsupported physics_state_features"):
            physics_state_features(window_features, window_centers, _args(physics_state_features="not_a_feature"))


if __name__ == "__main__":
    unittest.main()
