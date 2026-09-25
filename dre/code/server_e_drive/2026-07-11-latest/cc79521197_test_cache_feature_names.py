from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from ez_dataset import flatten_window_samples
from neuroez_c.evidence_views import WINDOW_NODE_FEATURE_NAMES, physics_state_features


def _args(**overrides):
    values = {
        "physics_state_features": "fast_slow_ratio",
        "physics_feature_parts": "abs",
        "self_compare_eps": 1e-5,
        "window_feature_names": list(WINDOW_NODE_FEATURE_NAMES) + ["fast_slow_ratio"],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class CacheFeatureNameTests(unittest.TestCase):
    def test_physics_state_features_uses_cache_provided_feature_names(self):
        names = list(WINDOW_NODE_FEATURE_NAMES) + ["fast_slow_ratio"]
        window_features = np.zeros((3, 2, len(names)), dtype=np.float32)
        window_features[:, :, -1] = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        centers = np.asarray([-1.0, 1.0, 3.0], dtype=np.float32)

        out = physics_state_features(window_features, centers, _args(window_feature_names=names))

        self.assertEqual(out.shape, (3, 2, 1))
        np.testing.assert_allclose(out[:, :, 0], window_features[:, :, -1])

    def test_missing_physics_feature_lists_available_names(self):
        window_features = np.zeros((3, 2, len(WINDOW_NODE_FEATURE_NAMES)), dtype=np.float32)
        centers = np.asarray([-1.0, 1.0, 3.0], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "Available features"):
            physics_state_features(
                window_features,
                centers,
                _args(physics_state_features="missing_feature", window_feature_names=list(WINDOW_NODE_FEATURE_NAMES)),
            )

    def test_flatten_window_samples_preserves_sample_feature_names(self):
        names = list(WINDOW_NODE_FEATURE_NAMES) + ["fast_slow_ratio"]
        run_records = [
            {
                "subject_id": "p1",
                "run_id": "r1",
                "channel_names_norm": ["A1", "A2"],
                "labels": np.asarray([1.0, 0.0], dtype=np.float32),
                "sample": {
                    "sample_id": "s1",
                    "window_features": np.zeros((1, 2, len(names)), dtype=np.float32),
                    "window_adjacency": np.zeros((1, 2, 2), dtype=np.float32),
                    "window_relative_centers_sec": np.asarray([0.0], dtype=np.float32),
                    "window_feature_names": names,
                },
            }
        ]

        samples = flatten_window_samples(run_records)

        self.assertEqual(samples[0]["window_feature_names"], names)


if __name__ == "__main__":
    unittest.main()
