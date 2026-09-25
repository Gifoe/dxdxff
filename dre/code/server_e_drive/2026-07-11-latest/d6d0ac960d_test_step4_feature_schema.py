from __future__ import annotations

import unittest

import numpy as np

from ez_features import (
    CLINICAL_ONSET_CORE_FEATURE_NAMES,
    WINDOW_NODE_FEATURE_NAMES,
    compute_window_feature_tensors,
    compute_window_feature_tensors_with_names,
    parse_extra_window_feature_groups,
)


def _window_segments() -> list[dict[str, float | int]]:
    sfreq = 128.0
    starts = [0, 128, 256, 384]
    centers = [-3.0, -1.0, 1.0, 3.0]
    return [
        {
            "start_sample": start,
            "end_sample": start + int(sfreq),
            "relative_center_sec": center,
        }
        for start, center in zip(starts, centers)
    ]


class Step4FeatureSchemaTests(unittest.TestCase):
    def test_legacy_window_tensor_return_shape_remains_three_values(self):
        rng = np.random.default_rng(7)
        data = rng.normal(size=(3, 640)).astype(np.float32)

        window_features, window_adjacency, centers = compute_window_feature_tensors(
            data,
            _window_segments(),
            sfreq=128.0,
            spectral_max_freq=100.0,
        )

        self.assertEqual(window_features.shape[-1], len(WINDOW_NODE_FEATURE_NAMES))
        self.assertEqual(window_adjacency.shape, (4, 3, 3))
        self.assertEqual(centers.shape, (4,))

    def test_clinical_onset_core_appends_named_finite_features(self):
        rng = np.random.default_rng(11)
        data = rng.normal(size=(3, 640)).astype(np.float32)

        window_features, _, _, feature_names = compute_window_feature_tensors_with_names(
            data,
            _window_segments(),
            sfreq=128.0,
            spectral_max_freq=100.0,
            extra_window_feature_groups="clinical_onset_core",
        )

        self.assertEqual(feature_names[: len(WINDOW_NODE_FEATURE_NAMES)], list(WINDOW_NODE_FEATURE_NAMES))
        self.assertEqual(feature_names[-len(CLINICAL_ONSET_CORE_FEATURE_NAMES) :], list(CLINICAL_ONSET_CORE_FEATURE_NAMES))
        self.assertEqual(window_features.shape[-1], len(feature_names))
        self.assertTrue(np.isfinite(window_features).all())

    def test_unknown_extra_feature_group_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown extra window feature group"):
            parse_extra_window_feature_groups("clinical_onset_core,not_a_group")


if __name__ == "__main__":
    unittest.main()
