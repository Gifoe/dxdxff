from __future__ import annotations

import unittest

from scripts.build_latent_core_targets import classify_phys_core_feature


class A9v8PhysFeatureSignTests(unittest.TestCase):
    def test_peak_time_features_are_latency_rank_negative(self):
        for name in ("peak_time_high_gamma", "peak_time_line_length"):
            group, sign, _ = classify_phys_core_feature(name)
            self.assertEqual(group, "latency_rank")
            self.assertEqual(sign, -1)

    def test_required_core_feature_groups_and_signs(self):
        cases = {
            "onset_latency_high_gamma": ("latency_rank", -1),
            "onset_rank_line_length": ("latency_rank", -1),
            "high_gamma_top20pct_mean": ("positive_core", 1),
            "line_length_top20pct_mean": ("positive_core", 1),
            "hfo80_150_event_rate": ("positive_core", 1),
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                group, sign, _ = classify_phys_core_feature(name)
                self.assertEqual((group, sign), expected)


if __name__ == "__main__":
    unittest.main()
