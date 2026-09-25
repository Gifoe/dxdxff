from __future__ import annotations

import unittest

from scripts.build_latent_core_targets import select_phys_core_features


EXACT_S5_12 = [
    "early_high_gamma_slope",
    "early_line_length_slope",
    "onset_latency_high_gamma",
    "onset_latency_line_length",
    "onset_rank_high_gamma",
    "onset_rank_line_length",
    "high_gamma_top20pct_mean",
    "line_length_top20pct_mean",
    "hfo80_150_event_rate",
    "hfo80_150_duration_fraction",
    "hfo80_150_mean_envelope_z",
    "hfo80_150_max_envelope_z",
]


class A9v8PhysFeatureSetModeTests(unittest.TestCase):
    def test_exact_s5_12_uses_only_original_twelve_features(self):
        names = EXACT_S5_12 + [
            "high_gamma_max_to_mean",
            "line_length_max_to_mean",
            "peak_time_high_gamma",
            "peak_time_line_length",
            "generic_gamma_peak",
        ]

        selected = select_phys_core_features(names, feature_set="exact_s5_12")

        self.assertEqual([item.feature_name for item in selected], EXACT_S5_12)
        self.assertNotIn("peak_time_high_gamma", [item.feature_name for item in selected])

    def test_expanded_s5_16_adds_max_to_mean_and_negative_peak_time(self):
        names = EXACT_S5_12 + [
            "high_gamma_max_to_mean",
            "line_length_max_to_mean",
            "peak_time_high_gamma",
            "peak_time_line_length",
        ]

        selected = select_phys_core_features(names, feature_set="expanded_s5_16")
        by_name = {item.feature_name: item for item in selected}

        self.assertEqual(len(selected), 16)
        self.assertEqual(by_name["high_gamma_max_to_mean"].sign, 1)
        self.assertEqual(by_name["line_length_max_to_mean"].sign, 1)
        self.assertEqual(by_name["peak_time_high_gamma"].sign, -1)
        self.assertEqual(by_name["peak_time_line_length"].matched_group, "latency_rank")

    def test_auto_does_not_match_peak_time_as_positive_core(self):
        selected = select_phys_core_features(
            ["peak_time_high_gamma", "peak_time_line_length", "generic_gamma_peak"],
            feature_set="auto",
        )
        by_name = {item.feature_name: item for item in selected}

        self.assertEqual(by_name["peak_time_high_gamma"].matched_group, "latency_rank")
        self.assertEqual(by_name["peak_time_high_gamma"].sign, -1)
        self.assertEqual(by_name["peak_time_line_length"].sign, -1)


if __name__ == "__main__":
    unittest.main()
