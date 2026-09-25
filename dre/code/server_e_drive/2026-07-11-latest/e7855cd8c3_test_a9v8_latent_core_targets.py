"""A9v8 latent core target tests — fake cache, strict feature matcher, rho."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_a9v3_oof_teacher import build_oof_teacher_dataframe, recompute_metrics
from scripts.build_latent_core_targets import (
    _compute_q_threshold_sigmoid,
    _entropy,
    _match_feature,
    _patient_zscore,
    build_latent_core_targets_dataframe,
    inspect_and_compute_phys_core,
    split_fold_train_targets,
    _write_summaries,
)


class _FakeSample:
    """Module-level fake sample for pickle compatibility."""
    pass


class A9v8LatentCoreTargetTests(unittest.TestCase):
    # ---- teacher ----

    def test_teacher_rejects_multiple_heldout_folds(self):
        ledger = pd.DataFrame([
            {"subject_id": "p1", "patient_id": "p1", "fold_id": 0,
             "split_role": "test", "channel_name": "a", "label_ez": 1,
             "score_eval": 0.9, "rank_eval": 1, "pred_topk": 1,
             "is_top1": 1, "center": "hup", "center_id": 0, "patient_ez_count": 1},
            {"subject_id": "p1", "patient_id": "p1", "fold_id": 1,
             "split_role": "test", "channel_name": "a", "label_ez": 1,
             "score_eval": 0.8, "rank_eval": 1, "pred_topk": 1,
             "is_top1": 1, "center": "hup", "center_id": 0, "patient_ez_count": 1},
        ])
        with self.assertRaisesRegex(ValueError, "multiple held-out folds"):
            build_oof_teacher_dataframe([ledger])

    def test_teacher_metrics_recomputation(self):
        teacher = pd.DataFrame([
            {"subject_id": "p1", "patient_id": "p1", "fold_id": 0, "center": "hup", "center_id": 0,
             "channel_name": "a", "label_ez": 1, "patient_ez_count": 1,
             "a9v3_oof_score": 0.9, "a9v3_rank_eval": 1, "a9v3_pred_topk": 1, "a9v3_top1": 1},
            {"subject_id": "p1", "patient_id": "p1", "fold_id": 0, "center": "hup", "center_id": 0,
             "channel_name": "b", "label_ez": 0, "patient_ez_count": 1,
             "a9v3_oof_score": 0.1, "a9v3_rank_eval": 2, "a9v3_pred_topk": 0, "a9v3_top1": 0},
        ])
        m = recompute_metrics(teacher)
        self.assertAlmostEqual(float(m["patient_macro_ez_mrr"]), 1.0)
        self.assertAlmostEqual(float(m["top1_is_ez_rate"]), 1.0)

    # ---- q computation ----

    def test_q_properties(self):
        scores = pd.Series([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        q = _compute_q_threshold_sigmoid(scores, 2.0, 0.1)
        self.assertTrue((q >= 0).all())
        self.assertTrue((q <= 1).all())
        self.assertAlmostEqual(float(q.sum()), 2.0, places=4)

    # ---- strict feature matcher ----

    def test_feature_matcher_positive_core(self):
        for name in ["early_high_gamma_slope", "line_length_top20pct_mean",
                      "hfo80_150_event_rate", "fast_ripple_max_envelope"]:
            g, s, r = _match_feature(name)
            self.assertEqual(g, "positive_core", f"{name}: expected positive_core, got {g}")
            self.assertEqual(s, 1, f"{name}: sign should be +1")

    def test_feature_matcher_latency_rank(self):
        for name in ["onset_latency_high_gamma", "onset_rank_high_gamma",
                      "latency_line_length", "rank_line_length"]:
            g, s, _ = _match_feature(name)
            self.assertEqual(g, "latency_rank", f"{name}: expected latency_rank, got {g}")
            self.assertEqual(s, -1)

    def test_feature_matcher_rejects_generic_gamma(self):
        g, s, _ = _match_feature("generic_gamma_power")
        self.assertIsNone(g)
        self.assertEqual(s, 0)

    def test_feature_matcher_rejects_plain_power(self):
        g, s, _ = _match_feature("log_bp_high_gamma")
        # "high_gamma" present but no stat token → should NOT match
        self.assertIsNone(g, "log_bp_high_gamma without core stat should not match")

    # ---- latent core strong/broad rules ----

    def test_strong_center_q_equals_label(self):
        teacher = pd.DataFrame([
            {"subject_id": "p1", "patient_id": "p1", "fold_id": 0, "center": "hup", "center_id": 0,
             "channel_id": 0, "channel_name": "a", "label_ez": 1, "patient_ez_count": 1,
             "a9v3_oof_score": 0.9, "a9v3_rank_eval": 1},
            {"subject_id": "p1", "patient_id": "p1", "fold_id": 0, "center": "hup", "center_id": 0,
             "channel_id": 1, "channel_name": "b", "label_ez": 0, "patient_ez_count": 1,
             "a9v3_oof_score": 0.1, "a9v3_rank_eval": 2},
        ])
        targets, _, _ = build_latent_core_targets_dataframe(
            teacher, broad_centers={"lzu", "pediatric"}, rho=0.3, tau_q=0.1,
        )
        hup = targets[targets["center"] == "hup"].sort_values("channel_id")
        self.assertEqual(hup["pseudo_core_q"].tolist(), [1.0, 0.0])
        self.assertEqual(hup["core_target_type"].tolist(), ["strong_label", "strong_label"])

    def test_broad_center_nez_q_zero(self):
        teacher = pd.DataFrame([
            {"subject_id": "p2", "patient_id": "p2", "fold_id": 1, "center": "lzu", "center_id": 1,
             "channel_id": 0, "channel_name": "x", "label_ez": 1, "patient_ez_count": 2,
             "a9v3_oof_score": 0.9, "a9v3_rank_eval": 1},
            {"subject_id": "p2", "patient_id": "p2", "fold_id": 1, "center": "lzu", "center_id": 1,
             "channel_id": 1, "channel_name": "y", "label_ez": 1, "patient_ez_count": 2,
             "a9v3_oof_score": 0.5, "a9v3_rank_eval": 2},
            {"subject_id": "p2", "patient_id": "p2", "fold_id": 1, "center": "lzu", "center_id": 1,
             "channel_id": 2, "channel_name": "z", "label_ez": 0, "patient_ez_count": 2,
             "a9v3_oof_score": 0.7, "a9v3_rank_eval": 3},
        ])
        targets, _, _ = build_latent_core_targets_dataframe(
            teacher, broad_centers={"lzu", "pediatric"}, rho=0.3, tau_q=0.1,
        )
        lzu = targets[targets["center"] == "lzu"]
        nez = lzu[lzu["label_ez"] == 0]
        self.assertTrue((nez["pseudo_core_q"] == 0.0).all(), "NEZ q must be 0")
        ez = lzu[lzu["label_ez"] > 0.5]
        self.assertTrue((ez["pseudo_core_q"] >= 0).all())
        self.assertTrue((ez["pseudo_core_q"] <= 1).all())
        # mass = max(1.0, 0.3*2) = 1.0
        self.assertAlmostEqual(float(lzu["pseudo_core_q"].sum()), 1.0, places=5)

    def test_fold_specific_excludes_test(self):
        teacher = pd.DataFrame([
            {"subject_id": f"p{pid}", "patient_id": f"p{pid}", "fold_id": pid % 2,
             "center": "hup", "center_id": 0, "channel_id": 0, "channel_name": "a",
             "label_ez": 1, "patient_ez_count": 1,
             "a9v3_oof_score": 0.9, "a9v3_rank_eval": 1}
            for pid in range(4)
        ])
        targets, _, _ = build_latent_core_targets_dataframe(
            teacher, broad_centers={"lzu", "pediatric"},
        )
        folds = split_fold_train_targets(targets, n_splits=2)
        self.assertFalse((folds[0]["fold_id"] == 0).any())
        self.assertFalse((folds[1]["fold_id"] == 1).any())

    def test_audit_required_fields(self):
        teacher = pd.DataFrame([
            {"subject_id": "p1", "patient_id": "p1", "fold_id": 0, "center": "hup", "center_id": 0,
             "channel_id": 0, "channel_name": "a", "label_ez": 1, "patient_ez_count": 1,
             "a9v3_oof_score": 0.9, "a9v3_rank_eval": 1},
        ])
        _, audit, _ = build_latent_core_targets_dataframe(
            teacher, broad_centers={"lzu", "pediatric"}, rho=0.3, tau_q=0.1,
        )
        for key in ("n_patients", "broad_centers", "rho", "tau_q", "phys_core_score_available"):
            self.assertIn(key, audit, f"Missing audit key: {key}")

    # ---- fake cache phys_core_score ----

    def test_fake_cache_produces_nonzero_phys_core(self):
        import pickle, tempfile
        fake_cache = [{
            "subject_id": "p1",
            "run_id": "r1",
            "channel_names_norm": ["a", "b"],
            "window_feature_names": [
                "early_high_gamma_slope",
                "line_length_top20pct_mean",
                "onset_latency_high_gamma",
                "generic_gamma_power",
            ],
            "window_features": np.random.randn(3, 2, 4).astype(np.float32),
        }]
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(fake_cache, f)
            cache_path = f.name

        teacher = pd.DataFrame([
            {"subject_id": "p1", "patient_id": "p1", "fold_id": 0, "center": "hup", "center_id": 0,
             "channel_id": 0, "channel_name": "a", "label_ez": 1, "patient_ez_count": 1,
             "a9v3_oof_score": 0.9, "a9v3_rank_eval": 1},
            {"subject_id": "p1", "patient_id": "p1", "fold_id": 0, "center": "hup", "center_id": 0,
             "channel_id": 1, "channel_name": "b", "label_ez": 0, "patient_ez_count": 1,
             "a9v3_oof_score": 0.1, "a9v3_rank_eval": 2},
        ])

        phys, audit, feaudit = inspect_and_compute_phys_core(teacher, cache_path)
        self.assertTrue(audit["phys_core_score_available"],
                        f"Expected available but got: {audit['phys_core_reason']}")
        self.assertFalse((phys == 0).all() or phys.std() < 1e-8,
                         "phys_core_score should not be all zero with usable features")

        # feature audit: generic_gamma_power should not be used
        used_names = [r["feature_name"] for r in feaudit if r["used"]]
        self.assertNotIn("generic_gamma_power", used_names,
                         "generic_gamma_power should not be used")

        # clean up
        Path(cache_path).unlink()

    def test_no_cache_phys_core_all_zero(self):
        teacher = pd.DataFrame([
            {"subject_id": "p1", "patient_id": "p1", "fold_id": 0, "center": "hup", "center_id": 0,
             "channel_id": 0, "channel_name": "a", "label_ez": 1, "patient_ez_count": 1,
             "a9v3_oof_score": 0.9, "a9v3_rank_eval": 1},
        ])
        phys, audit, _ = inspect_and_compute_phys_core(teacher, None)
        self.assertFalse(audit["phys_core_score_available"])
        self.assertTrue((phys == 0).all())

    # ---- summary uses rho ----

    def test_write_summaries_uses_rho(self):
        import tempfile
        teacher = pd.DataFrame([
            {"subject_id": "p1", "patient_id": "p1", "fold_id": 0, "center": "hup", "center_id": 0,
             "channel_id": 0, "channel_name": "a", "label_ez": 1, "patient_ez_count": 1,
             "a9v3_oof_score": 0.9, "a9v3_rank_eval": 1},
            {"subject_id": "p1", "patient_id": "p1", "fold_id": 0, "center": "hup", "center_id": 0,
             "channel_id": 1, "channel_name": "b", "label_ez": 0, "patient_ez_count": 1,
             "a9v3_oof_score": 0.1, "a9v3_rank_eval": 2},
        ])
        targets, _, _ = build_latent_core_targets_dataframe(
            teacher, broad_centers={"lzu", "pediatric"}, rho=0.5,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            _write_summaries(targets, out, rho=0.5)
            ps = pd.read_csv(out / "latent_core_patient_summary.csv")
            # n_ez for p1 = 1, target_mass = max(1.0, 0.5 * 1) = 1.0
            self.assertAlmostEqual(float(ps["target_mass"].iloc[0]), 1.0)

    # ---- helpers ----

    def test_zscore_groupby(self):
        df = pd.DataFrame({
            "patient_id": ["a", "a", "a", "b", "b"],
            "score": [1.0, 2.0, 3.0, 10.0, 10.0],
        })
        result = df.groupby("patient_id")["score"].transform(_patient_zscore).to_numpy()
        # patient a: mean=2, ddof=0 std=sqrt(2/3)≈0.8165
        self.assertAlmostEqual(result[0], (1 - 2) / np.sqrt(2 / 3), places=4)
        self.assertEqual(result[3], 0.0)
        self.assertEqual(result[4], 0.0)

    def test_entropy_uniform(self):
        self.assertAlmostEqual(_entropy([0.25, 0.25, 0.25, 0.25]), 1.38629, places=4)

    # ---- patient-channel key (not channel-only) ----

    def test_phys_core_uses_patient_channel_key_not_channel_only(self):
        import pickle, tempfile
        fake_cache = [
            {
                "subject_id": "p1", "run_id": "r1",
                "channel_names_norm": ["a1", "b1"],
                "window_feature_names": ["early_high_gamma_slope"],
                "window_features": np.array([
                    [[10.0], [0.0]],
                    [[10.0], [0.0]],
                    [[10.0], [0.0]],
                ], dtype=np.float32),
            },
            {
                "subject_id": "p2", "run_id": "r1",
                "channel_names_norm": ["a1", "b1"],
                "window_feature_names": ["early_high_gamma_slope"],
                "window_features": np.array([
                    [[-10.0], [0.0]],
                    [[-10.0], [0.0]],
                    [[-10.0], [0.0]],
                ], dtype=np.float32),
            },
        ]
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(fake_cache, f)
            cache_path = f.name

        teacher = pd.DataFrame([
            {"subject_id": "p1", "patient_id": "p1", "fold_id": 0, "center": "hup", "center_id": 0,
             "channel_id": 0, "channel_name": "a1", "label_ez": 1, "patient_ez_count": 2,
             "a9v3_oof_score": 0.9, "a9v3_rank_eval": 1},
            {"subject_id": "p1", "patient_id": "p1", "fold_id": 0, "center": "hup", "center_id": 0,
             "channel_id": 1, "channel_name": "b1", "label_ez": 1, "patient_ez_count": 2,
             "a9v3_oof_score": 0.5, "a9v3_rank_eval": 2},
            {"subject_id": "p2", "patient_id": "p2", "fold_id": 1, "center": "hup", "center_id": 0,
             "channel_id": 0, "channel_name": "a1", "label_ez": 1, "patient_ez_count": 2,
             "a9v3_oof_score": 0.7, "a9v3_rank_eval": 1},
            {"subject_id": "p2", "patient_id": "p2", "fold_id": 1, "center": "hup", "center_id": 0,
             "channel_id": 1, "channel_name": "b1", "label_ez": 1, "patient_ez_count": 2,
             "a9v3_oof_score": 0.3, "a9v3_rank_eval": 2},
        ])

        phys, audit, feaudit = inspect_and_compute_phys_core(teacher, cache_path)
        self.assertTrue(audit["phys_core_score_available"], audit.get("phys_core_reason", ""))
        self.assertEqual(audit["phys_core_match_key"], "subject_or_patient_id + normalized_channel_name")
        # both rows matched
        self.assertEqual(audit["phys_core_matched_teacher_rows"], 4,
                         "All 4 teacher rows should match via patient+channel key")
        # p1 a1 should have positive phys, p2 a1 negative (after z-score within each patient)
        p1_a1 = phys[0]
        p1_b1 = phys[1]
        p2_a1 = phys[2]
        p2_b1 = phys[3]
        # within p1: a1 > b1 because 10 > 0
        self.assertGreater(p1_a1, p1_b1, "p1 a1 (raw=10) should score higher than p1 b1 (raw=0)")
        # within p2: a1 < b1 because -10 < 0, sign is positive so lower raw → lower z → lower phys
        self.assertLess(p2_a1, p2_b1, "p2 a1 (raw=-10) should score lower than p2 b1 (raw=0)")

        Path(cache_path).unlink()

    # ---- feature-level z-score before sum ----

    def test_phys_core_feature_level_zscore_before_sum(self):
        """Feature-level z-score: high_gamma (sign=+1) and onset_latency (sign=-1)
        must each be z-scored per-patient before summing.  a1 has high gamma + low
        latency (both "good"), a2 has low gamma + high latency (both "bad")."""
        import pickle, tempfile
        fake_cache = [
            {
                "subject_id": "p1", "run_id": "r1",
                "channel_names_norm": ["a1", "a2"],
                "window_feature_names": [
                    "early_high_gamma_slope",       # index 0, sign +1
                    "onset_latency_high_gamma",      # index 1, sign -1
                ],
                # a1 = good core (high gamma=1000, low latency=0.1)
                # a2 = poor core (low gamma=0, high latency=0.9)
                "window_features": np.array([
                    [[1000.0, 0.1], [0.0, 0.9]],
                    [[1000.0, 0.1], [0.0, 0.9]],
                ], dtype=np.float32),
            },
        ]
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(fake_cache, f)
            cache_path = f.name

        teacher = pd.DataFrame([
            {"subject_id": "p1", "patient_id": "p1", "fold_id": 0, "center": "hup", "center_id": 0,
             "channel_id": 0, "channel_name": "a1", "label_ez": 1, "patient_ez_count": 2,
             "a9v3_oof_score": 0.9, "a9v3_rank_eval": 1},
            {"subject_id": "p1", "patient_id": "p1", "fold_id": 0, "center": "hup", "center_id": 0,
             "channel_id": 1, "channel_name": "a2", "label_ez": 1, "patient_ez_count": 2,
             "a9v3_oof_score": 0.5, "a9v3_rank_eval": 2},
        ])

        phys, audit, feaudit = inspect_and_compute_phys_core(teacher, cache_path)
        self.assertTrue(audit["phys_core_score_available"], audit.get("phys_core_reason", ""))
        used_count = sum(1 for r in feaudit if r["used"])
        self.assertEqual(used_count, 2, f"Expected 2 used features, got {used_count}")
        self.assertFalse((phys.abs() < 1e-8).all(),
                         "phys_core_score should not be all zero")
        # a1 > a2: both features favour a1
        self.assertGreater(float(phys.iloc[0]), float(phys.iloc[1]),
                           "a1 (high gamma, low latency) should rank above a2")

        Path(cache_path).unlink()

    def test_usable_features_but_zero_match_raises(self):
        import pickle, tempfile
        fake_cache = [
            {
                "subject_id": "p_other", "run_id": "r1",
                "channel_names_norm": ["ch_x"],
                "window_feature_names": ["early_high_gamma_slope"],
                "window_features": np.ones((3, 1, 1), dtype=np.float32),
            },
        ]
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(fake_cache, f)
            cache_path = f.name

        teacher = pd.DataFrame([
            {"subject_id": "p_diff", "patient_id": "p_diff", "fold_id": 0, "center": "hup", "center_id": 0,
             "channel_id": 0, "channel_name": "ch_y", "label_ez": 1, "patient_ez_count": 1,
             "a9v3_oof_score": 0.9, "a9v3_rank_eval": 1},
        ])
        with self.assertRaises(ValueError):
            inspect_and_compute_phys_core(teacher, cache_path)

        Path(cache_path).unlink()

    # ---- subject_id fallback to patient_id ----

    def test_teacher_subject_id_fallback_to_patient_id(self):
        import pickle, tempfile
        fake_cache = [{
            "subject_id": "cache_patient_001",
            "channel_names_norm": ["a1", "b1"],
            "window_feature_names": ["early_high_gamma_slope"],
            "window_features": np.array([
                [[10.0], [0.0]],
                [[10.0], [0.0]],
            ], dtype=np.float32),
        }]
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(fake_cache, f)
            cache_path = f.name

        teacher = pd.DataFrame([
            {"subject_id": "teacher_subject_format_different",
             "patient_id": "cache_patient_001",
             "fold_id": 0, "center": "hup", "center_id": 0,
             "channel_id": 0, "channel_name": "a1",
             "label_ez": 1, "patient_ez_count": 2,
             "a9v3_oof_score": 0.9, "a9v3_rank_eval": 1},
            {"subject_id": "teacher_subject_format_different",
             "patient_id": "cache_patient_001",
             "fold_id": 0, "center": "hup", "center_id": 0,
             "channel_id": 1, "channel_name": "b1",
             "label_ez": 1, "patient_ez_count": 2,
             "a9v3_oof_score": 0.5, "a9v3_rank_eval": 2},
        ])

        phys, audit, _ = inspect_and_compute_phys_core(teacher, cache_path)
        self.assertTrue(audit["phys_core_score_available"], audit.get("phys_core_reason", ""))
        self.assertEqual(audit["phys_core_matched_teacher_rows"], 2,
                         "Both rows should match via patient_id fallback")
        self.assertGreater(float(phys.iloc[0]), float(phys.iloc[1]),
                           "a1 (raw=10) should score above b1 (raw=0)")
        Path(cache_path).unlink()

    # ---- object sample with patient_id only ----

    def test_object_sample_with_patient_id_only(self):
        import pickle, tempfile

        sample = _FakeSample()
        sample.patient_id = "p_obj"
        sample.channel_names_norm = ["a1", "b1"]
        sample.window_feature_names = ["early_high_gamma_slope"]
        sample.window_features = np.array([
            [[5.0], [0.0]],
            [[5.0], [0.0]],
        ], dtype=np.float32)
        fake_cache = [sample]

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(fake_cache, f)
            cache_path = f.name

        teacher = pd.DataFrame([
            {"subject_id": "p_obj", "patient_id": "p_obj",
             "fold_id": 0, "center": "hup", "center_id": 0,
             "channel_id": 0, "channel_name": "a1",
             "label_ez": 1, "patient_ez_count": 2,
             "a9v3_oof_score": 0.9, "a9v3_rank_eval": 1},
            {"subject_id": "p_obj", "patient_id": "p_obj",
             "fold_id": 0, "center": "hup", "center_id": 0,
             "channel_id": 1, "channel_name": "b1",
             "label_ez": 1, "patient_ez_count": 2,
             "a9v3_oof_score": 0.5, "a9v3_rank_eval": 2},
        ])

        phys, audit, _ = inspect_and_compute_phys_core(teacher, cache_path)
        self.assertTrue(audit["phys_core_score_available"], audit.get("phys_core_reason", ""))
        self.assertEqual(audit["phys_core_matched_teacher_rows"], 2,
                         "Both rows should match via patient_id on object sample")
        self.assertGreater(float(phys.iloc[0]), float(phys.iloc[1]),
                           "a1 (raw=5) should score above b1 (raw=0)")
        Path(cache_path).unlink()


if __name__ == "__main__":
    unittest.main()
