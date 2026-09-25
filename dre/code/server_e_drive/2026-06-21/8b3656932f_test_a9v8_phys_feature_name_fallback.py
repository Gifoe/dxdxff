import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_latent_core_targets import (
    _parse_physics_state_features_arg,
    build_latent_core_targets_dataframe,
    inspect_and_compute_phys_core,
)


PHYSICS_FEATURES = (
    "early_high_gamma_slope,early_line_length_slope,onset_latency_high_gamma,"
    "onset_latency_line_length,onset_rank_high_gamma,onset_rank_line_length,"
    "high_gamma_top20pct_mean,line_length_top20pct_mean,hfo80_150_event_rate,"
    "hfo80_150_duration_fraction,hfo80_150_mean_envelope_z,hfo80_150_max_envelope_z"
)


def _teacher() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": "p1",
                "patient_id": "p1",
                "fold_id": 1,
                "center": "hup",
                "center_id": 4,
                "channel_id": 0,
                "channel_name": "A1",
                "label_ez": 1,
                "patient_ez_count": 2,
                "a9v3_oof_score": 0.9,
                "a9v3_rank_eval": 1,
            },
            {
                "subject_id": "p1",
                "patient_id": "p1",
                "fold_id": 1,
                "center": "hup",
                "center_id": 4,
                "channel_id": 1,
                "channel_name": "A2",
                "label_ez": 1,
                "patient_ez_count": 2,
                "a9v3_oof_score": 0.4,
                "a9v3_rank_eval": 2,
            },
        ]
    )


def _deterministic_phys_window_features() -> np.ndarray:
    wf = np.zeros((3, 2, 12), dtype=np.float32)
    positive_idx = [0, 1, 6, 7, 8, 9, 10, 11]
    latency_idx = [2, 3, 4, 5]
    wf[:, 0, positive_idx] = 2.0
    wf[:, 1, positive_idx] = -2.0
    wf[:, 0, latency_idx] = -2.0
    wf[:, 1, latency_idx] = 2.0
    return wf


class A9v8PhysFeatureNameFallbackTests(unittest.TestCase):
    def _write_cache(self, cache) -> str:
        handle = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        with handle:
            pickle.dump(cache, handle)
        return handle.name

    def test_cache_without_feature_names_uses_cli_names_when_lengths_match(self):
        cache_path = self._write_cache(
            {
                "run_records": [
                    {
                        "sample": {
                            "subject_id": "p1",
                            "channel_names": ["A1", "A2"],
                            "window_features": _deterministic_phys_window_features(),
                        }
                    }
                ]
            }
        )

        targets, audit, feaudit = build_latent_core_targets_dataframe(
            _teacher(),
            broad_centers={"lzu", "pediatric"},
            cache_path=cache_path,
            phys_core_mode="auto_s5",
            require_phys_core=True,
            physics_state_features=PHYSICS_FEATURES,
            feature_name_source="auto",
            physics_feature_slice="all",
        )

        self.assertTrue(audit["phys_core_score_available"])
        self.assertEqual(audit["feature_name_source_used"], "cli")
        self.assertEqual(audit["physics_state_features_cli_len"], 12)
        self.assertEqual(audit["window_feature_dim"], 12)
        self.assertEqual(len([row for row in feaudit if row["used"]]), 12)
        self.assertFalse((targets["phys_core_score"].abs() < 1e-8).all())
        Path(cache_path).unlink()

    def test_cli_feature_name_dim_mismatch_fails_closed(self):
        cache_path = self._write_cache(
            {
                "run_records": [
                    {
                        "sample": {
                            "subject_id": "p1",
                            "channel_names": ["A1", "A2"],
                            "window_features": np.random.randn(3, 2, 13).astype(np.float32),
                        }
                    }
                ]
            }
        )

        with self.assertRaisesRegex(ValueError, "does not match window_features dim"):
            inspect_and_compute_phys_core(
                _teacher(),
                cache_path,
                cli_feature_names=_parse_physics_state_features_arg(PHYSICS_FEATURES),
                feature_name_source="auto",
                physics_feature_slice="all",
            )
        Path(cache_path).unlink()

    def test_nested_sample_unwrap_extracts_window_features(self):
        cache_path = self._write_cache(
            {
                "run_records": [
                    {
                        "subject_id": "outer_should_not_break",
                        "sample": {
                            "subject_id": "p1",
                            "channel_names_norm": ["a1", "a2"],
                            "window_features": _deterministic_phys_window_features(),
                        },
                    }
                ]
            }
        )

        phys, audit, _ = inspect_and_compute_phys_core(
            _teacher(),
            cache_path,
            cli_feature_names=_parse_physics_state_features_arg(PHYSICS_FEATURES),
            feature_name_source="auto",
            physics_feature_slice="all",
        )

        self.assertTrue(audit["phys_core_score_available"])
        self.assertEqual(audit["phys_core_matched_teacher_rows"], 2)
        self.assertFalse((phys.abs() < 1e-8).all())
        Path(cache_path).unlink()

    def test_feature_name_source_cache_requires_cache_names(self):
        cache_path = self._write_cache(
            {
                "run_records": [
                    {
                        "sample": {
                            "subject_id": "p1",
                            "channel_names": ["A1", "A2"],
                            "window_features": np.random.randn(3, 2, 12).astype(np.float32),
                        }
                    }
                ]
            }
        )

        with self.assertRaisesRegex(ValueError, "cache_schema_missing_window_feature_names"):
            inspect_and_compute_phys_core(
                _teacher(),
                cache_path,
                cli_feature_names=_parse_physics_state_features_arg(PHYSICS_FEATURES),
                feature_name_source="cache",
                physics_feature_slice="all",
            )
        Path(cache_path).unlink()


if __name__ == "__main__":
    unittest.main()
