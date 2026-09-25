from __future__ import annotations

import unittest

from neuroez_c.raw_brainbert_data import audit_raw_cache_onset_timing, subject_sets_for_fold


class CleanNEZNoLeakageTests(unittest.TestCase):
    def test_success_train_only_ssl_excludes_test_and_failure_subjects(self) -> None:
        splits = [
            {"fold_idx": 1, "train_subjects": ["s1", "s2"], "test_subjects": ["s3"]},
            {"fold_idx": 2, "train_subjects": ["s1", "s3"], "test_subjects": ["s2"]},
        ]
        result = subject_sets_for_fold(splits, fold_idx=1, failure_subjects=["f1", "f2"])

        self.assertEqual(result["ssl_subjects"], ["s1", "s2"])
        self.assertEqual(result["leakage_success_test_subjects_in_ssl"], [])
        self.assertEqual(result["failure_ssl_subjects"], [])

    def test_raw_cache_without_onset_metadata_is_not_marked_onset_centered(self) -> None:
        audit = audit_raw_cache_onset_timing(
            [
                {
                    "subject_id": "s1",
                    "raw_temporal_duration_sec": 20.0,
                    "raw_waveform": "present",
                }
            ]
        )

        self.assertFalse(audit["onset_timing_verified"])
        self.assertEqual(audit["rawbb_temporal_pooling_basis"], "record_midpoint_fallback")


if __name__ == "__main__":
    unittest.main()
