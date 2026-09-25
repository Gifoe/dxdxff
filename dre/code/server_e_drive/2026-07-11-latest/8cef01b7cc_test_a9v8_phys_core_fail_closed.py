import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.build_latent_core_targets import build_latent_core_targets_dataframe


def _teacher() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"subject_id": "lzu:p1", "patient_id": "lzu:p1", "fold_id": 1, "center": "lzu", "center_id": 4, "channel_id": 0, "channel_name": "a", "label_ez": 1, "patient_ez_count": 1, "a9v3_oof_score": 0.9, "a9v3_rank_eval": 1},
            {"subject_id": "lzu:p1", "patient_id": "lzu:p1", "fold_id": 1, "center": "lzu", "center_id": 4, "channel_id": 1, "channel_name": "b", "label_ez": 0, "patient_ez_count": 1, "a9v3_oof_score": 0.1, "a9v3_rank_eval": 2},
        ]
    )


class A9v8PhysCoreFailClosedTests(unittest.TestCase):
    def test_auto_s5_without_usable_features_raises_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_cache = Path(tmpdir) / "missing.pkl"
            with self.assertRaisesRegex(ValueError, "No usable physiology features were extracted from cache"):
                build_latent_core_targets_dataframe(
                    _teacher(),
                    broad_centers={"lzu", "pediatric"},
                    rho=0.3,
                    tau_q=0.1,
                    cache_path=missing_cache,
                    phys_core_mode="auto_s5",
                    allow_teacher_only_core=False,
                    require_phys_core=True,
                )

    def test_teacher_only_mode_requires_explicit_allow_and_marks_audit(self):
        targets, audit, _ = build_latent_core_targets_dataframe(
            _teacher(),
            broad_centers={"lzu", "pediatric"},
            rho=0.3,
            tau_q=0.1,
            phys_core_mode="teacher_only",
            allow_teacher_only_core=True,
        )

        self.assertFalse(audit["phys_core_score_available"])
        self.assertTrue(audit["teacher_only_fallback_used"])
        self.assertEqual(audit["phys_core_reason"], "explicit_teacher_only_mode")
        self.assertTrue((targets["phys_core_score"] == 0.0).all())


if __name__ == "__main__":
    unittest.main()
