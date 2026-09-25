import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.run_a9v8_phase2_shadow_diagnostic import run_shadow_diagnostic


class A9v8ShadowDiagnosticModeTests(unittest.TestCase):
    def test_teacher_only_latent_core_does_not_emit_teacher_plus_phys_scores(self):
        latent = pd.DataFrame(
            [
                {"patient_id": "p1", "fold_id": 1, "center": "lzu", "channel_name": "a", "label_ez": 1, "patient_ez_count": 1, "a9v3_oof_score": 0.9, "core_score": 1.0, "phys_core_score": 0.0, "phys_core_score_available": False, "teacher_only_fallback_used": True},
                {"patient_id": "p1", "fold_id": 1, "center": "lzu", "channel_name": "b", "label_ez": 0, "patient_ez_count": 1, "a9v3_oof_score": 0.1, "core_score": -1.0, "phys_core_score": 0.0, "phys_core_score_available": False, "teacher_only_fallback_used": True},
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = Path(tmpdir) / "latent_core_audit.json"
            audit_path.write_text(
                '{"phys_core_score_available": false, "teacher_only_fallback_used": true, "phys_core_mode": "teacher_only"}',
                encoding="utf-8",
            )

            summary, _, _ = run_shadow_diagnostic(
                latent,
                audit_json=audit_path,
                gammas=[0.05, 0.10],
                allow_teacher_only=True,
            )

        score_types = set(summary["score_type"])
        self.assertEqual(score_types, {"a9v3_oof_score", "teacher_only_core_score"})
        self.assertFalse(any(item.startswith("teacher_plus_phys_gamma") for item in score_types))
        self.assertTrue((summary["teacher_only_fallback_used"] == True).all())


if __name__ == "__main__":
    unittest.main()
