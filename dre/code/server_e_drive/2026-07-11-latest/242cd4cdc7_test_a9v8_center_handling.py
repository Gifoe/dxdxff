import unittest

import pandas as pd

from scripts.build_a9v3_oof_teacher import build_center_mapping_audit
from scripts.build_latent_core_targets import build_latent_core_targets_dataframe, normalize_center


class A9v8CenterHandlingTests(unittest.TestCase):
    def test_broad_center_rule_uses_center_string_when_center_id_is_noninformative(self):
        teacher = pd.DataFrame(
            [
                {"subject_id": "hup:p1", "patient_id": "hup:p1", "fold_id": 1, "center": "hup", "center_id": 4, "channel_id": 0, "channel_name": "a", "label_ez": 1, "patient_ez_count": 1, "a9v3_oof_score": 0.9, "a9v3_rank_eval": 1},
                {"subject_id": "lzu:p2", "patient_id": "lzu:p2", "fold_id": 2, "center": "lzu", "center_id": 4, "channel_id": 0, "channel_name": "a", "label_ez": 1, "patient_ez_count": 1, "a9v3_oof_score": 0.8, "a9v3_rank_eval": 1},
                {"subject_id": "multi:p3", "patient_id": "multi:p3", "fold_id": 3, "center": "multicenter", "center_id": 4, "channel_id": 0, "channel_name": "a", "label_ez": 1, "patient_ez_count": 1, "a9v3_oof_score": 0.7, "a9v3_rank_eval": 1},
                {"subject_id": "ped:p4", "patient_id": "ped:p4", "fold_id": 4, "center": "pediatric", "center_id": 4, "channel_id": 0, "channel_name": "a", "label_ez": 1, "patient_ez_count": 1, "a9v3_oof_score": 0.6, "a9v3_rank_eval": 1},
            ]
        )

        self.assertEqual(normalize_center(" Pediatric "), "pediatric")
        targets, audit, _ = build_latent_core_targets_dataframe(
            teacher,
            broad_centers={"lzu", "pediatric"},
            rho=0.3,
            tau_q=0.1,
            phys_core_mode="teacher_only",
            allow_teacher_only_core=True,
        )

        by_center = dict(zip(targets["center"], targets["core_target_type"]))
        self.assertEqual(by_center["hup"], "strong_label")
        self.assertEqual(by_center["multicenter"], "strong_label")
        self.assertIn(by_center["lzu"], {"teacher_only_latent_core", "latent_core"})
        self.assertIn(by_center["pediatric"], {"teacher_only_latent_core", "latent_core"})
        self.assertFalse(audit["center_id_is_informative"])
        self.assertEqual(audit["broad_center_rule"], "normalized_center_string")

        center_audit = build_center_mapping_audit(teacher)
        self.assertFalse(bool(center_audit["center_id_is_informative"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
