from __future__ import annotations

import unittest

from a12_vcsn.calibration import ProbabilityCalibrator
from a12_vcsn.protocol import ProtocolError, assert_no_leakage


class A12NoLeakageProvenanceTests(unittest.TestCase):
    def test_calibrator_records_real_fit_subjects(self):
        calibrator = ProbabilityCalibrator().fit([.1, .9], [0, 1], subject_ids=["train-a", "train-b"])
        self.assertEqual(calibrator.fit_subjects, {"train-a", "train-b"})

    def test_every_fit_and_selection_role_rejects_outer_test(self):
        roles = ["scaler_subjects", "calibration_subjects", "gate_subjects",
                 "gate_threshold_subjects", "hyperparameter_search_subjects",
                 "checkpoint_selection_subjects", "ensemble_selection_subjects"]
        for role in roles:
            kwargs = {name: ["train"] for name in roles}
            kwargs[role] = ["test"]
            with self.subTest(role=role), self.assertRaises(ProtocolError):
                assert_no_leakage(train_subjects=["train"], test_subjects=["test"],
                                  feature_names=["score_delta"], **kwargs)


if __name__ == "__main__":
    unittest.main()
