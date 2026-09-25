import unittest

from a12_vcsn.protocol import ProtocolError, assert_frozen_anchor, assert_no_leakage
from tests.test_a12_schema import raw_ledger


class A12ProtocolTests(unittest.TestCase):
    def test_requires_one_source_fold_per_subject(self):
        ledger, _ = __import__("a12_vcsn.schemas", fromlist=["build_canonical_ledger"]).build_canonical_ledger(raw_ledger())
        ledger.loc[0, "outer_fold"] = 2
        with self.assertRaises(ProtocolError):
            assert_frozen_anchor(ledger, expected_patients=None, expected_folds=1)

    def test_rejects_train_test_overlap_and_forbidden_features(self):
        with self.assertRaises(ProtocolError):
            assert_no_leakage(
                train_subjects={"p1", "p2"},
                test_subjects={"p2"},
                feature_names=["v3_score", "clinical_true_ez"],
                calibration_subjects={"p1"},
                gate_subjects={"p1"},
            )

