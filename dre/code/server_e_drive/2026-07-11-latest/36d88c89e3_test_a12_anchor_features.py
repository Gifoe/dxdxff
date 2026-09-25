import unittest

from a12_vcsn.anchor_features import build_patient_nez_anchor_features
from tests.test_a12_schema import raw_ledger


class A12AnchorFeatureTests(unittest.TestCase):
    def test_anchor_does_not_depend_on_clinical_label(self):
        ledger, _ = __import__("a12_vcsn.schemas", fromlist=["build_canonical_ledger"]).build_canonical_ledger(raw_ledger())
        features = {"p1": {"A01REF": [4.0, 0.0], "A02REF": [0.0, 0.0], "B1": [0.1, 0.0], "B02": [3.0, 0.0]}}
        first, _ = build_patient_nez_anchor_features(ledger, features, min_channels=1)
        ledger["clinical_true_ez"] = 1 - ledger["clinical_true_ez"]
        second, _ = build_patient_nez_anchor_features(ledger, features, min_channels=1)
        self.assertEqual(first["distance_to_patient_nez_anchor_l2"].tolist(), second["distance_to_patient_nez_anchor_l2"].tolist())
