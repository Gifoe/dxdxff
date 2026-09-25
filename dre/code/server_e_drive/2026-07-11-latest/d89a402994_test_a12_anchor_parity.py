import unittest

from a12_vcsn.evaluation import evaluate_patient_predictions
from a12_vcsn.protocol import anchor_parity
from tests.test_a12_schema import raw_ledger


class A12AnchorParityTests(unittest.TestCase):
    def test_recomputes_frozen_anchor_exactly(self):
        ledger, _ = __import__("a12_vcsn.schemas", fromlist=["build_canonical_ledger"]).build_canonical_ledger(raw_ledger())
        expected = evaluate_patient_predictions(ledger)["patient_macro_f1"]
        report = anchor_parity(ledger, expected_macro_f1=expected, tolerance=1e-12)
        self.assertTrue(report["passed"])
        self.assertEqual(report["n_patients"], 1)

