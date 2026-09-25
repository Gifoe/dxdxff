import unittest

from a12_vcsn.pair_labels import compute_pair_label
from tests.test_a12_schema import raw_ledger


class A12PairLabelTests(unittest.TestCase):
    def test_evaluator_derived_beneficial_and_harmful_pairs(self):
        raw = raw_ledger()
        raw["predicted_ez"] = [1, 1, 0, 0]  # A02 is a false positive and B02 a false negative.
        ledger, _ = __import__("a12_vcsn.schemas", fromlist=["build_canonical_ledger"]).build_canonical_ledger(raw)
        harmful = compute_pair_label(ledger, "A01REF", "B1")
        beneficial = compute_pair_label(ledger, "A02REF", "B02")
        self.assertLess(harmful["delta_patient_macro_f1"], 0.0)
        self.assertGreater(beneficial["delta_patient_macro_f1"], 0.0)
