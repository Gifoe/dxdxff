from __future__ import annotations

import unittest

import pandas as pd

from a12_vcsn.reporting import PATIENT_DELTA_COLUMNS, build_patient_delta_table, patient_bootstrap


class A12ReportingSchemaTests(unittest.TestCase):
    def test_patient_delta_public_schema(self):
        anchor = pd.DataFrame({"subject_id": ["P"], "outer_fold": [1], "center": ["x"],
                               "patient_macro_f1": [.5], "patient_ez_f1": [.4], "patient_nez_f1": [.6]})
        candidate = pd.DataFrame({"subject_id": ["P"], "outer_fold": [1], "center": ["x"],
                                  "patient_macro_f1": [.7], "patient_ez_f1": [.8], "patient_nez_f1": [.6]})
        result = build_patient_delta_table(anchor, candidate, pd.DataFrame({"subject_id": ["P", "P"]}))
        self.assertEqual(result.columns.tolist(), PATIENT_DELTA_COLUMNS)
        self.assertAlmostEqual(result.loc[0, "delta_patient_macro_f1"], .2)
        self.assertEqual(result.loc[0, "n_swaps"], 2)

    def test_bootstrap_contains_probability_positive_and_counts(self):
        anchor = pd.DataFrame({"subject_id": ["A", "B", "C"], "patient_macro_f1": [.5, .5, .5]})
        candidate = pd.DataFrame({"subject_id": ["A", "B", "C"], "patient_macro_f1": [.6, .4, .5]})
        result = patient_bootstrap(anchor, candidate, n_bootstrap=100, seed=1)
        self.assertTrue({"probability_delta_positive", "n_improved", "n_harmed", "n_unchanged"}.issubset(result))
        self.assertEqual(result["n_improved"] + result["n_harmed"] + result["n_unchanged"], 3)


if __name__ == "__main__":
    unittest.main()
