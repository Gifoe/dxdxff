from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from a12_vcsn.io import load_hnc_oof_ledger
from a12_vcsn.schemas import LedgerSchemaError


class A12HNCSemanticsTests(unittest.TestCase):
    def _write(self, root: str, semantics: str):
        path = Path(root) / "hnc.csv"
        pd.DataFrame({"subject_id": ["P"], "fold_idx": [2], "channel_name": ["la01"],
                      "score": [.8], "hnc_score_semantics": [semantics]}).to_csv(path, index=False)
        return path

    def test_hnc_normalized_fold_join_and_p_nez_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame, _ = load_hnc_oof_ledger(self._write(tmp, "p_nez"))
        self.assertEqual(frame.loc[0, "channel_name_norm"], "LA1")
        self.assertEqual(frame.loc[0, "fold_idx"], 2)
        self.assertEqual(frame.loc[0, "hnc_eject_priority"], .8)
        self.assertEqual(frame.loc[0, "hnc_add_priority"], -.8)

    def test_hnc_p_ez_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame, _ = load_hnc_oof_ledger(self._write(tmp, "p_ez"))
        self.assertEqual(frame.loc[0, "hnc_eject_priority"], -.8)
        self.assertEqual(frame.loc[0, "hnc_add_priority"], .8)

    def test_unknown_hnc_semantics_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(LedgerSchemaError):
                load_hnc_oof_ledger(self._write(tmp, "mystery"))


if __name__ == "__main__":
    unittest.main()
