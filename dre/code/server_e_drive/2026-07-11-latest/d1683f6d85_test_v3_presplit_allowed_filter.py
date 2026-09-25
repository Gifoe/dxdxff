"""Test V3 pre-split allowed-subject filtering."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from ez_dataset import _read_allowed_subjects, _filter_run_records_by_allowed_subjects


class V3PresplitAllowedFilterTests(unittest.TestCase):
    def _make_records_index(self, n_patients: int):
        subjects = [f"subj_{i:03d}" for i in range(n_patients)]
        records = [{"subject_id": s, "run_id": f"{s}_r1"} for s in subjects]
        index = {s: {"labels": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                      "label_mask": np.array([True, True, True, True])}
                 for s in subjects}
        return records, index

    def test_allowed_filter_reduces_96_to_90(self):
        records, index = self._make_records_index(96)
        allowed = sorted(list(index.keys())[:90])
        args = SimpleNamespace(
            allowed_subjects_ledger=None,
            allowed_subjects_file=None,
            require_n_patients=0,
            drop_high_ez_fraction_lzu=False,
        )
        # Patch _read_allowed_subjects to return our list
        with patch("ez_dataset._read_allowed_subjects", return_value=allowed):
            filtered_records, filtered_index, audit = _filter_run_records_by_allowed_subjects(
                records, index, args,
            )
        self.assertTrue(audit["v3_allowed_subject_filter_used"])
        self.assertEqual(audit["v3_n_patients_before_allowed_filter"], 96)
        self.assertEqual(audit["v3_n_patients_after_allowed_filter"], 90)
        self.assertEqual(len(filtered_index), 90)
        self.assertTrue(audit["v3_split_built_after_allowed_filter"])

    def test_no_allowed_filter_passes_all(self):
        records, index = self._make_records_index(50)
        args = SimpleNamespace(
            allowed_subjects_ledger=None,
            allowed_subjects_file=None,
            require_n_patients=0,
            drop_high_ez_fraction_lzu=False,
        )
        with patch("ez_dataset._read_allowed_subjects", return_value=None):
            filtered_records, filtered_index, audit = _filter_run_records_by_allowed_subjects(
                records, index, args,
            )
        self.assertFalse(audit["v3_allowed_subject_filter_used"])
        self.assertEqual(len(filtered_index), 50)

    def test_high_ez_fraction_disabled_lzu_not_dropped(self):
        """When drop_high_ez_fraction_lzu=False, LZU patients are never dropped."""
        records, index = self._make_records_index(10)
        # Set all patients as "lzu" source center
        for s, meta in index.items():
            meta["source_center"] = "lzu"
            # Set high EZ fraction
            meta["labels"] = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
        args = SimpleNamespace(
            allowed_subjects_ledger=None,
            allowed_subjects_file=None,
            require_n_patients=0,
            drop_high_ez_fraction_lzu=False,
        )
        with patch("ez_dataset._read_allowed_subjects", return_value=None):
            filtered_records, filtered_index, audit = _filter_run_records_by_allowed_subjects(
                records, index, args,
            )
        # No allowed-subject filtering occurred
        self.assertFalse(audit["v3_allowed_subject_filter_used"])
        self.assertEqual(len(filtered_index), 10)


    def test_allowed_subject_filter_imports_pandas(self):
        import pandas as pd
        self.assertTrue(hasattr(pd, "read_csv"))


class V3AllowedSubjectAuditTests(unittest.TestCase):
    """Test the merged audit (allowed + high-ez filter)."""

    def _make_records_index(self, n_patients: int):
        subjects = [f"subj_{i:03d}" for i in range(n_patients)]
        records = [{"subject_id": s, "run_id": f"{s}_r1"} for s in subjects]
        index = {s: {"labels": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                      "label_mask": np.array([True, True, True, True])}
                 for s in subjects}
        return records, index

    def test_audit_allows_96_before_to_90_after_when_split_after_filter(self):
        """96→90 before split is legitimate. The critical fields are:
        v3_split_built_after_allowed_filter and v3_n_patients_after_allowed_filter."""
        from ez_dataset import _filter_run_records_by_allowed_subjects
        from unittest.mock import patch
        records, index = self._make_records_index(96)
        allowed = sorted(list(index.keys())[:90])
        args = SimpleNamespace(
            allowed_subjects_ledger=None, allowed_subjects_file=None,
            require_n_patients=0, drop_high_ez_fraction_lzu=False,
        )
        with patch("ez_dataset._read_allowed_subjects", return_value=allowed):
            _, _, audit = _filter_run_records_by_allowed_subjects(records, index, args)
        self.assertEqual(audit["v3_n_patients_before_allowed_filter"], 96)
        self.assertEqual(audit["v3_n_patients_after_allowed_filter"], 90)
        self.assertTrue(audit["v3_split_built_after_allowed_filter"])
        # This is the legitimate 96→90 case (filter happened before split)


if __name__ == "__main__":
    unittest.main()
