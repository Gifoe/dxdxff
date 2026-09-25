"""Test that drop_high_ez_fraction_lzu=False preserves all patients."""

import unittest
from types import SimpleNamespace

import numpy as np

from ez_dataset import _filter_high_ez_fraction_lzu


class DisableHighEZFractionFilterTests(unittest.TestCase):
    def _make_lzu_data(self, n: int, ez_fraction: float):
        records = []
        index = {}
        for i in range(n):
            sid = f"lzu_{i:03d}"
            records.append({"subject_id": sid})
            n_ez = max(1, int(round(ez_fraction * 4)))
            labels = np.zeros(4, dtype=np.float32)
            labels[:n_ez] = 1.0
            index[sid] = {
                "source_center": "lzu",
                "labels": labels,
                "label_mask": np.ones(4, dtype=bool),
            }
        return records, index

    def test_disabled_filter_keeps_all_lzu(self):
        records, index = self._make_lzu_data(5, ez_fraction=0.5)
        args = SimpleNamespace(drop_high_ez_fraction_lzu=False, lzu_max_ez_fraction=0.40)
        filtered_records, filtered_index, audit = _filter_high_ez_fraction_lzu(records, index, args)
        self.assertEqual(len(filtered_index), 5)

    def test_enabled_filter_drops_high_ez_lzu(self):
        records, index = self._make_lzu_data(5, ez_fraction=0.5)
        args = SimpleNamespace(drop_high_ez_fraction_lzu=True, lzu_max_ez_fraction=0.40)
        filtered_records, filtered_index, audit = _filter_high_ez_fraction_lzu(records, index, args)
        self.assertLess(len(filtered_index), 5)

    def test_non_lzu_centers_never_dropped(self):
        records = [{"subject_id": f"hup_{i}"} for i in range(3)]
        index = {
            f"hup_{i}": {
                "source_center": "hup",
                "labels": np.ones(4, dtype=np.float32),
                "label_mask": np.ones(4, dtype=bool),
            }
            for i in range(3)
        }
        args = SimpleNamespace(drop_high_ez_fraction_lzu=True, lzu_max_ez_fraction=0.40)
        _, filtered_index, _ = _filter_high_ez_fraction_lzu(records, index, args)
        self.assertEqual(len(filtered_index), 3)

    def test_high_ez_filter_audit_reports_zero_when_disabled(self):
        records, index = self._make_lzu_data(10, ez_fraction=0.5)
        args = SimpleNamespace(drop_high_ez_fraction_lzu=False, lzu_max_ez_fraction=0.40)
        _, _, audit = _filter_high_ez_fraction_lzu(records, index, args)
        self.assertFalse(audit["drop_high_ez_fraction_lzu_enabled"])
        self.assertEqual(audit["n_patients_dropped_by_high_ez_fraction"], 0)
        self.assertEqual(audit["subjects_dropped_by_high_ez_fraction"], [])

    def test_high_ez_filter_audit_reports_dropped_when_enabled(self):
        records, index = self._make_lzu_data(10, ez_fraction=0.5)
        args = SimpleNamespace(drop_high_ez_fraction_lzu=True, lzu_max_ez_fraction=0.40)
        _, _, audit = _filter_high_ez_fraction_lzu(records, index, args)
        self.assertTrue(audit["drop_high_ez_fraction_lzu_enabled"])
        self.assertGreater(audit["n_patients_dropped_by_high_ez_fraction"], 0)
        self.assertGreater(len(audit["subjects_dropped_by_high_ez_fraction"]), 0)


if __name__ == "__main__":
    unittest.main()
