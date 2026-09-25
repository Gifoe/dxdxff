from __future__ import annotations

import copy
import unittest

import numpy as np

from a12_vcsn.cache_filtering import filter_valid_cache_records


def _record(subject: str, run: str, values, names=("A1", "A2")):
    return {"subject_id": subject, "run_id": run, "canonical_channels": list(names),
            "sample": {"window_features": np.asarray(values, dtype=float)}}


class A12CacheFilteringContractTests(unittest.TestCase):
    def test_invalid_records_are_dropped_and_duplicate_group_is_fully_dropped(self):
        good = np.ones((2, 2, 3))
        cache = {"feature_names": ["f0", "f1", "f2"], "patient_index": {}, "run_records": [
            _record("P1", "good", good),
            _record("P1", "dup", good), _record("P1", "dup", good),
            _record("P2", "shape", np.ones((2, 2))),
            _record("P3", "axis", good, names=("A1",)),
            _record("P4", "finite", np.full((2, 2, 3), np.nan)),
            _record("P5", "dim", np.ones((2, 2, 4))),
        ]}
        original = copy.deepcopy(cache)
        filtered, audit = filter_valid_cache_records(cache, invalid_record_policy="drop",
                                                     strict_patient_coverage=False)
        self.assertEqual([(r["subject_id"], r["run_id"]) for r in filtered["run_records"]], [("P1", "good")])
        reasons = {row["reason"] for row in audit["excluded_records"]}
        self.assertTrue({"duplicate_subject_run", "invalid_shape", "channel_axis_mismatch",
                         "all_nonfinite", "feature_dimension_mismatch"}.issubset(reasons))
        self.assertEqual(cache["run_records"][0]["sample"]["window_features"].tolist(),
                         original["run_records"][0]["sample"]["window_features"].tolist())
        self.assertNotIn("P1", str(audit["excluded_records"]))

    def test_zero_valid_run_patient_fails_strict(self):
        cache = {"feature_names": ["f0"], "patient_index": {"P": {}}, "run_records": [
            _record("P", "bad", np.ones((2, 1, 1)), names=()),
        ]}
        with self.assertRaises(ValueError):
            filter_valid_cache_records(cache, invalid_record_policy="drop", strict_patient_coverage=True)


if __name__ == "__main__":
    unittest.main()
