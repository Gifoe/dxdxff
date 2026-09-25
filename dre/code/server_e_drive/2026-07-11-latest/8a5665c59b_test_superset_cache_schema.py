from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.inspect_superset_cache_schema import (
    EXPECTED_FEATURE_DIM,
    REQUIRED_FEATURE_GROUPS,
    REQUIRED_FEATURE_NAMES,
    inspect_superset_cache_schema,
)


class SupersetCacheSchemaTests(unittest.TestCase):
    def test_superset_schema_report_accepts_complete_payload(self):
        feature_names = list(REQUIRED_FEATURE_NAMES)
        payload = {
            "window_feature_groups": {"base": True, "extra": list(REQUIRED_FEATURE_GROUPS)},
            "window_feature_names": feature_names,
            "run_records": [
                {
                    "sample": {
                        "window_features": np.zeros((2, 3, len(feature_names)), dtype=np.float32),
                        "window_feature_names": feature_names,
                    }
                }
            ],
            "patient_index": {"lzu:p1": {"run_ids": ["r1"]}},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "all_window_cache.pkl"
            with open(cache_path, "wb") as fout:
                pickle.dump(payload, fout)

            report = inspect_superset_cache_schema(cache_path)

        self.assertEqual(report["feature_dim"], EXPECTED_FEATURE_DIM)
        self.assertEqual(report["missing_features"], [])
        self.assertEqual(report["missing_feature_groups"], [])
        self.assertTrue(report["feature_names_unique"])
        self.assertEqual(report["num_patients"], 1)

    def test_superset_schema_report_flags_missing_features(self):
        feature_names = list(REQUIRED_FEATURE_NAMES[:-1])
        payload = {
            "window_feature_groups": {"base": True, "extra": list(REQUIRED_FEATURE_GROUPS[:-1])},
            "window_feature_names": feature_names,
            "run_records": [
                {
                    "sample": {
                        "window_features": np.zeros((1, 2, len(feature_names)), dtype=np.float32),
                        "window_feature_names": feature_names,
                    }
                }
            ],
            "patient_index": {},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "all_window_cache.pkl"
            with open(cache_path, "wb") as fout:
                pickle.dump(payload, fout)

            report = inspect_superset_cache_schema(cache_path)

        self.assertNotEqual(report["missing_features"], [])
        self.assertNotEqual(report["missing_feature_groups"], [])


if __name__ == "__main__":
    unittest.main()
