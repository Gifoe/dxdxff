from __future__ import annotations

from pathlib import Path
import pickle
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from ez_features import WINDOW_NODE_FEATURE_NAMES
from run_baseline_three_centers.build_center_caches_from_patient_records import _merge_center_caches


def _payload(center: str) -> dict:
    names = list(WINDOW_NODE_FEATURE_NAMES)
    return {
        "cache_version": "test",
        "source_center": center,
        "window_feature_names": names,
        "window_feature_groups": {"base": True, "extra": []},
        "run_records": [
            {
                "subject_id": f"{center}:p1",
                "run_id": f"{center}:r1",
                "sample": {
                    "sample_id": f"{center}:s1",
                    "window_features": np.zeros((1, 1, len(names)), dtype=np.float32),
                    "window_feature_names": names,
                },
            }
        ],
        "patient_index": {f"{center}:p1": {"subject_id": f"{center}:p1"}},
    }


class CenterCacheMergeTests(unittest.TestCase):
    def test_direct_all_merge_includes_pediatric_cache_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            for center in ("lzu", "hup", "multicenter", "pediatric"):
                with open(cache_dir / f"{center}_window_cache.pkl", "wb") as fout:
                    pickle.dump(_payload(center), fout)

            summary = _merge_center_caches(
                cache_dir,
                cache_dir / "all_window_cache.pkl",
                SimpleNamespace(patient_records=Path("records.pkl"), force_cache_build=True),
            )

            self.assertEqual(summary["run_records"], 4)
            self.assertEqual({item["center"] for item in summary["merged_from"]}, {"lzu", "hup", "multicenter", "pediatric"})


if __name__ == "__main__":
    unittest.main()
