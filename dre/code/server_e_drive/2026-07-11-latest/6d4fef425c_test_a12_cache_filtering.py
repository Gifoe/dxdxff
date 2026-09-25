from __future__ import annotations

import unittest

import numpy as np

from a12_vcsn.cache_filtering import filter_cache_records


class A12CacheFilteringTests(unittest.TestCase):
    def test_drop_filters_missing_channel_source_and_duplicate_subject_run(self):
        records = [
            {"subject_id": "p", "run_id": "r1", "sample": {"window_features": np.ones((2, 2, 1))}},
            {"subject_id": "p", "run_id": "r1", "sample": {"window_features": np.ones((2, 2, 1))}},
            {"subject_id": "q", "run_id": "r2", "sample": {"window_features": np.ones((2, 2, 1))}},
        ]
        kept, audit = filter_cache_records(records, {"p": {"canonical_channels": ["A1", "A2"]}}, policy="drop")
        self.assertEqual(len(kept), 0)
        self.assertEqual(audit["dropped_missing_channel_source"], 1)
        self.assertEqual(audit["dropped_duplicate_subject_run"], 2)


if __name__ == "__main__": unittest.main()
