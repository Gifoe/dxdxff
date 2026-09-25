from __future__ import annotations

import unittest

import pandas as pd

from a12_vcsn.protocol import ProtocolError
from a12_vcsn.provenance import (assert_resume_compatible, execution_fingerprint,
                                 git_state)


class A12ResumeFingerprintTests(unittest.TestCase):
    def test_git_state_contains_diff_and_untracked_hashes(self):
        state = git_state()
        self.assertTrue({"commit", "diff_hash", "cached_diff_hash", "a12_untracked_source_hash"}.issubset(state))

    def test_resume_rejects_changed_filter_manifest_registry_and_config(self):
        ledger = pd.DataFrame({"subject_id": ["P"], "channel_name_original": ["A1"], "old_v3_selected": [1]})
        base = execution_fingerprint(ledger=ledger, cache_audit={"cache_sha256": "cache", "filter_audit": {"x": 1}},
                                     feature_registry={"f": 1}, config={"max_swaps": 1}, variant="A12-V1", outer_fold=1)
        mutations = [
            execution_fingerprint(ledger=ledger, cache_audit={"cache_sha256": "cache", "filter_audit": {"x": 2}}, feature_registry={"f": 1}, config={"max_swaps": 1}, variant="A12-V1", outer_fold=1),
            execution_fingerprint(ledger=ledger, cache_audit={"cache_sha256": "cache", "filter_audit": {"x": 1}}, feature_registry={"f": 2}, config={"max_swaps": 1}, variant="A12-V1", outer_fold=1),
            execution_fingerprint(ledger=ledger, cache_audit={"cache_sha256": "cache", "filter_audit": {"x": 1}}, feature_registry={"f": 1}, config={"max_swaps": 2}, variant="A12-V1", outer_fold=1),
        ]
        for changed in mutations:
            with self.assertRaises(ProtocolError) as caught:
                assert_resume_compatible(base, changed)
            self.assertIn("differing fields", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
