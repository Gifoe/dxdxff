from __future__ import annotations

import unittest

import pandas as pd

from a12_vcsn.candidate_pool import CandidateConfig, build_candidate_pools


class A12CandidateTopologyTests(unittest.TestCase):
    def test_topology_sources_and_flags_are_preserved_after_cap(self):
        ledger = pd.DataFrame({
            "subject_id": ["P"] * 6, "outer_fold": [1] * 6,
            "channel_name_original": ["A1", "A2", "A3", "A5", "B1", "bad"],
            "channel_name_norm": ["A1", "A2", "A3", "A5", "B1", "BAD"],
            "old_v3_selected": [1, 1, 0, 1, 0, 0],
            "old_v3_score_ez": [.9, .8, .7, .6, .5, .4], "old_v3_rank": [1, 2, 3, 4, 5, 6],
        })
        pairs = build_candidate_pools(ledger, CandidateConfig(max_eject_candidates=3,
                                                               max_add_candidates=3,
                                                               max_pairs_per_patient=9))
        self.assertIn("eject_source_isolated", pairs.columns)
        self.assertIn("eject_source_segment_edge", pairs.columns)
        self.assertIn("add_source_same_shaft", pairs.columns)
        self.assertIn("add_source_segment_neighbor", pairs.columns)
        self.assertTrue(pairs.loc[pairs["add_channel"] == "A3", "add_source_same_shaft"].eq(1).all())
        self.assertFalse(pairs.loc[pairs["add_channel"] == "bad",
                                   ["add_source_same_shaft", "add_source_segment_neighbor"]].any().any())


if __name__ == "__main__":
    unittest.main()
