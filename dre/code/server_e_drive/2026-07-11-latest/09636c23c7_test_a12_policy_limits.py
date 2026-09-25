from __future__ import annotations

import unittest

import pandas as pd

from a12_vcsn.thresholds import select_thresholds


class A12PolicyLimitTests(unittest.TestCase):
    def test_max_swaps_limit_is_respected_by_grid(self):
        ledger = pd.DataFrame({"subject_id": ["p"] * 2, "outer_fold": [1, 1], "center": ["x", "x"], "channel_name_original": ["A1", "A2"], "clinical_true_ez": [1, 0], "old_v3_selected": [1, 0]})
        inner = pd.DataFrame({"subject_id": ["p"], "eject_channel": ["A1"], "add_channel": ["A2"], "p_benefit": [.9], "p_harm": [.1], "pred_delta": [.2], "utility": [.2], "seed_positive_fraction": [1.]})
        _, grid = select_thresholds(inner, ledger=ledger, max_swaps_limit=1, return_grid=True)
        self.assertEqual(set(grid["max_swaps"]), {0, 1})


if __name__ == "__main__":
    unittest.main()
