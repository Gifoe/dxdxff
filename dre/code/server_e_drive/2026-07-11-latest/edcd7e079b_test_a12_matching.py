import unittest

import pandas as pd

from a12_vcsn.matching import select_max_weight_swaps


class A12MatchingTests(unittest.TestCase):
    def test_matching_is_one_to_one_and_k_preserving(self):
        edges = pd.DataFrame(
            {"eject_channel": ["e1", "e1", "e2"], "add_channel": ["a1", "a2", "a1"], "utility": [0.9, 0.8, 0.7]}
        )
        selected = select_max_weight_swaps(edges, max_swaps=2, utility_threshold=0.0)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected["eject_channel"].nunique(), 2)
        self.assertEqual(selected["add_channel"].nunique(), 2)

