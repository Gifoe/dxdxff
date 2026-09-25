from __future__ import annotations

import unittest

import pandas as pd

from neuroez_c.settopo_reranker import generate_clean_nez_candidates


def _toy_rows() -> pd.DataFrame:
    rows = []
    for idx in range(1, 11):
        rows.append(
            {
                "fold_idx": 1,
                "subject_id": "p1",
                "channel_name": f"A{idx}",
                "feature_suspicious_z": 100.0 - idx if idx <= 4 else 0.0,
                "raw_dist_onset_z": 50.0 if idx in {9, 10} else 0.0,
                "raw_dist_all_z": 40.0 if idx == 8 else 0.0,
                "true_ez": 1 if idx in {1, 9} else 0,
            }
        )
    return pd.DataFrame(rows)


class CleanNEZCandidateRuleTests(unittest.TestCase):
    def test_top20_top30_top40_candidate_counts_differ_without_neighbors(self) -> None:
        rows = _toy_rows()

        top20, audit20 = generate_clean_nez_candidates(rows, candidate_rule="feature_top20_only")
        top30, audit30 = generate_clean_nez_candidates(rows, candidate_rule="feature_top30_only")
        top40, audit40 = generate_clean_nez_candidates(rows, candidate_rule="feature_top40_only")

        self.assertEqual(int(top20["is_candidate"].sum()), 2)
        self.assertEqual(int(top30["is_candidate"].sum()), 3)
        self.assertEqual(int(top40["is_candidate"].sum()), 4)
        self.assertEqual(audit20["top_fraction"], 0.20)
        self.assertEqual(audit30["top_fraction"], 0.30)
        self.assertEqual(audit40["top_fraction"], 0.40)

    def test_top_fraction_selects_ceil_fraction_even_when_only_one_score_positive(self) -> None:
        rows = _toy_rows()
        rows["feature_suspicious_z"] = [10.0] + [0.0] * 9

        out, _ = generate_clean_nez_candidates(rows, candidate_rule="feature_top30_only")

        self.assertEqual(int(out["is_candidate"].sum()), 3)
        self.assertEqual(set(out.loc[out["is_candidate"].astype(bool), "channel_name"]), {"A1", "A2", "A3"})

    def test_feature_top30_only_excludes_raw_only_high_candidates(self) -> None:
        out, audit = generate_clean_nez_candidates(_toy_rows(), candidate_rule="feature_top30_only")
        selected = set(out.loc[out["is_candidate"].astype(bool), "channel_name"])

        self.assertEqual(selected, {"A1", "A2", "A3"})
        self.assertNotIn("A9", selected)
        self.assertTrue(audit["diagnostic_uses_true_labels"])

    def test_raw_top30_only_excludes_feature_only_high_candidates(self) -> None:
        rows = _toy_rows()
        rows.loc[rows["channel_name"].eq("A1"), ["raw_dist_onset_z", "raw_dist_all_z"]] = -100.0
        out, _ = generate_clean_nez_candidates(rows, candidate_rule="raw_top30_only")
        selected = set(out.loc[out["is_candidate"].astype(bool), "channel_name"])

        self.assertIn("A9", selected)
        self.assertNotIn("A1", selected)

    def test_unknown_candidate_rule_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown candidate_rule"):
            generate_clean_nez_candidates(_toy_rows(), candidate_rule="bogus")


if __name__ == "__main__":
    unittest.main()
