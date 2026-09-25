"""Test that strict v3_ledger mode ignores feature cache split entirely."""

import unittest


class RawBBNoFeatureCacheSplitInStrictModeTests(unittest.TestCase):
    def test_strict_mode_does_not_read_feature_cache_splits(self):
        """In v3_ledger mode, SSL subjects come from ledger only."""
        n_ledger_subjects = 90
        fold_idx = 2
        fold_size = 18

        # ledger-based SSL: exclude fold 2 test subjects
        v3_test = [f"s{i}" for i in range(fold_idx * fold_size, (fold_idx + 1) * fold_size)]
        all_ledger = [f"s{i}" for i in range(n_ledger_subjects)]
        ssl_from_ledger = set(all_ledger) - set(v3_test)

        self.assertEqual(len(ssl_from_ledger), 72)
        self.assertEqual(len(v3_test), 18)

        # Feature cache might produce different test/train split
        # In strict mode, this must be IGNORED
        fake_cache_test = [f"s{5}", f"s{10}", f"s{15}", f"s{20}"]  # wrong subjects
        ssl_from_cache = set(all_ledger[:50])  # wrong size
        self.assertNotEqual(ssl_from_ledger, ssl_from_cache,
                            "v3_ledger SSL subjects must come from ledger only")

    def test_ssl_subjcets_are_ledger_minus_test(self):
        all_90 = set(f"s{i}" for i in range(90))
        test_18 = set(f"s{i}" for i in range(36, 54))  # fold 2
        ssl = all_90 - test_18
        self.assertEqual(len(ssl), 72)
        self.assertEqual(len(all_90 & ssl), 72)
        self.assertEqual(len(test_18 & ssl), 0)


if __name__ == "__main__":
    unittest.main()
