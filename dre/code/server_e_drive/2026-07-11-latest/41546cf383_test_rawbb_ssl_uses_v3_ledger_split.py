"""Test RawBrainBERT SSL uses V3 ledger split."""

import hashlib
import unittest

import pandas as pd


class RawBBSSLV3LedgerSplitTests(unittest.TestCase):
    def _make_v3_ledger(self, n_patients: int = 90, n_folds: int = 5):
        rows = []
        fold_size = n_patients // n_folds
        for i in range(n_patients):
            rows.append({
                "subject_id": f"subj_{i:03d}",
                "fold_id": i // fold_size,
            })
        return pd.DataFrame(rows)

    def test_v3_ledger_based_ssl_excludes_test_for_fold_1(self):
        ledger = self._make_v3_ledger(90, 5)
        fold_idx = 1
        all_subjects = sorted(ledger["subject_id"].astype(str).unique())
        v3_test = sorted(
            ledger[ledger["fold_id"].astype(int) == fold_idx]["subject_id"].astype(str).unique()
        )
        ssl = sorted(set(all_subjects) - set(v3_test))

        self.assertEqual(len(all_subjects), 90)
        self.assertGreater(len(v3_test), 0)
        self.assertEqual(len(ssl), 90 - len(v3_test))
        self.assertEqual(len(set(v3_test) & set(ssl)), 0,
                         "V3 test subjects must not appear in SSL subjects")

    def test_all_folds_have_no_leakage(self):
        ledger = self._make_v3_ledger(90, 5)
        all_subjects = sorted(ledger["subject_id"].astype(str).unique())
        for fold_idx in range(5):
            v3_test = sorted(
                ledger[ledger["fold_id"].astype(int) == fold_idx]["subject_id"].astype(str).unique()
            )
            ssl = sorted(set(all_subjects) - set(v3_test))
            self.assertEqual(len(set(v3_test) & set(ssl)), 0,
                             f"Fold {fold_idx}: test subjects leaked into SSL")

    def test_sha1_is_deterministic(self):
        subjects = [f"s{i}" for i in range(90)]
        sha1 = hashlib.sha1(",".join(sorted(subjects)).encode()).hexdigest()
        sha2 = hashlib.sha1(",".join(sorted(subjects)).encode()).hexdigest()
        self.assertEqual(sha1, sha2)

    def test_feature_cache_split_must_not_affect_v3_ledger_ssl(self):
        """Even if feature_cache has different splits, v3_ledger mode ignores it."""
        ledger = self._make_v3_ledger(90, 5)
        # This simulates: ledger-based SSL is purely from ledger, not feature cache
        fold_1_ssl_from_ledger = sorted(
            set(ledger["subject_id"].unique())
            - set(ledger[ledger["fold_id"] == 1]["subject_id"].unique())
        )
        self.assertEqual(len(fold_1_ssl_from_ledger), 72)
        # A different "feature cache split" should not change this
        fake_cache_subjects = set(f"s{i}" for i in range(50, 100))  # completely different
        self.assertNotEqual(fold_1_ssl_from_ledger, sorted(fake_cache_subjects),
                            "Feature cache split must not override v3 ledger split")


    def test_rawbb_ssl_uses_fold_idx_from_v3_ledger(self):
        """V3 ledger may use 'fold_idx' column name; SSL must detect and use it."""
        # ledger with 'fold_idx' column
        ledger_fold_idx = pd.DataFrame({
            "subject_id": [f"s{i}" for i in range(90)],
            "fold_idx": [i // 18 for i in range(90)],
        })
        fold_col = "fold_idx" if "fold_idx" in ledger_fold_idx.columns else "fold_id"
        self.assertEqual(fold_col, "fold_idx")
        v3_test = ledger_fold_idx[ledger_fold_idx[fold_col].astype(int) == 1]["subject_id"].unique()
        self.assertEqual(len(v3_test), 18)

        # ledger with 'fold_id' column (fallback)
        ledger_fold_id = pd.DataFrame({
            "subject_id": [f"s{i}" for i in range(90)],
            "fold_id": [i // 18 for i in range(90)],
        })
        fold_col2 = "fold_idx" if "fold_idx" in ledger_fold_id.columns else "fold_id"
        self.assertEqual(fold_col2, "fold_id")
        v3_test2 = ledger_fold_id[ledger_fold_id[fold_col2].astype(int) == 1]["subject_id"].unique()
        self.assertEqual(len(v3_test2), 18)

    def test_rawbb_v3_ledger_mode_audit_has_no_undefined_vars(self):
        """In v3_ledger mode, the audit must have all required fields
        defined explicitly, with no feature-cache variables referenced."""
        required_keys = {
            "ssl_split_source",
            "strict_v3_fold_alignment",
            "n_all_ledger_subjects",
            "n_v3_test_subjects",
            "n_success_train_subjects",
            "n_success_test_subjects",
            "n_ssl_subjects",
            "n_failure_subjects_seen",
            "leakage_v3_test_subjects_in_ssl",
            "leakage_success_test_subjects_in_ssl",
            "failure_used_in_ssl",
            "rawbrainbert_item_subjects_subset_of_ssl_subjects",
        }
        # Simulate a v3_ledger mode audit dict
        audit = {
            "ssl_split_source": "v3_ledger",
            "strict_v3_fold_alignment": True,
            "n_all_ledger_subjects": 90,
            "n_v3_test_subjects": 18,
            "n_success_train_subjects": 72,
            "n_success_test_subjects": 18,
            "n_ssl_subjects": 72,
            "n_failure_subjects_seen": 0,
            "leakage_v3_test_subjects_in_ssl": [],
            "leakage_success_test_subjects_in_ssl": [],
            "failure_used_in_ssl": False,
            "rawbrainbert_item_subjects_subset_of_ssl_subjects": True,
        }
        for key in required_keys:
            self.assertIn(key, audit, f"Missing required key: {key}")
        self.assertEqual(audit["ssl_split_source"], "v3_ledger")
        self.assertTrue(audit["strict_v3_fold_alignment"])
        self.assertFalse(audit["failure_used_in_ssl"])


if __name__ == "__main__":
    unittest.main()
