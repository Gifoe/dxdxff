import unittest

import pandas as pd

from a12_vcsn.schemas import LedgerSchemaError, build_canonical_ledger


def raw_ledger() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_idx": [1, 1, 1, 1],
            "subject_id": ["p1"] * 4,
            "center": ["hup"] * 4,
            "channel_name": ["A01REF", "A02REF", "B1", "B02"],
            "true_nez": [0, 1, 1, 0],
            "true_ez": [1, 0, 0, 1],
            "score_ez_probability": [0.9, 0.2, 0.1, 0.8],
            "rank_ez_desc": [1, 3, 4, 2],
            "predicted_ez": [1, 0, 0, 1],
        }
    )


class A12SchemaTests(unittest.TestCase):
    def test_maps_real_v3_columns_and_explicit_semantics(self):
        ledger, resolved = build_canonical_ledger(raw_ledger(), strict=True)
        by_channel = ledger.set_index("channel_name_original")
        self.assertEqual(by_channel.loc[["A01REF", "A02REF", "B1", "B02"], "clinical_true_ez"].tolist(), [1, 0, 0, 1])
        self.assertEqual(by_channel.loc[["A01REF", "A02REF", "B1", "B02"], "clinical_true_nez"].tolist(), [0, 1, 1, 0])
        self.assertEqual(by_channel.loc[["A01REF", "A02REF", "B1", "B02"], "old_v3_selected"].tolist(), [1, 0, 0, 1])
        self.assertEqual(ledger["label_semantics"].iloc[0], "clinical_true_nez=1,clinical_true_ez=1")
        self.assertEqual(resolved["mapping"]["old_v3_score_ez"], "score_ez_probability")

    def test_rejects_inconsistent_clinical_labels(self):
        frame = raw_ledger()
        frame.loc[0, "true_nez"] = 1
        with self.assertRaises(LedgerSchemaError):
            build_canonical_ledger(frame, strict=True)

    def test_rejects_duplicate_normalized_channels(self):
        frame = raw_ledger()
        frame.loc[0, "channel_name"] = "A01"
        frame.loc[1, "channel_name"] = "A1"
        with self.assertRaises(LedgerSchemaError):
            build_canonical_ledger(frame, strict=True)

    def test_reuses_repository_normalizer_and_preserves_prime_contacts(self):
        frame = raw_ledger()
        frame.loc[0, "channel_name"] = "POLA3"
        frame.loc[1, "channel_name"] = "POLA'3"
        ledger, _ = build_canonical_ledger(frame, strict=True)
        self.assertNotEqual(ledger.loc[ledger["channel_name_original"] == "POLA3", "channel_name_norm"].iloc[0], ledger.loc[ledger["channel_name_original"] == "POLA'3", "channel_name_norm"].iloc[0])
