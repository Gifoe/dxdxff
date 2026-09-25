import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from a12_vcsn.calibration import ProbabilityCalibrator
from a12_vcsn.candidate_pool import build_candidate_pools
from a12_vcsn.cv import inner_subject_splits
from a12_vcsn.io import CacheContractError, load_hnc_oof_ledger, load_window_feature_store
from a12_vcsn.protocol import ProtocolError, audit_cache_subject_coverage
from a12_vcsn.pair_dataset import build_pair_dataset
from tests.test_a12_schema import raw_ledger


class A12IOAndCVTests(unittest.TestCase):
    def test_cache_adapter_reads_existing_feature_tensor_without_labels(self):
        cache = {"run_records": [{"subject_id": "p1", "canonical_channels": ["A1", "A2"], "sample": {"window_features": np.ones((2, 2, 3))}}], "patient_index": {"p1": {"canonical_channels": ["A1", "A2"]}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.pkl"
            with path.open("wb") as handle:
                pickle.dump(cache, handle)
            static, trajectories, audit = load_window_feature_store(path)
        self.assertEqual(audit["feature_dim"], 3)
        self.assertEqual(static["p1"]["A1"], [1.0, 1.0, 1.0])
        values, mask, seizure_mask = trajectories.get_channel_trajectory("p1", "A2")
        self.assertEqual(values.shape, (1, 2, 3))
        self.assertTrue(mask.all())
        self.assertEqual(seizure_mask.tolist(), [True])

    def test_cache_adapter_fails_closed_without_window_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.pkl"
            with path.open("wb") as handle:
                pickle.dump({"run_records": [], "patient_index": {}}, handle)
            with self.assertRaises(CacheContractError):
                load_window_feature_store(path)

    def test_pair_weights_and_inner_folds_are_subject_safe(self):
        ledger, _ = __import__("a12_vcsn.schemas", fromlist=["build_canonical_ledger"]).build_canonical_ledger(raw_ledger())
        pairs = build_pair_dataset(ledger, build_candidate_pools(ledger), include_labels=True)
        self.assertAlmostEqual(float(pairs.groupby("subject_id")["patient_pair_weight"].sum().iloc[0]), 1.0)
        for train, test in inner_subject_splits(["p1", "p2", "p3", "p4"], 2):
            self.assertFalse(train & test)

    def test_calibrator_and_hnc_contract(self):
        calibrated = ProbabilityCalibrator().fit([0.1, 0.9], [0, 1]).predict([0.5])
        self.assertTrue(0.0 <= calibrated[0] <= 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hnc.csv"
            pd.DataFrame({"subject_id": ["p1"], "fold_idx": [1], "channel_name": ["A1"], "p_hard_negative": [0.2], "hnc_score_semantics": ["p_nez"]}).to_csv(path, index=False)
            frame, audit = load_hnc_oof_ledger(path)
        self.assertEqual(frame["optional_hnc_score"].iloc[0], 0.2)
        self.assertEqual(audit["score_column"], "p_hard_negative")

    def test_cache_requires_every_frozen_v3_subject_but_audits_missing_channels(self):
        raw = pd.concat([raw_ledger(), raw_ledger().assign(subject_id="p2", fold_idx=2)], ignore_index=True)
        ledger, _ = __import__("a12_vcsn.schemas", fromlist=["build_canonical_ledger"]).build_canonical_ledger(raw)
        with self.assertRaises(ProtocolError):
            audit_cache_subject_coverage(ledger, {"p1": {"A01REF": [1.0]}} , strict=True)
        audit = audit_cache_subject_coverage(ledger, {"p1": {"A01REF": [1.0]}, "p2": {"A01REF": [1.0]}}, strict=False)
        self.assertEqual(audit["missing_subjects"], [])
        self.assertGreater(audit["missing_channel_count"], 0)
