from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import pandas as pd

import torch

from neuroez_c.settopo_reranker import CandidateSetDataset, SetTopoReranker, _settopo_losses, run_settopo_reranker


def _ledger() -> pd.DataFrame:
    rows = []
    for fold_idx, subject_id in [(1, "p1"), (2, "p2"), (3, "p3")]:
        for idx, channel in enumerate(["A1", "A2", "A3", "A4", "A5"]):
            rows.append(
                {
                    "fold_idx": fold_idx,
                    "subject_id": subject_id,
                    "center": "hup",
                    "channel_id": idx,
                    "channel_name": channel,
                    "true_ez": 1 if idx >= 3 else 0,
                    "true_nez": 1 if idx < 3 else 0,
                    "p_clean_nez": [0.95, 0.85, 0.70, 0.20, 0.10][idx],
                    "feature_non_nez_score": [0.05, 0.15, 0.30, 0.80, 0.90][idx],
                    "feature_suspicious_z": [-1.2, -0.7, -0.2, 0.8, 1.1][idx],
                    "raw_dist_all_z": [-1.0, -0.4, -0.1, 0.5, 1.0][idx],
                    "raw_dist_onset_z": [-1.1, -0.5, 0.0, 0.6, 1.2][idx],
                    "raw_dist_preictal_z": [-0.8, -0.3, 0.0, 0.4, 0.7][idx],
                    "is_pseudo_clean_nez_anchor": 1 if idx < 2 else 0,
                    "n_records": 2,
                }
            )
    return pd.DataFrame(rows)


class SetTopoRealModelTests(unittest.TestCase):
    def test_real_settopo_writes_model_audit_loss_curve_and_preserves_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ledger_path = tmp_path / "distance.csv"
            _ledger().to_csv(ledger_path, index=False)

            output, audit = run_settopo_reranker(
                ledger_path,
                tmp_path / "settopo",
                model_type="real_settopo",
                candidate_rule="union_top30_feature_raw_neighbors",
                epochs=1,
                batch_size=2,
                d_model=16,
                num_layers=1,
                num_heads=4,
                dropout=0.0,
                seed=7,
            )

            self.assertIsInstance(audit.get("architecture"), dict)
            self.assertEqual(audit["model_type"], "real_settopo")
            self.assertTrue(audit["real_settopo_used"])
            self.assertEqual(len(output), len(_ledger()))
            self.assertTrue((tmp_path / "settopo" / "settopo_loss_curve.csv").exists())
            self.assertGreater(len(pd.read_csv(tmp_path / "settopo" / "settopo_loss_curve.csv")), 0)
            self.assertTrue((tmp_path / "settopo" / "settopo_model_fold_1.pt").exists())
            self.assertEqual(audit["forbidden_feature_intersection"], [])
            self.assertTrue(audit["no_center_features"])
            self.assertTrue(audit["true_ez_count_not_used_as_feature"])

    def test_model_class_outputs_candidate_delta_shape(self) -> None:
        model = SetTopoReranker(input_dim=5, d_model=8, num_layers=1, num_heads=2, dropout=0.0)

        delta = model(torch.zeros((2, 4, 5)), torch.ones((2, 4), dtype=torch.bool))
        self.assertEqual(tuple(delta.shape), (2, 4))

    def test_candidate_dataset_uses_raw_contact_integer_for_topology_loss(self) -> None:
        rows = _ledger().head(3).copy()
        rows["is_candidate"] = True
        rows["base_suspicious_logit"] = 0.0
        rows["contact_number_norm"] = [0.0, 0.5, 1.0]
        rows["contact_number_raw"] = [1, 3, 10]
        dataset = CandidateSetDataset(rows, ["base_suspicious_logit", "contact_number_norm"])

        self.assertEqual(dataset[0]["contact_number"].tolist(), [1.0, 3.0, 10.0])

    def test_topology_loss_smooths_raw_contacts_within_two_positions(self) -> None:
        batch = {
            "mask": torch.tensor([[True, True, True]]),
            "base_suspicious_logit": torch.zeros((1, 3)),
            "true_nez": torch.ones((1, 3)),
            "true_ez": torch.zeros((1, 3)),
            "is_pseudo_clean_nez_anchor": torch.zeros((1, 3)),
            "raw_dist_onset_z": torch.zeros((1, 3)),
            "shaft_code": torch.tensor([[1.0, 1.0, 1.0]]),
            "contact_number": torch.tensor([[1.0, 3.0, 10.0]]),
        }

        _, parts = _settopo_losses(
            torch.tensor([[0.0, 2.0, 6.0]]),
            batch,
            alpha=1.0,
            lambda_clean_nez=0.0,
            lambda_noisy_pos_rank=0.0,
            lambda_topology=1.0,
            lambda_residual=0.0,
            ranking_margin=0.0,
        )

        self.assertGreater(parts["topology_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
