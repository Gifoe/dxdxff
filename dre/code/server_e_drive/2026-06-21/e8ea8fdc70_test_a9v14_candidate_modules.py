from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from neuroez_c.a9v14_candidate_modules import (
    A9V14_LEDGER_REQUIRED_COLUMNS,
    A9V14_MODEL_INPUT_FEATURE_NAMES,
    ClinicalPositiveMixtureHead,
    GatedShaftLocalResidualModule,
    PatientContextResidualReranker,
    build_consistency_features,
    build_local_context_features,
    build_positive_core_targets,
    compute_patient_rank_features,
    parse_channel_topology,
    validate_a9v14_ledger_columns,
)
from scripts.a9v14_candidate_configs import get_candidate_config, list_candidate_configs
from scripts.source_propagation_diagnostic import run_source_propagation_diagnostic


class A9v14CandidateModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_deepset_reranker_output_shape_mask_and_bound(self):
        embeddings = torch.randn(2, 4, 8)
        base_logits = torch.tensor([[2.0, 1.0, 0.0, -1.0], [0.5, -0.5, 1.0, 0.0]])
        mask = torch.tensor([[True, True, True, False], [True, False, True, True]])
        reranker = PatientContextResidualReranker(
            input_dim=8,
            reranker_type="deepset",
            hidden_dim=16,
            dropout=0.0,
            residual_scale=0.2,
            use_patient_gate=True,
        )
        out = reranker(embeddings, base_logits, mask)

        self.assertEqual(tuple(out["delta"].shape), (2, 4))
        self.assertEqual(tuple(out["gate"].shape), (2, 4))
        self.assertTrue(torch.all(out["delta"][~mask] == 0.0))
        self.assertTrue(torch.all(out["gate"][~mask] == 0.0))
        residual = out["final_logits"] - base_logits
        self.assertLessEqual(float(residual[mask].abs().max().detach()), 0.200001)

    def test_set_transformer_reranker_output_shape_and_permutation_equivariance(self):
        embeddings = torch.randn(1, 5, 8)
        base_logits = torch.tensor([[1.0, 3.0, 2.0, -1.0, 0.5]])
        mask = torch.tensor([[True, True, True, False, True]])
        rank_features = compute_patient_rank_features(torch.sigmoid(base_logits), mask)
        reranker = PatientContextResidualReranker(
            input_dim=8,
            reranker_type="set_transformer",
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
            residual_scale=0.2,
            use_rank_features=True,
            rank_feature_dim=rank_features.shape[-1],
        )
        reranker.eval()

        perm = torch.tensor([2, 0, 4, 1, 3])
        out = reranker(embeddings, base_logits, mask, rank_features=rank_features)
        out_perm = reranker(
            embeddings[:, perm],
            base_logits[:, perm],
            mask[:, perm],
            rank_features=rank_features[:, perm],
        )

        self.assertEqual(tuple(out["delta"].shape), (1, 5))
        self.assertTrue(torch.allclose(out["delta"][:, perm], out_perm["delta"], atol=1e-5))
        self.assertTrue(torch.allclose(out["gate"][:, perm], out_perm["gate"], atol=1e-5))

    def test_rank_features_no_nan_and_stable_with_tied_scores(self):
        scores = torch.tensor([[0.5, 0.5, 0.2, 0.0]])
        mask = torch.tensor([[True, True, True, False]])
        features = compute_patient_rank_features(scores, mask)

        self.assertEqual(tuple(features.shape), (1, 4, 8))
        self.assertFalse(torch.isnan(features).any())
        self.assertTrue(torch.all(features[0, 3] == 0.0))

    def test_consistency_features_single_and_multi_seizure_no_nan(self):
        one_record_scores = torch.tensor([[[0.9, 0.1, 0.2]]])
        one_record_mask = torch.tensor([[True]])
        one_record_channel_mask = torch.tensor([[[True, True, True]]])
        channel_mask = torch.tensor([[True, True, True]])
        one = build_consistency_features(
            one_record_scores,
            one_record_mask,
            one_record_channel_mask,
            channel_mask,
        )

        multi_scores = torch.tensor([[[0.9, 0.1, 0.2], [0.7, 0.3, 0.4], [0.8, 0.2, 0.1]]])
        multi_mask = torch.tensor([[True, True, True]])
        multi_channel_mask = torch.ones(1, 3, 3, dtype=torch.bool)
        multi = build_consistency_features(multi_scores, multi_mask, multi_channel_mask, channel_mask)

        self.assertFalse(torch.isnan(one.features).any())
        self.assertFalse(torch.isnan(multi.features).any())
        self.assertTrue(torch.all(one.features[..., one.feature_names.index("std_score_across_seizures")] == 0.0))
        self.assertGreater(float(multi.features[0, 0, multi.feature_names.index("top1_frequency")]), 0.0)

    def test_shaft_parser_handles_common_names_and_unknown(self):
        self.assertEqual(parse_channel_topology("A1").shaft_id, "A")
        self.assertEqual(parse_channel_topology("A2").contact_index, 2.0)
        self.assertEqual(parse_channel_topology("LA10").shaft_id, "LA")
        self.assertEqual(parse_channel_topology("L-A3").contact_index, 3.0)
        self.assertEqual(parse_channel_topology("HH1-2").shaft_id, "HH")
        self.assertAlmostEqual(parse_channel_topology("HH1-2").contact_index, 1.5)
        self.assertFalse(parse_channel_topology("not_a_contact").parse_success)

    def test_local_features_zero_without_neighbors_and_module_is_gated(self):
        scores = torch.tensor([[0.8, 0.2, 0.4]])
        mask = torch.tensor([[True, True, True]])
        no_neighbors = build_local_context_features(scores, [["A1", "B1", "C1"]], mask, local_window=1)
        self.assertTrue(torch.all(no_neighbors.features == 0.0))

        with_neighbors = build_local_context_features(scores, [["A1", "A2", "A3"]], mask, local_window=1)
        module = GatedShaftLocalResidualModule(feature_dim=with_neighbors.features.shape[-1], hidden_dim=8, dropout=0.0, residual_scale=0.1)
        out = module(torch.zeros_like(scores), with_neighbors.features, mask)

        self.assertIn("gate", out)
        self.assertIn("delta", out)
        self.assertLessEqual(float((out["final_logits"] - 0.0)[mask].abs().max().detach()), 0.100001)
        self.assertGreater(float(with_neighbors.features[..., with_neighbors.feature_names.index("neighbor_count")].sum()), 0.0)

    def test_mixture_head_score_range_and_positive_core_targets(self):
        embeddings = torch.randn(1, 4, 8)
        labels_ez = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        base_score = torch.tensor([[0.9, 0.4, 0.2, 0.1]])
        mask = torch.tensor([[True, True, True, True]])
        head = ClinicalPositiveMixtureHead(input_dim=8, hidden_dim=12, dropout=0.0)
        out = head(embeddings, mask)
        targets = build_positive_core_targets(labels_ez, base_score, mask)

        self.assertTrue(torch.all(out["score_clinical"][mask] >= 0.0))
        self.assertTrue(torch.all(out["score_clinical"][mask] <= 1.0))
        self.assertTrue(torch.all(targets[(labels_ez > 0.5) & mask] >= 0.5))
        self.assertTrue(torch.all(targets[(labels_ez > 0.5) & mask] > 0.0))

    def test_no_center_feature_in_model_input_names(self):
        forbidden = {"center", "center_id", "ez_fraction", "true_ez_count", "label", "labels_ez"}
        self.assertTrue(forbidden.isdisjoint(set(A9V14_MODEL_INPUT_FEATURE_NAMES)))

    def test_source_diagnostic_missing_time_features_writes_insufficient_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "A9v3_Reproduce"
            run_dir.mkdir(parents=True)
            ledger = pd.DataFrame(
                [
                    {
                        "config_name": "A9v3_Reproduce",
                        "fold": 1,
                        "patient_id": "p1",
                        "record_id": "r1",
                        "channel_id": 0,
                        "channel_name": "A1",
                        "center": "hup",
                        "y_true": 0,
                        "final_score": 0.9,
                    }
                ]
            )
            ledger.to_csv(run_dir / "a9v14_prediction_ledger.csv", index=False)

            summary = run_source_propagation_diagnostic(root=root, config_name="A9v3_Reproduce")

            self.assertEqual(summary["diagnostic_status"], "insufficient_features")
            missing_path = run_dir / "source_propagation_missing_features.json"
            self.assertTrue(missing_path.exists())
            self.assertTrue((run_dir / "insufficient_feature_report.json").exists())
            missing = json.loads(missing_path.read_text(encoding="utf-8"))
            self.assertIn("onset_latency", missing["missing_required_fields"])

    def test_ledger_required_columns_exist(self):
        row = {column: 0 for column in A9V14_LEDGER_REQUIRED_COLUMNS}
        row.update(
            {
                "config_name": "cfg",
                "patient_id": "p1",
                "record_id": "r1",
                "channel_name": "A1",
                "center": "hup",
            }
        )
        frame = pd.DataFrame([row])
        validate_a9v14_ledger_columns(frame)

    def test_config_registry_contains_required_configs(self):
        names = set(list_candidate_configs())
        for name in (
            "A9v3_Reproduce",
            "A9v14_M1_DeepsetReranker",
            "A9v14_M1_SetTransformerRank",
            "A9v14_M2_ConsistencyOnly",
            "A9v14_M3_LocalOnly",
            "A9v14_M4_MixtureOnly",
            "A9v14_M1M2_RankConsistency",
            "A9v14_M1M2M3_RankConsistencyLocal",
            "A9v14_M1M2M4_RankConsistencyMixture",
            "A9v14_AutoTop2_Combo",
            "A9v14_M5_SourcePropagationDiagnostic_A9v3",
            "A9v14_M5_SourcePropagationDiagnostic_BestCandidate",
        ):
            self.assertIn(name, names)
            self.assertEqual(get_candidate_config(name).name, name)


if __name__ == "__main__":
    unittest.main()
