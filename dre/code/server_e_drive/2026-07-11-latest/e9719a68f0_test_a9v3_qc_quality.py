import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import exp_ez_hybrid
from exp_ez_hybrid import Exp_EZHybridLocalization, _ez_pairwise_ranking_loss, _patient_balanced_bce_loss, _quality_patient_weights_vector
from neuroez_c.evidence_views import BASE_SPECTRAL_FEATURE_NAMES
from neuroez_c.dataset import build_patient_examples, collate_patient_ez_batch
from run_neuroez_c import build_parser
from scripts.audit_edf_quality_fields import audit_quality_fields
from scripts.run_v3_truebest_qc import build_command, build_effective_args, validate_outputs
from scripts.summarize_v3_truebest_qc_grid import iter_config_dirs, summarize
from seizure_aggregator import CrossSeizureMILAggregator


REPO_ROOT = Path(__file__).resolve().parents[1]


def _patient_index():
    return {
        "p1": {
            "center": "hup",
            "canonical_channels": ["a", "b"],
            "labels": np.asarray([1.0, 0.0], dtype=np.float32),
        }
    }


def _patient_index_many(subject_ids):
    return {
        subject_id: {
            "center": "hup",
            "canonical_channels": ["a", "b"],
            "labels": np.asarray([1.0, 0.0], dtype=np.float32),
            "label_mask": np.asarray([True, True]),
        }
        for subject_id in subject_ids
    }


def _sample(run_id: str, quality: str, value: float = 1.0) -> dict:
    return {
        "subject_id": "p1",
        "run_id": run_id,
        "sample_id": run_id,
        "center": "hup",
        "channel_names_norm": ["a", "b"],
        "labels": np.asarray([1.0, 0.0], dtype=np.float32),
        "window_features": np.full((1, 2, 13), value, dtype=np.float32),
        "window_adjacency": np.zeros((1, 2, 2), dtype=np.float32),
        "window_relative_centers_sec": np.asarray([0.0], dtype=np.float32),
        "window_feature_names": list(BASE_SPECTRAL_FEATURE_NAMES),
        "quality_metadata": {"edf_quality": quality},
    }


class A9v3QualityAwareTests(unittest.TestCase):
    def test_cli_exposes_quality_flags(self):
        args = build_parser().parse_args(
            [
                "--use_edf_quality_weighting",
                "--edf_quality_field",
                "edf_quality",
                "--review_weight",
                "0.7",
                "--poor_weight",
                "0.0",
                "--quality_weight_loss",
                "--quality_weight_record_aggregation",
                "weighted_mean_std",
                "--quality_curriculum_epochs",
                "15",
            ]
        )

        self.assertTrue(args.use_edf_quality_weighting)
        self.assertEqual(args.edf_quality_field, "edf_quality")
        self.assertAlmostEqual(args.review_weight, 0.7)
        self.assertAlmostEqual(args.poor_weight, 0.0)
        self.assertTrue(args.quality_weight_loss)
        self.assertEqual(args.quality_weight_record_aggregation, "weighted_mean_std")
        self.assertEqual(args.quality_curriculum_epochs, 15)

    def test_audit_quality_fields_finds_existing_cache_field(self):
        cache = {
            "patient_index": _patient_index(),
            "run_records": [
                {
                    "subject_id": "p1",
                    "run_id": "r1",
                    "source_center": "hup",
                    "edf_quality": "review",
                    "channel_names_norm": ["a", "b"],
                    "labels": np.asarray([1.0, 0.0], dtype=np.float32),
                    "sample": _sample("r1", "review"),
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.pkl"
            with cache_path.open("wb") as fout:
                pickle.dump(cache, fout)

            result = audit_quality_fields(cache_path, output_dir=Path(tmp))

            self.assertIn("edf_quality", result["candidate_fields"])
            self.assertEqual(result["record_count"], 1)
            self.assertTrue((Path(tmp) / "edf_quality_field_audit.json").exists())

    def test_dataset_and_collate_preserve_record_quality_weights(self):
        args = SimpleNamespace(
            positive_label="ez",
            use_diffusion_residual=False,
            use_edf_quality_weighting=True,
            edf_quality_field="edf_quality",
            review_weight=0.5,
            poor_weight=0.0,
        )

        examples = build_patient_examples(
            [_sample("r1", "good"), _sample("r2", "review")],
            _patient_index(),
            args=args,
        )
        batch = collate_patient_ez_batch(examples)

        self.assertEqual(examples[0]["record_quality_labels"], ["good", "review"])
        np.testing.assert_allclose(examples[0]["record_quality_weights"], [1.0, 0.5])
        self.assertIn("record_quality_weight", batch)
        torch.testing.assert_close(batch["record_quality_weight"][0, :2], torch.tensor([1.0, 0.5]))
        self.assertEqual(batch["record_quality_label"][0][:2], ["good", "review"])

    def test_weighted_record_aggregation_uses_quality_weights_for_mean_and_std(self):
        aggregator = CrossSeizureMILAggregator(model_dim=1, pooling="mean")
        embeddings = torch.tensor([[[[1.0]], [[3.0]]]])
        seizure_mask = torch.tensor([[True, True]])
        seizure_channel_mask = torch.tensor([[[True], [True]]])
        quality_weight = torch.tensor([[1.0, 0.0]])

        pooled, seizure_weights = aggregator(
            embeddings,
            seizure_mask,
            seizure_channel_mask,
            seizure_weights=quality_weight,
        )

        torch.testing.assert_close(pooled[0, 0], torch.tensor([1.0, 0.0001]), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(seizure_weights[0, 0], torch.tensor([1.0, 0.0]))

    def test_patient_loss_quality_weight_can_zero_review_only_patient(self):
        args = SimpleNamespace(patient_loss_weighting="uniform", quality_weight_loss=True)
        logits = torch.zeros((1, 2), dtype=torch.float32)
        batch = {
            "labels": torch.tensor([[1.0, 0.0]]),
            "labels_ez": torch.tensor([[1.0, 0.0]]),
            "channel_mask": torch.tensor([[True, True]]),
            "record_quality_weight_active": torch.tensor([[0.0, 0.0]]),
            "seizure_mask": torch.tensor([[True, True]]),
        }

        loss, parts = _patient_balanced_bce_loss(logits, batch, args)

        torch.testing.assert_close(loss, torch.tensor(0.0))
        self.assertEqual(parts["quality_zero_weight_patient_count"], 1.0)

    def test_ranking_loss_quality_weight_can_zero_review_only_patient(self):
        args = SimpleNamespace(quality_weight_loss=True)
        logits = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        batch = {
            "record_quality_weight_active": torch.tensor([[0.0, 0.0]]),
            "seizure_mask": torch.tensor([[True, True]]),
        }

        patient_weights = _quality_patient_weights_vector(batch, args, logits)
        loss = _ez_pairwise_ranking_loss(
            logits,
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[True, True]]),
            positive_label="ez",
            patient_weights=patient_weights,
        )

        torch.testing.assert_close(loss, torch.tensor(0.0))

    def test_qc_runner_declares_configs_and_disables_a9v14(self):
        script = (REPO_ROOT / "scripts" / "run_v3_truebest_qc_grid.ps1").read_text(encoding="utf-8")

        for name in (
            "V3_TrueBest_Reproduce",
            "V3_QC_review_w070",
            "V3_QC_review_w050",
            "V3_QC_review_w030",
            "V3_QC_curriculum_good15_review_w050",
            "V3_QC_good_only_train_all_eval",
        ):
            self.assertIn(name, script)
        runner = (REPO_ROOT / "scripts" / "run_v3_truebest_qc.py").read_text(encoding="utf-8")
        self.assertIn('"use_patient_context_reranker": False', runner)
        self.assertIn('"use_multi_seizure_consistency": False', runner)
        self.assertIn('"use_shaft_local_residual": False', runner)
        self.assertIn('"use_clinical_mixture_head": False', runner)
        self.assertIn('"freeze_a9v3_backbone": False', runner)
        self.assertIn('"base_aux_loss_weight": 0.0', runner)
        self.assertIn("audit_edf_quality_fields.py", script)
        self.assertIn("Start-Process", script)
        self.assertIn("RedirectStandardOutput", script)
        self.assertIn("RedirectStandardError", script)
        self.assertNotIn("*>&1 | Tee-Object", script)
        self.assertNotIn("run_a9v14_candidate.py", script)

    def test_v3_truebest_qc_runner_strictly_inherits_base_contract(self):
        base = {
            "use_negative_anchor_head": True,
            "negative_anchor_loss_weight": 0.02,
            "use_ez_ranking_loss": True,
            "ez_ranking_loss_weight": 0.05,
            "ez_ranking_margin": 0.05,
            "use_physics_dynamics": True,
            "loss_mode": "patient_balanced_bce",
            "patient_loss_weighting": "uniform",
            "positive_label": "ez",
            "drop_high_ez_fraction_lzu": False,
            "split_strategy": "5fold",
            "n_splits": 5,
            "random_seed": 42,
            "epochs": 40,
            "patience": 8,
            "physics_view_slices": "",
        }
        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp) / "run_args_b0_pruned.json"
            base_path.write_text(json.dumps(base), encoding="utf-8")
            effective, diff = build_effective_args(
                config_name="V3_TrueBest_Reproduce",
                base_run_args_path=base_path,
                output_dir=Path(tmp) / "out",
                cache_path=Path(tmp) / "cache.pkl",
            )

            self.assertEqual(effective["use_negative_anchor_head"], True)
            self.assertEqual(effective["use_ez_ranking_loss"], True)
            self.assertEqual(effective["use_physics_dynamics"], True)
            self.assertEqual(effective["loss_mode"], "patient_balanced_bce")
            self.assertEqual(effective["patient_loss_weighting"], "uniform")
            self.assertFalse(effective["drop_high_ez_fraction_lzu"])
            self.assertFalse(effective["use_patient_context_reranker"])
            self.assertFalse(effective["use_multi_seizure_consistency"])
            self.assertFalse(effective["use_shaft_local_residual"])
            self.assertFalse(effective["use_clinical_mixture_head"])
            command = build_command("python", effective)
            self.assertNotIn("--physics_view_slices", command)
            self.assertIn("truebest_contract_validated_not_overridden", diff)

            drifted = dict(base)
            drifted["use_negative_anchor_head"] = False
            base_path.write_text(json.dumps(drifted), encoding="utf-8")
            with self.assertRaises(ValueError):
                build_effective_args(
                    config_name="V3_TrueBest_Reproduce",
                    base_run_args_path=base_path,
                    output_dir=Path(tmp) / "out",
                    cache_path=Path(tmp) / "cache.pkl",
                )

    def test_v3_qc_summarizer_skips_logs_debug_tmp_and_requires_summary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "V3_TrueBest_Reproduce"
            valid.mkdir()
            pd.DataFrame(
                [
                    {
                        "config_name": "V3_TrueBest_Reproduce",
                        "n_patient_rows": 90,
                        "n_unique_subjects": 90,
                        "positive_label": "ez",
                        "score_semantics": "ez_probability",
                        "drop_high_ez_fraction_lzu": False,
                        "patient_macro_f1": 0.6427135781,
                        "patient_macro_ez_f1": 0.4705912943,
                        "patient_macro_auprc_ez": 0.5184277652,
                        "patient_macro_ez_mrr": 0.7099510182,
                    }
                ]
            ).to_csv(valid / "heldout_summary_neuroez_v3.csv", index=False)
            for name in ("logs", "debug", "tmp"):
                bad = root / name
                bad.mkdir()
                pd.DataFrame([{"config_name": name, "patient_macro_f1": 0.0}]).to_csv(
                    bad / "heldout_summary_neuroez_v3.csv", index=False
                )
            no_summary = root / "V3_QC_review_w050"
            no_summary.mkdir()

            dirs = iter_config_dirs(root)
            self.assertEqual(dirs, [valid])
            df = summarize(root)
            self.assertEqual(df["config_name"].tolist(), ["V3_TrueBest_Reproduce"])
            self.assertTrue(bool(df.iloc[0]["protocol_ok"]))
            self.assertTrue(bool(df.iloc[0]["reproduces_v3_truebest"]))

    def test_v3_qc_validation_checks_protocol_and_quality_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "config_name": "V3_QC_review_w050",
                        "n_patient_rows": 90,
                        "n_unique_subjects": 90,
                        "positive_label": "ez",
                        "score_semantics": "ez_probability",
                        "drop_high_ez_fraction_lzu": False,
                        "patient_macro_f1": 0.64,
                    }
                ]
            ).to_csv(out / "heldout_summary_neuroez_v3.csv", index=False)
            for name in (
                "quality_by_center_summary.csv",
                "quality_by_patient_summary.csv",
                "quality_by_fold_split_summary.csv",
                "quality_audit_summary.json",
            ):
                (out / name).write_text("{}" if name.endswith(".json") else "x\n", encoding="utf-8")

            result = validate_outputs("V3_QC_review_w050", out, metric_tolerance=0.003)

            self.assertTrue(result["passed"])

    def test_qc_synthetic_smoke_writes_quality_outputs(self):
        subjects = [f"p{i}" for i in range(4)]
        patient_index = _patient_index_many(subjects)
        qualities = ["good", "review", "good", "poor"]
        run_records = []
        for idx, subject_id in enumerate(subjects):
            sample = _sample(f"{subject_id}_r1", qualities[idx], value=float(idx + 1))
            sample["subject_id"] = subject_id
            sample["run_id"] = f"{subject_id}_r1"
            sample["sample_id"] = f"{subject_id}_s1"
            run_records.append(
                {
                    "subject_id": subject_id,
                    "run_id": sample["run_id"],
                    "source_center": "hup",
                    "edf_quality": qualities[idx],
                    "channel_names_norm": sample["channel_names_norm"],
                    "labels": sample["labels"],
                    "sample": {
                        "sample_id": sample["sample_id"],
                        "edf_quality": qualities[idx],
                        "window_features": sample["window_features"],
                        "window_adjacency": sample["window_adjacency"],
                        "window_relative_centers_sec": sample["window_relative_centers_sec"],
                        "window_feature_names": sample["window_feature_names"],
                    },
                }
            )
        splits = [
            {"fold_idx": 1, "train_subjects": ["p0", "p1"], "test_subjects": ["p2", "p3"]},
            {"fold_idx": 2, "train_subjects": ["p2", "p3"], "test_subjects": ["p0", "p1"]},
        ]
        original_data_provider = exp_ez_hybrid.data_provider
        try:
            exp_ez_hybrid.data_provider = lambda args: (run_records, patient_index, splits)
            with tempfile.TemporaryDirectory() as tmp:
                args = SimpleNamespace(
                    b0_feature_parts="abs,delta,zdelta,ratio",
                    b0_feature_groups="spectral_classical",
                    self_compare_eps=1e-5,
                    model_dim=8,
                    num_heads=2,
                    dropout=0.0,
                    use_patient_relative_z=True,
                    positive_label="ez",
                    class_weight_mode="none",
                    device="cpu",
                    output_dir=tmp,
                    num_workers=0,
                    epochs=1,
                    patience=0,
                    batch_size=2,
                    patient_batch_size=2,
                    split_strategy="5fold",
                    n_splits=2,
                    random_seed=42,
                    loss_mode="patient_balanced_bce",
                    patient_loss_weighting="uniform",
                    use_edf_quality_weighting=True,
                    edf_quality_field="edf_quality",
                    review_weight=0.5,
                    poor_weight=0.0,
                    quality_weight_loss=True,
                    quality_weight_record_aggregation="weighted_mean_std",
                    quality_curriculum_epochs=0,
                )

                records = Exp_EZHybridLocalization(args).run()
                output = Path(tmp)

                self.assertEqual(len(records), 4)
                self.assertTrue((output / "quality_by_center_summary.csv").exists())
                self.assertTrue((output / "quality_by_patient_summary.csv").exists())
                self.assertTrue((output / "quality_by_fold_split_summary.csv").exists())
                self.assertTrue((output / "quality_ablation_summary.csv").exists())
                patient_rows = pd.read_csv(output / "test_patient_predictions_neuroez_v2_fold_1.csv")
                self.assertIn("quality_good_records", patient_rows.columns)
        finally:
            exp_ez_hybrid.data_provider = original_data_provider


if __name__ == "__main__":
    unittest.main()
