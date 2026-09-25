from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from neuroez_c.raw_brainbert import RawBrainBERTEncoder, masked_patch_loss
from neuroez_c.raw_brainbert_data import (
    RawBrainBERTPatchDataset,
    RawRecordChannelItem,
    build_spectrogram_patches,
    normalize_channel_name,
)
from neuroez_c.v3_hnc_features import run_hnc_pipeline
from scripts.audit_rawbrainbert_hnc_pipeline import audit_pipeline
from scripts.evaluate_rawbrainbert_hnc import evaluate_hnc_outputs
from scripts.export_v3_oof_channel_ledger import export_v3_oof_channel_ledger


def _write_feature_cache(path: Path) -> None:
    subjects = ["s1", "s2", "s3", "s4"]
    patient_index = {}
    run_records = []
    for subject_id in subjects:
        channels = ["A1", "A2", "A3", "B1"]
        labels = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        channel_meta = [
            {"channel_name_norm": "A1", "contact_group": "A", "contact_number": 1},
            {"channel_name_norm": "A2", "contact_group": "A", "contact_number": 2},
            {"channel_name_norm": "A3", "contact_group": "A", "contact_number": 3},
            {"channel_name_norm": "B1", "contact_group": "B", "contact_number": 1},
        ]
        patient_index[subject_id] = {
            "canonical_channels": channels,
            "labels": labels,
            "outcome_group": "success",
            "surgery_success": True,
            "channel_meta": channel_meta,
        }
        run_records.append(
            {
                "subject_id": subject_id,
                "run_id": f"{subject_id}_run",
                "source_center": "hup",
                "channel_names_norm": channels,
                "labels": labels,
                "channel_meta": channel_meta,
                "sample": {
                    "sample_id": f"{subject_id}_sample",
                    "window_features": np.zeros((1, 4, 2), dtype=np.float32),
                    "window_adjacency": np.zeros((1, 4, 4), dtype=np.float32),
                    "window_relative_centers_sec": np.asarray([0.0], dtype=np.float32),
                },
            }
        )
    with path.open("wb") as fout:
        pickle.dump({"run_records": run_records, "patient_index": patient_index}, fout)


def _write_ledger(path: Path) -> pd.DataFrame:
    rows = []
    for fold_idx, subjects in ((1, ["s3", "s4"]), (2, ["s1", "s2"])):
        for subject_id in subjects:
            for idx, channel in enumerate(["A1", "A2", "A3", "B1"]):
                rows.append(
                    {
                        "fold_idx": fold_idx,
                        "subject_id": subject_id,
                        "center": "hup",
                        "channel_id": idx,
                        "channel_name": channel,
                        "true_ez": 1 if channel == "A1" else 0,
                        "score_ez_probability": [0.90, 0.85, 0.20, 0.10][idx],
                        "rank_ez_desc": idx + 1,
                        "predicted_ez": 1 if idx == 0 else 0,
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def _write_embeddings(embedding_dir: Path, fold_idx: int, subjects: list[str]) -> None:
    rows = []
    for subject_id in subjects:
        for idx, channel in enumerate(["A1", "A2", "A3", "B1"]):
            base = float(idx + fold_idx)
            row = {"fold_idx": fold_idx, "subject_id": subject_id, "channel_name": channel, "n_records": 1}
            for prefix in ("rawbb_all", "rawbb_preictal", "rawbb_onset", "rawbb_early", "rawbb_std", "rawbb_max"):
                row[f"{prefix}_0"] = base
                row[f"{prefix}_1"] = base + 0.25
            rows.append(row)
    pd.DataFrame(rows).to_csv(embedding_dir / f"rawbrainbert_patient_channel_embeddings_fold_{fold_idx}.csv", index=False)


class RawBrainBERTHNCPipelineTests(unittest.TestCase):
    def test_export_v3_oof_channel_ledger_maps_aliases_and_writes_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            v3_dir = tmp_path / "v3"
            v3_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "subject_id": "s1",
                        "center": "hup",
                        "channel_id": 0,
                        "channel_names_norm": "A01",
                        "true_ez": 1,
                        "score_ez_probability": 0.9,
                        "rank_ez_desc": 1,
                        "predicted_ez": 1,
                    }
                ]
            ).to_csv(v3_dir / "test_channel_predictions_neuroez_v2_fold_1.csv", index=False)

            output = tmp_path / "out" / "v3_oof_channel_ledger.csv"
            ledger, audit = export_v3_oof_channel_ledger(v3_dir, output, fold_start=1, fold_end=1)

            self.assertEqual(len(ledger), 1)
            self.assertEqual(ledger.loc[0, "fold_idx"], 1)
            self.assertEqual(ledger.loc[0, "channel_name"], "A01")
            self.assertEqual(audit["num_rows"], 1)
            self.assertEqual(audit["missing_required_columns"], [])
            self.assertTrue((output.parent / "v3_oof_ledger_audit.json").exists())

    def test_export_v3_oof_channel_ledger_rejects_duplicate_normalized_channels_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            v3_dir = tmp_path / "v3"
            v3_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "subject_id": "s1",
                        "center": "hup",
                        "channel_id": 0,
                        "channel_name": "A01",
                        "true_ez": 1,
                        "score_ez_probability": 0.9,
                        "rank_ez_desc": 1,
                        "predicted_ez": 1,
                    },
                    {
                        "subject_id": "s1",
                        "center": "hup",
                        "channel_id": 1,
                        "channel_name": "A1",
                        "true_ez": 1,
                        "score_ez_probability": 0.8,
                        "rank_ez_desc": 2,
                        "predicted_ez": 0,
                    },
                ]
            ).to_csv(v3_dir / "test_channel_predictions_neuroez_v2_fold_1.csv", index=False)

            output = tmp_path / "out" / "v3_oof_channel_ledger.csv"
            with self.assertRaisesRegex(ValueError, "Duplicate V3 OOF subject-channel rows"):
                export_v3_oof_channel_ledger(v3_dir, output, fold_start=1, fold_end=1)
            audit = json.loads((output.parent / "v3_oof_ledger_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["duplicated_subject_channel_rows"], 2)
            self.assertEqual(audit["duplicated_subject_channel_preview"][0]["normalized_channel_name"], "A1")

    def test_rawbrainbert_patch_builder_and_encoder_shapes(self):
        raw = np.sin(np.linspace(0, 20, 1000, dtype=np.float32))
        patches = build_spectrogram_patches(
            raw,
            sfreq=500.0,
            resample_sfreq=250.0,
            n_fft=64,
            hop_length=16,
            freq_min=1.0,
            freq_max=120.0,
            patch_time=2,
            patch_freq=4,
            mean=0.0,
            std=1.0,
        )
        self.assertGreater(patches.patches.shape[0], 0)
        self.assertEqual(patches.patches.shape[1], 8)

        model = RawBrainBERTEncoder(
            patch_dim=8,
            d_model=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            max_time_patches=128,
            max_freq_patches=128,
        )
        batch = patches.patches[None, :, :]
        time_ids = patches.time_ids[None, :]
        freq_ids = patches.freq_ids[None, :]
        mask = np.zeros(batch.shape[:2], dtype=bool)
        mask[:, ::2] = True
        out = model.from_numpy(batch, time_ids, freq_ids, mask)
        self.assertEqual(out["hidden"].shape[-1], 16)
        loss = masked_patch_loss(out["reconstruction"], model.tensor_from_numpy(batch), model.tensor_from_numpy(mask))
        self.assertTrue(float(loss.detach().cpu()) >= 0.0)

    def test_rawbrainbert_patch_dataset_clips_large_items_and_reports_counts(self):
        item = RawRecordChannelItem(
            subject_id="s1",
            run_id="r1",
            sample_id="sample1",
            channel_name="A1",
            channel_index=0,
            raw=np.sin(np.linspace(0, 300, 12000, dtype=np.float32)),
            sfreq=250.0,
            duration_sec=48.0,
        )
        preproc = {
            "resample_sfreq": 250.0,
            "n_fft": 64,
            "hop_length": 8,
            "freq_min": 1.0,
            "freq_max": 125.0,
            "patch_time": 1,
            "patch_freq": 1,
            "mean": 0.0,
            "std": 1.0,
        }

        dataset = RawBrainBERTPatchDataset([item], preproc, max_patches_per_item=32, random_seed=7)
        row = dataset[0]

        self.assertLessEqual(row["patches"].patches.shape[0], 32)
        self.assertGreaterEqual(row["patches"].patches.shape[0], 1)
        self.assertGreater(row["patch_count_before_clip"], row["patch_count_after_clip"])
        self.assertEqual(dataset.num_items_clipped, 1)

    def test_channel_name_normalization_collapses_common_variants(self):
        self.assertEqual(normalize_channel_name(" EEG A-01 "), normalize_channel_name("a1"))
        self.assertEqual(normalize_channel_name("A_002"), normalize_channel_name("A2"))

    def test_hnc_pipeline_preserves_rows_and_noncandidate_logits(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            feature_cache = tmp_path / "feature_cache.pkl"
            ledger_path = tmp_path / "ledger.csv"
            embedding_dir = tmp_path / "embeddings"
            output_dir = tmp_path / "hnc"
            embedding_dir.mkdir()
            _write_feature_cache(feature_cache)
            input_ledger = _write_ledger(ledger_path)
            for fold_idx in (1, 2):
                _write_embeddings(embedding_dir, fold_idx, ["s1", "s2", "s3", "s4"])

            audit = run_hnc_pipeline(
                SimpleNamespace(
                    v3_oof_ledger=str(ledger_path),
                    embedding_dir=str(embedding_dir),
                    feature_cache_path=str(feature_cache),
                    output_dir=str(output_dir),
                    candidate_rule="top20pct_plus_neighbors",
                    beta_list="0.10",
                    pca_dim=2,
                    classifier="l2_logreg",
                    split_strategy="5fold",
                    n_splits=2,
                    random_seed=42,
                    logit_eps=1e-5,
                )
            )

            corrected = pd.read_csv(output_dir / "corrected_oof_rawbrainbert_hnc_top20pct_plus_neighbors_beta0.10.csv")
            self.assertEqual(len(corrected), len(input_ledger))
            noncandidate = corrected[~corrected["is_candidate"].astype(bool)]
            self.assertTrue(np.allclose(noncandidate["final_logit"], noncandidate["patient_zscore_logit"]))
            forbidden = {"center_id", "outcome_group", "surgery_success", "true_ez_count", "subject_id", "fold_idx"}
            self.assertTrue(forbidden.isdisjoint(set(audit["classifier_feature_columns"])))
            self.assertTrue(audit["row_count_equal"])
            self.assertGreater(audit["embedding_columns_count_by_fold"]["1"], 0)
            self.assertEqual(audit["row_count_before_embedding_merge_by_fold"]["1"], 16)
            self.assertEqual(audit["row_count_after_embedding_merge_by_fold"]["1"], 16)
            self.assertLess(audit["missing_embedding_fraction_by_fold"]["1"], 0.2)

    def test_hnc_pipeline_rejects_duplicate_embedding_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            feature_cache = tmp_path / "feature_cache.pkl"
            ledger_path = tmp_path / "ledger.csv"
            embedding_dir = tmp_path / "embeddings"
            output_dir = tmp_path / "hnc"
            embedding_dir.mkdir()
            _write_feature_cache(feature_cache)
            _write_ledger(ledger_path)
            for fold_idx in (1, 2):
                _write_embeddings(embedding_dir, fold_idx, ["s1", "s2", "s3", "s4"])
            dup = pd.read_csv(embedding_dir / "rawbrainbert_patient_channel_embeddings_fold_1.csv")
            dup = pd.concat([dup, dup.iloc[[0]]], ignore_index=True)
            dup.to_csv(embedding_dir / "rawbrainbert_patient_channel_embeddings_fold_1.csv", index=False)

            with self.assertRaisesRegex(ValueError, "Duplicate Raw-BrainBERT embedding keys"):
                run_hnc_pipeline(
                    SimpleNamespace(
                        v3_oof_ledger=str(ledger_path),
                        embedding_dir=str(embedding_dir),
                        feature_cache_path=str(feature_cache),
                        output_dir=str(output_dir),
                        candidate_rule="top20pct_plus_neighbors",
                        beta_list="0.10",
                        pca_dim=2,
                        classifier="l2_logreg",
                        split_strategy="5fold",
                        n_splits=2,
                        random_seed=42,
                        logit_eps=1e-5,
                    )
                )

    def test_hnc_pipeline_rejects_embedding_tables_without_rawbb_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            feature_cache = tmp_path / "feature_cache.pkl"
            ledger_path = tmp_path / "ledger.csv"
            embedding_dir = tmp_path / "embeddings"
            output_dir = tmp_path / "hnc"
            embedding_dir.mkdir()
            _write_feature_cache(feature_cache)
            _write_ledger(ledger_path)
            for fold_idx in (1, 2):
                pd.DataFrame(
                    [
                        {"fold_idx": fold_idx, "subject_id": subject, "channel_name": channel, "n_records": 1}
                        for subject in ["s1", "s2", "s3", "s4"]
                        for channel in ["A1", "A2", "A3", "B1"]
                    ]
                ).to_csv(embedding_dir / f"rawbrainbert_patient_channel_embeddings_fold_{fold_idx}.csv", index=False)

            with self.assertRaisesRegex(ValueError, "No Raw-BrainBERT embedding columns"):
                run_hnc_pipeline(
                    SimpleNamespace(
                        v3_oof_ledger=str(ledger_path),
                        embedding_dir=str(embedding_dir),
                        feature_cache_path=str(feature_cache),
                        output_dir=str(output_dir),
                        candidate_rule="top20pct_plus_neighbors",
                        beta_list="0.10",
                        pca_dim=2,
                        classifier="l2_logreg",
                        split_strategy="5fold",
                        n_splits=2,
                        random_seed=42,
                        logit_eps=1e-5,
                    )
                )

    def test_evaluate_hnc_outputs_writes_baseline_and_corrected_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ledger_path = tmp_path / "ledger.csv"
            corrected_dir = tmp_path / "hnc"
            output_dir = tmp_path / "eval"
            corrected_dir.mkdir()
            ledger = _write_ledger(ledger_path)
            corrected = ledger.copy()
            corrected["final_score_ez_probability"] = corrected["score_ez_probability"]
            corrected.to_csv(corrected_dir / "corrected_oof_rawbrainbert_hnc_top20pct_beta0.10.csv", index=False)

            summary = evaluate_hnc_outputs(ledger_path, corrected_dir, output_dir)

            self.assertIn("V3", set(summary["method"]))
            self.assertIn("rawbrainbert_hnc_top20pct_beta0.10", set(summary["method"]))
            self.assertTrue((output_dir / "rawbrainbert_hnc_metrics_summary.csv").exists())
            self.assertTrue((output_dir / "rawbrainbert_hnc_delta_vs_v3.json").exists())

    def test_audit_pipeline_reports_missing_required_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = audit_pipeline(
                SimpleNamespace(
                    base_dir=str(tmp_path / "base"),
                    feature_cache_path=str(tmp_path / "missing_feature.pkl"),
                    raw_cache_path=str(tmp_path / "missing_raw.pkl"),
                    v3_output_dir=str(tmp_path / "v3"),
                )
            )
            self.assertFalse(result["ok"])
            self.assertGreater(len(result["failed_checks"]), 0)


if __name__ == "__main__":
    unittest.main()
