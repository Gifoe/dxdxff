from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.build_fm_raw_window_manifest import build_raw_window_manifest
from scripts.extract_fm_embeddings_from_manifest import extract_fm_embeddings
from scripts.fm_baselines.extractors import RandomProjectionExtractor, build_extractor
from scripts.fm_baselines.evaluation import fm_feature_columns, run_fm_frozen_head, write_fm_outputs
from scripts.run_fm_frozen_head_baseline import _validate_and_align_metadata, run_fm_frozen_head_baseline
from scripts.traditional_baselines.core import MethodResult, patient_topk_metrics
from tests.test_fm_raw_waveform_prep import _raw_cache, _write_cache
from tests.test_fm_true_adapters import _make_fake_biot, _make_fake_cbramod, _make_fake_labram


class FMEmbeddingDebugTests(unittest.TestCase):
    def _build_manifest(self, root: Path) -> tuple[Path, Path]:
        cache_path = _write_cache(root, _raw_cache())
        build_raw_window_manifest(
            cache_path,
            root / "manifest",
            target_sfreq=200,
            window_sec=2.0,
            stride_sec=1.0,
            max_windows_per_record=2,
            positive_label="ez",
            split_strategy="5fold",
            n_splits=2,
            random_seed=42,
        )
        return cache_path, root / "manifest" / "fm_raw_window_manifest.csv"

    def test_random_projection_extractor_encodes_synthetic_windows(self):
        extractor = RandomProjectionExtractor(embedding_dim=32, random_seed=7)
        extractor.load(None, None, "cpu")
        batch = np.ones((3, 400), dtype=np.float32)

        embeddings = extractor.encode_batch(batch)

        self.assertEqual(embeddings.shape, (3, 32))
        self.assertTrue(np.isfinite(embeddings).all())

    def test_debug_embedding_script_creates_npy_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_path, manifest_path = self._build_manifest(root)

            audit = extract_fm_embeddings(
                window_cache_path=cache_path,
                manifest_path=manifest_path,
                output_dir=root / "embeddings",
                fm_model="random_projection",
                target_sfreq=200,
                window_sec=2.0,
                batch_size=8,
                device="cpu",
            )

            self.assertTrue((root / "embeddings" / "fm_window_embeddings.npy").exists())
            self.assertTrue((root / "embeddings" / "fm_window_metadata.csv").exists())
            self.assertGreater(audit["n_embeddings_written"], 0)

    def test_embedding_metadata_row_count_matches_npy_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_path, manifest_path = self._build_manifest(root)

            extract_fm_embeddings(
                window_cache_path=cache_path,
                manifest_path=manifest_path,
                output_dir=root / "embeddings",
                fm_model="random_projection",
                target_sfreq=200,
                window_sec=2.0,
                batch_size=8,
                device="cpu",
            )

            embeddings = np.load(root / "embeddings" / "fm_window_embeddings.npy")
            metadata = pd.read_csv(root / "embeddings" / "fm_window_metadata.csv")
            audit = pd.read_json(root / "embeddings" / "fm_embedding_audit.json", typ="series")

            self.assertEqual(len(metadata), embeddings.shape[0])
            self.assertTrue(audit["metadata_embedding_alignment_check"]["metadata_rows_equals_embeddings_rows"])

    def test_batch_encode_failure_clears_batch_and_logs_all_rows(self):
        class FailingExtractor(RandomProjectionExtractor):
            def encode_batch(self, batch_waveforms):
                raise RuntimeError("forced encoder failure")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_path, manifest_path = self._build_manifest(root)
            with patch("scripts.extract_fm_embeddings_from_manifest.build_extractor", return_value=FailingExtractor()):
                with self.assertRaisesRegex(RuntimeError, "No FM embeddings were written"):
                    extract_fm_embeddings(
                        window_cache_path=cache_path,
                        manifest_path=manifest_path,
                        output_dir=root / "embeddings",
                        fm_model="random_projection",
                        target_sfreq=200,
                        window_sec=2.0,
                        batch_size=3,
                        device="cpu",
                    )

            manifest = pd.read_csv(manifest_path)
            skipped = pd.read_csv(root / "embeddings" / "fm_window_skipped_windows.csv")
            audit = pd.read_json(root / "embeddings" / "fm_embedding_audit.json", typ="series")

            self.assertEqual(audit["skipped_windows_count"], len(manifest))
            self.assertTrue(skipped["skip_reason"].str.contains("batch encode failed").all())
            self.assertFalse(audit["extraction_success"])

    def test_allow_empty_embeddings_for_diagnostic(self):
        class FailingExtractor(RandomProjectionExtractor):
            def encode_batch(self, batch_waveforms):
                raise RuntimeError("forced encoder failure")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_path, manifest_path = self._build_manifest(root)
            with patch("scripts.extract_fm_embeddings_from_manifest.build_extractor", return_value=FailingExtractor()):
                audit = extract_fm_embeddings(
                    window_cache_path=cache_path,
                    manifest_path=manifest_path,
                    output_dir=root / "embeddings",
                    fm_model="random_projection",
                    target_sfreq=200,
                    window_sec=2.0,
                    batch_size=3,
                    device="cpu",
                    allow_empty_embeddings=True,
                    allow_skipped_windows=True,
                )

            self.assertEqual(audit["n_embeddings_written"], 0)
            self.assertTrue(audit["allow_empty_embeddings"])

    def test_skipped_windows_raise_by_default(self):
        class FailsFirstBatch(RandomProjectionExtractor):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def encode_batch(self, batch_waveforms):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("first batch failure")
                return super().encode_batch(batch_waveforms)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_path, manifest_path = self._build_manifest(root)
            with patch("scripts.extract_fm_embeddings_from_manifest.build_extractor", return_value=FailsFirstBatch()):
                with self.assertRaisesRegex(RuntimeError, "skipped windows"):
                    extract_fm_embeddings(
                        window_cache_path=cache_path,
                        manifest_path=manifest_path,
                        output_dir=root / "embeddings",
                        fm_model="random_projection",
                        target_sfreq=200,
                        window_sec=2.0,
                        batch_size=1,
                        device="cpu",
                    )
            self.assertTrue((root / "embeddings" / "fm_window_skipped_windows.csv").exists())
            self.assertTrue((root / "embeddings" / "fm_embedding_audit.json").exists())

    def test_allow_skipped_windows_for_diagnostic(self):
        class FailsFirstBatch(RandomProjectionExtractor):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def encode_batch(self, batch_waveforms):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("first batch failure")
                return super().encode_batch(batch_waveforms)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_path, manifest_path = self._build_manifest(root)
            with patch("scripts.extract_fm_embeddings_from_manifest.build_extractor", return_value=FailsFirstBatch()):
                audit = extract_fm_embeddings(
                    window_cache_path=cache_path,
                    manifest_path=manifest_path,
                    output_dir=root / "embeddings",
                    fm_model="random_projection",
                    target_sfreq=200,
                    window_sec=2.0,
                    batch_size=1,
                    device="cpu",
                    allow_skipped_windows=True,
                )
            self.assertGreater(audit["skipped_windows_count"], 0)
            self.assertTrue(audit["extraction_success"])

    def test_empty_manifest_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_path = _write_cache(root, _raw_cache())
            manifest_path = root / "empty_manifest.csv"
            pd.DataFrame(columns=["row_id"]).to_csv(manifest_path, index=False)

            with self.assertRaisesRegex(ValueError, "manifest is empty"):
                extract_fm_embeddings(
                    window_cache_path=cache_path,
                    manifest_path=manifest_path,
                    output_dir=root / "embeddings",
                    fm_model="random_projection",
                    target_sfreq=200,
                    window_sec=2.0,
                    batch_size=8,
                    device="cpu",
                )

    def test_frozen_head_trains_logistic_l2_on_synthetic_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_path, manifest_path = self._build_manifest(root)
            extract_fm_embeddings(
                window_cache_path=cache_path,
                manifest_path=manifest_path,
                output_dir=root / "embeddings",
                fm_model="random_projection",
                target_sfreq=200,
                window_sec=2.0,
                batch_size=8,
                device="cpu",
            )

            audit = run_fm_frozen_head_baseline(
                Namespace(
                    embedding_path=str(root / "embeddings" / "fm_window_embeddings.npy"),
                    metadata_path=str(root / "embeddings" / "fm_window_metadata.csv"),
                    output_dir=str(root / "head"),
                    split_strategy="5fold",
                    n_splits=2,
                    random_seed=42,
                    positive_label="ez",
                    val_ratio=0.5,
                    head_methods="logistic_l2",
                    allow_incomplete_methods=False,
                )
            )

            self.assertEqual(audit["incomplete_methods"], [])
            summary = pd.read_csv(root / "head" / "fm_frozen_head_summary.csv")
            self.assertEqual(set(summary["head_method"]), {"logistic_l2"})
            self.assertEqual(audit["split_strategy"], "5fold")
            self.assertIn("split roles are read from fm_window_metadata.csv", audit["warning"])

    def test_topk_evaluator_predicts_true_ez_count_per_patient(self):
        rows = pd.DataFrame(
            [
                {"subject_id": "p1", "center": "hup", "channel_name": "a", "label_ez": 1, "score_ez": 0.9},
                {"subject_id": "p1", "center": "hup", "channel_name": "b", "label_ez": 0, "score_ez": 0.8},
                {"subject_id": "p1", "center": "hup", "channel_name": "c", "label_ez": 1, "score_ez": 0.1},
            ]
        )

        channel_rows, _ = patient_topk_metrics(rows, method="fm", fold_idx=1, selected_params={})

        self.assertEqual(sum(row["pred_ez_topk"] for row in channel_rows), 2)

    def test_no_center_id_or_center_string_used_as_model_input(self):
        frame = pd.DataFrame(
            {
                "subject_id": ["p1"],
                "center": ["hup"],
                "center_id": [0],
                "channel_name": ["a"],
                "label_ez": [1],
                "emb_0000": [0.1],
            }
        )

        self.assertEqual(fm_feature_columns(frame), ["emb_0000"])

    def test_incomplete_fold_guard_works_for_fm_outputs(self):
        result = MethodResult(
            method="random_projection__logistic_l2",
            method_group="frozen_fm",
            fold_idx=1,
            selected_params={},
            channel_predictions=[],
            patient_rows=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(RuntimeError, "random_projection__logistic_l2"):
                write_fm_outputs(
                    output_dir=tmpdir,
                    fm_model="random_projection",
                    results=[result],
                    skipped=[],
                    requested_methods=["random_projection__logistic_l2"],
                    expected_n_folds=2,
                    split_strategy="5fold",
                    allow_incomplete_methods=False,
                    embedding_path="emb.npy",
                    metadata_path="meta.csv",
                    head_methods=["logistic_l2"],
                )

    def test_frozen_head_raises_when_metadata_embedding_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            np.save(root / "emb.npy", np.zeros((2, 4), dtype=np.float32))
            pd.DataFrame(
                [
                    {
                        "embedding_row_idx": 0,
                        "fold_idx": 1,
                        "split_role": "train",
                        "subject_id": "p1",
                        "center": "hup",
                        "channel_name": "a",
                        "run_id": "r1",
                        "label_ez": 1,
                        "fm_model": "random_projection",
                    }
                ]
            ).to_csv(root / "meta.csv", index=False)

            with self.assertRaisesRegex(ValueError, "metadata row count"):
                run_fm_frozen_head_baseline(
                    Namespace(
                        embedding_path=str(root / "emb.npy"),
                        metadata_path=str(root / "meta.csv"),
                        output_dir=str(root / "head"),
                        split_strategy="5fold",
                        n_splits=1,
                        random_seed=42,
                        positive_label="ez",
                        val_ratio=0.5,
                        head_methods="logistic_l2",
                        allow_incomplete_methods=False,
                    )
                )

    def test_frozen_head_raises_when_embedding_row_idx_not_contiguous(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            np.save(root / "emb.npy", np.zeros((2, 4), dtype=np.float32))
            meta = pd.DataFrame(
                [
                    {"embedding_row_idx": 0, "fold_idx": 1, "split_role": "train", "subject_id": "p1", "center": "hup", "channel_name": "a", "run_id": "r1", "label_ez": 1, "fm_model": "random_projection"},
                    {"embedding_row_idx": 2, "fold_idx": 1, "split_role": "test", "subject_id": "p2", "center": "hup", "channel_name": "b", "run_id": "r2", "label_ez": 0, "fm_model": "random_projection"},
                ]
            )
            meta.to_csv(root / "meta.csv", index=False)

            with self.assertRaisesRegex(ValueError, "current embedding row order"):
                run_fm_frozen_head_baseline(
                    Namespace(embedding_path=str(root / "emb.npy"), metadata_path=str(root / "meta.csv"), output_dir=str(root / "head"), split_strategy="5fold", n_splits=1, random_seed=42, positive_label="ez", val_ratio=0.5, head_methods="logistic_l2", allow_incomplete_methods=False)
                )

    def test_frozen_head_raises_when_embedding_row_idx_shuffled(self):
        metadata = pd.DataFrame(
            [
                {"embedding_row_idx": 1, "fold_idx": 1, "split_role": "train", "subject_id": "p1", "center": "hup", "channel_name": "a", "run_id": "r1", "label_ez": 1, "fm_model": "random_projection"},
                {"embedding_row_idx": 0, "fold_idx": 1, "split_role": "test", "subject_id": "p2", "center": "hup", "channel_name": "b", "run_id": "r2", "label_ez": 0, "fm_model": "random_projection"},
            ]
        )
        with self.assertRaisesRegex(ValueError, "current embedding row order"):
            _validate_and_align_metadata(np.zeros((2, 4), dtype=np.float32), metadata, n_splits=1)

    def test_frozen_head_accepts_ordered_embedding_row_idx(self):
        metadata = pd.DataFrame(
            [
                {"embedding_row_idx": 0, "fold_idx": 1, "split_role": "train", "subject_id": "p1", "center": "hup", "channel_name": "a", "run_id": "r1", "label_ez": 1, "fm_model": "random_projection"},
                {"embedding_row_idx": 1, "fold_idx": 1, "split_role": "test", "subject_id": "p2", "center": "hup", "channel_name": "b", "run_id": "r2", "label_ez": 0, "fm_model": "random_projection"},
            ]
        )
        aligned = _validate_and_align_metadata(np.zeros((2, 4), dtype=np.float32), metadata, n_splits=1)
        self.assertEqual(aligned["embedding_row_idx"].tolist(), [0, 1])

    def test_deduplication_reduces_encoder_forward_windows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_path, manifest_path = self._build_manifest(root)
            audit = extract_fm_embeddings(
                window_cache_path=cache_path,
                manifest_path=manifest_path,
                output_dir=root / "embeddings",
                fm_model="random_projection",
                target_sfreq=200,
                window_sec=2.0,
                batch_size=8,
                device="cpu",
            )

            self.assertTrue(audit["deduplicate_physical_windows"])
            self.assertLess(audit["encoder_forward_windows"], audit["n_embeddings_written"])
            self.assertGreater(audit["reused_embedding_rows"], 0)

    def test_deduplication_preserves_metadata_embedding_alignment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_path, manifest_path = self._build_manifest(root)
            extract_fm_embeddings(
                window_cache_path=cache_path,
                manifest_path=manifest_path,
                output_dir=root / "embeddings",
                fm_model="random_projection",
                target_sfreq=200,
                window_sec=2.0,
                batch_size=8,
                device="cpu",
            )
            embeddings = np.load(root / "embeddings" / "fm_window_embeddings.npy")
            metadata = pd.read_csv(root / "embeddings" / "fm_window_metadata.csv")
            self.assertEqual(len(metadata), embeddings.shape[0])
            self.assertEqual(metadata["embedding_row_idx"].tolist(), list(range(len(metadata))))

    def test_audit_validator_passes_and_fails(self):
        from scripts.validate_fm_embedding_audit import validate_embedding_audit

        valid = {
            "n_manifest_rows": 2,
            "n_embeddings_written": 2,
            "metadata_embedding_alignment_check": {"pass": True},
            "skipped_windows_count": 0,
            "true_pretrained_fm": True,
            "debug_only": False,
            "paper_baseline": True,
            "comparison_only_baseline": True,
            "frozen_encoder": True,
            "fine_tuned": False,
            "adapter_tuned": False,
            "backbone_trainable_params": 0,
            "adapter_status": "implemented",
            "checkpoint_path": "ckpt.pt",
            "external_repo_path": "repo",
        }
        validate_embedding_audit(valid, model_name="biot", require_true_pretrained_fm=True, require_no_skipped_windows=True, require_full_manifest_coverage=True)
        invalid = dict(valid)
        invalid["skipped_windows_count"] = 1
        with self.assertRaisesRegex(ValueError, "skipped_windows_count"):
            validate_embedding_audit(invalid, model_name="biot", require_true_pretrained_fm=True, require_no_skipped_windows=True, require_full_manifest_coverage=True)

    def test_frozen_head_audit_validator_catches_incomplete_methods(self):
        from scripts.validate_fm_frozen_head_audit import validate_frozen_head_audit

        audit = {
            "incomplete_methods": [],
            "completed_folds_by_method": {"biot__logistic_l2": [1, 2, 3, 4, 5]},
            "comparison_only_baseline": True,
            "frozen_or_finetuned": "frozen",
            "true_pretrained_fm": True,
            "n_folds": 5,
        }
        validate_frozen_head_audit(audit, expected_n_folds=5)
        bad = dict(audit)
        bad["incomplete_methods"] = [{"method": "biot__linear_svm"}]
        with self.assertRaisesRegex(ValueError, "incomplete_methods"):
            validate_frozen_head_audit(bad, expected_n_folds=5)

    def test_preflight_works_on_fake_true_fm_repos(self):
        from scripts.preflight_true_fm_repos import run_preflight

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            biot_repo, biot_ckpt = _make_fake_biot(root)
            cbramod_repo, cbramod_ckpt = _make_fake_cbramod(root)
            labram_repo, labram_ckpt = _make_fake_labram(root)
            report = run_preflight(
                models=["biot", "cbramod", "labram"],
                repo_paths={"biot": str(biot_repo), "cbramod": str(cbramod_repo), "labram": str(labram_repo)},
                checkpoint_paths={"biot": str(biot_ckpt), "cbramod": str(cbramod_ckpt), "labram": str(labram_ckpt)},
                device="cpu",
                output_dir=root / "preflight",
                allow_partial=False,
            )
            self.assertTrue(all(item["passed"] for item in report["models"].values()))
            self.assertTrue((root / "preflight" / "true_fm_preflight_report.json").exists())

    def test_run_script_exists_and_contains_required_model_commands(self):
        script = Path("scripts/run_true_fm_all90.sh")
        self.assertTrue(script.exists())
        text = script.read_text(encoding="utf-8")
        for needle in ["set -euo pipefail", "preflight_true_fm_repos.py", "validate_fm_embedding_audit.py", "validate_fm_frozen_head_audit.py", "biot", "cbramod", "labram"]:
            self.assertIn(needle, text)

    def test_frozen_head_raises_on_train_test_subject_leakage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            np.save(root / "emb.npy", np.zeros((2, 4), dtype=np.float32))
            pd.DataFrame(
                [
                    {"embedding_row_idx": 0, "fold_idx": 1, "split_role": "train", "subject_id": "p1", "center": "hup", "channel_name": "a", "run_id": "r1", "label_ez": 1, "fm_model": "random_projection"},
                    {"embedding_row_idx": 1, "fold_idx": 1, "split_role": "test", "subject_id": "p1", "center": "hup", "channel_name": "a", "run_id": "r2", "label_ez": 1, "fm_model": "random_projection"},
                ]
            ).to_csv(root / "meta.csv", index=False)

            with self.assertRaisesRegex(ValueError, "train/test subject leakage"):
                run_fm_frozen_head_baseline(
                    Namespace(embedding_path=str(root / "emb.npy"), metadata_path=str(root / "meta.csv"), output_dir=str(root / "head"), split_strategy="5fold", n_splits=1, random_seed=42, positive_label="ez", val_ratio=0.5, head_methods="logistic_l2", allow_incomplete_methods=False)
                )

    def test_requested_head_zero_completed_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(RuntimeError, "random_projection__logistic_l2"):
                write_fm_outputs(
                    output_dir=tmpdir,
                    fm_model="random_projection",
                    results=[],
                    skipped=[{"method": "random_projection__logistic_l2", "fold_idx": 1, "reason": "single-class fit data"}],
                    requested_methods=["random_projection__logistic_l2"],
                    expected_n_folds=2,
                    split_strategy="5fold",
                    allow_incomplete_methods=False,
                    embedding_path="emb.npy",
                    metadata_path="meta.csv",
                    head_methods=["logistic_l2"],
                )

    def test_biot_adapter_requires_real_repo_and_checkpoint(self):
        with self.assertRaisesRegex(FileNotFoundError, "external_repo_path"):
            build_extractor("biot", target_sfreq=200, window_sec=2.0).load(None, None, "cpu")

    def test_cbramod_labram_require_real_repo_and_checkpoint(self):
        for model_name in ("cbramod", "labram"):
            with self.subTest(model_name=model_name):
                with self.assertRaisesRegex(FileNotFoundError, "external_repo_path"):
                    build_extractor(model_name, target_sfreq=200, window_sec=2.0).load(None, None, "cpu")


if __name__ == "__main__":
    unittest.main()
