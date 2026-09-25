from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_meta_ranker_score_bank import build_score_bank, check_duplicate_keys, merge_source_frame, _standardize_source
from scripts.build_persistent_rank_features import build_persistent_rank_features, get_forbidden_meta_feature_columns, validate_persistent_rows_against_patient_index
from scripts.discover_meta_ranker_sources import classify_run_dir, discover_sources
from scripts.meta_ranker_common import (
    A9V3_GATE,
    REQUIRED_SUMMARY_COLUMNS,
    compute_rank_robust_composite,
    passes_a9v3_gate,
    patient_topk_evaluate,
    patientwise_rankpct,
    patientwise_zscore,
)
from scripts.run_meta_ranker_v1 import build_subject_splits_from_fold_idx, parse_args, run_meta_ranker


class MetaRankerV1Tests(unittest.TestCase):
    def _write_synthetic_runner_inputs(self, root: Path) -> tuple[Path, Path]:
        bank = pd.DataFrame(
            [
                {"subject_id": "p1", "channel_name": "a", "center": "hup", "fold_idx": 1, "label_ez": 1, "score_a": 0.9, "z_score_a": 1.0, "rankpct_score_a": 1.0},
                {"subject_id": "p1", "channel_name": "b", "center": "hup", "fold_idx": 1, "label_ez": 0, "score_a": 0.1, "z_score_a": -1.0, "rankpct_score_a": 0.0},
                {"subject_id": "p2", "channel_name": "a", "center": "hup", "fold_idx": 1, "label_ez": 1, "score_a": 0.8, "z_score_a": 1.0, "rankpct_score_a": 1.0},
                {"subject_id": "p2", "channel_name": "b", "center": "hup", "fold_idx": 1, "label_ez": 0, "score_a": 0.2, "z_score_a": -1.0, "rankpct_score_a": 0.0},
                {"subject_id": "p3", "channel_name": "a", "center": "lzu", "fold_idx": 2, "label_ez": 1, "score_a": 0.85, "z_score_a": 1.0, "rankpct_score_a": 1.0},
                {"subject_id": "p3", "channel_name": "b", "center": "lzu", "fold_idx": 2, "label_ez": 0, "score_a": 0.15, "z_score_a": -1.0, "rankpct_score_a": 0.0},
                {"subject_id": "p4", "channel_name": "a", "center": "lzu", "fold_idx": 2, "label_ez": 1, "score_a": 0.75, "z_score_a": 1.0, "rankpct_score_a": 1.0},
                {"subject_id": "p4", "channel_name": "b", "center": "lzu", "fold_idx": 2, "label_ez": 0, "score_a": 0.25, "z_score_a": -1.0, "rankpct_score_a": 0.0},
            ]
        )
        bank_path = root / "bank.csv"
        features_path = root / "features.json"
        bank.to_csv(bank_path, index=False)
        features_path.write_text(json.dumps({"F0_scores_only": ["z_score_a", "rankpct_score_a"]}), encoding="utf-8")
        return bank_path, features_path

    def _run_synthetic_meta_ranker(self, root: Path, output_name: str, *, n_jobs: int = 1, progress: bool = True, progress_every: int = 1) -> Path:
        bank_path, features_path = self._write_synthetic_runner_inputs(root)
        out = root / output_name
        run_meta_ranker(
            Namespace(
                score_bank=str(bank_path),
                feature_columns_json=str(features_path),
                output_dir=str(out),
                split_strategy="5fold",
                n_splits=2,
                random_seed=42,
                positive_label="ez",
                drop_high_ez_fraction_lzu=False,
                allow_small_subject_count=True,
                n_jobs=n_jobs,
                progress=progress,
                progress_every=progress_every,
            )
        )
        return out

    def test_source_discovery_classifies_diagnostic_folders(self):
        for name in ("A9v8_NoPediatric_ReRun", "LCBO_Stage1c", "TeacherAnchor_debug"):
            status, usable, reason = classify_run_dir(Path(name), {"n_folds_detected": 5, "n_unique_subjects": 90})
            self.assertEqual(status, "DIAGNOSTIC_ONLY")
            self.assertFalse(usable)
            self.assertTrue(reason)

    def test_score_bank_duplicate_subject_channel_fails(self):
        rows = pd.DataFrame(
            [
                {"subject_id": "p1", "channel_name": "a"},
                {"subject_id": "p1", "channel_name": "a"},
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                check_duplicate_keys(rows, Path(tmpdir))
            self.assertTrue((Path(tmpdir) / "duplicate_key_rows.csv").exists())

    def test_label_conflict_detection_writes_rows_and_raises(self):
        base = pd.DataFrame([{"subject_id": "p1", "channel_name": "a", "label_ez": 1, "score_a": 0.9}])
        source = pd.DataFrame([{"subject_id": "p1", "channel_name": "a", "label_ez": 0, "score_b": 0.1}])
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                merge_source_frame(base, source, "b", Path(tmpdir))
            conflicts = pd.read_csv(Path(tmpdir) / "label_conflict_rows.csv")
            self.assertIn("source_name", conflicts.columns)
            self.assertIn("conflict_type", conflicts.columns)
            self.assertIn("base_value", conflicts.columns)
            self.assertIn("source_value", conflicts.columns)

    def test_forbidden_columns_are_excluded_from_feature_sets(self):
        forbidden = get_forbidden_meta_feature_columns(
            [
                "subject_id",
                "center",
                "center_id",
                "label_ez",
                "true_ez_count",
                "valid_channel_count",
                "score_a",
            ]
        )
        self.assertIn("center", forbidden)
        self.assertIn("center_id", forbidden)
        self.assertIn("valid_channel_count", forbidden)
        self.assertNotIn("score_a", forbidden)

    def test_center_columns_are_not_model_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            recs = {"cache_path": "", "a9v3_primary": {}, "a10_sources": {}, "traditional_source": {}, "fm_optional_sources": []}
            rec_path = root / "recommended.json"
            rec_path.write_text(json.dumps(recs), encoding="utf-8")
            persistent = pd.DataFrame(
                [
                    {"subject_id": "p1", "channel_name": "a", "center": "hup", "center_id": 0, "fold_idx": 1, "label_ez": 1, "high_gamma_rank_median": 1.0},
                    {"subject_id": "p1", "channel_name": "b", "center": "hup", "center_id": 0, "fold_idx": 1, "label_ez": 0, "high_gamma_rank_median": 2.0},
                ]
            )
            persistent_path = root / "persistent.csv"
            persistent.to_csv(persistent_path, index=False)
            build_score_bank(rec_path, persistent_path, root, allow_small_subject_count=True)
            feature_sets = json.loads((root / "meta_ranker_feature_columns.json").read_text(encoding="utf-8"))
            all_features = set().union(*[set(v) for v in feature_sets.values()])
            self.assertNotIn("center", all_features)
            self.assertNotIn("center_id", all_features)

    def test_within_patient_zscore_uses_only_same_subject(self):
        rows = pd.DataFrame({"subject_id": ["p1", "p1", "p2", "p2"], "score": [1.0, 3.0, 100.0, 102.0]})
        z = patientwise_zscore(rows, "score")
        self.assertAlmostEqual(float(z.iloc[0]), -1.0)
        self.assertAlmostEqual(float(z.iloc[1]), 1.0)
        self.assertAlmostEqual(float(z.iloc[2]), -1.0)
        self.assertAlmostEqual(float(z.iloc[3]), 1.0)

    def test_rankpct_highest_score_is_one(self):
        rows = pd.DataFrame({"subject_id": ["p1", "p1", "p1"], "score": [0.2, 0.9, 0.1]})
        pct = patientwise_rankpct(rows, "score")
        self.assertAlmostEqual(float(pct.iloc[1]), 1.0)
        self.assertAlmostEqual(float(pct.iloc[2]), 0.0)

    def test_topk_evaluator_predicts_true_ez_count_per_patient(self):
        rows = pd.DataFrame(
            [
                {"subject_id": "p1", "center": "hup", "channel_name": "a", "label_ez": 1, "score": 0.8},
                {"subject_id": "p1", "center": "hup", "channel_name": "b", "label_ez": 0, "score": 0.7},
                {"subject_id": "p1", "center": "hup", "channel_name": "c", "label_ez": 1, "score": 0.6},
            ]
        )
        channel_rows, patient_rows, _ = patient_topk_evaluate(rows, score_col="score", method="m", fold_idx=1)
        self.assertEqual(int(pd.DataFrame(channel_rows)["pred_ez_topk"].sum()), 2)
        self.assertEqual(len(patient_rows), 1)

    def test_meta_ranker_uses_existing_fold_idx(self):
        subjects = pd.DataFrame(
            [
                {"subject_id": f"p{i}", "center": "hup", "fold_idx": (i % 5) + 1, "label_ez": 1}
                for i in range(90)
            ]
        )
        splits = build_subject_splits_from_fold_idx(subjects, n_splits=5, random_seed=42)
        for split in splits:
            fold_idx = int(split["fold_idx"])
            self.assertEqual(split["split_source"], "fold_idx")
            self.assertEqual(set(split["test_subjects"]), set(subjects.loc[subjects["fold_idx"].eq(fold_idx), "subject_id"]))

    def test_subject_split_has_no_overlap_between_fit_val_test(self):
        subjects = pd.DataFrame({"subject_id": [f"p{i}" for i in range(10)], "center": ["hup"] * 10, "fold_idx": [1] * 5 + [2] * 5, "label_ez": [1] * 10})
        splits = build_subject_splits_from_fold_idx(subjects, n_splits=2, random_seed=42, allow_small_subject_count=True)
        for split in splits:
            fit = set(split["fit_subjects"])
            val = set(split["val_subjects"])
            test = set(split["test_subjects"])
            self.assertFalse(fit & val)
            self.assertFalse(fit & test)
            self.assertFalse(val & test)

    def test_rejects_missing_fold_idx(self):
        subjects = pd.DataFrame({"subject_id": [f"p{i}" for i in range(90)]})
        with self.assertRaises(ValueError):
            build_subject_splits_from_fold_idx(subjects, n_splits=5, random_seed=42)

    def test_rejects_incomplete_folds(self):
        subjects = pd.DataFrame({"subject_id": [f"p{i}" for i in range(90)], "fold_idx": [(i % 4) + 1 for i in range(90)]})
        with self.assertRaises(ValueError):
            build_subject_splits_from_fold_idx(subjects, n_splits=5, random_seed=42)

    def test_rejects_subject_in_multiple_folds(self):
        subjects = pd.DataFrame(
            [{"subject_id": f"p{i}", "fold_idx": (i % 5) + 1} for i in range(90)]
            + [{"subject_id": "p0", "fold_idx": 2}]
        )
        with self.assertRaises(ValueError):
            build_subject_splits_from_fold_idx(subjects, n_splits=5, random_seed=42)

    def test_selected_config_uses_validation_only_not_test(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bank = pd.DataFrame(
                [
                    {"subject_id": "p1", "channel_name": "a", "center": "hup", "fold_idx": 1, "label_ez": 1, "z_score_a": 1.0, "z_score_b": 0.0},
                    {"subject_id": "p1", "channel_name": "b", "center": "hup", "fold_idx": 1, "label_ez": 0, "z_score_a": 0.0, "z_score_b": 1.0},
                    {"subject_id": "p2", "channel_name": "a", "center": "hup", "fold_idx": 1, "label_ez": 1, "z_score_a": 1.0, "z_score_b": 0.0},
                    {"subject_id": "p2", "channel_name": "b", "center": "hup", "fold_idx": 1, "label_ez": 0, "z_score_a": 0.0, "z_score_b": 1.0},
                    {"subject_id": "p3", "channel_name": "a", "center": "hup", "fold_idx": 2, "label_ez": 1, "z_score_a": 0.0, "z_score_b": 1.0},
                    {"subject_id": "p3", "channel_name": "b", "center": "hup", "fold_idx": 2, "label_ez": 0, "z_score_a": 1.0, "z_score_b": 0.0},
                    {"subject_id": "p4", "channel_name": "a", "center": "hup", "fold_idx": 2, "label_ez": 1, "z_score_a": 0.0, "z_score_b": 1.0},
                    {"subject_id": "p4", "channel_name": "b", "center": "hup", "fold_idx": 2, "label_ez": 0, "z_score_a": 1.0, "z_score_b": 0.0},
                ]
            )
            bank_path = root / "bank.csv"
            features_path = root / "features.json"
            bank.to_csv(bank_path, index=False)
            features_path.write_text(json.dumps({"F0_scores_only": ["z_score_a", "z_score_b"]}), encoding="utf-8")
            run_meta_ranker(
                Namespace(
                    score_bank=str(bank_path),
                    feature_columns_json=str(features_path),
                    output_dir=str(root),
                    split_strategy="5fold",
                    n_splits=2,
                    random_seed=42,
                    positive_label="ez",
                    drop_high_ez_fraction_lzu=False,
                    allow_small_subject_count=True,
                )
            )
            selected = pd.read_csv(root / "meta_ranker_selected_params.csv")
            self.assertIn("selection_split", selected.columns)
            self.assertTrue((selected["selection_split"] == "val").all())

    def test_parse_args_accepts_n_jobs(self):
        args = parse_args(
            [
                "--score_bank",
                "bank.csv",
                "--feature_columns_json",
                "features.json",
                "--output_dir",
                "out",
                "--split_strategy",
                "5fold",
                "--n_splits",
                "5",
                "--random_seed",
                "42",
                "--positive_label",
                "ez",
                "--drop_high_ez_fraction_lzu",
                "false",
                "--n_jobs",
                "5",
            ]
        )
        self.assertEqual(args.n_jobs, 5)

    def test_parse_args_accepts_progress_args(self):
        args = parse_args(
            [
                "--score_bank",
                "bank.csv",
                "--feature_columns_json",
                "features.json",
                "--output_dir",
                "out",
                "--split_strategy",
                "5fold",
                "--n_splits",
                "5",
                "--random_seed",
                "42",
                "--positive_label",
                "ez",
                "--drop_high_ez_fraction_lzu",
                "false",
                "--no-progress",
                "--progress_every",
                "7",
            ]
        )
        self.assertFalse(args.progress)
        self.assertEqual(args.progress_every, 7)

    def test_parallel_meta_ranker_matches_serial_on_synthetic_score_bank(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for dirname, n_jobs in (("serial", 1), ("parallel", 2)):
                self._run_synthetic_meta_ranker(root, dirname, n_jobs=n_jobs)

            serial_summary = pd.read_csv(root / "serial" / "meta_ranker_summary.csv")
            parallel_summary = pd.read_csv(root / "parallel" / "meta_ranker_summary.csv")
            for col in REQUIRED_SUMMARY_COLUMNS:
                self.assertIn(col, serial_summary.columns)
                self.assertIn(col, parallel_summary.columns)
            self.assertEqual(len(pd.read_csv(root / "serial" / "meta_ranker_patient_rows.csv")), len(pd.read_csv(root / "parallel" / "meta_ranker_patient_rows.csv")))
            self.assertEqual(len(pd.read_csv(root / "serial" / "meta_ranker_channel_predictions.csv")), len(pd.read_csv(root / "parallel" / "meta_ranker_channel_predictions.csv")))
            parallel_audit = json.loads((root / "parallel" / "meta_ranker_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(parallel_audit["n_jobs"], 2)

    def test_progress_files_written_serial(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = self._run_synthetic_meta_ranker(Path(tmpdir), "serial", n_jobs=1, progress=True, progress_every=1)
            for fold_idx in (1, 2):
                self.assertTrue((out / f"fold_{fold_idx}" / "progress.csv").exists())
                self.assertTrue((out / f"fold_{fold_idx}" / "progress.jsonl").exists())
                progress_df = pd.read_csv(out / f"fold_{fold_idx}" / "progress.csv")
                for col in ("timestamp", "fold_idx", "stage", "feature_set", "model_type", "candidate_idx", "candidate_total", "elapsed_s", "message"):
                    self.assertIn(col, progress_df.columns)
                self.assertIn("search_start", set(progress_df["stage"]))
                self.assertIn("fold_done", set(progress_df["stage"]))
            self.assertTrue((out / "meta_ranker_progress.jsonl").exists())

    def test_progress_summary_contains_fold_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = self._run_synthetic_meta_ranker(Path(tmpdir), "serial", n_jobs=1, progress=True, progress_every=1)
            summary = pd.read_csv(out / "meta_ranker_progress_summary.csv")
            self.assertEqual(sorted(summary["fold_idx"].astype(int).tolist()), [1, 2])
            for col in (
                "last_stage",
                "candidate_total",
                "candidates_evaluated",
                "selected_model_type",
                "selected_summary_patient_macro_f1",
                "selected_summary_patient_macro_ez_mrr",
            ):
                self.assertIn(col, summary.columns)

    def test_parallel_meta_ranker_writes_progress_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = self._run_synthetic_meta_ranker(Path(tmpdir), "parallel", n_jobs=2, progress=True, progress_every=1)
            self.assertTrue((out / "meta_ranker_progress.jsonl").exists())
            self.assertTrue((out / "meta_ranker_progress_summary.csv").exists())
            for fold_idx in (1, 2):
                self.assertTrue((out / f"fold_{fold_idx}" / "progress.jsonl").exists())
                self.assertTrue((out / f"fold_{fold_idx}" / "progress.csv").exists())

    def test_standardize_true_ez_mapping(self):
        df = pd.DataFrame([{"subject_id": "p1", "channel_name": "a", "true_ez": 1, "true_nez": 0, "score_ez": 0.7}])
        out = _standardize_source(df, "score_a")
        self.assertIn("label_ez", out.columns)
        self.assertIn("label_nez", out.columns)
        self.assertEqual(int(out.loc[0, "label_ez"]), 1)

    def test_standardize_infers_missing_label_complement(self):
        ez_only = pd.DataFrame([{"subject_id": "p1", "channel_name": "a", "label_ez": 1, "score_ez": 0.7}])
        ez_out = _standardize_source(ez_only, "score_a")
        self.assertEqual(int(ez_out.loc[0, "label_nez"]), 0)
        nez_only = pd.DataFrame([{"subject_id": "p1", "channel_name": "a", "label_nez": 1, "score_ez": 0.2}])
        nez_out = _standardize_source(nez_only, "score_a")
        self.assertEqual(int(nez_out.loc[0, "label_ez"]), 0)

    def test_score_bank_missing_labels_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rec_path = root / "recommended.json"
            rec_path.write_text(json.dumps({"a9v3_primary": {}, "a10_sources": {}, "traditional_source": {}, "fm_optional_sources": []}), encoding="utf-8")
            persistent = pd.DataFrame([{"subject_id": "p1", "channel_name": "a", "fold_idx": 1, "score_like_feature": 0.1}])
            persistent_path = root / "persistent.csv"
            persistent.to_csv(persistent_path, index=False)
            with self.assertRaises(ValueError):
                build_score_bank(rec_path, persistent_path, root, allow_small_subject_count=True)
            self.assertTrue((root / "missing_label_rows.csv").exists())

    def test_missing_label_row_not_filled_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rec_path = root / "recommended.json"
            rec_path.write_text(json.dumps({"a9v3_primary": {}, "a10_sources": {}, "traditional_source": {}, "fm_optional_sources": []}), encoding="utf-8")
            persistent = pd.DataFrame(
                [
                    {"subject_id": "p1", "channel_name": "a", "fold_idx": 1, "label_ez": 1, "feature": 0.1},
                    {"subject_id": "p1", "channel_name": "b", "fold_idx": 1, "label_ez": np.nan, "feature": 0.2},
                ]
            )
            persistent_path = root / "persistent.csv"
            persistent.to_csv(persistent_path, index=False)
            with self.assertRaises(ValueError):
                build_score_bank(rec_path, persistent_path, root, allow_small_subject_count=True)
            missing = pd.read_csv(root / "missing_label_rows.csv")
            self.assertEqual(missing.loc[0, "channel_name"], "b")

    def test_numeric_label_conflict(self):
        base = pd.DataFrame([{"subject_id": "p1", "channel_name": "a", "label_ez": "1", "score_a": 0.9}])
        source = pd.DataFrame([{"subject_id": "p1", "channel_name": "a", "label_ez": 0.0, "score_b": 0.1}])
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                merge_source_frame(base, source, "b", Path(tmpdir))
            self.assertTrue((Path(tmpdir) / "label_conflict_rows.csv").exists())
            self.assertTrue((Path(tmpdir) / "label_conflict_summary.json").exists())

    def test_score_bank_missing_fold_idx_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rec_path = root / "recommended.json"
            rec_path.write_text(json.dumps({"a9v3_primary": {}, "a10_sources": {}, "traditional_source": {}, "fm_optional_sources": []}), encoding="utf-8")
            persistent = pd.DataFrame([{"subject_id": "p1", "channel_name": "a", "label_ez": 1, "feature": 0.1}])
            persistent_path = root / "persistent.csv"
            persistent.to_csv(persistent_path, index=False)
            with self.assertRaises(ValueError):
                build_score_bank(rec_path, persistent_path, root, allow_small_subject_count=True)
            self.assertTrue((root / "missing_fold_rows.csv").exists())

    def test_score_bank_subject_multiple_fold_idx_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rec_path = root / "recommended.json"
            rec_path.write_text(json.dumps({"a9v3_primary": {}, "a10_sources": {}, "traditional_source": {}, "fm_optional_sources": []}), encoding="utf-8")
            persistent = pd.DataFrame(
                [
                    {"subject_id": "p1", "channel_name": "a", "fold_idx": 1, "label_ez": 1, "feature": 0.1},
                    {"subject_id": "p1", "channel_name": "b", "fold_idx": 2, "label_ez": 0, "feature": 0.2},
                ]
            )
            persistent_path = root / "persistent.csv"
            persistent.to_csv(persistent_path, index=False)
            with self.assertRaises(ValueError):
                build_score_bank(rec_path, persistent_path, root, allow_small_subject_count=True)
            self.assertTrue((root / "subject_fold_conflict_rows.csv").exists())

    def test_score_bank_detects_five_folds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rec_path = root / "recommended.json"
            rec_path.write_text(json.dumps({"a9v3_primary": {}, "a10_sources": {}, "traditional_source": {}, "fm_optional_sources": []}), encoding="utf-8")
            rows = []
            for i in range(90):
                fold = (i % 5) + 1
                rows.append({"subject_id": f"p{i}", "channel_name": "a", "fold_idx": fold, "label_ez": 1, "score_marker": float(i)})
                rows.append({"subject_id": f"p{i}", "channel_name": "b", "fold_idx": fold, "label_ez": 0, "score_marker": float(i) / 2.0})
            persistent_path = root / "persistent.csv"
            pd.DataFrame(rows).to_csv(persistent_path, index=False)
            build_score_bank(rec_path, persistent_path, root)
            audit = json.loads((root / "meta_ranker_score_bank_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["folds_detected"], [1, 2, 3, 4, 5])

    def test_score_bank_each_subject_has_ez(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rec_path = root / "recommended.json"
            rec_path.write_text(json.dumps({"a9v3_primary": {}, "a10_sources": {}, "traditional_source": {}, "fm_optional_sources": []}), encoding="utf-8")
            persistent = pd.DataFrame(
                [
                    {"subject_id": "p1", "channel_name": "a", "fold_idx": 1, "label_ez": 0, "feature": 0.1},
                    {"subject_id": "p1", "channel_name": "b", "fold_idx": 1, "label_ez": 0, "feature": 0.2},
                ]
            )
            persistent_path = root / "persistent.csv"
            persistent.to_csv(persistent_path, index=False)
            with self.assertRaises(ValueError):
                build_score_bank(rec_path, persistent_path, root, allow_small_subject_count=True)
            self.assertTrue((root / "no_ez_subject_rows.csv").exists())

    def test_feature_sets_no_forbidden_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rec_path = root / "recommended.json"
            rec_path.write_text(json.dumps({"a9v3_primary": {}, "a10_sources": {}, "traditional_source": {}, "fm_optional_sources": []}), encoding="utf-8")
            persistent = pd.DataFrame(
                [
                    {"subject_id": "p1", "channel_name": "a", "center": "hup", "fold_idx": 1, "label_ez": 1, "feature": 0.1},
                    {"subject_id": "p1", "channel_name": "b", "center": "hup", "fold_idx": 1, "label_ez": 0, "feature": 0.2},
                ]
            )
            persistent_path = root / "persistent.csv"
            persistent.to_csv(persistent_path, index=False)
            build_score_bank(rec_path, persistent_path, root, allow_small_subject_count=True)
            feature_sets = json.loads((root / "meta_ranker_feature_columns.json").read_text(encoding="utf-8"))
            forbidden = get_forbidden_meta_feature_columns(list(persistent.columns))
            for features in feature_sets.values():
                self.assertFalse(set(features) & forbidden)

    def test_rank_robust_composite_formula_is_correct(self):
        row = {
            "patient_macro_f1": 0.6,
            "patient_macro_ez_f1": 0.4,
            "patient_macro_auprc_ez": 0.5,
            "patient_macro_ez_mrr": 0.7,
            "center_gap_f1": 0.2,
        }
        self.assertAlmostEqual(compute_rank_robust_composite(row), 0.965)

    def test_a9v3_gate_fails_if_any_threshold_missed(self):
        row = {
            "patient_macro_f1": A9V3_GATE["patient_macro_f1"] + 0.01,
            "patient_macro_ez_f1": A9V3_GATE["patient_macro_ez_f1"] + 0.01,
            "patient_macro_auprc_ez": A9V3_GATE["patient_macro_auprc_ez"] - 0.001,
            "patient_macro_ez_mrr": A9V3_GATE["patient_macro_ez_mrr"] + 0.01,
            "top1_is_ez_rate": A9V3_GATE["top1_is_ez_rate"] + 0.01,
        }
        self.assertFalse(passes_a9v3_gate(row))

    def test_summary_schema_contains_required_columns(self):
        self.assertIn("patient_macro_f1", REQUIRED_SUMMARY_COLUMNS)
        self.assertIn("passes_aaai_target_070", REQUIRED_SUMMARY_COLUMNS)
        self.assertIn("center_pediatric_patient_macro_f1", REQUIRED_SUMMARY_COLUMNS)

    def test_persistent_rank_builder_outputs_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_path = root / "cache.pkl"
            cache = {
                "patient_index": {"p1": {"source_center": "hup"}},
                "run_records": [
                    {
                        "subject_id": "p1",
                        "labels": np.array([1.0, 0.0], dtype=np.float32),
                        "channel_names_norm": ["a", "b"],
                        "sample": {
                            "window_features": np.array([[[3.0, 1.0], [1.0, 2.0]], [[2.0, 1.5], [1.0, 2.5]]], dtype=np.float32),
                            "window_feature_names": ["log_bp_high_gamma", "line_length_per_sec"],
                        },
                    }
                ],
            }
            with cache_path.open("wb") as fout:
                pickle.dump(cache, fout)
            build_persistent_rank_features(cache_path, root)
            out = pd.read_csv(root / "persistent_rank_features.csv")
            self.assertIn("high_gamma_rank_median", out.columns)
            self.assertIn("high_gamma_mean_rank_median", out.columns)
            self.assertIn("high_gamma_max_rank_median", out.columns)
            self.assertIn("high_gamma_top20pct_mean_rank_median", out.columns)
            self.assertEqual(len(out), 2)

    def test_persistent_uses_patient_index_labels_over_run_record_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_path = root / "cache.pkl"
            cache = {
                "patient_index": {"p1": {"source_center": "hup", "canonical_channels": ["a", "b"], "labels": np.array([1.0, 0.0], dtype=np.float32)}},
                "run_records": [
                    {
                        "subject_id": "p1",
                        "labels": np.array([0.0, 1.0], dtype=np.float32),
                        "channel_names_norm": ["a", "b"],
                        "sample": {
                            "window_features": np.array([[[3.0], [1.0]], [[2.0], [1.5]]], dtype=np.float32),
                            "window_feature_names": ["log_bp_high_gamma"],
                        },
                    }
                ],
            }
            with cache_path.open("wb") as fout:
                pickle.dump(cache, fout)
            build_persistent_rank_features(cache_path, root)
            out = pd.read_csv(root / "persistent_rank_features.csv").set_index("channel_name")
            self.assertEqual(int(out.loc["a", "label_ez"]), 1)
            self.assertEqual(int(out.loc["b", "label_ez"]), 0)

    def test_persistent_outputs_all_canonical_channels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_path = root / "cache.pkl"
            cache = {
                "patient_index": {"p1": {"source_center": "hup", "canonical_channels": ["a", "b", "c"], "labels": np.array([1.0, 0.0, 1.0], dtype=np.float32)}},
                "run_records": [
                    {
                        "subject_id": "p1",
                        "labels": np.array([1.0, 0.0], dtype=np.float32),
                        "channel_names_norm": ["a", "b"],
                        "sample": {
                            "window_features": np.array([[[3.0], [1.0]], [[2.0], [1.5]]], dtype=np.float32),
                            "window_feature_names": ["log_bp_high_gamma"],
                        },
                    }
                ],
            }
            with cache_path.open("wb") as fout:
                pickle.dump(cache, fout)
            build_persistent_rank_features(cache_path, root)
            out = pd.read_csv(root / "persistent_rank_features.csv").set_index("channel_name")
            self.assertEqual(list(out.index), ["a", "b", "c"])
            self.assertEqual(int(out.loc["c", "label_ez"]), 1)
            self.assertEqual(float(out.loc["c", "high_gamma_rank_median"]), 0.0)

    def test_persistent_rejects_noncanonical_output(self):
        rows = pd.DataFrame([{"subject_id": "p1", "channel_name": "x", "label_ez": 1}])
        patient_index = {"p1": {"canonical_channels": ["a"], "labels": np.array([1.0], dtype=np.float32)}}
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                validate_persistent_rows_against_patient_index(rows, patient_index, Path(tmpdir))
            self.assertTrue((Path(tmpdir) / "persistent_noncanonical_channel_rows.csv").exists())

    def test_persistent_label_validation_detects_mismatch(self):
        rows = pd.DataFrame([{"subject_id": "p1", "channel_name": "a", "label_ez": 0}])
        patient_index = {"p1": {"canonical_channels": ["a"], "labels": np.array([1.0], dtype=np.float32)}}
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                validate_persistent_rows_against_patient_index(rows, patient_index, Path(tmpdir))
            self.assertTrue((Path(tmpdir) / "persistent_label_conflict_rows.csv").exists())

    def test_source_discovery_writes_inventory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "A10_RankSimple_All90" / "A10_01_rank_w001_m003"
            run_dir.mkdir(parents=True)
            pd.DataFrame([{"patient_macro_f1": 0.64, "n_patient_rows": 90, "positive_label": "ez", "drop_high_ez_fraction_lzu": False}]).to_csv(
                run_dir / "heldout_summary_neuroez_v3.csv", index=False
            )
            pd.DataFrame(
                [
                    {"fold_idx": 1, "subject_id": f"p{i}", "channel_name": "a", "label_ez": 1, "score_ez_probability": 0.9}
                    for i in range(90)
                ]
            ).to_csv(run_dir / "test_channel_predictions_neuroez_v2_fold_1.csv", index=False)
            for fold in range(2, 6):
                pd.DataFrame(
                    [{"fold_idx": fold, "subject_id": f"p{fold}_{i}", "channel_name": "a", "label_ez": 1, "score_ez_probability": 0.9} for i in range(1)]
                ).to_csv(run_dir / f"test_channel_predictions_neuroez_v2_fold_{fold}.csv", index=False)
            (run_dir / "fixed_all90_protocol_audit.json").write_text(
                json.dumps({"positive_label": "ez", "drop_high_ez_fraction_lzu": False, "n_outer_splits": 5, "n_patients": 90}),
                encoding="utf-8",
            )
            discover_sources(root, root / "out")
            self.assertTrue((root / "out" / "source_inventory.csv").exists())

    def _write_traditional_source(self, root: Path, *, folder_name: str = "Traditional_Baselines_All90_Final", n_folds: int = 5) -> Path:
        run_dir = root / folder_name
        run_dir.mkdir(parents=True)
        rows = []
        for i in range(90):
            fold = (i % n_folds) + 1
            rows.append({"fold_idx": fold, "subject_id": f"p{i}", "channel_name": "a", "label_ez": 1, "score_single_high_gamma": 0.9})
            rows.append({"fold_idx": fold, "subject_id": f"p{i}", "channel_name": "b", "label_ez": 0, "score_single_high_gamma": 0.1})
        pd.DataFrame(rows).to_csv(run_dir / "traditional_baseline_channel_predictions.csv", index=False)
        pd.DataFrame([{"n_patient_rows": 90, "positive_label": "ez", "drop_high_ez_fraction_lzu": False, "patient_macro_f1": 0.61}]).to_csv(
            run_dir / "traditional_baseline_summary.csv", index=False
        )
        (run_dir / "traditional_baseline_audit.json").write_text(
            json.dumps({"n_subjects": 90, "folds_detected": list(range(1, n_folds + 1)), "positive_label": "ez", "drop_high_ez_fraction_lzu": False}),
            encoding="utf-8",
        )
        return run_dir

    def test_discovery_finds_traditional_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_traditional_source(root)
            discover_sources(root, root / "out")
            inventory = pd.read_csv(root / "out" / "source_inventory.csv")
            self.assertEqual(inventory.loc[0, "status_class"], "TRADITIONAL_SOURCE")
            rec = json.loads((root / "out" / "recommended_meta_ranker_sources.json").read_text(encoding="utf-8"))
            self.assertTrue(rec["traditional_source"]["channel_predictions"].endswith("traditional_baseline_channel_predictions.csv"))

    def test_discovery_rejects_traditional_with_four_folds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_traditional_source(root, n_folds=4)
            discover_sources(root, root / "out")
            inventory = pd.read_csv(root / "out" / "source_inventory.csv")
            self.assertEqual(inventory.loc[0, "status_class"], "DIAGNOSTIC_ONLY")

    def test_discovery_rejects_traditional_no_lzu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_traditional_source(root, folder_name="Traditional_Baselines_All90_no_lzu")
            discover_sources(root, root / "out")
            inventory = pd.read_csv(root / "out" / "source_inventory.csv")
            self.assertEqual(inventory.loc[0, "status_class"], "DIAGNOSTIC_ONLY")

    def test_runner_rejects_forbidden_feature_in_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bank = pd.DataFrame(
                [
                    {"subject_id": "p1", "channel_name": "a", "center": "hup", "fold_idx": 1, "label_ez": 1, "z_score_a": 1.0},
                    {"subject_id": "p1", "channel_name": "b", "center": "hup", "fold_idx": 1, "label_ez": 0, "z_score_a": 0.0},
                    {"subject_id": "p2", "channel_name": "a", "center": "hup", "fold_idx": 2, "label_ez": 1, "z_score_a": 1.0},
                    {"subject_id": "p2", "channel_name": "b", "center": "hup", "fold_idx": 2, "label_ez": 0, "z_score_a": 0.0},
                ]
            )
            bank_path = root / "bank.csv"
            features_path = root / "features.json"
            bank.to_csv(bank_path, index=False)
            features_path.write_text(json.dumps({"F0_scores_only": ["z_score_a", "center", "label_ez"]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_meta_ranker(
                    Namespace(
                        score_bank=str(bank_path),
                        feature_columns_json=str(features_path),
                        output_dir=str(root),
                        split_strategy="5fold",
                        n_splits=2,
                        random_seed=42,
                        positive_label="ez",
                        drop_high_ez_fraction_lzu=False,
                        allow_small_subject_count=True,
                    )
                )
            self.assertTrue((root / "forbidden_feature_columns_error.json").exists())


if __name__ == "__main__":
    unittest.main()
