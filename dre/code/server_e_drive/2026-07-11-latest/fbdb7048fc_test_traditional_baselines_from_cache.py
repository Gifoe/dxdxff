from __future__ import annotations

import pickle
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from data_factory import build_outer_splits
from scripts.traditional_baselines.core import (
    CHANNEL_PREDICTION_COLUMNS,
    MethodResult,
    PATIENT_ROW_COLUMNS,
    build_patient_channel_rows,
    classical_specs,
    compute_completed_and_incomplete_methods,
    feature_columns,
    load_window_cache,
    patient_topk_metrics,
    parse_args,
    resolve_requested_methods,
    run_classical_fold,
    run_single_marker_fold,
    validate_traditional_rows_against_patient_index,
    validate_traditional_output_against_patient_index,
    write_outputs,
)


def _synthetic_cache() -> dict:
    run_records = []
    patient_index = {}
    for center, subject, offset in [
        ("hup", "hup:p1", 0.0),
        ("lzu", "lzu:p2", 1.0),
        ("multicenter", "multicenter:p3", 2.0),
        ("pediatric", "pediatric:p4", 3.0),
    ]:
        patient_index[subject] = {"center": center, "canonical_channels": ["a", "b", "c"], "labels": np.asarray([1, 0, 0], dtype=np.float32)}
        features = np.asarray(
            [
                [[1.0 + offset, 0.5], [0.1 + offset, 0.2], [0.3 + offset, 0.4]],
                [[1.2 + offset, 0.6], [0.2 + offset, 0.1], [0.4 + offset, 0.3]],
            ],
            dtype=np.float32,
        )
        run_records.append(
            {
                "subject_id": subject,
                "run_id": f"{subject}:r1",
                "center": center,
                "labels": np.asarray([1, 0, 0], dtype=np.float32),
                "channel_names_norm": ["a", "b", "c"],
                "sfreq": 256.0,
                "sample": {
                    "window_features": features,
                    "window_feature_names": ["high_gamma_power", "line_length"],
                    "window_relative_centers_sec": [-5.0, 0.0],
                },
            }
        )
    return {"run_records": run_records, "patient_index": patient_index}


class TraditionalBaselineTests(unittest.TestCase):
    def test_core_cli_help_imports_from_repo_root(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "traditional_baselines" / "core.py"
        result = subprocess.run([sys.executable, str(script), "--help"], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_core_cli_executes_main_for_missing_cache(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "traditional_baselines" / "core.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--window_cache_path",
                    str(Path(tmpdir) / "missing.pkl"),
                    "--output_dir",
                    str(Path(tmpdir) / "out"),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing.pkl", result.stderr)

    def test_synthetic_cache_loads_and_builds_patient_channel_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.pkl"
            with path.open("wb") as handle:
                pickle.dump(_synthetic_cache(), handle)

            rows = build_patient_channel_rows(load_window_cache(path))

            self.assertEqual(len(rows), 12)
            self.assertEqual(set(rows["center"]), {"hup", "lzu", "multicenter", "pediatric"})
            self.assertIn("record_mean__top20pct_mean__high_gamma_power", rows.columns)

    def test_traditional_uses_patient_index_labels_over_run_record_labels(self):
        cache = {
            "patient_index": {"p1": {"center": "hup", "canonical_channels": ["a", "b"], "labels": np.asarray([1, 0], dtype=np.float32)}},
            "run_records": [
                {
                    "subject_id": "p1",
                    "center": "hup",
                    "labels": np.asarray([0, 1], dtype=np.float32),
                    "channel_names_norm": ["a", "b"],
                    "sample": {
                        "window_features": np.asarray([[[3.0], [1.0]], [[2.0], [1.5]]], dtype=np.float32),
                        "window_feature_names": ["high_gamma_power"],
                    },
                }
            ],
        }

        rows = build_patient_channel_rows(cache).set_index("channel_name")

        self.assertEqual(int(rows.loc["a", "label_ez"]), 1)
        self.assertEqual(int(rows.loc["b", "label_ez"]), 0)
        self.assertEqual(int(rows.attrs["n_label_conflict_with_run_record_labels"]), 2)

    def test_traditional_rejects_noncanonical_output(self):
        cache = {
            "patient_index": {"p1": {"center": "hup", "canonical_channels": ["a", "b"], "labels": np.asarray([1, 0], dtype=np.float32)}},
            "run_records": [
                {
                    "subject_id": "p1",
                    "center": "hup",
                    "labels": np.asarray([1, 0, 1], dtype=np.float32),
                    "channel_names_norm": ["a", "b", "x"],
                    "sample": {
                        "window_features": np.asarray([[[3.0], [1.0], [4.0]], [[2.0], [1.5], [4.5]]], dtype=np.float32),
                        "window_feature_names": ["high_gamma_power"],
                    },
                }
            ],
        }

        rows = build_patient_channel_rows(cache)

        self.assertNotIn("x", set(rows["channel_name"]))
        self.assertEqual(int(rows.attrs["n_channels_skipped_not_in_canonical"]), 1)

    def test_traditional_label_validation_against_patient_index(self):
        rows = pd.DataFrame([{"subject_id": "p1", "center": "hup", "channel_name": "a", "label_ez": 0}])
        patient_index = {"p1": {"canonical_channels": ["a"], "labels": np.asarray([1], dtype=np.float32)}}
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                validate_traditional_rows_against_patient_index(rows, patient_index, Path(tmpdir))
            self.assertTrue((Path(tmpdir) / "traditional_label_conflict_rows.csv").exists())

    def test_traditional_final_output_no_duplicate_per_method(self):
        channel_rows = pd.DataFrame(
            [
                {"method": "m", "fold_idx": 1, "subject_id": "p1", "channel_name": "a", "label_ez": 1},
                {"method": "m", "fold_idx": 1, "subject_id": "p1", "channel_name": "a", "label_ez": 1},
            ]
        )
        patient_index = {"p1": {"canonical_channels": ["a"], "labels": np.asarray([1], dtype=np.float32)}}
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                validate_traditional_output_against_patient_index(channel_rows, patient_index, Path(tmpdir), expected_n_splits=1)
            self.assertTrue((Path(tmpdir) / "traditional_output_duplicate_method_channel_rows.csv").exists())

    def test_traditional_final_output_label_validation(self):
        channel_rows = pd.DataFrame([{"method": "m", "fold_idx": 1, "subject_id": "p1", "channel_name": "a", "label_ez": 0}])
        patient_index = {"p1": {"canonical_channels": ["a"], "labels": np.asarray([1], dtype=np.float32)}}
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                validate_traditional_output_against_patient_index(channel_rows, patient_index, Path(tmpdir), expected_n_splits=1)
            self.assertTrue((Path(tmpdir) / "traditional_output_label_conflict_rows.csv").exists())

    def test_patient_split_is_subject_wise(self):
        patient_index = _synthetic_cache()["patient_index"]

        for split in build_outer_splits(patient_index, split_strategy="5fold", n_splits=2, random_seed=42):
            self.assertFalse(set(split["train_subjects"]) & set(split["test_subjects"]))

    def test_topk_evaluator_predicts_true_ez_count_per_patient(self):
        rows = pd.DataFrame(
            [
                {"subject_id": "p1", "center": "hup", "channel_name": "a", "label_ez": 1, "score_ez": 0.9},
                {"subject_id": "p1", "center": "hup", "channel_name": "b", "label_ez": 0, "score_ez": 0.8},
                {"subject_id": "p1", "center": "hup", "channel_name": "c", "label_ez": 1, "score_ez": 0.7},
                {"subject_id": "p2", "center": "hup", "channel_name": "a", "label_ez": 0, "score_ez": 0.9},
                {"subject_id": "p2", "center": "hup", "channel_name": "b", "label_ez": 0, "score_ez": 0.8},
            ]
        )

        channel_rows, patient_rows = patient_topk_metrics(rows, method="m", fold_idx=1, selected_params={})

        predicted_by_subject = pd.DataFrame(channel_rows).groupby("subject_id")["pred_ez_topk"].sum().to_dict()
        ez_count_by_subject = {row["subject_id"]: row["ez_channel_count"] for row in patient_rows}
        self.assertEqual(predicted_by_subject["p1"], 2)
        self.assertEqual(predicted_by_subject["p2"], 0)
        self.assertEqual(ez_count_by_subject["p1"], 2)
        self.assertEqual(ez_count_by_subject["p2"], 0)

    def test_single_marker_orientation_uses_validation_only(self):
        rows = build_patient_channel_rows(_synthetic_cache())
        marker_col = "record_mean__top20pct_mean__high_gamma_power"
        rows.loc[rows["subject_id"].eq("hup:p1"), marker_col] = [0.1, 0.9, 0.8]
        rows.loc[rows["subject_id"].eq("lzu:p2"), marker_col] = [0.1, 0.9, 0.8]
        rows.loc[rows["subject_id"].eq("multicenter:p3"), marker_col] = [0.9, 0.1, 0.2]
        rows.loc[rows["subject_id"].eq("pediatric:p4"), marker_col] = [0.9, 0.1, 0.2]

        result = run_single_marker_fold(
            rows,
            "single_high_gamma",
            1,
            fit_subjects=["hup:p1"],
            val_subjects=["lzu:p2"],
            test_subjects=["multicenter:p3", "pediatric:p4"],
            candidate_cols=feature_columns(rows),
        )

        self.assertFalse(isinstance(result, str))
        self.assertEqual(result.selected_params["orientation"], -1)
        self.assertEqual(result.selected_params["marker_column"], marker_col)

    def test_single_marker_prefers_record_mean_top20pct_over_record_max(self):
        rows = build_patient_channel_rows(_synthetic_cache())
        result = run_single_marker_fold(
            rows,
            "single_high_gamma",
            1,
            fit_subjects=["hup:p1"],
            val_subjects=["lzu:p2"],
            test_subjects=["multicenter:p3"],
            candidate_cols=feature_columns(rows),
        )

        self.assertFalse(isinstance(result, str))
        self.assertEqual(result.selected_params["marker_column"], "record_mean__top20pct_mean__high_gamma_power")

    def test_output_files_contain_required_columns(self):
        rows = build_patient_channel_rows(_synthetic_cache())
        scored = rows[rows["subject_id"].eq("hup:p1")].copy()
        scored["score_ez"] = [0.9, 0.1, 0.2]
        channel_rows, patient_rows = patient_topk_metrics(scored, method="single_high_gamma", fold_idx=1, selected_params={})
        result = Namespace(
            method="single_high_gamma",
            method_group="single_marker",
            fold_idx=1,
            selected_params={},
            channel_predictions=channel_rows,
            patient_rows=patient_rows,
        )
        args = Namespace(
            window_cache_path="cache.pkl",
            split_strategy="5fold",
            n_splits=5,
            random_seed=42,
            val_ratio=0.2,
            allow_incomplete_methods=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            write_outputs(tmpdir, rows, [result], [], _synthetic_cache(), args, ["single_high_gamma"])
            patient_out = pd.read_csv(Path(tmpdir) / "traditional_baseline_patient_rows.csv")
            channel_out = pd.read_csv(Path(tmpdir) / "traditional_baseline_channel_predictions.csv")
            audit = pd.read_json(Path(tmpdir) / "traditional_baseline_audit.json", typ="series")

        self.assertTrue(set(PATIENT_ROW_COLUMNS).issubset(patient_out.columns))
        self.assertTrue(set(CHANNEL_PREDICTION_COLUMNS).issubset(channel_out.columns))
        self.assertEqual(audit["label_source"], "patient_index")

    def test_missing_marker_skips_without_crashing(self):
        rows = build_patient_channel_rows(_synthetic_cache())
        result = run_single_marker_fold(
            rows,
            "single_hfo_event_rate",
            1,
            fit_subjects=["hup:p1"],
            val_subjects=["lzu:p2"],
            test_subjects=["multicenter:p3"],
            candidate_cols=feature_columns(rows),
        )

        self.assertIsInstance(result, str)
        self.assertIn("missing", result)

    def test_center_id_and_label_derived_counts_are_not_input_features(self):
        rows = build_patient_channel_rows(_synthetic_cache())
        rows["center_id"] = 1

        cols = feature_columns(rows)

        self.assertNotIn("center_id", cols)
        self.assertNotIn("ez_channel_count", cols)
        self.assertNotIn("valid_channel_count", cols)
        self.assertNotIn("record_count_feature", cols)

    def test_methods_cli_accepts_subset_for_smoke_runs(self):
        args = parse_args(
            [
                "--window_cache_path",
                "cache.pkl",
                "--output_dir",
                "out",
                "--methods",
                "single_high_gamma,logistic_l2",
            ]
        )

        marker_methods, ml_specs = resolve_requested_methods(args.methods, random_seed=42, include_mlp=False)

        self.assertEqual(marker_methods, ["single_high_gamma"])
        self.assertEqual(list(ml_specs.keys()), ["logistic_l2"])

    def test_logistic_l2_classical_fold_runs_on_synthetic_cache(self):
        rows = build_patient_channel_rows(_synthetic_cache())
        method_group, grid, factory = classical_specs(42, include_mlp=False)["logistic_l2"]

        result = run_classical_fold(
            rows,
            "logistic_l2",
            method_group,
            grid[:1],
            factory,
            fold_idx=1,
            fit_subjects=["hup:p1", "lzu:p2"],
            val_subjects=["multicenter:p3"],
            train_subjects=["hup:p1", "lzu:p2", "multicenter:p3"],
            test_subjects=["pediatric:p4"],
            cols=feature_columns(rows),
        )

        self.assertFalse(isinstance(result, str))
        self.assertEqual(result.method, "logistic_l2")
        self.assertTrue(result.patient_rows)
        self.assertTrue(result.channel_predictions)

    def test_incomplete_methods_are_detected_and_raise(self):
        rows = build_patient_channel_rows(_synthetic_cache())
        result = MethodResult(
            method="logistic_l2",
            method_group="classical_ml",
            fold_idx=1,
            selected_params={},
            channel_predictions=[],
            patient_rows=[],
        )
        args = Namespace(
            window_cache_path="cache.pkl",
            split_strategy="5fold",
            n_splits=2,
            random_seed=42,
            val_ratio=0.2,
            allow_incomplete_methods=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(RuntimeError, "logistic_l2.*missing folds.*2"):
                write_outputs(tmpdir, rows, [result], [], _synthetic_cache(), args, ["logistic_l2"])

    def test_allow_incomplete_methods_writes_audit(self):
        rows = build_patient_channel_rows(_synthetic_cache())
        result = MethodResult(
            method="logistic_l2",
            method_group="classical_ml",
            fold_idx=1,
            selected_params={},
            channel_predictions=[],
            patient_rows=[],
        )
        args = Namespace(
            window_cache_path="cache.pkl",
            split_strategy="5fold",
            n_splits=2,
            random_seed=42,
            val_ratio=0.2,
            allow_incomplete_methods=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            audit = write_outputs(tmpdir, rows, [result], [], _synthetic_cache(), args, ["logistic_l2"])
            incomplete_path = Path(tmpdir) / "traditional_baseline_incomplete_methods.csv"

            self.assertTrue(incomplete_path.exists())
            self.assertEqual(audit["expected_n_folds"], 2)
            self.assertEqual(audit["incomplete_methods"][0]["method"], "logistic_l2")

    def test_xgboost_unavailable_not_required_when_not_requested(self):
        completed_results = [
            MethodResult("single_high_gamma", "single_marker", fold_idx, {}, [], [])
            for fold_idx in range(1, 6)
        ] + [
            MethodResult("logistic_l2", "classical_ml", fold_idx, {}, [], [])
            for fold_idx in range(1, 6)
        ]
        completed, incomplete = compute_completed_and_incomplete_methods(
            completed_results,
            requested_methods=["single_high_gamma", "logistic_l2"],
            expected_n_folds=5,
            skipped=[{"method": "xgboost_optional", "fold_idx": "all", "reason": "xgboost unavailable"}],
        )

        self.assertEqual(completed["single_high_gamma"], [1, 2, 3, 4, 5])
        self.assertEqual(completed["logistic_l2"], [1, 2, 3, 4, 5])
        self.assertEqual(incomplete, [])

    def test_requested_method_zero_completed_is_incomplete(self):
        _, incomplete = compute_completed_and_incomplete_methods(
            [],
            requested_methods=["logistic_l2"],
            expected_n_folds=2,
            skipped=[{"method": "logistic_l2", "fold_idx": 1, "reason": "all candidates failed"}],
        )

        self.assertEqual(incomplete[0]["method"], "logistic_l2")
        self.assertEqual(incomplete[0]["completed_n_folds"], 0)
        self.assertEqual(incomplete[0]["missing_folds"], [1, 2])

    def test_xgboost_unavailable_zero_completed_is_not_incomplete(self):
        _, incomplete = compute_completed_and_incomplete_methods(
            [],
            requested_methods=["xgboost_optional"],
            expected_n_folds=2,
            skipped=[{"method": "xgboost_optional", "fold_idx": "all", "reason": "xgboost unavailable"}],
        )

        self.assertEqual(incomplete, [])

    def test_requested_subset_only_enforced(self):
        result = MethodResult(
            method="single_high_gamma",
            method_group="single_marker",
            fold_idx=1,
            selected_params={},
            channel_predictions=[],
            patient_rows=[],
        )

        _, incomplete = compute_completed_and_incomplete_methods(
            [result],
            requested_methods=["single_high_gamma"],
            expected_n_folds=2,
            skipped=[],
        )

        self.assertEqual(incomplete[0]["method"], "single_high_gamma")
        self.assertNotIn("logistic_l2", {item["method"] for item in incomplete})


if __name__ == "__main__":
    unittest.main()
