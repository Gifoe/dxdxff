from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_raw_waveform_for_fm import audit_raw_waveform_cache
from scripts.build_fm_raw_window_manifest import _leakage_report, _validate_manifest_integrity, build_raw_window_manifest


def _raw_cache(*, include_missing: bool = False, label_mismatch: bool = False) -> dict:
    run_records = []
    patient_index = {}
    for idx, center in enumerate(["hup", "lzu", "multicenter", "pediatric"], start=1):
        subject_id = f"{center}:p{idx}"
        patient_index[subject_id] = {"center": center}
        raw = np.vstack(
            [
                np.linspace(0.0, 1.0, 1200, dtype=np.float32) + idx,
                np.linspace(1.0, 0.0, 1200, dtype=np.float32) + idx,
            ]
        )
        labels = np.asarray([1, 0], dtype=np.float32)
        if label_mismatch and idx == 1:
            labels = np.asarray([1], dtype=np.float32)
        sample = {
            "raw_waveform": raw,
            "raw_temporal_sfreq": 200.0,
            "window_features": "must_not_be_used_by_fm_stage0",
        }
        if include_missing and idx == 4:
            sample.pop("raw_waveform")
        run_records.append(
            {
                "subject_id": subject_id,
                "run_id": f"{subject_id}:run1",
                "center": center,
                "labels": labels,
                "channel_names_norm": ["a", "b"],
                "sfreq": 200.0,
                "sample": sample,
            }
        )
    return {"run_records": run_records, "patient_index": patient_index}


def _write_cache(root: Path, cache: dict) -> Path:
    path = root / "cache.pkl"
    with path.open("wb") as handle:
        pickle.dump(cache, handle)
    return path


class FMRawWaveformPrepTests(unittest.TestCase):
    def test_synthetic_cache_with_raw_waveform_creates_audit_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_path = _write_cache(root, _raw_cache())

            audit = audit_raw_waveform_cache(cache_path, root / "audit")

            self.assertTrue((root / "audit" / "fm_raw_waveform_records.csv").exists())
            self.assertTrue((root / "audit" / "fm_raw_waveform_audit.json").exists())
            self.assertEqual(audit["n_records_total"], 4)
            self.assertEqual(audit["n_records_usable_for_fm"], 4)

    def test_manifest_has_required_columns_and_no_subject_leakage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_path = _write_cache(root, _raw_cache())

            manifest, audit = build_raw_window_manifest(
                cache_path,
                root / "manifest",
                target_sfreq=200,
                window_sec=2.0,
                stride_sec=1.0,
                max_windows_per_record=3,
                positive_label="ez",
                split_strategy="5fold",
                n_splits=2,
                random_seed=42,
            )

            required = {
                "row_id",
                "physical_window_id",
                "fold_idx",
                "split_role",
                "subject_id",
                "run_id",
                "center",
                "channel_name",
                "channel_idx",
                "label_ez",
                "sfreq",
                "target_sfreq",
                "raw_n_times",
                "window_idx",
                "window_start_sample",
                "window_end_sample",
                "window_center_sample",
                "window_sec",
                "stride_sec",
            }
            self.assertTrue(required.issubset(manifest.columns))
            self.assertFalse(audit["train_test_subject_leakage_check"]["has_leakage"])

    def test_manifest_has_physical_window_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_path = _write_cache(root, _raw_cache())

            manifest, audit = build_raw_window_manifest(
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

            self.assertIn("physical_window_id", manifest.columns)
            self.assertTrue(manifest["physical_window_id"].astype(str).str.contains("|", regex=False).all())
            self.assertLess(manifest["physical_window_id"].nunique(), len(manifest))
            self.assertEqual(audit["n_unique_physical_windows"], int(manifest["physical_window_id"].nunique()))

    def test_manifest_has_all_expected_folds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_path = _write_cache(root, _raw_cache())

            manifest, audit = build_raw_window_manifest(
                cache_path,
                root / "manifest",
                target_sfreq=200,
                window_sec=2.0,
                stride_sec=1.0,
                max_windows_per_record=3,
                positive_label="ez",
                split_strategy="5fold",
                n_splits=2,
                random_seed=42,
            )

            self.assertEqual(sorted(manifest["fold_idx"].unique().tolist()), [1, 2])
            self.assertEqual(audit["missing_expected_folds"], [])

    def test_manifest_leakage_detection_raises(self):
        manifest = pd.DataFrame(
            [
                {"fold_idx": 1, "split_role": "train", "subject_id": "p1"},
                {"fold_idx": 1, "split_role": "test", "subject_id": "p1"},
            ]
        )
        leakage = _leakage_report(manifest)

        with self.assertRaisesRegex(RuntimeError, "train/test subject leakage"):
            _validate_manifest_integrity(manifest, expected_n_splits=1, leakage=leakage)

    def test_window_indices_are_within_raw_waveform_length(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_path = _write_cache(root, _raw_cache())

            manifest, _ = build_raw_window_manifest(
                cache_path,
                root / "manifest",
                target_sfreq=200,
                window_sec=2.0,
                stride_sec=1.0,
                max_windows_per_record=5,
                positive_label="ez",
                split_strategy="5fold",
                n_splits=2,
                random_seed=42,
            )

            self.assertTrue((manifest["window_start_sample"] >= 0).all())
            self.assertTrue((manifest["window_end_sample"] <= manifest["raw_n_times"]).all())
            self.assertTrue((manifest["window_start_sample"] < manifest["window_end_sample"]).all())

    def test_missing_raw_waveform_record_is_skipped_and_logged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_path = _write_cache(root, _raw_cache(include_missing=True))

            _, audit = build_raw_window_manifest(
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
            skipped = pd.read_csv(root / "manifest" / "fm_raw_window_skipped_records.csv")

            self.assertEqual(audit["skipped_records_count"], 1)
            self.assertIn("missing raw_waveform", set(skipped["skip_reason"]))

    def test_label_length_mismatch_is_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_path = _write_cache(root, _raw_cache(label_mismatch=True))

            audit = audit_raw_waveform_cache(cache_path, root / "audit")
            records = pd.read_csv(root / "audit" / "fm_raw_waveform_records.csv")

            self.assertLess(audit["n_records_usable_for_fm"], audit["n_records_total"])
            self.assertIn("labels length mismatch", set(records["skip_reason"].dropna()))

    def test_manifest_builder_does_not_use_engineered_window_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache = _raw_cache()
            for record in cache["run_records"]:
                record["sample"]["window_features"] = object()
            cache_path = _write_cache(root, cache)

            _, audit = build_raw_window_manifest(
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

            audit_json = json.loads((root / "manifest" / "fm_raw_window_manifest_audit.json").read_text())
            self.assertEqual(audit["n_manifest_rows"], audit_json["n_manifest_rows"])
            self.assertIn("no engineered window_features used", audit_json["warning"])


if __name__ == "__main__":
    unittest.main()
