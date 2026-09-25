"""Build a fold-aware raw-window manifest for frozen-FM NeuroEZ-C baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_factory import build_outer_splits
from scripts.audit_raw_waveform_for_fm import _raw_record_row, _sfreq
from scripts.traditional_baselines.core import infer_center, load_window_cache


def _window_starts(n_times: int, window_len: int, stride_len: int, max_windows: int) -> List[int]:
    if n_times < window_len or window_len <= 0:
        return []
    stride_len = max(1, int(stride_len))
    starts = list(range(0, n_times - window_len + 1, stride_len))
    if not starts:
        starts = [0]
    if len(starts) <= max_windows:
        return starts
    chosen_idx = np.linspace(0, len(starts) - 1, int(max_windows)).round().astype(int)
    return [int(starts[idx]) for idx in sorted(set(chosen_idx.tolist()))]


def _split_role(subject_id: str, split: Mapping[str, Sequence[str]]) -> str | None:
    if subject_id in set(split["train_subjects"]):
        return "train"
    if subject_id in set(split["test_subjects"]):
        return "test"
    return None


def _physical_window_id(
    *,
    subject_id: str,
    run_id: str,
    channel_idx: int,
    start: int,
    end: int,
    sfreq: float,
    target_sfreq: int,
    window_sec: float,
) -> str:
    return (
        f"{subject_id}|{run_id}|ch{int(channel_idx)}|{int(start)}|{int(end)}|"
        f"sfreq{float(sfreq):g}|target{int(target_sfreq)}|sec{float(window_sec):g}"
    )


def _leakage_report(manifest: pd.DataFrame) -> Dict[str, Any]:
    leaks = []
    if manifest.empty:
        return {"has_leakage": False, "leaks": leaks}
    for fold_idx, group in manifest.groupby("fold_idx", sort=True):
        train_subjects = set(group.loc[group["split_role"].eq("train"), "subject_id"].astype(str))
        test_subjects = set(group.loc[group["split_role"].eq("test"), "subject_id"].astype(str))
        overlap = sorted(train_subjects & test_subjects)
        if overlap:
            leaks.append({"fold_idx": int(fold_idx), "subjects": overlap})
    return {"has_leakage": bool(leaks), "leaks": leaks}


def _validate_manifest_integrity(
    manifest: pd.DataFrame,
    *,
    expected_n_splits: int,
    leakage: Mapping[str, Any],
) -> None:
    if bool(leakage.get("has_leakage", False)):
        leaks = leakage.get("leaks", [])
        raise RuntimeError(f"FM raw-window manifest has train/test subject leakage: {leaks}")
    observed = sorted(int(fold) for fold in manifest["fold_idx"].unique().tolist()) if not manifest.empty else []
    expected = list(range(1, int(expected_n_splits) + 1))
    missing = [fold for fold in expected if fold not in observed]
    if missing:
        raise RuntimeError(f"FM raw-window manifest missing expected folds: {missing}")


def build_raw_window_manifest(
    window_cache_path: str | Path,
    output_dir: str | Path,
    *,
    target_sfreq: int,
    window_sec: float,
    stride_sec: float,
    max_windows_per_record: int,
    positive_label: str,
    split_strategy: str,
    n_splits: int,
    random_seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if str(positive_label).lower() != "ez":
        raise ValueError("FM raw-window manifest currently requires positive_label='ez'.")
    cache = load_window_cache(window_cache_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    patient_index = cache["patient_index"]
    splits = build_outer_splits(patient_index, split_strategy=split_strategy, n_splits=int(n_splits), random_seed=int(random_seed))
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    row_id = 0

    for record in cache["run_records"]:
        audit_row = _raw_record_row(record, patient_index)
        subject_id = str(record.get("subject_id", ""))
        run_id = str(record.get("run_id", ""))
        sample = record.get("sample") or {}
        if not audit_row["usable_for_fm"]:
            skipped.append({"subject_id": subject_id, "run_id": run_id, "skip_reason": audit_row["skip_reason"]})
            continue
        raw = np.asarray(sample["raw_waveform"])
        labels = np.asarray(record["labels"]).astype(int)
        channel_names = list(record.get("channel_names_norm") or record.get("channel_names") or [])
        sfreq = _sfreq(record, sample)
        window_len = int(round(float(window_sec) * sfreq))
        stride_len = int(round(float(stride_sec) * sfreq))
        starts = _window_starts(raw.shape[1], window_len, stride_len, int(max_windows_per_record))
        if not starts:
            skipped.append({"subject_id": subject_id, "run_id": run_id, "skip_reason": "raw segment shorter than window_sec"})
            continue
        center = infer_center(subject_id, patient_index.get(subject_id, {}) or {}, record, sample)
        for split in splits:
            role = _split_role(subject_id, split)
            if role is None:
                continue
            for channel_idx, channel_name in enumerate(channel_names):
                for window_idx, start in enumerate(starts):
                    end = int(start + window_len)
                    rows.append(
                        {
                            "row_id": row_id,
                            "physical_window_id": _physical_window_id(
                                subject_id=subject_id,
                                run_id=run_id,
                                channel_idx=int(channel_idx),
                                start=int(start),
                                end=int(end),
                                sfreq=float(sfreq),
                                target_sfreq=int(target_sfreq),
                                window_sec=float(window_sec),
                            ),
                            "fold_idx": int(split["fold_idx"]),
                            "split_role": role,
                            "subject_id": subject_id,
                            "run_id": run_id,
                            "center": center,
                            "channel_name": str(channel_name),
                            "channel_idx": int(channel_idx),
                            "label_ez": int(labels[channel_idx] == 1),
                            "sfreq": float(sfreq),
                            "target_sfreq": int(target_sfreq),
                            "raw_n_times": int(raw.shape[1]),
                            "window_idx": int(window_idx),
                            "window_start_sample": int(start),
                            "window_end_sample": int(end),
                            "window_center_sample": int((start + end) // 2),
                            "window_sec": float(window_sec),
                            "stride_sec": float(stride_sec),
                        }
                    )
                    row_id += 1

    manifest = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped, columns=["subject_id", "run_id", "skip_reason"])
    manifest.to_csv(out / "fm_raw_window_manifest.csv", index=False)
    skipped_df.to_csv(out / "fm_raw_window_skipped_records.csv", index=False)
    leakage = _leakage_report(manifest)
    observed_folds = sorted(int(fold) for fold in manifest["fold_idx"].unique().tolist()) if not manifest.empty else []
    expected_folds = list(range(1, int(n_splits) + 1))
    missing_expected_folds = [fold for fold in expected_folds if fold not in observed_folds]
    audit = {
        "input_cache_path": str(window_cache_path),
        "output_dir": str(out),
        "target_sfreq": int(target_sfreq),
        "window_sec": float(window_sec),
        "stride_sec": float(stride_sec),
        "max_windows_per_record": int(max_windows_per_record),
        "n_manifest_rows": int(len(manifest)),
        "n_unique_physical_windows": int(manifest["physical_window_id"].nunique()) if not manifest.empty and "physical_window_id" in manifest.columns else 0,
        "n_unique_subjects": int(manifest["subject_id"].nunique()) if not manifest.empty else 0,
        "n_unique_run_records": int(manifest["run_id"].nunique()) if not manifest.empty else 0,
        "n_unique_patient_channels": int(manifest[["subject_id", "channel_name"]].drop_duplicates().shape[0]) if not manifest.empty else 0,
        "rows_by_center": manifest["center"].value_counts().sort_index().to_dict() if not manifest.empty else {},
        "rows_by_fold": manifest["fold_idx"].value_counts().sort_index().to_dict() if not manifest.empty else {},
        "rows_by_split_role": manifest["split_role"].value_counts().sort_index().to_dict() if not manifest.empty else {},
        "train_test_subject_leakage_check": leakage,
        "observed_folds": observed_folds,
        "expected_folds": expected_folds,
        "missing_expected_folds": missing_expected_folds,
        "skipped_records_count": int(len(skipped_df)),
        "skipped_records_by_reason": skipped_df["skip_reason"].value_counts().sort_index().to_dict() if not skipped_df.empty else {},
        "expected_n_splits": int(n_splits),
        "random_seed": int(random_seed),
        "positive_label": str(positive_label),
        "warning": "raw-waveform FM baseline only; no engineered window_features used.",
    }
    (out / "fm_raw_window_manifest_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    _validate_manifest_integrity(manifest, expected_n_splits=int(n_splits), leakage=leakage)
    return manifest, audit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window_cache_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_sfreq", type=int, default=200)
    parser.add_argument("--window_sec", type=float, default=4.0)
    parser.add_argument("--stride_sec", type=float, default=2.0)
    parser.add_argument("--max_windows_per_record", type=int, default=20)
    parser.add_argument("--positive_label", choices=["ez"], default="ez")
    parser.add_argument("--split_strategy", default="5fold")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    build_raw_window_manifest(
        args.window_cache_path,
        args.output_dir,
        target_sfreq=args.target_sfreq,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        max_windows_per_record=args.max_windows_per_record,
        positive_label=args.positive_label,
        split_strategy=args.split_strategy,
        n_splits=args.n_splits,
        random_seed=args.random_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
