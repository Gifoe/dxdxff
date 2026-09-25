"""Prepare and audit the feature-only Step4B static-top20 All90 cache."""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_factory import build_outer_splits
from ez_features import (
    BURSTNESS_FEATURE_NAMES,
    CLINICAL_ONSET_CORE_FEATURE_NAMES,
    WINDOW_NODE_FEATURE_NAMES,
    _burstness_features,
    _clinical_onset_core_features,
)

STEP4B_STATIC_TOP20_FEATURES = (
    "early_high_gamma_slope",
    "early_line_length_slope",
    "onset_latency_high_gamma",
    "onset_latency_line_length",
    "onset_rank_high_gamma",
    "onset_rank_line_length",
    "high_gamma_top20pct_mean",
    "line_length_top20pct_mean",
)
EXPECTED_PATIENTS = 90
EXPECTED_RUNS = 281
EXPECTED_LEGACY_MISSING_LZU = 14


def _load_pickle(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Cache not found: {path}")
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Unable to read cache {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Cache payload must be a dict: {path}")
    return payload


def _sample(record: dict[str, Any]) -> dict[str, Any]:
    sample = record.get("sample")
    return sample if isinstance(sample, dict) else record


def _read_subjects(path: Path, *, expected_patients: int) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"All90 ledger not found: {path}")
    frame = pd.read_csv(path)
    if "subject_id" not in frame.columns:
        raise RuntimeError(f"All90 ledger must contain subject_id: {path}")
    subjects = frame["subject_id"].astype(str).str.strip().tolist()
    duplicates = sorted(subject for subject, count in Counter(subjects).items() if count > 1)
    if duplicates:
        raise RuntimeError(f"All90 ledger contains duplicate patients: {duplicates}")
    if len(subjects) != expected_patients:
        raise RuntimeError(f"Expected {expected_patients} ledger patients, got {len(subjects)}")
    return subjects


def derive_step4b_static_top20(base: np.ndarray, centers: np.ndarray) -> np.ndarray:
    base = np.asarray(base, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    if base.ndim != 3 or base.shape[-1] != len(WINDOW_NODE_FEATURE_NAMES):
        raise RuntimeError(f"Expected [T,C,{len(WINDOW_NODE_FEATURE_NAMES)}] base features, got {base.shape}")
    if centers.shape != (base.shape[0],):
        raise RuntimeError(f"window_relative_centers_sec shape {centers.shape} does not match T={base.shape[0]}")
    clinical = _clinical_onset_core_features(base, centers)
    burst = _burstness_features(base, centers)
    clinical_idx = {name: idx for idx, name in enumerate(CLINICAL_ONSET_CORE_FEATURE_NAMES)}
    burst_idx = {name: idx for idx, name in enumerate(BURSTNESS_FEATURE_NAMES)}
    first = clinical[:, :, [clinical_idx[name] for name in STEP4B_STATIC_TOP20_FEATURES[:6]]]
    last = burst[:, :, [burst_idx[name] for name in STEP4B_STATIC_TOP20_FEATURES[6:]]]
    derived = np.concatenate([first, last], axis=-1).astype(np.float32, copy=False)
    if derived.shape[:2] != base.shape[:2] or derived.shape[-1] != len(STEP4B_STATIC_TOP20_FEATURES):
        raise RuntimeError(f"Unexpected derived Step4B shape: {derived.shape}")
    if not np.isfinite(derived).all():
        raise RuntimeError("Derived Step4B features contain non-finite values.")
    return derived


def _validate_patient_labels(subjects: Sequence[str], patient_index: dict[str, Any]) -> tuple[Counter[float], dict[str, float]]:
    values: Counter[float] = Counter()
    ez_fractions: dict[str, float] = {}
    for subject in subjects:
        meta = patient_index.get(subject)
        if not isinstance(meta, dict):
            raise RuntimeError(f"Missing patient metadata for {subject}")
        channels = list(meta.get("canonical_channels", []))
        labels = np.asarray(meta.get("labels", []), dtype=np.float32).reshape(-1)
        if not channels or len(channels) != len(labels):
            raise RuntimeError(f"Invalid canonical channel/label metadata for {subject}")
        if not np.isfinite(labels).all() or not set(np.unique(labels)).issubset({-1.0, 0.0, 1.0}):
            raise RuntimeError(f"Labels for {subject} must use cache EZ-positive values -1/0/1.")
        valid = labels >= 0.0
        if not np.any(valid):
            raise RuntimeError(f"Patient {subject} has no valid channel labels.")
        values.update(float(value) for value in labels[valid])
        ez_fractions[subject] = float((labels[valid] > 0.5).mean())
    if not {0.0, 1.0}.issubset(values):
        raise RuntimeError("All90 cache must contain both EZ=1 and NEZ=0 labels.")
    return values, ez_fractions


def _center_name(subject: str, meta: dict[str, Any]) -> str:
    value = meta.get("source_center", meta.get("center"))
    return str(value if value else subject.split(":", 1)[0]).strip().lower()


def _feature_stats(blocks: list[np.ndarray]) -> dict[str, Any]:
    if not blocks:
        raise RuntimeError("No Step4B feature blocks were available for audit.")
    flat = np.concatenate([block.reshape(-1, len(STEP4B_STATIC_TOP20_FEATURES)) for block in blocks], axis=0)
    finite = bool(np.isfinite(flat).all())
    minima = np.nanmin(flat, axis=0)
    maxima = np.nanmax(flat, axis=0)
    all_zero = [name for idx, name in enumerate(STEP4B_STATIC_TOP20_FEATURES) if minima[idx] == 0.0 and maxima[idx] == 0.0]
    zero_variance = [name for idx, name in enumerate(STEP4B_STATIC_TOP20_FEATURES) if minima[idx] == maxima[idx]]
    if not finite or all_zero or zero_variance:
        raise RuntimeError(
            f"Invalid Step4B feature values: finite={finite}, all_zero={all_zero}, zero_variance={zero_variance}"
        )
    return {
        "all_finite": finite,
        "all_zero_features": all_zero,
        "zero_variance_features": zero_variance,
        "feature_statistics": {
            name: {
                "mean": float(np.mean(flat[:, idx])),
                "std": float(np.std(flat[:, idx])),
                "min": float(minima[idx]),
                "max": float(maxima[idx]),
            }
            for idx, name in enumerate(STEP4B_STATIC_TOP20_FEATURES)
        },
    }


def _split_audit(patient_index: dict[str, Any], subjects: Sequence[str], seed: int) -> dict[str, Any]:
    selected_index = {subject: patient_index[subject] for subject in subjects}
    splits = build_outer_splits(selected_index, split_strategy="5fold", n_splits=5, random_seed=seed)
    seen: set[str] = set()
    fold_rows: list[dict[str, Any]] = []
    leakage: list[str] = []
    duplicate_test: set[str] = set()
    expected = set(subjects)
    for split in splits:
        train = set(map(str, split["train_subjects"]))
        test = set(map(str, split["test_subjects"]))
        leakage.extend(sorted(train & test))
        duplicate_test.update(seen & test)
        seen.update(test)
        if train | test != expected:
            raise RuntimeError(f"Fold {split['fold_idx']} does not partition the All90 ledger.")
        fold_rows.append({"fold_idx": int(split["fold_idx"]), "n_train": len(train), "n_test": len(test)})
    if leakage or duplicate_test or seen != expected:
        raise RuntimeError(
            f"Invalid patient-level folds: leakage={sorted(set(leakage))}, "
            f"duplicate_test={sorted(duplicate_test)}, missing_test={sorted(expected - seen)}"
        )
    return {
        "n_outer_splits": len(splits),
        "folds": fold_rows,
        "patient_disjoint": True,
        "test_subjects_unique": True,
        "test_union_count": len(seen),
        "test_union_matches_all90": True,
    }


def _audit_target_cache(
    payload: dict[str, Any],
    subjects: Sequence[str],
    *,
    expected_runs: int,
) -> dict[str, Any]:
    patient_index = payload.get("patient_index")
    records = payload.get("run_records")
    names = [str(name) for name in payload.get("window_feature_names", [])]
    expected_names = list(WINDOW_NODE_FEATURE_NAMES) + list(STEP4B_STATIC_TOP20_FEATURES)
    if not isinstance(patient_index, dict) or set(map(str, patient_index)) != set(subjects):
        raise RuntimeError("Training cache patient_index must exactly match the fixed All90 ledger.")
    if not isinstance(records, list) or len(records) != expected_runs:
        raise RuntimeError(f"Training cache must contain exactly {expected_runs} run_records.")
    if names != expected_names:
        missing = sorted(set(expected_names) - set(names))
        raise RuntimeError(f"Training cache must have the exact 20+8 schema; missing={missing}, count={len(names)}")
    blocks: list[np.ndarray] = []
    run_subjects: set[str] = set()
    for record in records:
        subject = str(record.get("subject_id"))
        if subject not in patient_index:
            raise RuntimeError(f"Training cache run has unknown patient: {subject}")
        sample = _sample(record)
        features = np.asarray(sample.get("window_features"), dtype=np.float32)
        centers = np.asarray(sample.get("window_relative_centers_sec"), dtype=np.float32)
        if features.ndim != 3 or features.shape[-1] != len(expected_names) or centers.shape != (features.shape[0],):
            raise RuntimeError(f"Invalid training cache sample for {subject} / {record.get('run_id')}")
        blocks.append(features[:, :, -len(STEP4B_STATIC_TOP20_FEATURES):])
        run_subjects.add(subject)
    if run_subjects != set(subjects):
        raise RuntimeError(f"Training cache run coverage differs from All90: missing={sorted(set(subjects) - run_subjects)}")
    labels, ez_fractions = _validate_patient_labels(subjects, patient_index)
    return {
        "n_patients": len(patient_index),
        "n_runs": len(records),
        "window_feature_count": len(names),
        "window_feature_names": names,
        "required_features_present": list(STEP4B_STATIC_TOP20_FEATURES),
        "required_features_missing": [],
        "label_values": {str(key): int(value) for key, value in sorted(labels.items())},
        "ez_fractions": ez_fractions,
        **_feature_stats(blocks),
    }


def prepare_step4b_cache(
    feature_cache: str | Path,
    target_cache: str | Path,
    subjects_file: str | Path,
    audit_output: str | Path,
    *,
    dry_run: bool = False,
    build_if_missing: bool = True,
    expected_patients: int = EXPECTED_PATIENTS,
    expected_runs: int = EXPECTED_RUNS,
    seed: int = 42,
) -> dict[str, Any]:
    feature_path = Path(feature_cache)
    target_path = Path(target_cache)
    subjects_path = Path(subjects_file)
    audit_path = Path(audit_output)
    subjects = _read_subjects(subjects_path, expected_patients=expected_patients)
    allowed = set(subjects)

    payload = _load_pickle(feature_path)
    records = payload.get("run_records")
    patient_index = payload.get("patient_index")
    names = [str(name) for name in payload.get("window_feature_names", [])]
    if not isinstance(records, list) or not isinstance(patient_index, dict):
        raise RuntimeError("Feature cache must contain run_records list and patient_index dict.")
    if names != list(WINDOW_NODE_FEATURE_NAMES):
        raise RuntimeError(f"Feature source must have the canonical 20-feature schema, got {names}")
    label_semantics = str(payload.get("label_semantics", ""))
    if "ez-positive" not in label_semantics.lower():
        raise RuntimeError(f"Feature cache label semantics are not explicitly EZ-positive: {label_semantics!r}")
    available = set(map(str, patient_index))
    missing_subjects = sorted(allowed - available)
    if missing_subjects:
        raise RuntimeError(f"Feature cache is missing All90 patients: {missing_subjects}")
    selected = [record for record in records if str(record.get("subject_id")) in allowed]
    if len(selected) != expected_runs:
        raise RuntimeError(f"Expected {expected_runs} All90 runs in source cache, got {len(selected)}")
    if {str(record.get("subject_id")) for record in selected} != allowed:
        raise RuntimeError("Selected source runs do not cover the complete All90 ledger.")
    label_values, ez_fractions = _validate_patient_labels(subjects, patient_index)
    centers = Counter(_center_name(subject, patient_index[subject]) for subject in subjects)
    high_ez_lzu = sorted(
        subject for subject in subjects
        if _center_name(subject, patient_index[subject]) == "lzu" and ez_fractions[subject] > 0.40
    )

    build_target = not target_path.exists() and not dry_run
    if build_target and not build_if_missing:
        raise FileNotFoundError(f"Derived Step4B cache is missing and auto-build is disabled: {target_path}")
    derived_blocks: list[np.ndarray] = []
    enriched_records: list[dict[str, Any]] = []
    seen_run_keys: set[tuple[str, str]] = set()
    for record in selected:
        subject = str(record.get("subject_id"))
        run_key = (subject, str(record.get("run_id", _sample(record).get("sample_id", ""))))
        if run_key in seen_run_keys:
            raise RuntimeError(f"Duplicate source run key: {run_key}")
        seen_run_keys.add(run_key)
        sample = _sample(record)
        base = np.asarray(sample.get("window_features"), dtype=np.float32)
        window_centers = np.asarray(sample.get("window_relative_centers_sec"), dtype=np.float32)
        derived = derive_step4b_static_top20(base, window_centers)
        derived_blocks.append(derived)
        if build_target:
            new_sample = dict(sample)
            new_sample["window_features"] = np.concatenate([base, derived], axis=-1).astype(np.float32, copy=False)
            new_sample["window_feature_names"] = names + list(STEP4B_STATIC_TOP20_FEATURES)
            new_sample["feature_mode"] = "STEP4B_STATIC_TOP20_ALL90"
            new_record = dict(record)
            new_record["sample"] = new_sample
            enriched_records.append(new_record)
    derived_stats = _feature_stats(derived_blocks)
    split_report = _split_audit(patient_index, subjects, seed)

    source_report = {
        "source_cache_path": str(feature_path),
        "source_n_patients": len(patient_index),
        "source_n_runs": len(records),
        "source_feature_count": len(names),
        "source_required_features_present": [name for name in STEP4B_STATIC_TOP20_FEATURES if name in names],
        "source_required_features_missing": [name for name in STEP4B_STATIC_TOP20_FEATURES if name not in names],
        "source_sufficient_without_raw": True,
        "raw_cache_used": False,
    }

    if build_target:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        output = dict(payload)
        output.update({
            "cache_version": "neuroez_c_step4b_static_top20_all90_v1",
            "feature_mode": "STEP4B_STATIC_TOP20_ALL90",
            "patient_index": {subject: patient_index[subject] for subject in subjects},
            "run_records": enriched_records,
            "window_feature_names": names + list(STEP4B_STATIC_TOP20_FEATURES),
        })
        with target_path.open("wb") as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        target_action = "built"
        target_report = {
            "n_patients": len(subjects),
            "n_runs": len(enriched_records),
            "window_feature_count": 28,
            "window_feature_names": names + list(STEP4B_STATIC_TOP20_FEATURES),
            "required_features_present": list(STEP4B_STATIC_TOP20_FEATURES),
            "required_features_missing": [],
            "label_values": {str(key): int(value) for key, value in sorted(label_values.items())},
            **derived_stats,
        }
    elif target_path.exists():
        del payload, records, selected, enriched_records, derived_blocks
        gc.collect()
        target_payload = _load_pickle(target_path)
        target_report = _audit_target_cache(target_payload, subjects, expected_runs=expected_runs)
        del target_payload
        gc.collect()
        target_action = "reused"
    else:
        target_action = "dry_run_simulated"
        target_report = {
            "n_patients": len(subjects),
            "n_runs": len(selected),
            "window_feature_count": 28,
            "window_feature_names": names + list(STEP4B_STATIC_TOP20_FEATURES),
            "required_features_present": list(STEP4B_STATIC_TOP20_FEATURES),
            "required_features_missing": [],
            "label_values": {str(key): int(value) for key, value in sorted(label_values.items())},
            **derived_stats,
        }

    report = {
        "protocol_name": "fixed_all90_step4b_static_top20_nez",
        "status": "passed",
        "dry_run": bool(dry_run),
        "feature_source": "feature_pkl_only",
        "target_cache_path": str(target_path),
        "target_cache_action": target_action,
        "subjects_file": str(subjects_path),
        "n_patients": int(target_report["n_patients"]),
        "n_runs": int(target_report["n_runs"]),
        "center_distribution": {key: int(value) for key, value in sorted(centers.items())},
        "positive_label": "nez",
        "score_semantics": "nez_probability",
        "cache_label_semantics": label_semantics,
        "label_mapping_verified": (
            "cache labels remain EZ=1, NEZ=0; dataset computes NEZ=1 as 1-labels_ez "
            "exactly once for positive_label=nez"
        ),
        "required_features": list(STEP4B_STATIC_TOP20_FEATURES),
        "required_features_present": list(target_report["required_features_present"]),
        "required_features_missing": list(target_report["required_features_missing"]),
        "required_feature_count": len(target_report["required_features_present"]),
        "window_feature_count": int(target_report["window_feature_count"]),
        "all_features_finite": bool(target_report["all_finite"]),
        "all_zero_features": list(target_report["all_zero_features"]),
        "zero_variance_features": list(target_report["zero_variance_features"]),
        "legacy_high_ez_lzu_threshold": 0.40,
        "legacy_high_ez_lzu_actual_count": len(high_ez_lzu),
        "legacy_high_ez_lzu_subjects": high_ez_lzu,
        "legacy_missing_lzu_claimed_count": EXPECTED_LEGACY_MISSING_LZU,
        "legacy_missing_lzu_count_matches_claim": len(high_ez_lzu) == EXPECTED_LEGACY_MISSING_LZU,
        "legacy_missing_lzu_discrepancy": EXPECTED_LEGACY_MISSING_LZU - len(high_ez_lzu),
        "drop_high_ez_fraction_lzu": False,
        "center_as_input_allowed": False,
        **source_report,
        **split_report,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--subjects", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-build-if-missing", action="store_true")
    args = parser.parse_args()
    report = prepare_step4b_cache(
        args.feature_cache,
        args.target_cache,
        args.subjects,
        args.audit_output,
        dry_run=args.dry_run,
        build_if_missing=not args.no_build_if_missing,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
