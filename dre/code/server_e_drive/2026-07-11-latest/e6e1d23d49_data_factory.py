from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import KFold

from ez_dataset import build_or_load_run_records


def read_fixed_split_manifest(
    path: str | Path,
    patient_index: Dict[str, Dict[str, Any]],
    *,
    allow_partial_test_union: bool = False,
) -> List[Dict[str, List[str]]]:
    """Load an audited fit/validation/test manifest without recreating a split.

    The manifest is intentionally patient-level.  It keeps model seeds from
    changing any cohort membership or validation threshold source.
    """
    import pandas as pd

    frame = pd.read_csv(path)
    required = {"subject_id", "split_role"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Fixed split manifest requires {sorted(required)}")
    fold_column = "outer_fold" if "outer_fold" in frame.columns else "fold_idx"
    if fold_column not in frame.columns:
        raise ValueError("Fixed split manifest requires outer_fold or fold_idx")
    frame = frame[["subject_id", fold_column, "split_role"]].copy()
    frame["subject_id"] = frame["subject_id"].astype(str).str.strip()
    frame["split_role"] = frame["split_role"].astype(str).str.strip().str.lower()
    frame[fold_column] = pd.to_numeric(frame[fold_column], errors="raise").astype(int)
    if not frame["split_role"].isin({"fit", "validation", "test"}).all():
        raise ValueError("Fixed split manifest roles must be fit, validation, or test")
    subjects = set(map(str, patient_index))
    splits: List[Dict[str, List[str]]] = []
    test_occurrences: dict[str, int] = {}
    for fold_idx, group in frame.groupby(fold_column, sort=True):
        roles = {role: sorted(values.subject_id.tolist()) for role, values in group.groupby("split_role")}
        if set(roles) != {"fit", "validation", "test"}:
            raise ValueError(f"Fold {fold_idx} must contain fit, validation, and test roles")
        combined = roles["fit"] + roles["validation"] + roles["test"]
        if len(combined) != len(set(combined)):
            raise ValueError(f"Fold {fold_idx} has a patient in multiple split roles")
        if set(combined) != subjects:
            raise ValueError(f"Fold {fold_idx} does not exactly cover the active cohort")
        for subject in roles["test"]:
            test_occurrences[subject] = test_occurrences.get(subject, 0) + 1
        splits.append({
            "fold_idx": int(fold_idx),
            "fit_subjects": roles["fit"],
            "validation_subjects": roles["validation"],
            "train_subjects": roles["fit"] + roles["validation"],
            "test_subjects": roles["test"],
        })
    if not allow_partial_test_union and (set(test_occurrences) != subjects or any(count != 1 for count in test_occurrences.values())):
        raise ValueError("Every active patient must occur in exactly one fixed outer-test fold")
    return splits


def build_outer_splits(
    patient_index: Dict[str, Dict[str, Any]],
    *,
    split_strategy: str = "5fold",
    n_splits: int = 5,
    random_seed: int = 42,
) -> List[Dict[str, List[str]]]:
    subject_ids = sorted(patient_index.keys())
    if len(subject_ids) < 2:
        raise ValueError(
            "Patient-wise cross-validation requires at least two patients after data discovery and feature extraction. "
            f"Detected {len(subject_ids)} patient(s): {subject_ids}."
        )

    if split_strategy.lower() == "lopo":
        return [
            {
                "fold_idx": fold_idx + 1,
                "train_subjects": [sid for sid in subject_ids if sid != test_subject],
                "test_subjects": [test_subject],
            }
            for fold_idx, test_subject in enumerate(subject_ids)
        ]

    actual_splits = max(2, min(int(n_splits), len(subject_ids)))
    subject_array = np.asarray(subject_ids)
    kfold = KFold(n_splits=actual_splits, shuffle=True, random_state=random_seed)
    splits: List[Dict[str, List[str]]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(subject_array), start=1):
        splits.append(
            {
                "fold_idx": fold_idx,
                "train_subjects": subject_array[train_idx].tolist(),
                "test_subjects": subject_array[test_idx].tolist(),
            }
        )
    return splits


def split_train_val_subjects(
    train_subjects: List[str],
    *,
    val_ratio: float = 0.2,
    random_seed: int = 42,
    fold_idx: int = 0,
) -> Tuple[List[str], List[str]]:
    train_subjects = list(train_subjects)
    if len(train_subjects) < 2:
        return train_subjects, train_subjects

    rng = np.random.default_rng(seed=int(random_seed) + int(fold_idx))
    rng.shuffle(train_subjects)
    n_val = max(1, int(round(len(train_subjects) * float(val_ratio))))
    n_val = min(n_val, len(train_subjects) - 1)
    val_subjects = train_subjects[:n_val]
    fit_subjects = train_subjects[n_val:]
    return fit_subjects, val_subjects


def data_provider(args: Any):
    run_records, patient_index = build_or_load_run_records(args)
    fixed_manifest = str(getattr(args, "fixed_split_manifest", "") or "").strip()
    if fixed_manifest:
        outer_splits = read_fixed_split_manifest(
            fixed_manifest, patient_index,
            allow_partial_test_union=bool(getattr(args, "allow_partial_fixed_test_manifest", False)),
        )
        selected = int(getattr(args, "selected_outer_fold", 0) or 0)
        if selected:
            outer_splits = [split for split in outer_splits if int(split["fold_idx"]) == selected]
            if len(outer_splits) != 1:
                raise ValueError(f"selected_outer_fold={selected} is not present in fixed split manifest")
        return run_records, patient_index, outer_splits
    manifest_path = str(getattr(args, "v3_qbc_outer_fold_manifest", "") or "").strip()
    if bool(getattr(args, "use_v3_qbc", False)) and manifest_path:
        from neuroez_c.v3_qbc_protocol import read_outer_fold_manifest

        manifest = read_outer_fold_manifest(manifest_path)
        subjects = set(map(str, patient_index))
        manifest_subjects = set().union(*(set(values) for values in manifest.values()))
        if subjects != manifest_subjects:
            raise RuntimeError(
                "V3-QBC outer manifest does not match the active cache cohort: "
                f"missing={sorted(subjects-manifest_subjects)} extra={sorted(manifest_subjects-subjects)}"
            )
        outer_splits = [
            {
                "fold_idx": fold_idx,
                "train_subjects": sorted(subjects - set(test_subjects)),
                "test_subjects": list(test_subjects),
            }
            for fold_idx, test_subjects in sorted(manifest.items())
        ]
    else:
        outer_splits = build_outer_splits(
            patient_index,
            split_strategy=str(getattr(args, "split_strategy", "5fold")),
            n_splits=int(getattr(args, "n_splits", 5)),
            random_seed=int(getattr(args, "random_seed", 42)),
        )
    return run_records, patient_index, outer_splits


__all__ = ["build_outer_splits", "data_provider", "read_fixed_split_manifest", "split_train_val_subjects"]
