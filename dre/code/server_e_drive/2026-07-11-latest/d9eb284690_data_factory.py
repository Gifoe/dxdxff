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
    """Load an audited fit/validation/test partition without resampling patients."""
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
    sensitivity80 = str(getattr(args, "cohort_mode", "all90")).lower() == "sensitivity80"
    requested_count = getattr(args, "require_n_patients", None)
    if sensitivity80:
        # The source cache is the frozen All90 cache. The count assertion is
        # applied after exact manifest filtering, not before it.
        setattr(args, "require_n_patients", None)
    try:
        run_records, patient_index = build_or_load_run_records(args)
    finally:
        if sensitivity80:
            setattr(args, "require_n_patients", requested_count)
    fixed_manifest = str(getattr(args, "fixed_split_manifest", "") or "").strip()
    outer_splits = read_fixed_split_manifest(
        fixed_manifest, patient_index,
        allow_partial_test_union=bool(getattr(args, "allow_partial_fixed_test_manifest", False)),
    ) if fixed_manifest else build_outer_splits(
        patient_index,
        split_strategy=str(getattr(args, "split_strategy", "5fold")),
        n_splits=int(getattr(args, "n_splits", 5)),
        random_seed=int(getattr(args, "random_seed", 42)),
    )
    selected = int(getattr(args, "selected_outer_fold", 0) or 0)
    if fixed_manifest and selected:
        outer_splits = [split for split in outer_splits if int(split["fold_idx"]) == selected]
        if len(outer_splits) != 1:
            raise ValueError(f"selected_outer_fold={selected} is not present in fixed split manifest")
    if sensitivity80:
        if fixed_manifest:
            # The confirmatory runner supplies the already-filtered frozen
            # cohort.  Reapplying the legacy 90->80 exclusion would be wrong.
            setattr(args, "sensitivity80_cohort_audit", {
                "status": "passed", "cohort_status": "external_fixed_manifest",
                "n_patients": len(patient_index), "fixed_split_manifest": fixed_manifest,
            })
            return run_records, patient_index, outer_splits
        from pathlib import Path
        from neuroez_c.cane_path_cohort import build_sensitivity80_cohort

        patient_index, outer_splits, cohort_audit = build_sensitivity80_cohort(
            patient_index,
            outer_splits,
            getattr(args, "exclude_subjects_file"),
            output_dir=Path(getattr(args, "output_dir", "outputs")) / "audit",
        )
        retained = set(patient_index)
        run_records = [record for record in run_records if str(record.get("subject_id")) in retained]
        setattr(args, "sensitivity80_cohort_audit", cohort_audit)
    return run_records, patient_index, outer_splits


__all__ = ["build_outer_splits", "data_provider", "read_fixed_split_manifest", "split_train_val_subjects"]
