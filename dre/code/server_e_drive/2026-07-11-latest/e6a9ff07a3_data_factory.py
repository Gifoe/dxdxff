from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import KFold

from ez_dataset import build_or_load_run_records


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
    outer_splits = build_outer_splits(
        patient_index,
        split_strategy=str(getattr(args, "split_strategy", "5fold")),
        n_splits=int(getattr(args, "n_splits", 5)),
        random_seed=int(getattr(args, "random_seed", 42)),
    )
    if sensitivity80:
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


__all__ = ["build_outer_splits", "data_provider", "split_train_val_subjects"]
