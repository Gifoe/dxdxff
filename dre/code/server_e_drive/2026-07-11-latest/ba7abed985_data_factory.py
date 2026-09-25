from __future__ import annotations

from typing import Any, Dict, List, Tuple
import json
from pathlib import Path

import numpy as np
import pandas as pd
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


def _fixed_split_manifest(
    path: str | Path, patient_index: Dict[str, Dict[str, Any]], *, allow_partial_test_union: bool,
) -> tuple[Dict[str, Dict[str, Any]], List[Dict[str, List[str]]], dict[str, Any]]:
    """Load a frozen patient-level fit/validation/test contract without rerandomizing it."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Fixed split manifest not found: {source}")
    frame = pd.read_csv(source)
    required = {"subject_id", "outer_fold"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Fixed split manifest missing columns: {sorted(missing)}")
    role_column = next((name for name in ("split_role", "partition", "split", "role") if name in frame.columns), None)
    if role_column is None:
        raise ValueError("Fixed split manifest requires split_role/partition with fit, validation, and test")
    data = frame[["subject_id", "outer_fold", role_column]].copy()
    data.columns = ["subject_id", "outer_fold", "split_role"]
    data.subject_id = data.subject_id.astype(str).str.strip()
    data.outer_fold = pd.to_numeric(data.outer_fold, errors="raise").astype(int)
    data.split_role = data.split_role.astype(str).str.strip().str.lower().replace({"train": "fit", "val": "validation", "outer_test": "test", "heldout": "test"})
    if not data.split_role.isin({"fit", "validation", "test"}).all():
        raise ValueError("Fixed split manifest contains a role other than fit/validation/test")
    active = set(data.subject_id)
    unknown = sorted(active - set(patient_index))
    if unknown:
        raise ValueError(f"Fixed split manifest references unknown patients: {unknown}")
    if data.duplicated(["outer_fold", "subject_id"]).any():
        raise ValueError("Fixed split manifest assigns a patient more than one role within a fold")
    selected_index = {subject: patient_index[subject] for subject in sorted(active)}
    splits: List[Dict[str, List[str]]] = []
    test_occurrences: dict[str, int] = {}
    for fold, group in data.groupby("outer_fold", sort=True):
        roles = {role: sorted(values.subject_id.tolist()) for role, values in group.groupby("split_role")}
        if set(roles) != {"fit", "validation", "test"}:
            raise ValueError(f"Fixed split fold {fold} must contain fit, validation, and test")
        merged = set().union(*map(set, roles.values()))
        if merged != active:
            raise ValueError(f"Fixed split fold {fold} does not cover the active cohort exactly once")
        for subject in roles["test"]:
            test_occurrences[subject] = test_occurrences.get(subject, 0) + 1
        splits.append({
            "fold_idx": int(fold), "train_subjects": roles["fit"] + roles["validation"],
            "fit_subjects": roles["fit"], "validation_subjects": roles["validation"],
            "test_subjects": roles["test"],
        })
    if not allow_partial_test_union and (
        set(test_occurrences) != active or any(count != 1 for count in test_occurrences.values())
    ):
        raise ValueError("Fixed split manifest must assign every patient to exactly one outer-test fold")
    audit = {
        "status": "passed", "manifest_path": str(source), "n_patients": len(active),
        "n_outer_folds": len(splits), "allow_partial_test_union": bool(allow_partial_test_union),
        "folds": [{"fold_idx": split["fold_idx"], "n_fit": len(split["fit_subjects"]),
                    "n_validation": len(split["validation_subjects"]), "n_test": len(split["test_subjects"])} for split in splits],
    }
    return selected_index, splits, audit


def data_provider(args: Any):
    cohort_mode = str(getattr(args, "cohort_mode", "all90")).lower()
    sensitivity80 = cohort_mode == "sensitivity80"
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
    if fixed_manifest:
        patient_index, outer_splits, fixed_audit = _fixed_split_manifest(
            fixed_manifest, patient_index,
            allow_partial_test_union=bool(getattr(args, "allow_partial_fixed_test_manifest", False)),
        )
        if requested_count is not None and len(patient_index) != int(requested_count):
            raise ValueError(f"Fixed split manifest selects {len(patient_index)} patients, expected {requested_count}")
        retained = set(patient_index)
        run_records = [record for record in run_records if str(record.get("subject_id")) in retained]
        output = Path(getattr(args, "output_dir", "outputs")) / "audit"
        output.mkdir(parents=True, exist_ok=True)
        (output / "fixed_split_manifest_audit.json").write_text(json.dumps(fixed_audit, indent=2), encoding="utf-8")
        setattr(args, "fixed_split_manifest_audit", fixed_audit)
    else:
        outer_splits = build_outer_splits(
            patient_index,
            split_strategy=str(getattr(args, "split_strategy", "5fold")),
            n_splits=int(getattr(args, "n_splits", 5)),
            random_seed=int(getattr(args, "random_seed", 42)),
        )
    if sensitivity80 and not fixed_manifest:
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
    if bool(getattr(args, "use_p2_rtc_shift", False) or getattr(args, "use_p2_atc", False) or getattr(args, "use_p2_scope_v2", False)):
        from neuroez_c.task1_cohort_contract import validate_task1_cohort, write_task1_cohort_contract

        cohort_audit = validate_task1_cohort(
            patient_index, outer_splits,
            cohort_name=cohort_mode,
            all90_ledger_path=getattr(args, "p2_atc_all90_ledger_path", getattr(args, "rtc_all90_ledger_path")),
            exclusion_path=getattr(args, "exclude_subjects_file", "") if sensitivity80 else None,
        )
        audit_dir = Path(getattr(args, "output_dir", "outputs"))
        write_task1_cohort_contract(audit_dir, cohort_audit, outer_splits)
        setattr(args, "rtc_cohort_audit", cohort_audit)
    return run_records, patient_index, outer_splits


__all__ = ["build_outer_splits", "data_provider", "split_train_val_subjects"]
