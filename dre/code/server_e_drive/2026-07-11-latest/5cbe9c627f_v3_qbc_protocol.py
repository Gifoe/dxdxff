"""Fixed-subject and fixed-fold contracts for V3-QBC."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_allowed_subjects(path: str | Path) -> list[str]:
    frame = pd.read_csv(path)
    if "subject_id" not in frame.columns:
        raise ValueError("Allowed-subject ledger must contain subject_id")
    subjects = frame["subject_id"].astype(str).str.strip().tolist()
    if not subjects or len(subjects) != len(set(subjects)):
        raise ValueError("Allowed-subject ledger must contain unique non-empty subjects")
    return subjects


def read_outer_fold_manifest(path: str | Path) -> dict[int, list[str]]:
    frame = pd.read_csv(path)
    fold_column = "outer_fold" if "outer_fold" in frame.columns else "fold_idx"
    if "subject_id" not in frame.columns or fold_column not in frame.columns:
        raise ValueError("Outer-fold manifest requires subject_id and outer_fold/fold_idx")
    role_column = next(
        (name for name in ("split_role", "split", "partition", "role") if name in frame.columns),
        None,
    )
    if role_column:
        frame = frame[frame[role_column].astype(str).str.lower().isin({"test", "outer_test", "heldout"})]
    keyed = frame[["subject_id", fold_column]].copy()
    keyed["subject_id"] = keyed["subject_id"].astype(str).str.strip()
    if keyed["subject_id"].duplicated().any():
        raise ValueError("Every subject must occur in exactly one outer test fold")
    result: dict[int, list[str]] = {}
    for fold, group in keyed.groupby(fold_column):
        result[int(fold)] = sorted(group["subject_id"].tolist())
    if sorted(result) != list(range(1, len(result) + 1)):
        raise ValueError("Outer folds must be consecutively numbered from 1")
    return result


def validate_v3_qbc_protocol(
    *,
    patient_ids: list[str],
    allowed_subjects_path: str | Path,
    outer_fold_manifest_path: str | Path,
    require_n_patients: int,
    seed: int,
    fixed_split_manifest_path: str | Path | None = None,
    allow_partial_test_manifest: bool = False,
) -> dict[str, Any]:
    allowed = read_allowed_subjects(allowed_subjects_path)
    folds = read_outer_fold_manifest(outer_fold_manifest_path)
    patients = set(map(str, patient_ids))
    allowed_set = set(allowed)
    fold_union = set().union(*(set(values) for values in folds.values()))
    errors = []
    if int(seed) < 0:
        errors.append(f"seed must be non-negative, got {seed}")
    if len(allowed) != int(require_n_patients):
        errors.append(f"allowed ledger has {len(allowed)} patients, expected {require_n_patients}")
    if patients != allowed_set:
        errors.append(
            f"cache/ledger mismatch missing={sorted(allowed_set-patients)} extra={sorted(patients-allowed_set)}"
        )
    if not allow_partial_test_manifest and fold_union != allowed_set:
        errors.append(
            f"fold/ledger mismatch missing={sorted(allowed_set-fold_union)} extra={sorted(fold_union-allowed_set)}"
        )
    if allow_partial_test_manifest and (not fold_union or not fold_union.issubset(allowed_set)):
        errors.append("LOCO test subjects must be a non-empty subset of the allowed cohort")
    for left in folds:
        for right in folds:
            if left < right and set(folds[left]) & set(folds[right]):
                errors.append(f"outer test folds {left} and {right} overlap")
    fold_partitions = []
    for fold_idx, test_subjects in sorted(folds.items()):
        outer_train = sorted(allowed_set - set(test_subjects))
        shuffled = list(outer_train)
        rng = np.random.default_rng(int(seed) + int(fold_idx))
        rng.shuffle(shuffled)
        n_val = min(max(1, int(round(len(shuffled) * 0.20))), max(1, len(shuffled) - 1))
        validation = shuffled[:n_val]
        fit = shuffled[n_val:]
        partition_text = "\n".join(
            [*(f"fit|{subject}" for subject in sorted(fit)),
             *(f"validation|{subject}" for subject in sorted(validation)),
             *(f"test|{subject}" for subject in sorted(test_subjects))]
        )
        fold_partitions.append(
            {
                "outer_fold": int(fold_idx),
                "n_fit": len(fit),
                "n_validation": len(validation),
                "n_test": len(test_subjects),
                "fit_validation_test_disjoint": not (
                    set(fit) & set(validation) or set(fit) & set(test_subjects) or set(validation) & set(test_subjects)
                ),
                "outer_train_covered": set(fit) | set(validation) == set(outer_train),
                "partition_hash": hashlib.sha256(partition_text.encode("utf-8")).hexdigest(),
            }
        )
    audit = {
        "status": "passed" if not errors else "failed",
        "protocol_name": "v3_qbc_fixed_outer_validation_only_threshold",
        "n_patients": len(patients),
        "n_outer_folds": len(folds),
        "training_seed": int(seed),
        "allowed_subjects_hash": file_sha256(allowed_subjects_path),
        "outer_fold_manifest_hash": file_sha256(outer_fold_manifest_path),
        "fold_test_counts": {str(key): len(value) for key, value in folds.items()},
        "fixed_fit_validation_partitions": fold_partitions,
        "test_union_count": len(fold_union),
        "patient_disjoint": not any("overlap" in error for error in errors),
        "label_semantics": "NEZ=1,EZ=0; V3 supervised adapter uses labels_ez once",
        "threshold_protocol": "outer-train fixed validation global robust-z threshold",
        "test_labels_used_for_threshold": False,
        "center_used_as_model_input": False,
        "errors": errors,
    }
    if fixed_split_manifest_path:
        from data_factory import read_fixed_split_manifest

        fixed = read_fixed_split_manifest(
            fixed_split_manifest_path, {subject: {} for subject in allowed},
            allow_partial_test_union=bool(allow_partial_test_manifest),
        )
        fixed_test = {int(split["fold_idx"]): set(split["test_subjects"]) for split in fixed}
        if fixed_test != {int(fold): set(values) for fold, values in folds.items()}:
            errors.append("fixed split manifest test roles differ from the outer-fold manifest")
        audit["fixed_split_manifest_path"] = str(fixed_split_manifest_path)
        audit["fixed_validation_manifest_used"] = True
    else:
        audit["fixed_validation_manifest_used"] = False
    audit["status"] = "passed" if not errors else "failed"
    audit["allow_partial_test_manifest"] = bool(allow_partial_test_manifest)
    if errors:
        raise RuntimeError(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return audit


__all__ = [
    "file_sha256",
    "read_allowed_subjects",
    "read_outer_fold_manifest",
    "validate_v3_qbc_protocol",
]
