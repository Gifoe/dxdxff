"""Patient-level fixed-partition and LOCO contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from neuroez_c.p2_v3_fusion_protocol import discover_fold_ledger, read_subjects, validate_fold_manifest


CENTERS = ("hup", "lzu", "multicenter", "pediatric")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subject_center(subject_id: str) -> str:
    return str(subject_id).split(":", 1)[0].strip().lower()


def _read_subjects_with_center(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "subject_id" not in frame.columns:
        raise ValueError("Cohort ledger must contain subject_id")
    out = pd.DataFrame({"subject_id": frame.subject_id.astype(str).str.strip()})
    out["center"] = frame.center.astype(str).str.strip().str.lower() if "center" in frame else out.subject_id.map(subject_center)
    if out.subject_id.duplicated().any() or (out.subject_id == "").any():
        raise ValueError("Cohort ledger must contain unique non-empty subject IDs")
    if not out.center.isin(CENTERS).all():
        raise ValueError(f"Unexpected centers: {sorted(set(out.center) - set(CENTERS))}")
    return out.sort_values("subject_id", kind="mergesort").reset_index(drop=True)


def validate_partition(frame: pd.DataFrame, subjects: set[str], *, expected_folds: set[int] | None = None) -> dict[str, Any]:
    required = {"subject_id", "center", "outer_fold", "split_role"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Partition missing columns: {sorted(required - set(frame.columns))}")
    data = frame.copy()
    data.subject_id = data.subject_id.astype(str)
    data.center = data.center.astype(str).str.lower()
    data.split_role = data.split_role.astype(str).str.lower()
    data.outer_fold = pd.to_numeric(data.outer_fold, errors="raise").astype(int)
    if not data.split_role.isin({"fit", "validation", "test"}).all():
        raise ValueError("Only fit, validation, and test split roles are allowed")
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    observed_folds = set(map(int, data.outer_fold.unique()))
    if expected_folds is not None and observed_folds != expected_folds:
        errors.append(f"unexpected folds {sorted(observed_folds)}")
    test_occurrences: dict[str, int] = {}
    for fold, group in data.groupby("outer_fold", sort=True):
        if group.subject_id.duplicated().any():
            errors.append(f"fold {fold} has patient duplicates")
            continue
        roles = {role: set(values.subject_id) for role, values in group.groupby("split_role")}
        if set(roles) != {"fit", "validation", "test"}:
            errors.append(f"fold {fold} lacks a split role")
            continue
        combined = set().union(*roles.values())
        if combined != subjects:
            errors.append(f"fold {fold} does not cover active cohort")
        if any(roles[left] & roles[right] for left in roles for right in roles if left < right):
            errors.append(f"fold {fold} has split overlap")
        for subject in roles["test"]:
            test_occurrences[subject] = test_occurrences.get(subject, 0) + 1
        rows.append({
            "outer_fold": int(fold), "n_fit": len(roles["fit"]), "n_validation": len(roles["validation"]),
            "n_test": len(roles["test"]), "fit_validation_test_disjoint": not any(roles[left] & roles[right] for left in roles for right in roles if left < right),
        })
    if len(observed_folds) > 1 and (set(test_occurrences) != subjects or any(v != 1 for v in test_occurrences.values())):
        errors.append("outer-test union must cover every patient exactly once")
    return {"status": "passed" if not errors else "failed", "errors": errors, "folds": rows,
            "n_subjects": len(subjects), "test_union_count": len(test_occurrences)}


def build_reference_partition(
    *, cohort_ledger: str | Path, outer_fold_manifest: str | Path, p2_reference_root: str | Path,
    output_path: str | Path, require_n_patients: int = 80,
) -> dict[str, Any]:
    """Freeze the actual seed-42 P2 validation patients for all later seeds."""
    cohort = _read_subjects_with_center(cohort_ledger)
    subjects = set(cohort.subject_id)
    if len(subjects) != int(require_n_patients):
        raise ValueError(f"Expected {require_n_patients} cohort patients, got {len(subjects)}")
    expected_tests = validate_fold_manifest(outer_fold_manifest, subjects)
    rows: list[dict[str, Any]] = []
    roots = Path(p2_reference_root)
    for fold in range(1, 6):
        validation = pd.read_csv(discover_fold_ledger(roots, fold, "validation"))
        test = pd.read_csv(discover_fold_ledger(roots, fold, "test"))
        val_subjects = set(validation.subject_id.astype(str))
        test_subjects = set(test.subject_id.astype(str))
        if test_subjects != expected_tests[fold]:
            raise RuntimeError(f"Reference P2 fold {fold} test patients differ from frozen outer manifest")
        if val_subjects & test_subjects:
            raise RuntimeError(f"Reference P2 fold {fold} validation/test leakage")
        fit_subjects = subjects - val_subjects - test_subjects
        if not fit_subjects or not val_subjects:
            raise RuntimeError(f"Reference P2 fold {fold} has empty fit or validation set")
        for role, ids in (("fit", fit_subjects), ("validation", val_subjects), ("test", test_subjects)):
            for subject in sorted(ids):
                rows.append({"subject_id": subject, "center": subject_center(subject), "outer_fold": fold, "split_role": role})
    output = Path(output_path); output.parent.mkdir(parents=True, exist_ok=True)
    partition = pd.DataFrame(rows).sort_values(["outer_fold", "split_role", "subject_id"], kind="mergesort")
    audit = validate_partition(partition, subjects, expected_folds={1, 2, 3, 4, 5})
    if audit["status"] != "passed":
        raise RuntimeError(json.dumps(audit, sort_keys=True))
    partition.to_csv(output, index=False)
    return {**audit, "partition_path": str(output), "partition_sha256": sha256_file(output),
            "source": "seed42_p2_validation_ledgers", "cohort_sha256": sha256_file(cohort_ledger),
            "outer_fold_manifest_sha256": sha256_file(outer_fold_manifest)}


def _stable_order(seed: int, held_out: str, center: str, subjects: list[str]) -> list[str]:
    return sorted(subjects, key=lambda subject: hashlib.sha256(f"{seed}|{held_out}|{center}|{subject}".encode()).hexdigest())


def build_loco_manifests(
    *, cohort_ledger: str | Path, output_dir: str | Path, split_seed: int,
) -> dict[str, Any]:
    cohort = _read_subjects_with_center(cohort_ledger)
    subjects = set(cohort.subject_id)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    audits: dict[str, Any] = {}
    all_rows: list[pd.DataFrame] = []
    for held_out in CENTERS:
        target = cohort[cohort.center.eq(held_out)]
        source = cohort[~cohort.center.eq(held_out)]
        rows: list[dict[str, Any]] = []
        for center in CENTERS:
            group = source[source.center.eq(center)]
            if group.empty:
                continue
            ordered = _stable_order(split_seed, held_out, center, group.subject_id.astype(str).tolist())
            n_validation = min(len(ordered) - 1, max(2, int(round(0.2 * len(ordered)))))
            if n_validation < 1:
                raise RuntimeError(f"LOCO {held_out}: source center {center} cannot supply validation")
            validation = set(ordered[:n_validation])
            for subject in ordered:
                rows.append({"subject_id": subject, "center": center, "outer_fold": 1,
                             "split_role": "validation" if subject in validation else "fit",
                             "held_out_center": held_out, "source_split_seed": int(split_seed)})
        for subject in target.subject_id.astype(str):
            rows.append({"subject_id": subject, "center": held_out, "outer_fold": 1, "split_role": "test",
                         "held_out_center": held_out, "source_split_seed": int(split_seed)})
        frame = pd.DataFrame(rows).sort_values(["split_role", "center", "subject_id"], kind="mergesort")
        audit = validate_partition(frame, subjects, expected_folds={1})
        target_roles = set(frame.loc[frame.center.eq(held_out), "split_role"])
        if target_roles != {"test"}:
            audit["errors"].append("held-out center appears outside test")
            audit["status"] = "failed"
        for source_center in set(CENTERS) - {held_out}:
            present = set(frame.loc[frame.center.eq(source_center), "split_role"])
            if not {"fit", "validation"}.issubset(present):
                audit["errors"].append(f"source center {source_center} missing fit/validation")
                audit["status"] = "failed"
        if audit["status"] != "passed":
            raise RuntimeError(json.dumps({"held_out": held_out, **audit}, sort_keys=True))
        path = output / f"loco_heldout_{held_out}_manifest.csv"
        frame.to_csv(path, index=False)
        audits[held_out] = {**audit, "manifest_path": str(path), "manifest_sha256": sha256_file(path),
                            "held_out_center": held_out, "source_split_seed": int(split_seed)}
        all_rows.append(frame)
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(output / "loco_manifest.csv", index=False)
    report = {"status": "passed", "n_patients": len(subjects), "centers": list(CENTERS), "split_seed": int(split_seed), "held_out": audits}
    (output / "loco_manifest_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
