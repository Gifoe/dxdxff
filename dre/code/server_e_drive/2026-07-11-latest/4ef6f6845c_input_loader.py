"""Load the audited P2/V3 fold ledgers and optional component roots."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from neuroez_c.p2_v3_fusion_protocol import discover_fold_ledger, read_subjects, sha256_file, validate_fold_manifest
from .alignment import align_fold_ledgers, canonicalize_p2_fusion_ledger, canonicalize_v3_fusion_ledger


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_base_folds(input_manifest: str | Path, *, allowed_subjects_ledger: str | Path, fixed_fold_manifest: str | Path, require_n_patients: int, expected_n_channels: int, strict: bool) -> tuple[dict, dict, dict]:
    manifest = load_json(input_manifest)
    subjects = read_subjects(allowed_subjects_ledger)
    if len(subjects) != require_n_patients:
        raise RuntimeError(f"Expected {require_n_patients} patients, got {len(subjects)}")
    expected_test = validate_fold_manifest(fixed_fold_manifest, subjects)
    folds: dict[int, dict[str, pd.DataFrame]] = {}
    audits = []
    for item in manifest.get("folds", []):
        fold = int(item["outer_fold"])
        fold_data = {}
        for role in ("validation", "test"):
            p2_path = Path(item[f"p2_{role}_ledger_path"])
            v3_path = Path(item[f"v3_{role}_ledger_path"])
            for model, path in (("p2", p2_path), ("v3", v3_path)):
                expected_hash = item.get(f"{model}_{role}_sha256")
                if expected_hash and sha256_file(path) != expected_hash:
                    raise RuntimeError(f"Fold {fold} {model} {role} hash mismatch")
            p2 = canonicalize_p2_fusion_ledger(pd.read_csv(p2_path), split_role=role)
            v3 = canonicalize_v3_fusion_ledger(pd.read_csv(v3_path), split_role=role)
            aligned, audit, _ = align_fold_ledgers(p2, v3, outer_fold=fold, split_role=role)
            if role == "test" and set(aligned.subject_id) != expected_test[fold]:
                raise RuntimeError(f"Fold {fold} test patients differ from fixed manifest")
            fold_data[role] = aligned
            audits.append(audit)
        folds[fold] = fold_data
    test = pd.concat([data["test"] for data in folds.values()], ignore_index=True)
    errors = []
    if set(folds) != {1, 2, 3, 4, 5}: errors.append("outer folds are not exactly 1..5")
    if test.subject_id.nunique() != require_n_patients: errors.append("outer-test patient count mismatch")
    if len(test) != expected_n_channels: errors.append("outer-test channel count mismatch")
    if strict and errors:
        raise RuntimeError("; ".join(errors))
    audit = {
        "status": "passed" if not errors else "failed", "errors": errors, "n_patients": int(test.subject_id.nunique()),
        "n_test_channels": int(len(test)), "n_outer_folds": len(folds), "alignment_rows": audits,
        "input_manifest": str(input_manifest), "input_manifest_sha256": sha256_file(input_manifest),
        "allowed_subjects_ledger": str(allowed_subjects_ledger), "fixed_fold_manifest": str(fixed_fold_manifest),
    }
    return folds, manifest, audit


def load_optional_model(root: str | None, base_folds: dict, *, model_name: str) -> tuple[dict | None, dict]:
    if not root:
        return None, {"model": model_name, "status": "UNAVAILABLE_MISSING_INPUT"}
    path = Path(root)
    if not path.is_dir():
        return None, {"model": model_name, "status": "UNAVAILABLE_MISSING_INPUT", "root": str(path)}
    output = {}
    audit_rows = []
    for fold in range(1, 6):
        output[fold] = {}
        for role in ("validation", "test"):
            ledger_path = discover_fold_ledger(path, fold, role)
            optional = canonicalize_p2_fusion_ledger(pd.read_csv(ledger_path), split_role=role)
            reference = base_folds[fold][role]
            # Align against the P2 half of the already-audited common ledger.
            pseudo = reference.rename(columns={"p2_score_nez": "score_nez", "p2_score_ez": "score_ez"}).copy()
            pseudo["source_model"] = "P2_REFERENCE"; pseudo["split_role"] = role
            aligned, audit, _ = align_fold_ledgers(optional, pseudo, outer_fold=fold, split_role=role)
            output[fold][role] = aligned.rename(columns={"p2_score_nez": "model_score_nez"})
            audit_rows.append(audit)
    config_candidates = [path / name for name in ("run_args_p23.json", "run_args.json", "run_args_b0_pruned.json")]
    config_path = next((candidate for candidate in config_candidates if candidate.is_file()), None)
    return output, {"model": model_name, "status": "available", "root": str(path), "config_path": str(config_path) if config_path else None, "config": load_json(config_path) if config_path else None, "alignment": audit_rows}


def config_difference(audit_items: list[dict]) -> dict:
    available = [item for item in audit_items if item.get("config")]
    keys = sorted(set().union(*(item["config"].keys() for item in available))) if available else []
    differences = {}
    for key in keys:
        values = {item["model"]: item["config"].get(key) for item in available}
        if len({json.dumps(value, sort_keys=True, default=str) for value in values.values()}) > 1:
            differences[key] = values
    return {"models": audit_items, "different_fields": differences, "pure_q10_ablation_confirmed": False}

