#!/usr/bin/env python3
"""Create a fail-closed P2-Q10/V3-QBC conservative-fusion input manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from neuroez_c.p2_v3_fusion_protocol import (
    align_fold_ledgers, canonicalize_p2_fusion_ledger, canonicalize_v3_fusion_ledger,
    discover_fold_ledger, read_subjects, sha256_file, validate_fold_manifest,
)


def _load_optional_manifest(path: str | None) -> dict | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(f"Manifest not found: {candidate}")
    return json.loads(candidate.read_text(encoding="utf-8"))


def _confirm_v3_profile(root: Path, expected: str, expected_run_id: str | None) -> str:
    candidates = [root / "v3_qbc_protocol_audit.json", root / "v3_qbc_config_audit.json", root / "run_args.json", root / "run_args_b0_pruned.json"]
    text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in candidates if path.is_file()) + "\n" + str(root)
    if expected.lower() not in text.lower():
        raise RuntimeError(f"Cannot prove expected V3 profile {expected!r}; supply a matching V3-QBC root or expected profile metadata")
    if expected_run_id and expected_run_id.lower() not in text.lower():
        raise RuntimeError(f"Cannot prove expected V3 run id {expected_run_id!r}")
    return expected


def _confirm_p2_profile(root: Path) -> str:
    candidates = [root / "run_args_p23.json", root / "run_args.json", root / "run_args_b0_pruned.json", root / "p23_protocol_audit.json"]
    text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in candidates if path.is_file()) + "\n" + str(root)
    if "p2_temporal_q10" not in text.lower():
        raise RuntimeError("Cannot prove P2 root is P2_TEMPORAL_Q10")
    return "P2_TEMPORAL_Q10"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2_q10_root", required=True)
    parser.add_argument("--v3_qbc_root", required=True)
    parser.add_argument("--p2_manifest", default=None)
    parser.add_argument("--v3_manifest", default=None)
    parser.add_argument("--allowed_subjects_ledger", required=True)
    parser.add_argument("--fixed_fold_manifest", required=True)
    parser.add_argument("--require_n_patients", type=int, default=80)
    parser.add_argument("--expected_v3_profile", default="V3_QBC")
    parser.add_argument("--expected_v3_run_id", default=None)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    p2_root, v3_root = Path(args.p2_q10_root), Path(args.v3_qbc_root)
    if not p2_root.is_dir() or not v3_root.is_dir():
        raise FileNotFoundError("P2 and V3 roots must both exist")
    subjects = read_subjects(args.allowed_subjects_ledger)
    if len(subjects) != args.require_n_patients:
        raise RuntimeError(f"Expected {args.require_n_patients} allowed patients, got {len(subjects)}")
    expected_folds = validate_fold_manifest(args.fixed_fold_manifest, subjects)
    p2_manifest, v3_manifest = _load_optional_manifest(args.p2_manifest), _load_optional_manifest(args.v3_manifest)
    p2_profile = _confirm_p2_profile(p2_root)
    profile = _confirm_v3_profile(v3_root, args.expected_v3_profile, args.expected_v3_run_id)
    audit_rows, patient_rows, folds = [], [], []
    for fold in range(1, 6):
        paths = {}
        for model, root, manifest, canonicalize in (("p2", p2_root, p2_manifest, canonicalize_p2_fusion_ledger), ("v3", v3_root, v3_manifest, canonicalize_v3_fusion_ledger)):
            for role in ("validation", "test"):
                path = discover_fold_ledger(root, fold, role, manifest, model)
                frame = canonicalize(pd.read_csv(path), split_role=role)
                if set(frame.subject_id) - subjects:
                    raise RuntimeError(f"{model} fold {fold} {role} ledger includes patients outside allowed ledger")
                if role == "test" and set(frame.subject_id) != expected_folds[fold]:
                    raise RuntimeError(f"{model} fold {fold} test patients differ from fixed outer-fold manifest")
                paths[f"{model}_{role}_ledger_path"] = str(path)
                paths[f"{model}_{role}_sha256"] = sha256_file(path)
                paths[f"n_{model}_{role}_patients"] = int(frame.subject_id.nunique())
                paths[f"n_{model}_{role}_channels"] = int(len(frame))
                paths[f"_{model}_{role}_frame"] = frame
        for role in ("validation", "test"):
            aligned, audit, _ = align_fold_ledgers(paths[f"_p2_{role}_frame"], paths[f"_v3_{role}_frame"], outer_fold=fold, split_role=role)
            audit_rows.append(audit)
            patient_rows.extend(aligned.groupby(["subject_id", "center"], as_index=False).size().assign(outer_fold=fold, split_role=role).to_dict("records"))
        folds.append({key: value for key, value in paths.items() if not key.startswith("_")} | {"outer_fold": fold})
    audit = {"status": "passed", "p2_root": str(p2_root), "v3_root": str(v3_root), "p2_manifest": args.p2_manifest, "v3_manifest": args.v3_manifest, "p2_profile": p2_profile, "v3_profile": profile, "n_patients": len(subjects), "fold_manifest_path": args.fixed_fold_manifest, "fold_manifest_sha256": sha256_file(args.fixed_fold_manifest), "subject_ledger_path": args.allowed_subjects_ledger, "subject_ledger_sha256": sha256_file(args.allowed_subjects_ledger), "folds": folds}
    (output / "p2_v3_fusion_input_manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output / "p2_v3_fusion_alignment_audit.json").write_text(json.dumps({"status": "passed", "rows": audit_rows}, indent=2), encoding="utf-8")
    pd.DataFrame(audit_rows).to_csv(output / "p2_v3_fusion_alignment_by_fold.csv", index=False)
    pd.DataFrame(patient_rows).to_csv(output / "p2_v3_fusion_alignment_by_patient.csv", index=False)
    print(json.dumps({"status": "passed", "n_patients": len(subjects), "n_folds": len(folds), "v3_profile": profile}, indent=2))


if __name__ == "__main__":
    main()
