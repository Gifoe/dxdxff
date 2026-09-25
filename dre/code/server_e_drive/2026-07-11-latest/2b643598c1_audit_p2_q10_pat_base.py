#!/usr/bin/env python3
"""Fail-closed audit for a completed frozen P2_TEMPORAL_Q10 run used by PAT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from neuroez_c.p2_pat_decoder import probability_to_logit
from neuroez_c.p2_pat_protocol import (
    canonicalize_pat_channel_frame, evaluate_pat_ledger, sha256_file, threshold_probability,
)


def _subject_set(path: str) -> set[str]:
    frame = pd.read_csv(path)
    for column in ("subject_id", "subject", "patient_id"):
        if column in frame:
            return set(frame[column].astype(str))
    raise ValueError("Allowed-subject ledger lacks subject_id/subject/patient_id")


def _fold_manifest(path: str) -> dict[int, set[str]]:
    frame = pd.read_csv(path)
    if not {"subject_id", "outer_fold"}.issubset(frame):
        raise ValueError("Fixed fold manifest must contain subject_id and outer_fold")
    if frame.subject_id.duplicated().any():
        raise ValueError("Each patient must occur exactly once in the frozen outer-test manifest")
    return {int(fold): set(group.subject_id.astype(str)) for fold, group in frame.groupby("outer_fold")}


def _find(root: Path, fold: int, names: list[str]) -> Path | None:
    folders = [root/f"fold_{fold}", root/f"fold{fold}", root/f"outer_{fold}", root]
    for folder in folders:
        if not folder.is_dir():
            continue
        for name in names:
            matches = sorted(folder.glob(name))
            if matches:
                return matches[0]
    return None


def _paths(root: Path, fold: int) -> dict[str, Path | None]:
    return {
        "checkpoint": _find(root, fold, ["best_model.pt", "best*.pt", "best*.pth"]),
        "fit": _find(root, fold, [f"fit_channel_predictions_neuroez_v2_fold_{fold}.csv", f"*fit*channel*fold*{fold}*.csv"]),
        "validation": _find(root, fold, [f"val_channel_predictions_neuroez_v2_fold_{fold}.csv", f"*val*channel*fold*{fold}*.csv"]),
        "test": _find(root, fold, [f"test_channel_predictions_neuroez_v2_fold_{fold}.csv", f"*test*channel*fold*{fold}*.csv"]),
    }


def _run_args(root: Path) -> tuple[Path | None, dict]:
    candidates = sorted(root.rglob("run_args_p23.json")) + sorted(root.rglob("run_args_b0_pruned.json")) + sorted(root.rglob("run_args.json"))
    if not candidates:
        return None, {}
    return candidates[0], json.loads(candidates[0].read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2_q10_root", required=True)
    parser.add_argument("--allowed_subjects_ledger", required=True)
    parser.add_argument("--fixed_fold_manifest", required=True)
    parser.add_argument("--require_n_patients", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    root, output = Path(args.p2_q10_root), Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    allowed, expected = _subject_set(args.allowed_subjects_ledger), _fold_manifest(args.fixed_fold_manifest)
    args_path, run_args = _run_args(root)
    config = str(run_args.get("config_name", run_args.get("p23_profile", "")))
    if "P2" not in config.upper() or "Q10" not in config.upper():
        errors.append(f"base profile is not P2_TEMPORAL_Q10: {config!r}")
    if len(allowed) != args.require_n_patients:
        errors.append(f"allowed ledger has {len(allowed)} patients, expected {args.require_n_patients}")
    if set(expected) != {1,2,3,4,5}:
        errors.append(f"frozen manifest folds must be 1..5, got {sorted(expected)}")
    per_fold, tests = [], []
    for fold in range(1, 6):
        paths = _paths(root, fold); missing = [name for name, path in paths.items() if path is None]
        row = {"outer_fold": fold, **{f"{key}_path": str(value or "") for key, value in paths.items()}, "status": "passed"}
        if missing:
            row.update(status="failed", error=f"missing {missing}"); errors.append(f"fold {fold} missing {missing}"); per_fold.append(row); continue
        try:
            fit = canonicalize_pat_channel_frame(pd.read_csv(paths["fit"]))
            validation = canonicalize_pat_channel_frame(pd.read_csv(paths["validation"]))
            test = canonicalize_pat_channel_frame(pd.read_csv(paths["test"]))
            fit_subjects, val_subjects, test_subjects = map(lambda frame: set(frame.subject_id), (fit, validation, test))
            if fit_subjects & val_subjects or fit_subjects & test_subjects or val_subjects & test_subjects:
                raise ValueError("fit/validation/test patients overlap")
            if not val_subjects or val_subjects == fit_subjects:
                raise ValueError("validation ledger is missing or is an in-sample fit ledger")
            if test_subjects != expected[fold]:
                raise ValueError("outer-test patients differ from frozen fold manifest")
            probability = threshold_probability(validation, fold)
            if abs(probability-threshold_probability(test, fold)) > 1e-15:
                raise ValueError("validation and test frozen thresholds differ")
            if "true_count_used_for_prediction" in test and test.true_count_used_for_prediction.astype(str).str.lower().isin({"true","1"}).any():
                raise ValueError("formal base prediction used true K")
            if "threshold_source" in test and not test.threshold_source.astype(str).str.contains("validation|fixed", case=False, regex=True).all():
                raise ValueError("formal base threshold is not validation-only")
            score_column = "base_nez_logit"
            row.update({
                "n_fit_patients": len(fit_subjects), "n_validation_patients": len(val_subjects), "n_test_patients": len(test_subjects),
                "global_threshold_probability": probability, "global_threshold_logit": probability_to_logit(probability),
                "fit_subjects": sorted(fit_subjects), "validation_subjects": sorted(val_subjects), "test_subjects": sorted(test_subjects),
                "checkpoint_sha256": sha256_file(paths["checkpoint"]), "score_column": score_column,
            })
            test = test.copy(); test["global_threshold_probability"] = probability; test["patient_threshold_probability"] = probability
            test["global_threshold_logit"] = probability_to_logit(probability); test["patient_threshold_logit"] = probability_to_logit(probability)
            test["predicted_threshold_residual"] = 0.0; tests.append(test)
        except Exception as error:
            row.update(status="failed", error=str(error)); errors.append(f"fold {fold}: {error}")
        per_fold.append(row)
    combined = pd.concat(tests, ignore_index=True) if tests else pd.DataFrame()
    if not combined.empty:
        if combined[["subject_id","outer_fold"]].drop_duplicates().subject_id.duplicated().any():
            errors.append("a patient occurs in more than one outer-test fold")
        if set(combined.subject_id) != allowed:
            errors.append("outer-test union does not equal allowed patient ledger")
        patients, summary, by_fold, by_center = evaluate_pat_ledger(combined)
        combined.to_csv(output/"p2_pat_base_oof_channel_ledger.csv", index=False)
        patients.to_csv(output/"p2_pat_base_by_patient.csv", index=False)
        summary.to_csv(output/"p2_pat_base_summary.csv", index=False)
        by_fold.to_csv(output/"p2_pat_base_by_fold.csv", index=False)
        by_center.to_csv(output/"p2_pat_base_by_center.csv", index=False)
    else:
        summary = pd.DataFrame()
    pd.DataFrame([{key: json.dumps(value) if isinstance(value, list) else value for key,value in row.items()} for row in per_fold]).to_csv(output/"p2_pat_base_by_fold_audit.csv", index=False)
    manifest = {
        "p2_q10_root": str(root), "profile": "P2_TEMPORAL_Q10", "run_args_path": str(args_path or ""),
        "allowed_subjects_ledger": args.allowed_subjects_ledger, "fixed_fold_manifest": args.fixed_fold_manifest,
        "patient_manifest_hash": sha256_file(args.allowed_subjects_ledger), "fold_manifest_hash": sha256_file(args.fixed_fold_manifest),
        "label_semantics": "NEZ=1,EZ=0", "score_column": "base_nez_logit", "label_column": "label_nez",
        "per_fold": per_fold, "formal_metrics": summary.to_dict("records"),
    }
    (output/"p2_pat_base_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    audit = {"status": "passed" if not errors else "failed", "errors": errors, "profile": "P2_TEMPORAL_Q10", "n_patients": len(allowed), "n_outer_folds": 5, "formal_prediction_uses_true_k": False, "outer_test_labels_used_for_threshold": False, "label_semantics": "NEZ=1,EZ=0", "score_semantics": "base_nez_logit; larger means NEZ"}
    (output/"p2_pat_base_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError("P2-Q10 PAT base audit failed: " + "; ".join(errors))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
