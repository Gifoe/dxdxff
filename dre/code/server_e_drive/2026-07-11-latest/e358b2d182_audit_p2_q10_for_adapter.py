#!/usr/bin/env python3
"""Fail-closed audit for a completed P2_TEMPORAL_Q10 run before adaptation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from neuroez_c.p2_q10_lzu_adapter_protocol import canonicalize_channel_frame, evaluate_channel_ledger, frozen_threshold, sha256_file


def _subjects(path: str) -> set[str]:
    frame = pd.read_csv(path)
    for name in ("subject_id", "subject", "patient_id"):
        if name in frame:
            return set(frame[name].astype(str))
    raise ValueError("Allowed-subjects ledger has no subject_id/subject/patient_id column")


def _manifest(path: str) -> dict[int, set[str]]:
    frame = pd.read_csv(path)
    if not {"subject_id", "outer_fold"}.issubset(frame):
        raise ValueError("Fixed fold manifest must contain subject_id and outer_fold")
    if frame.subject_id.duplicated().any():
        raise ValueError("Fixed fold manifest must contain each subject exactly once as outer test")
    return {int(fold): set(group.subject_id.astype(str)) for fold, group in frame.groupby("outer_fold")}


def _first(root: Path, patterns: list[str]) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(root.glob(pattern))
    return sorted(set(matches))[0] if matches else None


def _paths(root: Path, fold: int) -> dict[str, Path | None]:
    folder_patterns = [f"fold_{fold}", f"fold{fold}", f"outer_{fold}"]
    folders = [root / name for name in folder_patterns if (root / name).is_dir()] + [root]
    def find(names: list[str]) -> Path | None:
        for folder in folders:
            found = _first(folder, names)
            if found: return found
        return None
    return {
        "checkpoint": find(["best_model.pt", "best_b0_pruned_model.pth", "best*.pt", "best*.pth"]),
        "fit": find([f"fit_channel_predictions_neuroez_v2_fold_{fold}.csv", f"fit_channel_predictions_fold_{fold}.csv", f"fold_{fold}_fit_channel_predictions.csv"]),
        "validation": find([f"val_channel_predictions_neuroez_v2_fold_{fold}.csv", f"validation_channel_predictions_neuroez_v2_fold_{fold}.csv", f"val_channel_predictions_fold_{fold}.csv"]),
        "test": find([f"test_channel_predictions_neuroez_v2_fold_{fold}.csv", f"outer_test_channel_predictions_fold_{fold}.csv", f"test_channel_predictions_fold_{fold}.csv"]),
    }


def _validate_p2_configuration(root: Path) -> list[str]:
    candidates = (sorted(root.rglob("run_args_b0_pruned.json")) + sorted(root.rglob("run_args_p23.json")) + sorted(root.rglob("run_args.json")))
    if not candidates:
        return ["missing run_args_b0_pruned.json/run_args_p23.json/run_args.json needed to prove frozen P2-Q10 protocol"]
    args = json.loads(candidates[0].read_text(encoding="utf-8"))
    config = str(args.get("config_name", ""))
    errors: list[str] = []
    if "P2" not in config.upper() or "Q10" not in config.upper():
        errors.append(f"run args config_name is not P2-Q10: {config!r}")
    for key in ("use_a9v8_lcbo", "scope_refit_full_outer_train", "use_v3_qbc", "use_v3_rcc", "use_cane_path_cp_nez"):
        if bool(args.get(key, False)):
            errors.append(f"P2-Q10 base has forbidden {key}=True")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2_q10_root", required=True); parser.add_argument("--allowed_subjects_ledger", required=True)
    parser.add_argument("--fixed_fold_manifest", required=True); parser.add_argument("--require_n_patients", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args(); root, output = Path(args.p2_q10_root), Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    errors: list[str] = _validate_p2_configuration(root); allowed = _subjects(args.allowed_subjects_ledger); expected = _manifest(args.fixed_fold_manifest)
    if len(allowed) != args.require_n_patients: errors.append(f"allowed ledger has {len(allowed)}, expected {args.require_n_patients}")
    if set(expected) != {1,2,3,4,5}: errors.append(f"fixed fold manifest must contain folds 1..5, got {sorted(expected)}")
    per_fold: list[dict] = []; all_test: list[pd.DataFrame] = []
    for fold in range(1, 6):
        paths = _paths(root, fold); missing = [name for name, path in paths.items() if path is None]
        row: dict = {"outer_fold": fold, **{f"{name}_path": str(path) if path else "" for name, path in paths.items()}, "status": "passed"}
        if missing:
            row.update({"status": "failed", "error": f"missing {missing}"}); errors.append(f"fold {fold} missing {missing}"); per_fold.append(row); continue
        try:
            test = canonicalize_channel_frame(pd.read_csv(paths["test"]))
            validation = canonicalize_channel_frame(pd.read_csv(paths["validation"]))
            fit = canonicalize_channel_frame(pd.read_csv(paths["fit"]))
            if "outer_fold" not in test or set(test.outer_fold.astype(int)) != {fold}: raise ValueError("test ledger outer_fold mismatch")
            test_subjects = set(test.subject_id.astype(str))
            if test_subjects != expected[fold]: raise ValueError(f"test subjects do not match frozen manifest: missing={sorted(expected[fold]-test_subjects)} extra={sorted(test_subjects-expected[fold])}")
            fit_subjects, val_subjects = set(fit.subject_id.astype(str)), set(validation.subject_id.astype(str))
            if fit_subjects & val_subjects or fit_subjects & test_subjects or val_subjects & test_subjects:
                raise ValueError("fit/validation/test patients are not pairwise disjoint")
            threshold = frozen_threshold(validation, fold)
            test_threshold = frozen_threshold(test, fold)
            if threshold != test_threshold: raise ValueError("test ledger threshold differs from validation frozen threshold")
            if "threshold_source" in test and not test.threshold_source.astype(str).str.contains("validation|fixed", case=False, regex=True).all(): raise ValueError("test ledger threshold source is not validation-only/fixed")
            if "true_count_used_for_prediction" in test and test.true_count_used_for_prediction.astype(str).str.lower().isin(["true", "1"]).any(): raise ValueError("formal test prediction used true count")
            row.update({"n_fit_patients": int(fit.subject_id.nunique()), "n_validation_patients": int(validation.subject_id.nunique()), "n_test_patients": int(test.subject_id.nunique()), "frozen_threshold": threshold, "checkpoint_sha256": sha256_file(paths["checkpoint"]), "status": "passed"})
            all_test.append(test)
        except Exception as exc:
            row.update({"status": "failed", "error": str(exc)}); errors.append(f"fold {fold}: {exc}")
        per_fold.append(row)
    if all_test:
        combined = pd.concat(all_test, ignore_index=True)
        subject_folds = combined[["subject_id", "outer_fold"]].drop_duplicates()
        if subject_folds.subject_id.duplicated().any(): errors.append("outer-test subject appears in more than one test ledger")
        if set(combined.subject_id.astype(str)) != allowed: errors.append("combined test patients do not equal allowed subject ledger")
        formal_patients, formal_summary, formal_fold, formal_center = evaluate_channel_ledger(combined, logit_column="base_nez_logit")
        truek_patients, truek_summary, _, truek_center = evaluate_channel_ledger(combined, logit_column="base_nez_logit", truek=True)
        formal_summary.to_csv(output / "p2_q10_adapter_base_summary.csv", index=False); formal_fold.to_csv(output / "p2_q10_adapter_base_by_fold.csv", index=False); formal_center.to_csv(output / "p2_q10_adapter_base_by_center.csv", index=False)
        combined.to_csv(output / "p2_q10_adapter_base_oof_channel_ledger.csv", index=False)
    else:
        formal_summary = pd.DataFrame(); truek_summary = pd.DataFrame(); truek_center = pd.DataFrame()
    pd.DataFrame(per_fold).to_csv(output / "p2_q10_adapter_base_audit_by_fold.csv", index=False)
    manifest = {"p2_q10_root": str(root), "allowed_subjects_ledger": str(args.allowed_subjects_ledger), "fixed_fold_manifest": str(args.fixed_fold_manifest), "patient_manifest_hash": sha256_file(args.allowed_subjects_ledger), "fold_manifest_hash": sha256_file(args.fixed_fold_manifest), "label_semantics": "NEZ=1,EZ=0", "score_column": "base_nez_logit", "P2_profile": "P2_TEMPORAL_Q10", "per_fold": per_fold, "formal_metrics": formal_summary.to_dict("records"), "truek_diagnostic_metrics": truek_summary.to_dict("records"), "truek_analysis_status": "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE"}
    (output / "p2_q10_adapter_base_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    audit = {"status": "passed" if not errors else "failed", "errors": errors, "n_allowed_patients": len(allowed), "n_outer_folds": 5, "base_root": str(root), "formal_prediction_uses_true_k": False, "profile": "P2_TEMPORAL_Q10", "scope_enabled": False, "robust_tail_enabled": False, "causal_enabled": False, "center_specific_threshold": False}
    (output / "p2_q10_adapter_base_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if errors: raise RuntimeError("P2-Q10 adapter base audit failed: " + "; ".join(errors))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__": main()
