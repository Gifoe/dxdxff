#!/usr/bin/env python3
"""Run fixed 90/10 P2-Q10 + V3-QBC probability fusion without model training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from neuroez_c.p2_v3_conservative_fusion import (
    FORMAL_DECISION_RULE, LOCKED_P2_WEIGHT, LOCKED_V3_WEIGHT, TRUEK_DECISION_RULE,
    conservative_probability_fusion, require_locked_weights,
)
from neuroez_c.p2_v3_fusion_protocol import (
    align_fold_ledgers, canonicalize_p2_fusion_ledger, canonicalize_v3_fusion_ledger,
    read_subjects, sha256_file, validate_fold_manifest,
)
from neuroez_c.p2_v3_fusion_reporting import (
    aggregate_patients, evaluate_probability_predictions, oracle_patient_thresholds, select_validation_threshold,
)


def _load(path: str | Path, model: str, role: str) -> pd.DataFrame:
    source = pd.read_csv(path)
    return canonicalize_p2_fusion_ledger(source, split_role=role) if model == "p2" else canonicalize_v3_fusion_ledger(source, split_role=role)


def _annotate_predictions(frame: pd.DataFrame, *, threshold: float, p2_threshold: float, v3_threshold: float, p2_weight: float, v3_weight: float, analysis_status: str) -> pd.DataFrame:
    result = frame.copy()
    result["selected_validation_threshold"] = threshold
    result["p2_selected_validation_threshold"] = p2_threshold
    result["v3_selected_validation_threshold"] = v3_threshold
    result["legal_pred_nez"] = (result.fused_score_nez >= threshold).astype(int)
    result["legal_pred_ez"] = 1 - result.legal_pred_nez
    result["p2_legal_pred_nez"] = (result.p2_score_nez >= p2_threshold).astype(int)
    result["v3_legal_pred_nez"] = (result.v3_score_nez >= v3_threshold).astype(int)
    for _, group in result.groupby("subject_id", sort=False):
        k = int(group.label_ez.sum())
        idx = group.index.to_numpy()[np.argsort(-group.fused_score_ez.to_numpy(float), kind="mergesort")[:k]]
        result.loc[group.index, "truek_pred_nez"] = 1
        result.loc[idx, "truek_pred_nez"] = 0
    result["truek_pred_nez"] = result.truek_pred_nez.astype(int)
    result["truek_pred_ez"] = 1 - result.truek_pred_nez
    result["p2_weight"] = p2_weight
    result["v3_weight"] = v3_weight
    result["decision_rule"] = FORMAL_DECISION_RULE
    result["truek_decision_rule"] = TRUEK_DECISION_RULE
    result["threshold_source"] = "outer_fold_validation_only"
    result["weight_selection"] = "locked_before_execution"
    result["true_count_used_for_prediction"] = False
    result["patient_oracle_used_for_prediction"] = False
    result["center_specific_threshold"] = False
    result["patient_specific_threshold"] = False
    result["analysis_status"] = analysis_status
    return result


def _label_patient_rows(frame: pd.DataFrame, *, model: str, threshold: float | None, truek: bool, analysis_status: str) -> pd.DataFrame:
    score_column = {"P2-Q10": "p2_score_nez", "V3-QBC": "v3_score_nez", "P2-Q10-CF": "fused_score_nez"}[model]
    patients = evaluate_probability_predictions(frame, score_nez_column=score_column, threshold=threshold, truek=truek)
    patients["model"] = model
    patients["decision_rule"] = TRUEK_DECISION_RULE if truek else FORMAL_DECISION_RULE
    patients["threshold_source"] = "not_used" if truek else "outer_fold_validation_only"
    patients["analysis_status"] = "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE" if truek else analysis_status
    patients["true_count_used_for_prediction"] = bool(truek)
    patients["formal_prediction"] = not truek
    return patients


def _table(patients: pd.DataFrame, *, by: str | None = None) -> pd.DataFrame:
    rows = []
    for model, group in patients.groupby("model", sort=True):
        summary = aggregate_patients(group, by)
        summary["model"] = model
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def _model_comparison_by(table: pd.DataFrame, *, by: str) -> pd.DataFrame:
    """Put P2, V3, and fusion on the same patient-equal aggregation rows."""
    metrics = [
        "patient_macro_f1", "patient_macro_ez_f1", "patient_macro_nez_f1",
        "patient_macro_auprc_ez", "patient_macro_ez_mrr", "top1_is_ez_rate",
        "predicted_ez_fraction_mae", "predicted_ez_count_mae",
    ]
    pivot = table.pivot(index=by, columns="model", values=metrics)
    pivot.columns = [f"{model.lower().replace('-', '_')}_{metric.removeprefix('patient_')}" for metric, model in pivot.columns]
    result = pivot.reset_index()
    for metric in metrics:
        suffix = metric.removeprefix("patient_")
        fusion = f"p2_q10_cf_{suffix}"
        p2 = f"p2_q10_{suffix}"
        if fusion in result and p2 in result:
            result[f"fusion_minus_p2_{suffix}"] = result[fusion] - result[p2]
    return result


def _reproduction_audit(comparison: pd.DataFrame, truek_comparison: pd.DataFrame, ledger: pd.DataFrame) -> dict:
    expected = {"p2_legal": .628464, "fusion_legal": .643265, "p2_truek": .648767, "fusion_truek": .650398}
    legal = comparison.set_index("model").patient_macro_f1.to_dict()
    truek = truek_comparison.set_index("model").patient_macro_f1.to_dict()
    actual = {"p2_legal": float(legal.get("P2-Q10", np.nan)), "fusion_legal": float(legal.get("P2-Q10-CF", np.nan)), "p2_truek": float(truek.get("P2-Q10", np.nan)), "fusion_truek": float(truek.get("P2-Q10-CF", np.nan))}
    tolerance = .0005
    passed = {key: bool(np.isfinite(actual[key]) and abs(actual[key] - expected[key]) <= tolerance) for key in expected}
    return {"expected_p2_legal_macro_f1": expected["p2_legal"], "actual_p2_legal_macro_f1": actual["p2_legal"], "expected_fusion_legal_macro_f1": expected["fusion_legal"], "actual_fusion_legal_macro_f1": actual["fusion_legal"], "expected_p2_truek_macro_f1": expected["p2_truek"], "actual_p2_truek_macro_f1": actual["p2_truek"], "expected_fusion_truek_macro_f1": expected["fusion_truek"], "actual_fusion_truek_macro_f1": actual["fusion_truek"], "legal_tolerance": tolerance, "truek_tolerance": tolerance, "n_patients_expected": 80, "n_patients_actual": int(ledger.subject_id.nunique()), "n_channels_reference": 7635, "n_channels_actual": int(len(ledger)), "p2_reproduction_passed": passed["p2_legal"], "fusion_reproduction_passed": passed["fusion_legal"], "truek_reproduction_passed": passed["p2_truek"] and passed["fusion_truek"], "overall_passed": all(passed.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2_q10_root", required=True)
    parser.add_argument("--v3_qbc_root", required=True)
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--allowed_subjects_ledger", required=True)
    parser.add_argument("--fixed_fold_manifest", required=True)
    parser.add_argument("--require_n_patients", type=int, default=80)
    parser.add_argument("--p2_weight", type=float, default=LOCKED_P2_WEIGHT)
    parser.add_argument("--v3_weight", type=float, default=LOCKED_V3_WEIGHT)
    parser.add_argument("--diagnostic_allow_nonlocked_weights", action="store_true")
    parser.add_argument("--threshold_step", type=float, default=.005)
    parser.add_argument("--analysis_status", choices=["EXPLORATORY_LOCKED_FUSION", "PRIMARY_CONFIRMATORY_LOCKED_FUSION"], default="EXPLORATORY_LOCKED_FUSION")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--max_outer_folds", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if not np.isclose(args.threshold_step, .005, atol=1e-12):
        raise ValueError("P2-Q10-CF fixes threshold_step at 0.005")
    weight_status = require_locked_weights(args.p2_weight, args.v3_weight, diagnostic_allow_nonlocked_weights=args.diagnostic_allow_nonlocked_weights)
    analysis_status = weight_status if weight_status.startswith("DIAGNOSTIC") else args.analysis_status
    manifest = json.loads(Path(args.input_manifest).read_text(encoding="utf-8"))
    subjects = read_subjects(args.allowed_subjects_ledger)
    if len(subjects) != args.require_n_patients:
        raise RuntimeError(f"Expected {args.require_n_patients} patients, got {len(subjects)}")
    fold_subjects = validate_fold_manifest(args.fixed_fold_manifest, subjects)
    if args.dry_run:
        print(json.dumps({"status": "passed", "mode": "dry_run", "n_patients": len(subjects), "n_outer_folds": len(fold_subjects), "p2_weight": args.p2_weight, "v3_weight": args.v3_weight, "analysis_status": analysis_status, "threshold_step": args.threshold_step}, indent=2))
        return
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    max_fold = args.max_outer_folds or 5
    aligned_audits, aligned_by_patient, searches, selected_rows, test_frames, legal_rows, truek_rows, oracle_rows = [], [], [], [], [], [], [], []
    for item in manifest.get("folds", []):
        fold = int(item["outer_fold"])
        if fold > max_fold:
            continue
        # The input audit binds every source ledger by content hash.  Recheck it
        # here so a later file replacement cannot silently change the analysis.
        for model in ("p2", "v3"):
            for role in ("validation", "test"):
                path_key = f"{model}_{role}_ledger_path"
                hash_key = f"{model}_{role}_sha256"
                if hash_key in item and sha256_file(item[path_key]) != item[hash_key]:
                    raise RuntimeError(f"Fold {fold}: {model} {role} ledger hash differs from the audited manifest")
        p2_val, v3_val = _load(item["p2_validation_ledger_path"], "p2", "validation"), _load(item["v3_validation_ledger_path"], "v3", "validation")
        p2_test, v3_test = _load(item["p2_test_ledger_path"], "p2", "test"), _load(item["v3_test_ledger_path"], "v3", "test")
        validation, audit_val, _ = align_fold_ledgers(p2_val, v3_val, outer_fold=fold, split_role="validation")
        test, audit_test, _ = align_fold_ledgers(p2_test, v3_test, outer_fold=fold, split_role="test")
        if set(test.subject_id) != fold_subjects[fold]:
            raise RuntimeError(f"Fold {fold}: test subject set changed after alignment")
        for frame in (validation, test):
            frame["fused_score_nez"], frame["fused_score_ez"] = conservative_probability_fusion(frame.p2_score_nez, frame.v3_score_nez, p2_weight=args.p2_weight, v3_weight=args.v3_weight)
        for model, score in (("P2-Q10", "p2_score_nez"), ("V3-QBC", "v3_score_nez"), ("P2-Q10-CF", "fused_score_nez")):
            threshold, search = select_validation_threshold(validation, score_nez_column=score)
            search["outer_fold"], search["model"] = fold, model
            searches.append(search)
            selected_rows.append({"outer_fold": fold, "model": model, **search.loc[search.selected].iloc[0].to_dict()})
            target = test if model == "P2-Q10-CF" else test.copy()
            legal_rows.append(_label_patient_rows(target, model=model, threshold=threshold, truek=False, analysis_status=analysis_status))
            truek_rows.append(_label_patient_rows(target, model=model, threshold=None, truek=True, analysis_status=analysis_status))
        selected = {row["model"]: float(row["threshold"]) for row in selected_rows if int(row["outer_fold"]) == fold}
        annotated = _annotate_predictions(test, threshold=selected["P2-Q10-CF"], p2_threshold=selected["P2-Q10"], v3_threshold=selected["V3-QBC"], p2_weight=args.p2_weight, v3_weight=args.v3_weight, analysis_status=analysis_status)
        test_frames.append(annotated)
        oracle_rows.append(oracle_patient_thresholds(test, score_nez_column="fused_score_nez"))
        aligned_audits.extend([audit_val, audit_test])
        aligned_by_patient.extend(test.groupby(["subject_id", "center"], as_index=False).size().assign(outer_fold=fold, split_role="test").to_dict("records"))
    if len(test_frames) != max_fold:
        raise RuntimeError("Missing fold outputs; cannot write partial fusion silently")
    ledger = pd.concat(test_frames, ignore_index=True)
    legal = pd.concat(legal_rows, ignore_index=True)
    truek = pd.concat(truek_rows, ignore_index=True)
    oracle = pd.concat(oracle_rows, ignore_index=True)
    legal_summary, truek_summary = _table(legal), _table(truek)
    legal_fold, legal_center = _table(legal, by="outer_fold"), _table(legal, by="center")
    truek_fold, truek_center = _table(truek, by="outer_fold"), _table(truek, by="center")
    comparison = legal_summary[["model", "patient_macro_f1", "patient_macro_ez_f1", "patient_macro_nez_f1", "patient_macro_auprc_ez", "patient_macro_ez_mrr", "top1_is_ez_rate"]].copy()
    comparison["decision_rule"] = FORMAL_DECISION_RULE; comparison["true_count_used"] = False; comparison["analysis_status"] = analysis_status
    truek_comparison = truek_summary[["model", "patient_macro_f1", "patient_macro_ez_f1", "patient_macro_nez_f1", "patient_macro_auprc_ez", "patient_macro_ez_mrr", "top1_is_ez_rate"]].copy()
    truek_comparison["decision_rule"] = TRUEK_DECISION_RULE; truek_comparison["true_count_used"] = True; truek_comparison["analysis_status"] = "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE"
    comparison = pd.concat([comparison, truek_comparison], ignore_index=True)
    reproduction = _reproduction_audit(legal_summary, truek_summary, ledger)
    expected_fold_scores = {
        1: {"p2": .594415, "fusion": .597769, "threshold": .450},
        2: {"p2": .650049, "fusion": .666682, "threshold": .365},
        3: {"p2": .613628, "fusion": .651834, "threshold": .425},
        4: {"p2": .615624, "fusion": .620085, "threshold": .430},
        5: {"p2": .668729, "fusion": .677972, "threshold": .445},
    }
    selected_frame = pd.DataFrame(selected_rows)
    fold_diagnostics = []
    for fold, values in expected_fold_scores.items():
        if fold > max_fold:
            continue
        actual_scores = legal_fold[legal_fold.outer_fold.eq(fold)].set_index("model").patient_macro_f1.to_dict()
        actual_threshold = selected_frame[(selected_frame.outer_fold.eq(fold)) & (selected_frame.model.eq("P2-Q10-CF"))].threshold.iloc[0]
        fold_diagnostics.append({
            "outer_fold": fold,
            "expected_p2_legal_macro_f1": values["p2"],
            "actual_p2_legal_macro_f1": float(actual_scores.get("P2-Q10", np.nan)),
            "expected_fusion_legal_macro_f1": values["fusion"],
            "actual_fusion_legal_macro_f1": float(actual_scores.get("P2-Q10-CF", np.nan)),
            "expected_fusion_threshold": values["threshold"],
            "actual_fusion_threshold": float(actual_threshold),
        })
    reproduction["fold_diagnostics"] = fold_diagnostics
    reproduction["input_version_note"] = "A failed reproduction audit is reported only; it never changes the locked weights or validation-only threshold selection."
    for item in manifest.get("folds", []):
        fold = int(item["outer_fold"])
        matching = selected_frame[(selected_frame.outer_fold.eq(fold)) & (selected_frame.model.eq("P2-Q10-CF"))]
        if not matching.empty:
            item["selected_fusion_validation_threshold"] = float(matching.threshold.iloc[0])
            item["n_validation_patients"] = int(item["n_p2_validation_patients"])
            item["n_test_patients"] = int(item["n_p2_test_patients"])
            item["n_validation_channels"] = int(item["n_p2_validation_channels"])
            item["n_test_channels"] = int(item["n_p2_test_channels"])
    run_args = vars(args) | {"resolved_analysis_status": analysis_status, "p2_weight": args.p2_weight, "v3_weight": args.v3_weight, "input_manifest_sha256": sha256_file(args.input_manifest)}
    (output / "run_args.json").write_text(json.dumps(run_args, indent=2), encoding="utf-8")
    (output / "p2_v3_fusion_input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "p2_v3_fusion_alignment_audit.json").write_text(json.dumps({"status": "passed", "rows": aligned_audits}, indent=2), encoding="utf-8")
    pd.DataFrame(aligned_audits).to_csv(output / "p2_v3_fusion_alignment_by_fold.csv", index=False)
    pd.DataFrame(aligned_by_patient).to_csv(output / "p2_v3_fusion_alignment_by_patient.csv", index=False)
    pd.concat(searches, ignore_index=True).to_csv(output / "p2_v3_fusion_threshold_search.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(output / "p2_v3_fusion_selected_thresholds.csv", index=False)
    ledger.to_csv(output / "p2_v3_fusion_oof_channel_ledger.csv", index=False)
    legal[legal.model.eq("P2-Q10-CF")].to_csv(output / "p2_v3_fusion_legal_by_patient.csv", index=False)
    legal_fold[legal_fold.model.eq("P2-Q10-CF")].to_csv(output / "p2_v3_fusion_legal_by_fold.csv", index=False)
    legal_center[legal_center.model.eq("P2-Q10-CF")].to_csv(output / "p2_v3_fusion_legal_by_center.csv", index=False)
    legal_summary[legal_summary.model.eq("P2-Q10-CF")].to_csv(output / "p2_v3_fusion_legal_summary.csv", index=False)
    truek[truek.model.eq("P2-Q10-CF")].to_csv(output / "p2_v3_fusion_truek_by_patient.csv", index=False)
    truek_fold[truek_fold.model.eq("P2-Q10-CF")].to_csv(output / "p2_v3_fusion_truek_by_fold.csv", index=False)
    truek_center[truek_center.model.eq("P2-Q10-CF")].to_csv(output / "p2_v3_fusion_truek_by_center.csv", index=False)
    truek_summary[truek_summary.model.eq("P2-Q10-CF")].to_csv(output / "p2_v3_fusion_truek_summary.csv", index=False)
    oracle.to_csv(output / "p2_v3_fusion_oracle_by_patient.csv", index=False)
    aggregate_patients(oracle).to_csv(output / "p2_v3_fusion_oracle_summary.csv", index=False)
    comparison.to_csv(output / "p2_v3_fusion_model_comparison.csv", index=False)
    _model_comparison_by(legal_fold, by="outer_fold").to_csv(output / "p2_v3_fusion_by_fold.csv", index=False)
    _model_comparison_by(legal_center, by="center").to_csv(output / "p2_v3_fusion_by_center.csv", index=False)
    (output / "p2_v3_fusion_reproduction_audit.json").write_text(json.dumps(reproduction, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "n_patients": int(ledger.subject_id.nunique()), "n_channels": len(ledger), "actual_fusion_legal_macro_f1": reproduction["actual_fusion_legal_macro_f1"], "reproduction_passed": reproduction["overall_passed"]}, indent=2))


if __name__ == "__main__":
    main()
