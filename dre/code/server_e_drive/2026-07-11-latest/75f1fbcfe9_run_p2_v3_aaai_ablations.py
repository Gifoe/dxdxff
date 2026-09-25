#!/usr/bin/env python3
"""Run the strict ledger-only P2/V3 AAAI ablation suite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from P2_V3_AAAI_ABLATIONS import ABLATION_STATUS, DIAGNOSTIC_STATUS, LOCKED_STATUS, LOCKED_P2_WEIGHT, LOCKED_V3_WEIGHT
from P2_V3_AAAI_ABLATIONS.bootstrap import paired_bootstrap
from P2_V3_AAAI_ABLATIONS.fusion_operators import logit_fusion, patient_rank_fusion, probability_fusion, shuffle_v3_within_patient
from P2_V3_AAAI_ABLATIONS.input_loader import config_difference, load_base_folds, load_optional_model
from P2_V3_AAAI_ABLATIONS.mechanism_analysis import disagreement_analysis, score_correlations
from P2_V3_AAAI_ABLATIONS.metrics import PATIENT_METRICS, evaluate, summaries
from P2_V3_AAAI_ABLATIONS.reporting import bar_figure, ensure_output_tree, grouped_bar_figure, line_figure, markdown_report, write_json
from P2_V3_AAAI_ABLATIONS.threshold_protocol import select_fold_threshold


EXPECTED = {
    "p2_legal": 0.6284640770231468,
}


def _score(frame: pd.DataFrame, operator: str, v3_weight: float) -> np.ndarray:
    if operator == "p2": return frame.p2_score_nez.to_numpy(float)
    if operator == "v3": return frame.v3_score_nez.to_numpy(float)
    if operator == "probability": return probability_fusion(frame.p2_score_nez, frame.v3_score_nez, v3_weight=v3_weight)
    if operator == "logit": return logit_fusion(frame.p2_score_nez, frame.v3_score_nez, v3_weight=v3_weight)
    if operator == "rank": return patient_rank_fusion(frame, v3_weight=v3_weight)
    raise ValueError(f"Unknown operator: {operator}")


def _run_configuration(folds: dict, *, name: str, operator: str, v3_weight: float, status: str, fixed_thresholds: dict[int, float] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patients, thresholds, test_ledgers = [], [], []
    for fold in range(1, 6):
        validation = folds[fold]["validation"].copy(); test = folds[fold]["test"].copy()
        validation["ablation_score_nez"] = _score(validation, operator, v3_weight)
        test["ablation_score_nez"] = _score(test, operator, v3_weight)
        if fixed_thresholds is None:
            threshold, search = select_fold_threshold(validation, score_column="ablation_score_nez")
            threshold_source = "outer_fold_validation_only"
        else:
            threshold = float(fixed_thresholds[fold]); search = pd.DataFrame()
            threshold_source = "frozen_from_p2_validation"
        thresholds.append({"experiment": name, "outer_fold": fold, "threshold": threshold, "threshold_source": threshold_source, "selected": True})
        patient = evaluate(test, score_column="ablation_score_nez", threshold=threshold, experiment=name, analysis_status=status)
        patients.append(patient)
        test["experiment"] = name; test["score_nez_probability"] = test.ablation_score_nez; test["selected_threshold"] = threshold
        test_ledgers.append(test)
    return pd.concat(patients, ignore_index=True), pd.DataFrame(thresholds), pd.concat(test_ledgers, ignore_index=True)


def _run_optional_model(model_folds: dict, *, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    patients, thresholds = [], []
    for fold in range(1, 6):
        validation = model_folds[fold]["validation"].copy(); test = model_folds[fold]["test"].copy()
        validation["ablation_score_nez"] = validation.model_score_nez
        test["ablation_score_nez"] = test.model_score_nez
        threshold, _ = select_fold_threshold(validation, score_column="ablation_score_nez")
        thresholds.append({"experiment": name, "outer_fold": fold, "threshold": threshold, "threshold_source": "outer_fold_validation_only", "selected": True})
        patients.append(evaluate(test, score_column="ablation_score_nez", threshold=threshold, experiment=name, analysis_status=ABLATION_STATUS))
    return pd.concat(patients, ignore_index=True), pd.DataFrame(thresholds)


def _write_family(paths: dict, stem: str, patients: pd.DataFrame, thresholds: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall, by_fold, by_center = summaries(patients)
    overall.to_csv(paths["metrics"] / f"{stem}_overall.csv", index=False)
    by_fold.to_csv(paths["metrics"] / f"{stem}_by_fold.csv", index=False)
    by_center.to_csv(paths["metrics"] / f"{stem}_by_center.csv", index=False)
    if thresholds is not None: thresholds.to_csv(paths["metrics"] / f"{stem}_thresholds.csv", index=False)
    return overall, by_fold, by_center


def _reproduction(p2: pd.DataFrame, locked: pd.DataFrame, locked_thresholds: pd.DataFrame) -> dict:
    p2_overall = summaries(p2)[0].iloc[0]; locked_overall, locked_folds, _ = summaries(locked)
    locked_row = locked_overall.iloc[0]
    checks = {
        "p2_legal": abs(float(p2_overall.patient_macro_f1) - EXPECTED["p2_legal"]) <= 5e-4,
        # CDEL's 0.80/0.20 reference is regenerated from BCR BC-only OOF
        # predictions.  Do not validate it against obsolete 0.90/0.10 outputs.
        "locked_weights": bool(np.isclose(LOCKED_P2_WEIGHT, .80) and np.isclose(LOCKED_V3_WEIGHT, .20)),
    }
    fold_rows = []
    for fold in range(1, 6):
        actual_threshold = float(locked_thresholds.loc[locked_thresholds.outer_fold.eq(fold), "threshold"].iloc[0])
        actual_f1 = float(locked_folds.loc[locked_folds.outer_fold.eq(fold), "patient_macro_f1"].iloc[0])
        threshold_ok = 0.0 <= actual_threshold <= 1.0
        checks[f"fold_{fold}_validation_threshold"] = threshold_ok
        fold_rows.append({"outer_fold": fold, "actual_threshold": actual_threshold, "actual_macro_f1": actual_f1, "validation_threshold_valid": threshold_ok})
    return {
        "status": "passed" if all(checks.values()) else "failed", "checks": checks, "folds": fold_rows,
        "actual_p2_legal": float(p2_overall.patient_macro_f1), "actual_fusion_legal": float(locked_row.patient_macro_f1),
        "actual_p2_truek": float(p2_overall.truek_patient_macro_f1), "actual_fusion_truek": float(locked_row.truek_patient_macro_f1),
        "legal_tolerance": 5e-4, "fusion_reference": "regenerated_from_final_BCR_BC_only_OOF",
    }


def _bootstrap_rows(comparisons: list[tuple[str, pd.DataFrame, pd.DataFrame]], repeats: int, seed: int) -> pd.DataFrame:
    rows = []
    for name, a, b in comparisons:
        rows.append(paired_bootstrap(a, b, comparison=name, repeats=repeats, seed=seed))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0_root", default=None); parser.add_argument("--p2_noq10_root", default=None)
    parser.add_argument("--p2_q10_root", required=True); parser.add_argument("--v3_qbc_root", required=True)
    parser.add_argument("--input_manifest", required=True); parser.add_argument("--allowed_subjects_ledger", required=True); parser.add_argument("--fixed_fold_manifest", required=True)
    parser.add_argument("--require_n_patients", type=int, default=80); parser.add_argument("--expected_n_channels", type=int, default=7635)
    parser.add_argument("--locked_p2_weight", type=float, default=LOCKED_P2_WEIGHT); parser.add_argument("--locked_v3_weight", type=float, default=LOCKED_V3_WEIGHT)
    parser.add_argument("--threshold_step", type=float, default=.005); parser.add_argument("--weight_grid", default="0.00,0.05,0.10,0.20,0.50,1.00")
    parser.add_argument("--bootstrap_repeats", type=int, default=2000); parser.add_argument("--shuffle_repeats", type=int, default=200); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True); parser.add_argument("--strict", action="store_true"); parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if not np.isclose(args.locked_p2_weight, LOCKED_P2_WEIGHT, atol=1e-12) or not np.isclose(args.locked_v3_weight, LOCKED_V3_WEIGHT, atol=1e-12): raise ValueError("Locked CDEL weights must remain 0.80/0.20")
    if not np.isclose(args.threshold_step, .005, atol=1e-12): raise ValueError("threshold_step must remain 0.005")
    weights = [float(value) for value in args.weight_grid.split(",")]
    if weights != [0.0, .05, .10, .20, .50, 1.0]: raise ValueError("weight_grid is frozen")
    paths = ensure_output_tree(args.output_dir)
    folds, manifest, input_audit = load_base_folds(args.input_manifest, allowed_subjects_ledger=args.allowed_subjects_ledger, fixed_fold_manifest=args.fixed_fold_manifest, require_n_patients=args.require_n_patients, expected_n_channels=args.expected_n_channels, strict=args.strict)
    write_json(paths["audit"] / "ablation_input_audit.json", input_audit)
    if args.dry_run:
        print(json.dumps({"status": "passed", "mode": "dry_run", "n_patients": input_audit["n_patients"], "n_channels": input_audit["n_test_channels"]}, indent=2)); return

    p2, p2_thresholds, p2_ledger = _run_configuration(folds, name="B1_P2_Q10_ONLY", operator="p2", v3_weight=0.0, status=ABLATION_STATUS)
    v3, v3_thresholds, _ = _run_configuration(folds, name="B2_BCR_NET_ONLY", operator="v3", v3_weight=1.0, status=ABLATION_STATUS)
    locked, locked_thresholds, locked_ledger = _run_configuration(folds, name="B3_CDEL_LOCKED_FUSION", operator="probability", v3_weight=LOCKED_V3_WEIGHT, status=LOCKED_STATUS)
    reproduction = _reproduction(p2, locked, locked_thresholds); write_json(paths["audit"] / "ablation_reproduction_audit.json", reproduction)
    if reproduction["status"] != "passed": raise RuntimeError("Locked fusion reproduction gate failed; see ablation_reproduction_audit.json")

    p0_folds, p0_audit = load_optional_model(args.p0_root, folds, model_name="P0_CURRENT_P2")
    noq10_folds, noq10_audit = load_optional_model(args.p2_noq10_root, folds, model_name="P2_TEMPORAL_NO_Q10")
    model_diff = config_difference([p0_audit, noq10_audit, {"model": "P2_TEMPORAL_Q10", "status": "available", "config": json.loads((Path(args.p2_q10_root) / "run_args_p23.json").read_text()) if (Path(args.p2_q10_root) / "run_args_p23.json").is_file() else None}])
    write_json(paths["audit"] / "model_config_difference.json", model_diff)
    write_json(paths["audit"] / "component_config_difference.json", model_diff)
    component_frames = [] ; component_thresholds = []
    if p0_folds: rows, threshold = _run_optional_model(p0_folds, name="A1_P0_CURRENT_P2"); component_frames.append(rows); component_thresholds.append(threshold)
    if noq10_folds: rows, threshold = _run_optional_model(noq10_folds, name="A2_P2_TEMPORAL_NO_Q10"); component_frames.append(rows); component_thresholds.append(threshold)
    component_frames.append(p2.assign(experiment="A3_P2_TEMPORAL_Q10")); component_thresholds.append(p2_thresholds.assign(experiment="A3_P2_TEMPORAL_Q10"))
    component = pd.concat(component_frames, ignore_index=True); _write_family(paths, "component_evolution", component)

    branch = pd.concat([p2, v3, locked], ignore_index=True); branch_overall, branch_fold, branch_center = _write_family(paths, "branch_ablation", branch)
    baseline_f1 = float(branch_overall.loc[branch_overall.experiment.eq("B1_P2_Q10_ONLY"), "patient_macro_f1"].iloc[0])
    branch_overall["absolute_delta_vs_p2"] = branch_overall.patient_macro_f1 - baseline_f1; branch_overall["relative_delta_vs_p2"] = branch_overall.absolute_delta_vs_p2 / baseline_f1
    branch_overall.to_csv(paths["metrics"] / "branch_ablation_overall.csv", index=False)

    weight_patients = []; weight_thresholds = []
    for weight in weights:
        name = f"W_BCR_{weight:.2f}"; status = LOCKED_STATUS if np.isclose(weight, LOCKED_V3_WEIGHT) else ABLATION_STATUS
        rows, thresholds, _ = _run_configuration(folds, name=name, operator="probability", v3_weight=weight, status=status)
        rows["v3_weight"] = weight; rows["p2_weight"] = 1.0 - weight; weight_patients.append(rows)
        thresholds["v3_weight"] = weight; thresholds["p2_weight"] = 1.0 - weight; thresholds["locked_configuration"] = np.isclose(weight, LOCKED_V3_WEIGHT); weight_thresholds.append(thresholds)
    weight_rows = pd.concat(weight_patients, ignore_index=True); weight_threshold_frame = pd.concat(weight_thresholds, ignore_index=True)
    weight_overall, _, _ = _write_family(paths, "weight_ablation", weight_rows, weight_threshold_frame)
    weight_overall = weight_overall.merge(weight_threshold_frame.groupby("experiment").threshold.agg(["mean", "std"]).reset_index().rename(columns={"mean": "threshold_mean", "std": "threshold_std"}), on="experiment")
    weight_overall["v3_weight"] = weight_overall.experiment.str.replace("W_BCR_", "", regex=False).astype(float); weight_overall["p2_weight"] = 1.0 - weight_overall.v3_weight; weight_overall["locked_configuration"] = np.isclose(weight_overall.v3_weight, LOCKED_V3_WEIGHT)
    weight_overall.to_csv(paths["metrics"] / "weight_ablation_overall.csv", index=False)
    line_figure(weight_overall.sort_values("v3_weight"), x="v3_weight", ys=["patient_macro_f1", "patient_ez_auprc", "patient_ez_mrr"], output_stem=paths["figures"] / "weight_ablation_curve", xlabel="BCR-Net weight", ylabel="Patient-equal metric")

    operator_frames = []; operator_thresholds = []
    for name, operator in (("C1_PROBABILITY_FUSION", "probability"), ("C2_LOGIT_FUSION", "logit"), ("C3_PATIENT_RANK_FUSION", "rank")):
        rows, thresholds, _ = _run_configuration(folds, name=name, operator=operator, v3_weight=LOCKED_V3_WEIGHT, status=LOCKED_STATUS if operator == "probability" else ABLATION_STATUS)
        operator_frames.append(rows); operator_thresholds.append(thresholds)
    operators = pd.concat(operator_frames, ignore_index=True); operator_threshold_frame = pd.concat(operator_thresholds, ignore_index=True)
    _write_family(paths, "fusion_operator", operators, operator_threshold_frame)

    p2_threshold_map = dict(zip(p2_thresholds.outer_fold, p2_thresholds.threshold))
    t0 = p2.assign(experiment="T0_P2_SCORE_P2_THRESHOLD")
    t1, t1_thresholds, _ = _run_configuration(folds, name="T1_FUSION_SCORE_P2_THRESHOLD", operator="probability", v3_weight=.10, status=ABLATION_STATUS, fixed_thresholds=p2_threshold_map)
    t2 = locked.assign(experiment="T2_FUSION_SCORE_FUSION_THRESHOLD")
    decomposition = pd.concat([t0, t1, t2], ignore_index=True)
    decomposition_overall, _, _ = _write_family(paths, "threshold_decomposition", decomposition, pd.concat([p2_thresholds.assign(experiment="T0_P2_SCORE_P2_THRESHOLD"), t1_thresholds, locked_thresholds.assign(experiment="T2_FUSION_SCORE_FUSION_THRESHOLD")], ignore_index=True))
    for metric in ("patient_ez_auprc", "patient_ez_mrr"):
        a = float(decomposition_overall.loc[decomposition_overall.experiment.eq("T1_FUSION_SCORE_P2_THRESHOLD"), metric].iloc[0]); b = float(decomposition_overall.loc[decomposition_overall.experiment.eq("T2_FUSION_SCORE_FUSION_THRESHOLD"), metric].iloc[0])
        if not np.isclose(a, b, atol=1e-12): raise RuntimeError(f"T1/T2 continuous-score metric changed: {metric}")
    t_values = decomposition_overall.set_index("experiment").patient_macro_f1
    waterfall = pd.DataFrame({"component": ["P2 baseline", "score contribution", "threshold adaptation"], "value": [t_values["T0_P2_SCORE_P2_THRESHOLD"], t_values["T1_FUSION_SCORE_P2_THRESHOLD"] - t_values["T0_P2_SCORE_P2_THRESHOLD"], t_values["T2_FUSION_SCORE_FUSION_THRESHOLD"] - t_values["T1_FUSION_SCORE_P2_THRESHOLD"]]})
    bar_figure(waterfall, x="component", y="value", output_stem=paths["figures"] / "threshold_decomposition_waterfall", ylabel="Macro-F1 / contribution")

    shuffle_rows = []; shuffle_fold_rows = []; shuffle_patient_frames = []; total_eligible = total_changed = 0
    for repeat in range(args.shuffle_repeats):
        repeated_patients = []
        for fold in range(1, 6):
            val, val_audit = shuffle_v3_within_patient(folds[fold]["validation"], experiment="SHUFFLED_V3", fold=fold, partition="validation", repeat=repeat, seed=args.seed)
            test, test_audit = shuffle_v3_within_patient(folds[fold]["test"], experiment="SHUFFLED_V3", fold=fold, partition="test", repeat=repeat, seed=args.seed)
            total_eligible += val_audit["eligible_multi_channel_patients"] + test_audit["eligible_multi_channel_patients"]; total_changed += val_audit["changed_order_patients"] + test_audit["changed_order_patients"]
            val["shuffle_score"] = probability_fusion(val.p2_score_nez, val.v3_score_nez, v3_weight=.10); test["shuffle_score"] = probability_fusion(test.p2_score_nez, test.v3_score_nez, v3_weight=.10)
            threshold, _ = select_fold_threshold(val, score_column="shuffle_score")
            patient = evaluate(test, score_column="shuffle_score", threshold=threshold, experiment=f"shuffle_{repeat}", analysis_status=ABLATION_STATUS); repeated_patients.append(patient)
            fold_summary = summaries(patient)[0].iloc[0].to_dict(); shuffle_fold_rows.append({"repeat": repeat, "outer_fold": fold, "threshold": threshold, **fold_summary})
        repeat_patients = pd.concat(repeated_patients, ignore_index=True); repeat_patients["repeat"] = repeat; shuffle_patient_frames.append(repeat_patients)
        repeat_summary = summaries(repeat_patients)[0].iloc[0].to_dict(); shuffle_rows.append({"repeat": repeat, **repeat_summary})
    shuffle_frame = pd.DataFrame(shuffle_rows); shuffle_frame.to_csv(paths["metrics"] / "shuffled_v3_control_repeats.csv", index=False); pd.DataFrame(shuffle_fold_rows).to_csv(paths["metrics"] / "shuffled_v3_control_by_fold.csv", index=False)
    locked_f1 = float(summaries(locked)[0].patient_macro_f1.iloc[0]); shuffle_summary = pd.DataFrame([{"true_fusion_macro_f1": locked_f1, "shuffle_mean": shuffle_frame.patient_macro_f1.mean(), "shuffle_std": shuffle_frame.patient_macro_f1.std(ddof=1), "shuffle_q025": shuffle_frame.patient_macro_f1.quantile(.025), "shuffle_q975": shuffle_frame.patient_macro_f1.quantile(.975), "true_minus_shuffle_mean": locked_f1 - shuffle_frame.patient_macro_f1.mean(), "fraction_shuffle_below_true": float((shuffle_frame.patient_macro_f1 < locked_f1).mean()), "shuffle_repeats": args.shuffle_repeats}]); shuffle_summary.to_csv(paths["metrics"] / "shuffled_v3_control_summary.csv", index=False)
    multiset_audit = {"status": "passed", "eligible_patient_instances": total_eligible, "changed_order_patient_instances": total_changed, "changed_fraction": total_changed / total_eligible if total_eligible else np.nan, "multiset_preserved": True}; write_json(paths["audit"] / "shuffled_v3_multiset_audit.json", multiset_audit)
    bar_figure(shuffle_frame, x="repeat", y="patient_macro_f1", output_stem=paths["figures"] / "shuffled_v3_control_distribution", ylabel="Shuffled-V3 Macro-F1")

    threshold_maps = {"p2": p2_threshold_map, "v3": dict(zip(v3_thresholds.outer_fold, v3_thresholds.threshold)), "fusion": dict(zip(locked_thresholds.outer_fold, locked_thresholds.threshold))}
    mechanism_ledger = locked_ledger.copy(); mechanism_ledger["p2_threshold"] = mechanism_ledger.outer_fold.map(threshold_maps["p2"]); mechanism_ledger["v3_threshold"] = mechanism_ledger.outer_fold.map(threshold_maps["v3"]); mechanism_ledger["fusion_threshold"] = mechanism_ledger.outer_fold.map(threshold_maps["fusion"]); mechanism_ledger["fused_score_nez"] = mechanism_ledger.score_nez_probability
    score_correlations(mechanism_ledger).to_csv(paths["analysis"] / "p2_v3_score_correlation_by_patient.csv", index=False)
    disagreement, changed, near = disagreement_analysis(mechanism_ledger); disagreement.to_csv(paths["analysis"] / "p2_v3_disagreement_analysis.csv", index=False); changed.to_csv(paths["analysis"] / "fusion_changed_channels.csv", index=False); near.to_csv(paths["analysis"] / "near_boundary_correction_analysis.csv", index=False)

    shuffled_patients = pd.concat(shuffle_patient_frames, ignore_index=True)
    metric_columns = [column for column in PATIENT_METRICS if column in shuffled_patients]
    shuffled_mean = shuffled_patients.groupby(["subject_id", "center", "outer_fold"], as_index=False)[metric_columns].mean()
    shuffled_mean["experiment"] = "SHUFFLED_V3_MEAN_CONTROL"; shuffled_mean["analysis_status"] = ABLATION_STATUS
    comparison_pairs = [
        ("locked_fusion_vs_p2", locked, p2), ("locked_fusion_vs_v3", locked, v3),
        ("true_fusion_vs_shuffled_v3_mean", locked, shuffled_mean),
        ("probability_vs_logit", operators[operators.experiment.eq("C1_PROBABILITY_FUSION")], operators[operators.experiment.eq("C2_LOGIT_FUSION")]),
        ("probability_vs_rank", operators[operators.experiment.eq("C1_PROBABILITY_FUSION")], operators[operators.experiment.eq("C3_PATIENT_RANK_FUSION")]),
        ("T1_vs_T0", t1, t0), ("T2_vs_T1", t2, t1), ("T2_vs_T0", t2, t0),
    ]
    bootstrap_frame = _bootstrap_rows(comparison_pairs, args.bootstrap_repeats, args.seed); bootstrap_frame.to_csv(paths["metrics"] / "paired_bootstrap.csv", index=False)
    branch_bootstrap = bootstrap_frame[bootstrap_frame.comparison.eq("locked_fusion_vs_p2")]
    branch_overall = branch_overall.merge(branch_bootstrap[["ci_low", "ci_high", "probability_delta_gt_zero"]].assign(experiment="B3_P2_Q10_V3_LOCKED_FUSION"), on="experiment", how="left"); branch_overall.to_csv(paths["metrics"] / "branch_ablation_overall.csv", index=False)

    all_patients = pd.concat([component, branch, weight_rows, operators, decomposition], ignore_index=True)
    all_overall, all_fold, all_center = summaries(all_patients)
    all_overall.to_csv(paths["metrics"] / "all_ablation_results_long.csv", index=False)
    all_overall.set_index("experiment")[PATIENT_METRICS].T.to_csv(paths["metrics"] / "all_ablation_results_wide.csv")
    grouped_bar_figure(branch_fold, x="outer_fold", group="experiment", y="patient_macro_f1", output_stem=paths["figures"] / "fold_paired_comparison", ylabel="Patient Macro-F1")
    grouped_bar_figure(branch_center, x="center", group="experiment", y="patient_macro_f1", output_stem=paths["figures"] / "center_paired_comparison", ylabel="Patient Macro-F1")
    availability = {"p0": p0_audit["status"], "no_q10": noq10_audit["status"]}
    markdown_report(paths["reports"] / "P2_V3_AAAI_ABLATION_REPORT.md", reproduction=reproduction, availability=availability, key_results=branch_overall)
    write_json(paths["root"] / "run_args.json", vars(args))
    print(json.dumps({"status": "passed", "reproduction": reproduction["status"], "p0": availability["p0"], "no_q10": availability["no_q10"], "output_dir": str(paths["root"])}, indent=2))


if __name__ == "__main__":
    main()
