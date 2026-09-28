"""Reproducibly assemble only compact A1-TF reports from frozen aggregates."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PROJECT = ROOT.parent
LOCK_SHA = "ace2017e01d0d21e3dda8ddd4d42551366704430b8a10f3666f955150150630d"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def put(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def pretty(value, digits=4):
    return f"{value:.{digits}f}"


def main():
    if sha(ROOT / "PROTOCOL_LOCK.json") != LOCK_SHA:
        raise RuntimeError("Protocol lock changed")
    arch = load(ROOT / "ARCHITECTURE_IDENTITY_AUDIT.json")
    gate = load(OUT / "ICTAL_STAGE2_GATE_AUDIT.json")
    score_freeze = load(OUT / "ICTAL_SCORE_FREEZE_AUDIT.json")
    nfreeze = load(OUT / "INTERICTAL_MODEL_FREEZE.json")
    naccess = load(OUT / "INTERICTAL_TEST_ACCESS_AUDIT.json")
    ip = pd.read_csv(OUT / "ICTAL_PRIMARY_METRICS.csv").set_index("model")
    ib = pd.read_csv(OUT / "ICTAL_PATIENT_BOOTSTRAP.csv").set_index("metric")
    np = load(OUT / "INTERICTAL_PRIMARY_METRICS.json")
    n0 = load(PROJECT / "omni_ieeg_a1_interictal_v2_seed42/outputs/PRIMARY_METRICS.json")
    nb = pd.read_csv(OUT / "INTERICTAL_PATIENT_BOOTSTRAP.csv")
    nb = nb[nb.comparison == "N1_minus_N0"].set_index("metric")
    iv = pd.read_csv(OUT / "ICTAL_TF_INTERVENTION.csv")
    nv = pd.read_csv(OUT / "INTERICTAL_TF_INTERVENTION.csv").set_index("setting")
    nr = pd.read_csv(OUT / "DESCRIPTOR_RECONSTRUCTION_OMNI.csv")
    ir = pd.read_csv(OUT / "DESCRIPTOR_RECONSTRUCTION_ICTAL.csv")
    recon = pd.concat([ir, nr], ignore_index=True)
    recon.to_csv(OUT / "DESCRIPTOR_RECONSTRUCTION_AUDIT.csv", index=False)
    (OUT / "INTERICTAL_THRESHOLD_SELECTION.csv").write_bytes(
        (OUT / "THRESHOLD_SELECTION.csv").read_bytes())
    pd.DataFrame([{"benchmark": b, "model": model, "parameters": arch[f"{model}_parameters"],
                   "parameter_topology_identical": arch["parameter_topology_identical"]}
                  for b, model in (("Ictal", "I1"), ("Omni_interictal", "N1"))]).to_csv(
                      OUT / "MODEL_PARAMETER_AUDIT.csv", index=False)
    selected = pd.read_csv(OUT / "ICTAL_CHECKPOINT_SELECTION.csv")
    put(OUT / "ICTAL_TRAINING_AUDIT.json", {
        "seed": 42, "stage1_folds_complete": 5, "stage1_epochs_per_fold": 15,
        "stage2_eligible_folds": gate["eligible_folds"],
        "stage2_folds_complete": len(gate["eligible_folds"]),
        "stage2_epochs_per_eligible_fold": 15, "stage2_eligible_target_cells": gate["eligible_cells"],
        "stage2_total_target_cells": 65,
        "selected_stage1_target_cells": int(selected.selected_stage1.sum()),
        "selected_stage2_target_cells": int(selected.selected_stage2.sum()),
        "score_freeze_pass": score_freeze["pass"],
        "score_files_frozen": score_freeze["score_files"],
        "target_labels_used_for_own_selection": False,
        "historical_validation_loader_materialized_labels": True,
        "selected_epoch_checkpoints_retained": False,
        "selected_epoch_score_snapshots_retained": True,
        "ictal_intervention_scope": "stage endpoint, not VLOO-selected checkpoint"})
    put(OUT / "INTERICTAL_TRAINING_AUDIT.json", {
        "seed": 42, "official_train_patients": 141, "inner_fit_patients": 112,
        "inner_validation_patients": 29, "inner_max_epochs": 30,
        "selected_inner_epoch": nfreeze["selected_inner_epoch"],
        "inner_validation_patient_equal_ap": nfreeze["selected_inner_patient_equal_AP"],
        "final_all_train_refit_epochs": nfreeze["selected_inner_epoch"],
        "final_checkpoint_sha256": nfreeze["final_checkpoint_sha256"],
        "frozen_threshold": nfreeze["threshold"],
        "model_frozen_before_test": nfreeze["model_frozen_before_official_test"],
        "test_used_for_tuning": False})
    put(OUT / "ICTAL_REFERENCE_AUDIT.json", {
        "reference": "exact historical same-channel pre-onset A1 four-view constructor",
        "historical_a1_model_sha256": arch["original_A1_model_sha256"],
        "real_train_batch_alpha_zero_logit_error_below_1e_6_all_folds": True,
        "raw_alignment_formal_coverage_all_active_channels_and_windows": True,
        "no_new_ictal_outer_test": True,
        "fixed_query_baseline_replay_ap": float(ip.loc["I0_A1", "ap"])})
    put(OUT / "INTERICTAL_REFERENCE_AUDIT.json", {
        "reference": "same-patient/EDF/window contemporaneous cross-channel median and 1.4826 MAD",
        "descriptor_views": ["ABS", "DELTA", "ZDELTA", "LOGR"],
        "historical_v2_feature_cache_reused_exactly": True,
        "tf_alignment_check": "EDF, channel order, 59 windows and original clip starts",
        "official_test_patients": np["patients"],
        "official_test_edf_channel_rows": np["edf_channel_records"]})
    ictal_ap = float(ip.loc["I1_A1_TF", "ap"])
    ictal_mrr_delta = float(ib.loc["mrr", "delta"])
    ictal_top1_delta = float(ib.loc["top1", "delta"])
    ap_delta = float(ib.loc["ap", "delta"])
    n_auroc_delta = np["pooled_auroc"] - n0["pooled_auroc"]
    n_patient_ap_delta = np["patient_equal_ap"] - n0["patient_equal_ap"]
    gates = {
        "ictal_AP_noninferiority": ictal_ap >= 0.5717,
        "ictal_MRR_within_0_01": ictal_mrr_delta >= -0.01,
        "ictal_Top1_within_0_01_or_CI_includes_zero":
            ictal_top1_delta >= -0.01 or
            float(ib.loc["top1", "ci_lower"]) <= 0 <= float(ib.loc["top1", "ci_upper"]),
        "ictal_overall_noninferiority": False,
        "ictal_improvement": ictal_ap > float(ip.loc["I0_A1", "ap"]) and
            float(ib.loc["ap", "ci_lower"]) > 0,
        "interictal_macroF1_at_least_0_65": np["macro_f1"] >= .65,
        "interictal_macroF1_at_least_0_68": np["macro_f1"] >= .68,
        "interictal_macroF1_at_least_0_70": np["macro_f1"] >= .70,
        "interictal_AUROC_at_least_0_78": np["pooled_auroc"] >= .78,
        "interictal_AUROC_at_least_0_80": np["pooled_auroc"] >= .80,
        "interictal_N1_vs_N0_AUROC_point_gain": n_auroc_delta > 0,
        "interictal_N1_vs_N0_patient_equal_AP_point_gain": n_patient_ap_delta > 0,
        "interictal_patient_equal_AP_delta_CI_lower_above_zero":
            float(nb.loc["delta_patient_equal_ap", "ci_lower"]) > 0,
        "interictal_TF_branch_isolated_benefit_proven": False,
        "one_architecture_meets_both_predeclared_performance_gates": False,
        "terminal": "A1_TF_UNIFIED_VIABILITY_NOT_SUPPORTED"}
    gates["ictal_overall_noninferiority"] = all(gates[key] for key in (
        "ictal_AP_noninferiority", "ictal_MRR_within_0_01",
        "ictal_Top1_within_0_01_or_CI_includes_zero"))
    put(OUT / "SUCCESS_GATES.json", gates)
    put(OUT / "LABEL_USAGE_AUDIT.json", {
        "ictal_FIT_labels_used_for_training": True,
        "ictal_validation_other_12_labels_used_for_each_target_gate_and_selection": True,
        "ictal_target_own_labels_used_for_own_gate_or_selection": False,
        "ictal_target_query_labels_used_only_for_evaluation": True,
        "omni_official_train_labels_used_for_training_selection_threshold": True,
        "omni_official_test_labels_used_for_training_or_selection": False,
        "omni_official_test_labels_used_for_final_metrics": True,
        "historical_I0_and_N0_outcomes_seen_before_new_model_development": True})
    put(OUT / "TEST_ACCESS_AUDIT.json", {
        "ictal_historical_fixed_query_evaluated": True,
        "ictal_new_outer_test_accessed": False,
        "omni_official_test_accessed": naccess["official_test_accessed"],
        "omni_test_patients": naccess["patients"],
        "omni_test_edf_channel_records": naccess["edf_channel_records"],
        "N1_model_frozen_before_test": naccess["model_frozen_before_test"],
        "N1_threshold_frozen_before_test": naccess["threshold_frozen_before_test"],
        "test_used_for_tuning": False,
        "fresh_sealed_confirmation": False})
    summary = pd.DataFrame([
        {"model": "Original A1", "benchmark": "Original", "regime": "Ictal",
         "ap": float(ip.loc["I0_A1", "ap"]), "auroc": float(ip.loc["I0_A1", "auc"]),
         "macro_f1": float(ip.loc["I0_A1", "macro_f1"]),
         "mrr": float(ip.loc["I0_A1", "mrr"]), "top1": float(ip.loc["I0_A1", "top1"])},
        {"model": "A1-TF", "benchmark": "Original", "regime": "Ictal",
         "ap": ictal_ap, "auroc": float(ip.loc["I1_A1_TF", "auc"]),
         "macro_f1": float(ip.loc["I1_A1_TF", "macro_f1"]),
         "mrr": float(ip.loc["I1_A1_TF", "mrr"]), "top1": float(ip.loc["I1_A1_TF", "top1"])},
        {"model": "A1-Interictal v2", "benchmark": "Omni", "regime": "Interictal",
         "ap": n0["patient_equal_ap"], "auroc": n0["pooled_auroc"],
         "macro_f1": n0["macro_f1"], "mrr": n0["patient_equal_mrr"],
         "top1": n0["patient_equal_top1"]},
        {"model": "A1-TF", "benchmark": "Omni", "regime": "Interictal",
         "ap": np["patient_equal_ap"], "auroc": np["pooled_auroc"],
         "macro_f1": np["macro_f1"], "mrr": np["mrr"], "top1": np["top1"]},
    ])
    summary.to_csv(OUT / "TWO_BENCHMARK_SUMMARY.csv", index=False)
    (ROOT / "TF_PREPROCESSING_AUDIT.md").write_text(
        "# TF preprocessing audit\n\n"
        "Both branches use the same 32 physical-frequency log-power centers (1–120 Hz), "
        "symmetric Hann 2-second STFT, 1-second hop, linear interpolation on the physical "
        "frequency axis, and train-only per-bin normalization. Ictal source sampling is "
        "250 Hz; Omni v2 source sampling is 300 Hz. Their Nyquist limits are "
        "125 Hz and 150 Hz, so both cover the locked 120 Hz upper bin. "
        "There is no per-window variance normalization, label-dependent preprocessing, "
        "patient ID embedding, or center embedding. Ictal TF comes from formally aligned "
        "raw 2-second windows; Omni TF matches each v2 EDF channel and clip start. "
        "The fixed preprocessing is recorded in PROTOCOL_LOCK.json.\n", encoding="utf-8")
    def fmtrow(r):
        return f"| {r.model} | {r.benchmark} | {r.regime} | {r.ap:.4f} | {r.auroc:.4f} | {r.macro_f1:.4f} | {r.mrr:.4f} | {r.top1:.4f} |"
    lines = ["# Unified A1-TF seed 42 — final report", "",
        "| Model | Benchmark | Regime | AP | AUROC | Macro-F1 | MRR | Top1 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
        *[fmtrow(r) for r in summary.itertuples(index=False)], "",
        "AP means 65-cell × 20-repetition matched-query AP for Ictal, but patient-equal AP "
        "over positive-label patients for Omni. The two AP columns are not directly comparable.", "",
        "## Verdict", "",
        "**A1_TF_UNIFIED_VIABILITY_NOT_SUPPORTED.** Module topology is identical "
        "(67,403 parameters each), but the predeclared performance gates fail. Ictal A1-TF "
        f"AP {ictal_ap:.6f} versus original A1 {float(ip.loc['I0_A1','ap']):.6f}; "
        f"paired ΔAP {ap_delta:+.6f}, 95% CI [{float(ib.loc['ap','ci_lower']):+.6f}, "
        f"{float(ib.loc['ap','ci_upper']):+.6f}]. The required ≥0.5717 AP is not met. "
        f"MRR changes {ictal_mrr_delta:+.6f}; Top1 changes {ictal_top1_delta:+.6f}, "
        f"its CI [{float(ib.loc['top1','ci_lower']):+.6f}, "
        f"{float(ib.loc['top1','ci_upper']):+.6f}].", "",
        f"Omni N1 versus N0: Macro-F1 {np['macro_f1']:.6f} versus {n0['macro_f1']:.6f}; "
        f"AUROC {np['pooled_auroc']:.6f} versus {n0['pooled_auroc']:.6f}; "
        f"patient-equal AP {np['patient_equal_ap']:.6f} versus {n0['patient_equal_ap']:.6f}. "
        f"The paired 10,000-draw CI for patient-equal AP Δ is "
        f"[{float(nb.loc['delta_patient_equal_ap','ci_lower']):+.6f}, "
        f"{float(nb.loc['delta_patient_equal_ap','ci_upper']):+.6f}]. "
        "The prespecified Macro-F1 ≥0.65 and AUROC ≥0.78 targets are not met. "
        "Macro-F1 ≥0.68/0.70 and AUROC ≥0.80 also fail.", "",
        "## Mechanism and limitations", "",
        f"Omni trained alpha = {np['trained_alpha']:+.6f}; alpha=0 gives "
        f"Macro-F1 {float(nv.loc['alpha_zero','macro_f1']):.6f}, AUROC "
        f"{float(nv.loc['alpha_zero','pooled_auroc']):.6f}, patient-equal AP "
        f"{float(nv.loc['alpha_zero','patient_equal_ap']):.6f}. "
        "The full model does not consistently outperform its own alpha-zero intervention, "
        "so N1-over-N0 is not evidence of an isolated TF benefit. Ictal endpoint alphas "
        f"range {iv.alpha.min():+.6f} to {iv.alpha.max():+.6f}; endpoint validation "
        "alpha-zero AP changes have mixed signs. Those endpoint checkpoints are not "
        "the VLOO-selected checkpoints and cannot establish the selected I1 TF effect: "
        "training retained score snapshots but only the final epoch weights. "
        "No post-outcome retraining was done to fill that gap.", "",
        "Descriptor reconstruction is mixed. Omni inner-validation descriptors mostly show "
        "positive correlation, but descriptor 6 has negative R². Several ictal descriptors "
        "have nearly constant targets and extremely negative or undefined R²; the TF branch "
        "has not reliably reconstructed all nine A1 descriptors. See the per-descriptor CSV.", "",
        "Omni source strata are heterogeneous: source-sink performs best, OpenIEEG is "
        "substantially weaker, and Zurich has zero pathological channels, making ranking "
        "and positive-class metrics non-estimable. The main bottleneck is low pathological "
        "sensitivity and limited ranking discrimination, not just threshold placement.", "",
        "The architecture topology, shared A1/TF modules, and zero-init fusion match. "
        "However the complete training protocols do **not** differ only in their physiological "
        "reference: Ictal starts from historical A1 and uses a teacher anchor; Omni trains "
        "from scratch without that anchor, as the supplied instructions also require. "
        "Consequently this is a same-topology two-benchmark study, not a controlled "
        "reference-operator-only intervention.", "",
        "Ictal uses previously viewed historical fixed-query development targets; no new "
        "outer test was opened. Omni N1 checkpoint and threshold were frozen before its "
        "official test was accessed once, and test outcomes were not used to tune N1. "
        "Historical I0/N0 outcomes were already visible, so neither comparison is a fresh "
        "blind confirmation. There is no detected within-run test tuning, but literal "
        "absence of leakage cannot be proved from code alone.", "",
        "## Audit index", "",
        "See `outputs/` for primary metrics, paired bootstrap, stage gate and selection, "
        "threshold/intervention, reconstruction, label usage and test-access audits. "
        "Only compact CSV/JSON and code are published; no checkpoint, raw iEEG, TF cache, "
        "or private ictal patient-channel query rows are uploaded.", ""]
    (ROOT / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "IMPLEMENTATION_AUDIT.md").write_text(
        "# Implementation audit\n\n"
        "- Same `A1TFModel` class and parameter topology for I1/N1; synthetic alpha-zero "
        "replay gave zero logit error/mismatch; Ictal real FIT batch replay passed.\n"
        "- Original A1 36-D descriptors and patient-relative downstream are imported from "
        "historical source; TF is a 32-bin physical-frequency residual.\n"
        "- Ictal exact 80-person/5-fold source and 47-ID/65-cell/20-query comparison replayed "
        "the historical I0 AP to 1e-9. Stage2 gate used the other 12 validation patients "
        "per target, and 135 score-only epoch files were frozen before target evaluation.\n"
        "- Omni exact official split: 141 train, 96 test patients, 8,104 EDF-channel test "
        "records. Checkpoint and threshold SHA were frozen before N1 test extraction.\n"
        "- Ictal per-epoch **weights** were not retained, only per-epoch score snapshots; "
        "its alpha-zero intervention is limited to final stage endpoints, not selected I1.\n"
        "- Historical I0/N0 outcomes were already seen before A1-TF design. This is "
        "exploratory and does not license a blinded generalization claim.\n", encoding="utf-8")
    print(gates["terminal"])


if __name__ == "__main__":
    main()
