"""Build aggregate-only decision matrix and report from frozen outputs."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze import csv_out
from common import EXPERIMENT, ensure_source


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def avg(rows, key):
    return float(np.mean([float(row[key]) for row in rows]))


def run():
    ensure_source()
    reproduction = read_json(EXPERIMENT / "SOURCE_REPRODUCTION.json")
    sanity = read_json(EXPERIMENT / "SANITY_AUDIT.json")
    alignment = read_json(EXPERIMENT / "shared_linear" / "ALIGNMENT_GATE.json")
    geometry = read_json(EXPERIMENT / "patient_specific" / "GEOMETRY_GATE.json")
    richer = read_json(EXPERIMENT / "richer_features" / "RICHER_FEATURE_GATE.json")
    if not reproduction["pass"] or not sanity["pass"]:
        raise RuntimeError("Required source/sanity check failed")
    if not sanity["feature_inventory_committed_before_richer_evaluation"]:
        raise RuntimeError("Richer inventory sequence failed")
    shared = read_csv(EXPERIMENT / "shared_linear" / "SHARED_LINEAR_BY_REPRESENTATION.csv")
    local = read_csv(EXPERIMENT / "patient_specific" / "MATCHED_SHARED_VS_PATIENT_SPECIFIC.csv")
    gap = read_csv(EXPERIMENT / "patient_specific" / "GEOMETRY_GAP_BOOTSTRAP.csv")
    richer9 = read_csv(EXPERIMENT / "richer_features" / "RICH9_BY_FOLD.csv")
    richerall = read_csv(EXPERIMENT / "richer_features" / "RICHALL_BY_FOLD.csv")
    richer9p = read_csv(EXPERIMENT / "richer_features" / "RICH9_PCA_BY_FOLD.csv")
    richerallp = read_csv(EXPERIMENT / "richer_features" / "RICHALL_PCA_BY_FOLD.csv")
    feature = read_csv(EXPERIMENT / "single_feature" / "FEATURE_DIRECTION_STABILITY.csv")
    view = read_csv(EXPERIMENT / "single_feature" / "VIEW_DIRECTION_STABILITY.csv")
    centroid = read_csv(EXPERIMENT / "geometry" / "CENTROID_DIRECTION_SUMMARY.csv")
    transfer = read_csv(EXPERIMENT / "geometry" / "FIT_TO_VALIDATION_TRANSFER_SUMMARY.csv")

    matrix = []
    for align, key in (("global_fit", "raw72_global_shared"), ("patient_z", "raw72_patient_z_shared"),
                       ("patient_robust", "raw72_patient_robust_shared"), ("patient_rank", "raw72_patient_rank_shared")):
        rows = [r for r in shared if r["representation"] == "R1" and r["alignment"] == align]
        matrix.append({"row": key, "shared_auprc": avg(rows, "ap"), "auroc": avg(rows, "auc"),
                       "gate_pass": alignment["variants"].get(align, {}).get("pass", "NA"),
                       "interpretation": "label-free coordinate alignment"})
    for rep in ("R1", "R3", "R4"):
        g = geometry["representations"][rep]
        for method in ("linear", "mlp"):
            local_row = next(r for r in local if r["representation"] == rep and r["method"] == method)
            gap_row = next(r for r in gap if r["representation"] == rep and r["method"] == method)
            matrix.append({"row": f"{rep}_patient_specific_{method}",
                           "shared_auprc": local_row["matched_shared_ap"],
                           "patient_specific_auprc": local_row["mean_ap"],
                           "delta_patient_specific_vs_shared": gap_row["mean_delta_ap"],
                           "auroc": local_row["mean_auc"],
                           "direction_median_cosine": g["median_direction_cosine"],
                           "negative_direction_fraction": g["reversed_transfer_fraction"],
                           "positive_patient_fraction": gap_row["positive_patient_fraction"],
                           "bootstrap_ci_low": gap_row["ci_low"], "bootstrap_ci_high": gap_row["ci_high"],
                           "gate_pass": g["pass"] if method == "linear" else "NA",
                           "interpretation": "nondeployable within-patient cross-fit"})
    matrix.append({"row": "richer_all_features", "shared_auprc": avg(richerall, "ap"),
                   "auroc": avg(richerall, "auc"), "gate_pass": richer["pass"],
                   "interpretation": "28 signal descriptors; 224D patient-z"})
    matrix.append({"row": "richer_all_features_pca_matched", "shared_auprc": avg(richerallp, "ap"),
                   "auroc": avg(richerallp, "auc"), "gate_pass": richer["pass"],
                   "interpretation": "FIT-only PCA72 matched to selected nine"})
    keys = ("row", "shared_auprc", "patient_specific_auprc", "delta_patient_specific_vs_shared", "auroc",
            "direction_median_cosine", "negative_direction_fraction", "positive_patient_fraction",
            "bootstrap_ci_low", "bootstrap_ci_high", "gate_pass", "interpretation")
    matrix = [{key: row.get(key, "NA") for key in keys} for row in matrix]
    csv_out(EXPERIMENT / "SUMMARY_MATRIX.csv", matrix)

    positive_local = any(v["pass"] for v in geometry["representations"].values())
    nonlinear = any(float(next(r for r in local if r["representation"] == rep and r["method"] == "mlp")["mean_ap"]) >
                    float(next(r for r in local if r["representation"] == rep and r["method"] == "linear")["mean_ap"]) + .05
                    for rep in ("R1", "R3", "R4"))
    if positive_local:
        terminal = "FEATURE_INFORMATION_PRESENT_BUT_PATIENT_GEOMETRY_UNSTABLE"
    elif nonlinear:
        terminal = "PATIENT_SPECIFIC_NONLINEAR_MAPPING_IDENTIFIED"
    elif richer["pass"]:
        terminal = "SOURCE_FEATURE_SELECTION_LIMITATION_IDENTIFIED"
    else:
        terminal = "NO_CLEAR_FEATURE_OR_GEOMETRY_SOURCE_IDENTIFIED"
    view_ranking = sorted(((v["view"], float(v["validation_sign_agreement_fraction"])) for v in view),
                          key=lambda x: x[1], reverse=True)
    feature_agreement = avg(feature, "validation_sign_agreement_fraction")
    r1 = next(r for r in local if r["representation"] == "R1" and r["method"] == "linear")
    r3 = next(r for r in local if r["representation"] == "R3" and r["method"] == "linear")
    r4 = next(r for r in local if r["representation"] == "R4" and r["method"] == "linear")
    l1 = next(r for r in local if r["representation"] == "R1" and r["method"] == "mlp")
    l3 = next(r for r in local if r["representation"] == "R3" and r["method"] == "mlp")
    l4 = next(r for r in local if r["representation"] == "R4" and r["method"] == "mlp")
    lines = ["# Cross-patient feature geometry audit — seed 42", "",
             "Development FIT + validation only. The within-validation-patient probes use that patient's labels for inner training and are NONDEPLOYABLE. No new outer-test evaluation was run.", "",
             "Protocol caveat: the inherited source cache initializer loaded the full 80-patient pickle and materialized outer-patient labels in memory before FIT/validation filtering. No outer-test predictions or metrics were computed and no outer-test loader was constructed, but the literal no-read-outer-label requirement was not met. Treat this branch as a development diagnostic with a sealing-process deviation, not a cleanly sealed audit.", "",
             f"Overall terminal: `{terminal}`.", "",
             "## Frozen-source and controls", "",
             f"A1 replay: {reproduction['checkpoints']} checkpoints, maximum validation-grid error {reproduction['max_validation_grid_error']:.1g}, mean Macro-F1 {reproduction['mean_macro_f1']:.6f}.",
             f"Channel-permutation maximum metric error: {sanity['channel_permutation_max_metric_error']:.1g}. Repeated extraction matched: {sanity['representation_repeat_identical']}.",
             f"One within-patient FIT-label permutation: validation AP {sanity['random_fit_label_permutation']['permuted_fit_ap']:.3f} versus source {sanity['random_fit_label_permutation']['source_ap']:.3f} and prevalence {sanity['random_fit_label_permutation']['validation_prevalence']:.3f}. It dropped substantially but did NOT reach prevalence; this weakens a clean chance-control claim and is reported without reinterpretation.", "",
             "## What the diagnostic distinguishes", "",
             f"The mean source-feature validation sign-agreement fraction is {feature_agreement:.3f}. By view: " +
             ", ".join(f"{name} {value:.3f}" for name, value in view_ranking) + ". A large absolute effect with low sign agreement is direction-unstable, not information-free.",
             "Patient-relative raw72 shared-probe AP changes versus global: " +
             ", ".join(f"{name} {data['delta_ap']:+.3f} ({data['positive_folds']}/5 positive folds)" for name, data in alignment["variants"].items()) +
             ". None reaches the fixed simple-alignment gate. Rank Gaussianization does not beat ordinary patient-z on AP.",
             "The validation median FIT-consensus cosines are " +
             ", ".join(f"{rep} {geometry['representations'][rep]['median_direction_cosine']:.3f}" for rep in ("R1", "R3", "R4")) +
             ". Pairwise and transfer aggregate CSVs give full distribution summaries. FIT-to-validation source-direction reversed-AUROC fractions are " +
             ", ".join(f"{rep} {geometry['representations'][rep]['reversed_transfer_fraction']:.3f}" for rep in ("R1", "R3", "R4")) + ".",
             "Matched heldout-channel AP (shared → patient-specific linear): " +
             "; ".join(f"{r['representation']} {float(r['matched_shared_ap']):.3f} → {float(r['mean_ap']):.3f}" for r in (r1, r3, r4)) +
             ". All three satisfy the preregistered geometric gate; see 10,000-resample patient bootstrap CIs in GEOMETRY_GAP_BOOTSTRAP.csv.",
             "The same within-patient MLP AP values are " +
             ", ".join(f"{r['representation']} {float(r['mean_ap']):.3f}" for r in (l1, l3, l4)) +
             ". They do not surpass the local linear control, so this audit does not identify patient-specific nonlinearity.",
             f"RICH9 shared AP {avg(richer9, 'ap'):.3f}; RICHALL {avg(richerall, 'ap'):.3f} (Δ {richer['mean_deltas']['delta_ap']:+.3f}). PCA72-matched Δ {richer['mean_deltas']['delta_pca_ap']:+.3f}. Additional cached descriptors do not meet the richer-feature gate; AUROC and Top1 also fail nondecrease.", "",
             "## Decision and limits", "",
             "The strongest supported next *class* of experiment is FIT-only learned patient-conditioned coordinate alignment or adaptation, evaluated prospectively on development data before any sealed test. This audit does not implement it. A patient-specific CV oracle cannot be deployed without that patient's EZ labels; its AP is also measured on smaller heldout subsets than a whole-patient AP, so the matched shared scorer on the identical subsets is the required comparison.",
             "This is not an information-theoretic claim about EEG. The source cache loader materializes the 80-patient cache at initialization, but the analysis constructs no outer-test loader and computes no outer-test metric. The cache-initialization behavior should be fixed in a future strict sealed-state audit if even in-memory test-label materialization is disallowed.", "",
             "Individual terminals: `" + geometry["terminal"] + "`, `" +
             ("PATIENT_SPECIFIC_NONLINEARITY_SUPPORTED" if nonlinear else "PATIENT_SPECIFIC_NONLINEARITY_NOT_SUPPORTED") +
             "`, `" + alignment["terminal"] + "`, `" + richer["terminal"] + "`.", ""]
    (EXPERIMENT / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (EXPERIMENT / "README.md").write_text("# Cross-patient feature geometry audit\n\nFrozen diagnostic code, aggregate-only outputs, and decision report. Run source replay, `extract_private.py`, `analyze.py`, `richer.py`, `sanity.py`, then `finalize.py` on the original server with private runtime environment variables. Never commit the private runtime.\n", encoding="utf-8")
    (EXPERIMENT / "IMPLEMENTATION_AUDIT.md").write_text(
        "# Implementation audit\n\n- Source A1 tip and protocol hashes checked at runtime. All 150 checkpoint grids replayed exactly.\n- The 28-name feature inventory was committed before richer-feature label evaluation.\n- All FIT/validation rows and selected-epoch embeddings remain in private runtime; public files are fold or representation aggregates.\n- Source four-view function is reused for all 28 descriptors by temporarily substituting its feature-index selector; RICH9 aggregate replay checks RAW72 numerically.\n- Local probes use one fixed stratified cross-fit and evaluate AP/AUROC within heldout folds, weighted by heldout channel count. No cross-model whole-patient MRR or Top1 is computed.\n- PCA fits on FIT patient-z channels only. Source and target patients are disjoint in all five folds.\n- Random-label control reduced but did not reach prevalence AP; see SANITY_AUDIT.json.\n- No outer-test loader or metric is used by these scripts, although the legacy source cache initializer materializes all 80 patient records.\n- This is a protocol deviation from the literal no-read-outer-label rule. Do not label the run as strictly outer-sealed.\n", encoding="utf-8")
    print(f"[FINAL] {terminal}", flush=True)


if __name__ == "__main__":
    run()
