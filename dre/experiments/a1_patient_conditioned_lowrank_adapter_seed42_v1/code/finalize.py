"""Public aggregate-only VLOO comparison, paired bootstrap, fixed gates and report."""
from __future__ import annotations

import json
import math
import pickle
from collections import defaultdict

import numpy as np
import torch

from adapter import FrozenR4Adapter
from common import ROOT, RUNTIME, preflight, read_csv, write_csv, write_json
from diagnostics import metrics

VARIANTS = ("P0", "P1", "P2", "P3")
PAIRS = (("P2", "P0"), ("P2", "P1"), ("P3", "P0"), ("P3", "P1"), ("P3", "P2"))


def fold_public(variant):
    folder = {"P0": "p0_source", "P1": "p1_global", "P2": "p2_conditioned", "P3": "p3_geometry_teacher"}[variant]
    return read_csv(ROOT / folder / f"{variant}_VLOO_BY_FOLD.csv")


def patient_rows(variant):
    results = {}
    for fold in range(1, 6):
        folder = RUNTIME / "private" / f"fold_{fold}"
        rows = read_csv(folder / f"{variant}_VLOO_PRIVATE.csv")
        if len(rows) != 13:
            raise RuntimeError("Expected 13 VLOO-selected patients")
        for row in rows:
            subject, epoch = row["subject_id"], int(row["selected_epoch"])
            if variant == "P0":
                with (folder / f"source_epoch_{epoch:02d}.pkl").open("rb") as stream:
                    payload = pickle.load(stream)
                pred = payload["val"][subject]
                logit = pred["source_nez_logit"]
                labels = pred["y_ez"]
            else:
                with (folder / variant / f"source_epoch_{epoch:02d}" / "validation_predictions.pkl").open("rb") as stream:
                    pred = pickle.load(stream)[subject]
                logit, labels = pred["nez_logit"], pred["y_ez"]
            fixed = metrics(subject, labels, logit)
            if abs(fixed["ap"] - float(row["patient_ez_auprc"])) > 1e-7 or abs(fixed["mrr"] - float(row["patient_ez_mrr"])) > 1e-7:
                raise RuntimeError("Selected-score ranking metric replay mismatch")
            results[(fold, subject)] = {"fold": fold, "subject": subject, "selected_epoch": epoch,
                                        "macro_f1": float(row["patient_macro_f1"]),
                                        "ez_f1": float(row["patient_ez_f1"]),
                                        "ap": fixed["ap"], "mrr": fixed["mrr"],
                                        "top1": fixed["top1"], "auroc": fixed["auroc"], "ndcg": fixed["ndcg"]}
    if len(results) != 65:
        raise RuntimeError("Expected 65 VLOO patient cases")
    return results


def summary(rows):
    return {metric: float(np.mean([row[metric] for row in rows.values()]))
            for metric in ("macro_f1", "ez_f1", "ap", "mrr", "top1", "auroc", "ndcg")}


def paired_bootstrap(records):
    rng = np.random.default_rng(42)
    output = []
    for candidate, reference in PAIRS:
        if set(records[candidate]) != set(records[reference]):
            raise RuntimeError("Bootstrap patient pairing changed")
        keys = sorted(records[candidate])
        for metric in ("ap", "mrr", "macro_f1"):
            delta = np.asarray([records[candidate][key][metric] - records[reference][key][metric] for key in keys])
            resampled = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(axis=1)
            output.append({"contrast": f"{candidate}-{reference}", "metric": metric, "patients": len(delta),
                           "mean_delta": float(delta.mean()), "median_delta": float(np.median(delta)),
                           "fraction_positive": float(np.mean(delta > 0)),
                           "ci_low": float(np.quantile(resampled, .025)), "ci_high": float(np.quantile(resampled, .975))})
    return output


def aggregate_controls():
    rows = read_csv(ROOT / "controls" / "SHUFFLED_CONTEXT_COMPARISON.csv")
    result = {}
    for variant in ("P2", "P3"):
        part = [row for row in rows if row["variant"] == variant]
        if len(part) != 5:
            raise RuntimeError("Wrong-context control missing fold")
        result[variant] = {"correct_ap": float(np.mean([float(r["mean_correct_ap"]) for r in part])),
                           "shuffled_ap": float(np.mean([float(r["mean_shuffled_ap"]) for r in part])),
                           "correct_mrr": float(np.mean([float(r["mean_correct_mrr"]) for r in part])),
                           "shuffled_mrr": float(np.mean([float(r["mean_shuffled_mrr"]) for r in part]))}
    return result


def coefficient_noncollapse(variant):
    folder = ROOT / ("p2_conditioned" if variant == "P2" else "p3_geometry_teacher")
    rows = read_csv(folder / f"{variant}_ADAPTER_DIAGNOSTICS.csv")
    if len(rows) != 5:
        raise RuntimeError("Adapter diagnostics missing fold")
    component_max = max(float(row[f"component_{k}_std"]) for row in rows for k in range(4))
    pairs = [float(row["pairwise_cosine_mean"]) for row in rows if row["pairwise_cosine_mean"] != "NA"]
    # Numerical-identity check, not an outcome-tuned effect threshold. Correct-vs-shuffled AP is the effect gate.
    return {"component_std_max": component_max, "same_epoch_pair_cosine_mean": float(np.mean(pairs)) if pairs else None,
            "not_numerically_constant": bool(component_max > 1e-6)}


def gates(records, means, folds):
    controls = aggregate_controls()
    conditioning = {}
    for variant in ("P2", "P3"):
        positive_p0 = sum(float(folds[variant][i]["patient_ez_auprc"]) > float(folds["P0"][i]["patient_ez_auprc"]) for i in range(5))
        positive_p1 = sum(float(folds[variant][i]["patient_ez_auprc"]) > float(folds["P1"][i]["patient_ez_auprc"]) for i in range(5))
        noncollapse = coefficient_noncollapse(variant)
        checks = {"ap_vs_p0_at_least_0_010": means[variant]["ap"] >= means["P0"]["ap"] + .010,
                  "ap_vs_p1_at_least_0_010": means[variant]["ap"] >= means["P1"]["ap"] + .010,
                  "ap_positive_vs_p0_4of5": positive_p0 >= 4,
                  "ap_positive_vs_p1_3of5": positive_p1 >= 3,
                  "mrr_nondecrease_vs_p0": means[variant]["mrr"] >= means["P0"]["mrr"],
                  "top1_nondecrease_vs_p0": means[variant]["top1"] >= means["P0"]["top1"],
                  "macro_f1_vs_p0": means[variant]["macro_f1"] >= means["P0"]["macro_f1"] - .005,
                  "ez_f1_vs_p0": means[variant]["ez_f1"] >= means["P0"]["ez_f1"] - .005,
                  "correct_context_ap_exceeds_shuffled_by_0_005": controls[variant]["correct_ap"] >= controls[variant]["shuffled_ap"] + .005,
                  "not_numerically_global": noncollapse["not_numerically_constant"]}
        conditioning[variant] = {"checks": checks, "pass": all(checks.values()),
                                 "positive_folds_vs_p0": positive_p0, "positive_folds_vs_p1": positive_p1,
                                 "control": controls[variant], "coefficient": noncollapse}
    positive_p3_p2 = sum(float(folds["P3"][i]["patient_ez_auprc"]) > float(folds["P2"][i]["patient_ez_auprc"]) for i in range(5))
    teacher_checks = {"ap_plus_0_005": means["P3"]["ap"] >= means["P2"]["ap"] + .005,
                      "positive_folds_3of5": positive_p3_p2 >= 3,
                      "mrr_nondecrease": means["P3"]["mrr"] >= means["P2"]["mrr"],
                      "top1_nondecrease": means["P3"]["top1"] >= means["P2"]["top1"],
                      "macro_f1_noninferior": means["P3"]["macro_f1"] >= means["P2"]["macro_f1"] - .005}
    condition_pass = any(row["pass"] for row in conditioning.values())
    teacher_pass = all(teacher_checks.values())
    write_json(ROOT / "comparisons" / "CONDITIONING_GATE.json", {"variants": conditioning, "pass": condition_pass,
               "terminal": "PATIENT_CONDITIONED_COORDINATE_ADAPTATION_SUPPORTED" if condition_pass else "PATIENT_CONDITIONED_COORDINATE_ADAPTATION_NOT_SUPPORTED"})
    write_json(ROOT / "comparisons" / "GEOMETRY_TEACHER_GATE.json", {"checks": teacher_checks,
               "positive_folds": positive_p3_p2, "pass": teacher_pass,
               "terminal": "GEOMETRY_TEACHER_SUPPORTED" if teacher_pass else "GEOMETRY_TEACHER_NOT_SUPPORTED"})
    return condition_pass, teacher_pass, controls


def main():
    preflight()
    reproduction = json.loads((ROOT / "SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))
    if not reproduction["pass"] or reproduction["checkpoints"] != 150:
        raise RuntimeError("A1 source not reproduced")
    records = {variant: patient_rows(variant) for variant in VARIANTS}
    if any(set(records[variant]) != set(records["P0"]) for variant in VARIANTS):
        raise RuntimeError("Variant VLOO patients not matched")
    folds = {variant: fold_public(variant) for variant in VARIANTS}
    means = {variant: summary(records[variant]) for variant in VARIANTS}
    main_rows = [{"variant": variant, "patients": 65, **means[variant],
                  "mean_predicted_ez_fraction": float(np.mean([float(r["predicted_ez_fraction"]) for r in folds[variant]])),
                  "mean_nez_f1": float(np.mean([float(r["patient_nez_f1"]) for r in folds[variant]])),
                  "mean_ba": float(np.mean([float(r["patient_balanced_accuracy"]) for r in folds[variant]]))}
                 for variant in VARIANTS]
    write_csv(ROOT / "comparisons" / "MAIN_COMPARISON.csv", main_rows)
    bootstrap = paired_bootstrap(records)
    write_csv(ROOT / "comparisons" / "PAIRED_BOOTSTRAP.csv", bootstrap)
    condition_pass, teacher_pass, controls = gates(records, means, folds)
    source_count = json.loads((RUNTIME.parent / "a1_a2_patient_equal_objective_seed42_v1" / "A1" / "fold_1" / "development_summary.json").read_text(encoding="utf-8"))["parameter_count"]
    param = {"frozen_a1_parameters": int(source_count), "r4_dimension": 64, "rank": 4}
    for variant in ("P1", "P2", "P3"):
        model = FrozenR4Adapter(64, variant)
        param[variant] = {"trainable_adapter_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                          "frozen_original_classifier": True, "frozen_a1": True}
    if param["P2"]["trainable_adapter_parameters"] != param["P3"]["trainable_adapter_parameters"]:
        raise RuntimeError("P2/P3 parameter count mismatch")
    write_json(ROOT / "PARAMETER_COUNT.json", param)
    write_json(ROOT / "LABEL_USAGE_AUDIT.json", {"fit_labels_classification_supervision": True,
               "fit_labels_geometry_teacher": "P3_only", "fit_labels_as_context_input": False,
               "validation_labels_adapter_training": False, "validation_labels_context_input": False,
               "validation_labels_geometry_teacher": False,
               "validation_labels_only_other_12_vloo_selection_and_excluded_patient_evaluation": True,
               "outer_predictions_metrics_training_selection": False,
               "legacy_monolithic_loader_materializes_all_80_labels": True,
               "strict_no_outer_label_materialization_satisfied": False})
    write_json(ROOT / "p3_geometry_teacher" / "FIT_TEACHER_AUDIT.json", {"folds": 5,
               "source_epochs_per_fold": 30, "fit_patients_per_fold": 52,
               "fit_patient_teacher_fits": 5 * 30 * 52,
               "validation_patient_teacher_fits": 0,
               "outer_patient_teacher_fits": 0,
               "positive_label": "EZ", "solver": "lbfgs", "C": 1.0})
    report = ["# Frozen A1 patient-conditioned low-rank adapter — development result", "",
              "This is development-only and exploratory on a historically viewed cohort. No outer predictions or metrics were computed. The monolithic legacy cache initializer nevertheless materialized all 80 patient labels; therefore the literal no-outer-label-read rule was not met and this is not a strictly sealed confirmation.", "",
              f"A1 replay: {reproduction['checkpoints']}/150 checkpoints, maximum grid error {reproduction['max_grid_error']:.1g}; exact R4 classifier-input dimension 64 obtained by hook because ordinary A1 forward does not expose the contextual embedding key.", "",
              "## VLOO patient-equal results", "",
              "| Variant | EZ-AP | EZ-MRR | Top1-EZ | EZ-AUROC | NDCG | Macro-F1 | EZ-F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        m = means[variant]
        report.append(f"| {variant} | {m['ap']:.4f} | {m['mrr']:.4f} | {m['top1']:.4f} | {m['auroc']:.4f} | {m['ndcg']:.4f} | {m['macro_f1']:.4f} | {m['ez_f1']:.4f} |")
    report += ["", "Primary matched AP contrasts: " + "; ".join(f"{a}-{b} {means[a]['ap']-means[b]['ap']:+.4f}" for a,b in PAIRS) + ".",
               "Correct-vs-wrong-context AP: " + "; ".join(f"{v} {controls[v]['correct_ap']:.4f} vs {controls[v]['shuffled_ap']:.4f}" for v in ("P2", "P3")) + ". Wrong context was diagnostic only and never selected a checkpoint.",
               "Coefficient variation, saturation, embedding perturbation and 50/70/100% context robustness are in the aggregate diagnostic CSVs. The patient-specific FIT teacher never uses validation labels. Patient-level paired bootstrap CIs are in PAIRED_BOOTSTRAP.csv.", "",
               "The previous patient-specific oracle was nondeployable; this zero-label context mechanism is supported only if its full fixed gate passes. A favorable Macro-F1 alone is insufficient.", "",
               "Terminals: `" + ("PATIENT_CONDITIONED_COORDINATE_ADAPTATION_SUPPORTED" if condition_pass else "PATIENT_CONDITIONED_COORDINATE_ADAPTATION_NOT_SUPPORTED") + "`; `" + ("GEOMETRY_TEACHER_SUPPORTED" if teacher_pass else "GEOMETRY_TEACHER_NOT_SUPPORTED") + "`.", ""]
    if not condition_pass:
        report += ["STOP_PATIENT_CONDITIONED_ADAPTER_EXPLORATION", ""]
    (ROOT / "FINAL_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (ROOT / "README.md").write_text("# Frozen A1 patient-conditioned rank-four adapter\n\nRun `prepare.py`, `train_adapters.py`, `diagnostics.py`, and `finalize.py` on the original server with private runtime variables. Source A1 remains frozen. Public outputs contain aggregate rows only; no patient-level records, embeddings, logits, teachers or checkpoints are committed.\n", encoding="utf-8")
    (ROOT / "IMPLEMENTATION_AUDIT.md").write_text(
        "# Implementation audit\n\n- A1 source modules/cache/split and protocol hashes are checked before execution. All 150 source validation grids replay within 1e-6.\n- Ordinary A1 forward does not expose `contextual_channel_embedding` in this branch; a pre-hook on the original classifier captures exact R4, verified by direct classifier logit replay. D=64.\n- A1 is frozen/eval; only adapter parameters enter AdamW. Zero-init P1/P2/P3 are identity. P2 and P3 have identical architecture/parameter count.\n- Context/query assignment is label-blind, deterministic by seed/fold/source epoch/adapter epoch/subject hash; only query labels enter A1 weighted BCE. P3 teacher fits FIT patient labels only.\n- The frozen A1 patient-attention stage forms each R4 from all patient channels before the adapter context/query split. Thus query *features* can affect context R4 indirectly; no query labels enter context. This is a representation-level dependency inherent to the mandated frozen R4 interface.\n- Adapter epoch 20 is fixed; source epoch and threshold use exact 13-patient VLOO. Wrong-context and subset controls are never selection inputs.\n- Each fold/source-epoch/variant saves only private resumable state, validation grids and patient arrays. Public outputs are aggregate only.\n- Same-source-epoch patient coefficient pairs are used for pairwise cosine; vectors from independently trained source epochs are not compared directly. Numerical noncollapse means coefficient component std exceeds 1e-6; the +0.005 correct-vs-shuffled AP gate supplies the material-effect test.\n- The inherited monolithic cache loader materializes all 80 patient labels at initialization, before role filtering. This violates the literal no-outer-label-read rule, although no outer loader/prediction/metric/training/selection is performed. Classify this run as development-only/non-sealed.\n", encoding="utf-8")
    print("[FINAL]", "PATIENT_CONDITIONED_COORDINATE_ADAPTATION_SUPPORTED" if condition_pass else "PATIENT_CONDITIONED_COORDINATE_ADAPTATION_NOT_SUPPORTED", flush=True)
    print("[FINAL]", "GEOMETRY_TEACHER_SUPPORTED" if teacher_pass else "GEOMETRY_TEACHER_NOT_SUPPORTED", flush=True)


if __name__ == "__main__":
    main()
