"""Aggregate private 65-cell audit into public, patient-anonymous diagnostics."""
from __future__ import annotations

import json
import pickle

import numpy as np
from scipy.stats import spearmanr

from common import LOCK_SHA, ROOT, RUNTIME, preflight, read_csv, write_csv, write_json

METHODS = [("D0", ""), ("D1", "")] + [(d, c) for d in ("D2", "D3", "D4")
                                           for c in ("R3_MARG", "R3_COV", "R4_MARG", "R4_COV")]
N_BOOT = 10000
RNG = np.random.default_rng(42)


def ci(values, weights):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) != 65 or not np.isfinite(values).all():
        raise RuntimeError("Bootstrap input must contain 65 finite target cells")
    if weights.shape != (N_BOOT, 65):
        raise RuntimeError("Patient-cluster bootstrap weight shape mismatch")
    samples = (weights @ values) / weights.sum(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def avg(values):
    return float(np.mean(values))


def q(values, quantile):
    return float(np.quantile(values, quantile))


def load_cells():
    cells = []
    for fold in range(1, 6):
        folder = RUNTIME / "private" / f"fold_{fold}"
        selected = read_csv(folder / "A1_VLOO_PRIVATE.csv")
        if len(selected) != 13:
            raise RuntimeError("Incomplete VLOO source")
        for row in selected:
            sid = row["subject_id"]
            path = folder / "cells" / f"cell_{sid.replace(':', '_')}.pkl"
            with path.open("rb") as f:
                cell = pickle.load(f)
            if cell["lock_sha"] != LOCK_SHA or cell["fold"] != fold or cell["epoch"] != int(row["selected_epoch"]) or cell["subject_id"] != sid:
                raise RuntimeError("Private cell provenance failed")
            if len(cell["predictions"]) != 14 or len(cell["wrong"]) != 8:
                raise RuntimeError("Incomplete predictor or wrong-context grid")
            cells.append(cell)
    if len(cells) != 65 or len({(r["fold"], r["subject_id"]) for r in cells}) != 65:
        raise RuntimeError("Expected 65 distinct fold-by-patient target cells")
    if len({r["subject_id"] for r in cells}) != 47:
        raise RuntimeError("Validation patient cluster count changed")
    return cells


def select(cells, method, context):
    output = []
    for cell in cells:
        rows = [r for r in cell["predictions"] if r["predictor"] == method and r["context"] == context]
        if len(rows) != 1:
            raise RuntimeError(f"Missing matched predictor {method}/{context}")
        output.append(rows[0])
    return output


def values(rows, key):
    return np.asarray([r[key] for r in rows], dtype=np.float64)


def fold_signs(cells, delta):
    return sum(avg(delta[np.array([c["fold"] == f for c in cells])]) > 0 for f in range(1, 6))


def main():
    preflight()
    replay = json.loads((ROOT / "SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))
    if not replay["pass"] or replay["checkpoints"] != 150 or replay["max_grid_error"] > 1e-6:
        raise RuntimeError("SOURCE_A1_REPRODUCTION_FAILED")
    cells = load_cells()
    amendment = json.loads((ROOT / "BOOTSTRAP_UNIT_AMENDMENT.json").read_text(encoding="utf-8"))
    if amendment["unique_validation_patients"] != 47 or amendment["evaluation_cells_unchanged"] != 65:
        raise RuntimeError("Bootstrap membership amendment changed")
    unique = sorted({cell["subject_id"] for cell in cells})
    cluster = np.asarray([unique.index(cell["subject_id"]) for cell in cells])
    draws = RNG.integers(0, len(unique), size=(N_BOOT, len(unique)))
    counts = np.zeros((N_BOOT, len(unique)), dtype=np.int16)
    np.add.at(counts, (np.arange(N_BOOT)[:, None], draws), 1)
    indices = counts[:, cluster].astype(np.float64)
    d0, d1 = select(cells, "D0", ""), select(cells, "D1", "")
    d0cos, d1cos, d1ap = values(d0, "cosine"), values(d1, "cosine"), values(d1, "ap")
    wrong_map = {}
    for method, context in METHODS:
        if method not in {"D3", "D4"}:
            continue
        wrong_map[(method, context)] = [next(r for r in c["wrong"] if r["predictor"] == method and r["context"] == context) for c in cells]

    # The label-using references are explicitly upper bounds, not deployable predictors.
    refs = []
    for label, path in (("A1_frozen", "source_a1"), ("D1_shared", None),
                        ("rank4_oracle_label_using", "rank4"), ("full_oracle_label_using", "oracle")):
        rows = d1 if path is None else [c[path] for c in cells]
        refs.append({"reference": label, "patients": 65,
                     **{f"mean_{m}": avg(values(rows, m)) for m in ("ap", "auc", "mrr", "top1", "ndcg")}})
    write_csv(ROOT / "oracle_directions" / "ORACLE_HEADROOM_SUMMARY.csv", refs)

    stability = np.asarray([c["oracle_stability"] for c in cells])
    stability_rows = []
    for context in ("R3_MARG", "R3_COV", "R4_MARG", "R4_COV"):
        pred = select(cells, "D4", context)
        stability_rows.append({"context": context, "patients": 65,
                               **{k: fn(stability) for k, fn in (("mean", avg), ("median", lambda x: q(x, .5)),
                                                               ("q10", lambda x: q(x, .1)), ("q25", lambda x: q(x, .25)),
                                                               ("q75", lambda x: q(x, .75)), ("q90", lambda x: q(x, .9)))},
                               "spearman_stability_vs_oracle_ap": float(spearmanr(stability, [c["oracle"]["ap"] for c in cells]).statistic),
                               "spearman_stability_vs_D4_cosine": float(spearmanr(stability, values(pred, "cosine")).statistic),
                               "spearman_stability_vs_D4_ap_gain_over_shared": float(spearmanr(stability, values(pred, "ap") - d1ap).statistic)})
    write_csv(ROOT / "oracle_directions" / "ORACLE_DIRECTION_STABILITY.csv", stability_rows)

    evr_rows = []
    for rank in (1, 2, 4, 8):
        v = np.asarray([c["evr"][str(rank)] for c in cells])
        evr_rows.append({"rank": rank, "target_cells": 65, "mean_cumulative_evr": avg(v),
                         "median_cumulative_evr": q(v, .5), "min_cumulative_evr": float(v.min()),
                         "max_cumulative_evr": float(v.max())})
    write_csv(ROOT / "oracle_directions" / "ORACLE_DIRECTION_SUBSPACE_EVR.csv", evr_rows)

    oracle_ap = np.asarray([c["oracle"]["ap"] for c in cells])
    rank4_ap = np.asarray([c["rank4"]["ap"] for c in cells])
    rank4_cos = np.asarray([c["rank4"]["cosine"] for c in cells])
    denominator = avg(oracle_ap) - avg(d1ap)
    retention = (avg(rank4_ap) - avg(d1ap)) / denominator if denominator > 1e-12 else float("-inf")
    rank_rows = []
    for fold in (0, 1, 2, 3, 4, 5):
        keep = np.ones(65, dtype=bool) if fold == 0 else np.asarray([c["fold"] == fold for c in cells])
        rank_rows.append({"fold": "all" if fold == 0 else fold, "patients": int(keep.sum()),
                          "median_reconstruction_cosine": q(rank4_cos[keep], .5),
                          "fraction_cosine_ge_0_75": avg(rank4_cos[keep] >= .75),
                          "mean_reconstruction_ap": avg(rank4_ap[keep]),
                          "mean_full_oracle_ap": avg(oracle_ap[keep]),
                          "mean_shared_ap": avg(d1ap[keep]),
                          "headroom_retention": retention if fold == 0 else None})
    write_csv(ROOT / "oracle_directions" / "RANK4_ORACLE_RECONSTRUCTION.csv", rank_rows)
    rank_checks = {"median_cosine_ge_0_85": q(rank4_cos, .5) >= .85,
                   "fraction_cosine_ge_0_75_at_least_0_75": avg(rank4_cos >= .75) >= .75,
                   "ap_headroom_retention_ge_0_90": retention >= .90}
    rank_pass = all(rank_checks.values())
    rank_terminal = "RANK4_ORACLE_SUBSPACE_SUFFICIENT" if rank_pass else "RANK4_ORACLE_SUBSPACE_INSUFFICIENT"
    write_json(ROOT / "oracle_directions" / "RANK4_EXPRESSIVITY_GATE.json",
               {"pass": rank_pass, "terminal": rank_terminal, "checks": rank_checks,
                "median_cosine": q(rank4_cos, .5), "fraction_ge_0_75": avg(rank4_cos >= .75),
                "headroom_retention": retention})

    association = []
    for context in ("R3_MARG", "R3_COV", "R4_MARG", "R4_COV"):
        rho = np.asarray([c["association"][context]["rho"] for c in cells])
        counts = np.asarray([c["association"][context]["n_pairs"] for c in cells])
        association.append({"context": context, "target_cells": 65, "pair_observations_with_repeated_fit_cohorts": int(counts.sum()),
                            "mean_fit_pair_spearman_rho": avg(rho), "median_fit_pair_spearman_rho": q(rho, .5),
                            "bootstrap_ci_low": ci(rho, indices)[0], "bootstrap_ci_high": ci(rho, indices)[1],
                            "bootstrap_unit": "47 patient-ID clusters; FIT pairs within repeated epochs are dependent"})
    write_csv(ROOT / "contexts" / "CONTEXT_DIRECTION_ASSOCIATION.csv", association)
    write_json(ROOT / "contexts" / "CONTEXT_SPECIFICATION.json",
               {"context_sources": ["exact R3 channel-attention input", "exact R4 original-classifier input"],
                "pca_dim": 8, "pca_fit": "all FIT patient channels, current fold and selected epoch only",
                "MARG": "8 means + 8 population standard deviations",
                "COV": "MARG + 36 upper covariance entries including diagonal; population normalization",
                "validation_label_inputs": False})

    summary, direction_boot, ranking_boot, wrong_public, gates = [], [], [], [], {}
    for method, context in METHODS:
        rows = select(cells, method, context)
        co, ap = values(rows, "cosine"), values(rows, "ap")
        dc0, dc1, da1 = co - d0cos, co - d1cos, ap - d1ap
        coci, dc0ci, apci, da1ci = ci(co, indices), ci(dc0, indices), ci(ap, indices), ci(da1, indices)
        wrong_cos = wrong_ap = None
        if (method, context) in wrong_map:
            wrong = wrong_map[(method, context)]
            wrong_cos = avg(values(wrong, "correct_cosine") - values(wrong, "wrong_cosine"))
            wrong_ap = avg(values(wrong, "correct_ap") - values(wrong, "wrong_ap"))
            wrong_public.append({"predictor": method, "context": context, "patients": 65,
                                 "mean_correct_cosine": avg(values(wrong, "correct_cosine")),
                                 "mean_wrong_cosine": avg(values(wrong, "wrong_cosine")),
                                 "mean_correct_ap": avg(values(wrong, "correct_ap")),
                                 "mean_wrong_ap": avg(values(wrong, "wrong_ap")),
                                 "correct_minus_wrong_cosine": wrong_cos, "correct_minus_wrong_ap": wrong_ap})
        gate = False
        if method in {"D3", "D4"}:
            checks = {"mean_cos_ge_0_50": avg(co) >= .50,
                      "median_cos_ge_0_50": q(co, .5) >= .50,
                      "fraction_cos_better_than_D0_ge_0_65": avg(dc0 > 0) >= .65,
                      "cos_delta_D0_bootstrap_lower_gt_0": dc0ci[0] > 0,
                      "ap_gain_D1_ge_0_020": avg(da1) >= .020,
                      "positive_ap_folds_D1_ge_4": fold_signs(cells, da1) >= 4,
                      "ap_delta_D1_bootstrap_lower_gt_0": da1ci[0] > 0,
                      "correct_minus_wrong_ap_ge_0_010": wrong_ap >= .010}
            gate = all(checks.values())
            gates[f"{method}_{context}"] = {"checks": checks, "pass": gate}
        row = {"predictor": method, "context_source": context.split("_")[0] if context else "",
               "context_summary": context.split("_")[1] if context else "", "patients": 65,
               "mean_cosine": avg(co), "median_cosine": q(co, .5),
               "fraction_cos_gt_0": avg(co > 0), "fraction_cos_gt_0_5": avg(co > .5),
               "fraction_cos_gt_0_75": avg(co > .75),
               "mean_ap": avg(ap), "mean_auc": avg(values(rows, "auc")),
               "mean_mrr": avg(values(rows, "mrr")), "mean_top1": avg(values(rows, "top1")),
               "mean_ndcg": avg(values(rows, "ndcg")),
               "delta_cos_vs_population": avg(dc0), "delta_cos_vs_shared": avg(dc1),
               "delta_ap_vs_shared": avg(da1), "positive_ap_folds_vs_shared": fold_signs(cells, da1),
               "mean_projection_recovery_l2": avg(values(rows, "projection_recovery_l2")) if method == "D4" else None,
               "wrong_context_delta_cos": wrong_cos, "wrong_context_delta_ap": wrong_ap,
               "bootstrap_cos_ci_low": coci[0], "bootstrap_cos_ci_high": coci[1],
               "bootstrap_cos_delta_D0_ci_low": dc0ci[0], "bootstrap_cos_delta_D0_ci_high": dc0ci[1],
               "bootstrap_ap_ci_low": apci[0], "bootstrap_ap_ci_high": apci[1],
               "bootstrap_ap_delta_ci_low": da1ci[0], "bootstrap_ap_delta_ci_high": da1ci[1],
               "gate_pass": gate}
        summary.append(row)
        direction_boot.append({"predictor": method, "context": context, "patients": 65,
                               "mean_cosine": avg(co), "median_cosine": q(co, .5),
                               "fraction_cos_gt_0": avg(co > 0), "fraction_cos_gt_0_5": avg(co > .5),
                               "fraction_cos_gt_0_75": avg(co > .75),
                               "mean_delta_vs_D0": avg(dc0), "mean_delta_vs_D1": avg(dc1),
                               "cos_ci_low": coci[0], "cos_ci_high": coci[1],
                               "delta_D0_ci_low": dc0ci[0], "delta_D0_ci_high": dc0ci[1]})
        ranking_boot.append({"predictor": method, "context": context, "patients": 65,
                             "mean_ap": avg(ap), "median_ap": q(ap, .5),
                             "mean_delta_ap_vs_D1": avg(da1), "ap_ci_low": apci[0], "ap_ci_high": apci[1],
                             "delta_ap_ci_low": da1ci[0], "delta_ap_ci_high": da1ci[1]})
    write_csv(ROOT / "PREDICTABILITY_SUMMARY.csv", summary)
    for method, file in (("D0", "D0_POPULATION_MEAN.csv"), ("D1", "D1_SHARED_LINEAR.csv"),
                         ("D2", "D2_KNN_BY_CONTEXT.csv"), ("D3", "D3_FULL_RIDGE_BY_CONTEXT.csv"),
                         ("D4", "D4_RANK4_RIDGE_BY_CONTEXT.csv")):
        write_csv(ROOT / "prediction" / file, [r for r in summary if r["predictor"] == method])
    write_csv(ROOT / "controls" / "WRONG_CONTEXT_COMPARISON.csv", wrong_public)
    write_csv(ROOT / "statistics" / "DIRECTION_BOOTSTRAP.csv", direction_boot)
    write_csv(ROOT / "statistics" / "RANKING_BOOTSTRAP.csv", ranking_boot)
    predict_pass = any(v["pass"] for v in gates.values())
    predict_terminal = ("UNLABELED_CONTEXT_PREDICTS_ORACLE_DIRECTION" if predict_pass else
                        "UNLABELED_CONTEXT_DOES_NOT_PREDICT_ORACLE_DIRECTION")
    d3_pass = any(v["pass"] for k, v in gates.items() if k.startswith("D3_"))
    d4_pass = any(v["pass"] for k, v in gates.items() if k.startswith("D4_"))
    if rank_pass and d4_pass:
        overall = "LOWRANK_PATIENT_READOUT_IS_JUSTIFIED"
    elif rank_pass and not d4_pass and not d3_pass:
        overall = "ORACLE_GEOMETRY_EXISTS_BUT_NOT_IDENTIFIABLE_FROM_CURRENT_CONTEXT"
    elif not rank_pass and d3_pass:
        overall = "PATIENT_DIRECTION_PREDICTABLE_BUT_RANK4_TOO_RESTRICTIVE"
    else:
        overall = "CURRENT_ORACLE_DIRECTION_TRANSFER_MECHANISM_UNRESOLVED"
    write_json(ROOT / "PREDICTABILITY_GATE.json", {"pass": predict_pass, "terminal": predict_terminal,
               "rank4_terminal": rank_terminal, "overall_terminal": overall, "predeclared_candidates": gates,
               "d3_any_pass": d3_pass, "d4_any_pass": d4_pass})
    write_json(ROOT / "LABEL_USAGE_AUDIT.json", {
        "fit_labels_oracle_direction": True, "fit_labels_shared_logistic": True,
        "validation_labels_target_oracle_and_metrics_only": True,
        "validation_labels_context_input_or_predictor_fit": False,
        "validation_labels_rank4_expressivity_upper_bound": True,
        "outer_prediction_metric_training_or_selection": False,
        "legacy_monolithic_cache_materializes_all_80_labels": True,
        "strict_no_outer_label_materialization_satisfied": False,
        "validation_target_cells": 65, "unique_validation_patient_clusters": 47})
    (ROOT / "README.md").write_text("# Oracle direction predictability audit\n\nFrozen A1 development-only diagnostic. Run `extract.py`, then `analyze.py`, then `finalize.py` on the original server. `ODPA_RUNTIME` and `A1_A2_RUNTIME` point to private absolute directories. Public outputs contain aggregate metrics only, not target-level rows, embeddings, labels, checkpoints or caches. No outer predictions or metrics were computed.\n", encoding="utf-8")
    (ROOT / "IMPLEMENTATION_AUDIT.md").write_text(
        "# Implementation audit\n\n- Source branch is exact A1 tip `b2871b32873b67d0e6155d2766ef040ee1e9df01`; source lock and input/source hashes checked.\n"
        "- All 150 A1 validation grids re-evaluated within 1e-6 before audit. Other-12 VLOO selects one source epoch per target. FIT and target representations use the same epoch.\n"
        "- R3 is captured at the original patient-channel attention input, after source patient-relative normalization. R4 is captured at the original classifier input; classifier replay is exact to 1e-6. No approximate R3 reimplementation.\n"
        "- FIT-only oracle directions, context PCA, SVD basis, scalers and predictors are rebuilt in each target's selected-epoch coordinates. Target labels are used only for the nondeployable target oracle, expressivity bound and evaluation after prediction.\n"
        "- The original frozen A1 attention computes each patient's channel representations jointly. All unlabeled channels contribute to R3/R4; no target label is an input to context or direction predictors.\n"
        "- The inherited monolithic cache initializer materializes labels for all 80 patients before role filtering. Literal no-outer-label-materialization is not met. No outer loader, prediction, metric or outcome-based model choice was made. This is exploratory development, not sealed confirmation.\n"
        "- Patient-level cell records, direction vectors, teacher models and caches remain in the private runtime. There are 65 fold-by-patient validation cells but only 47 distinct patients. CIs use 10,000 paired patient-ID cluster resamples, preserving all repeated cells; see BOOTSTRAP_UNIT_AMENDMENT.json. FIT patient pairs are repeated across selected epochs and are not independent.\n", encoding="utf-8")
    best = max((r for r in summary if r["predictor"] in {"D3", "D4"}), key=lambda r: r["mean_ap"])
    d2_best = max((r for r in summary if r["predictor"] == "D2"), key=lambda r: r["mean_ap"])
    association_text = "; ".join(f"{r['context']} rho={r['mean_fit_pair_spearman_rho']:+.3f}" for r in association)
    lines = ["# Oracle-direction predictability audit — development-only", "",
             "No outer predictions or metrics were computed. The legacy cache initializer did materialize all 80 labels, so the literal no-outer-label-read requirement was not satisfied; this is not a sealed result. The 65 validation cells comprise only 47 distinct patients; 10,000 paired bootstrap resamples use patient-ID clusters, as recorded before aggregate gating in BOOTSTRAP_UNIT_AMENDMENT.json.", "",
             f"Exact A1 replay: {replay['checkpoints']}/150 checkpoints, max grid error {replay['max_grid_error']:.2g}; mean VLOO Macro-F1 {replay['mean_macro_f1']:.6f}. R3/R4 dimension 64 and R4 classifier replay passed.", "",
             "## Fixed 65-cell findings", "",
             f"Oracle-direction inner-fit stability: mean {avg(stability):.3f}, median {q(stability,.5):.3f}, q10/q90 {q(stability,.1):.3f}/{q(stability,.9):.3f}. No low-stability patient was excluded.",
             "FIT oracle-direction cumulative variance explained: " + ", ".join(f"rank {r['rank']} {r['mean_cumulative_evr']:.3f}" for r in evr_rows) + ".",
             f"Label-using full oracle AP {avg(oracle_ap):.4f}; label-using rank-4 projection AP {avg(rank4_ap):.4f}, median direction cosine {q(rank4_cos,.5):.3f}, fraction cosine >=.75 {avg(rank4_cos>=.75):.3f}, shared FIT logistic AP {avg(d1ap):.4f}, rank-4 headroom retention {retention:.3f}. These oracle figures are nondeployable upper bounds.",
             f"FIT-only unlabeled context-distance/direction-cosine associations: {association_text}. Negative rho would mean closer contexts have more similar oracle directions; repeated FIT pairs make the bootstrap descriptive.",
             f"Best context KNN D2 AP {d2_best['mean_ap']:.4f} ({d2_best['context_source']}_{d2_best['context_summary']}); best D3/D4 mean AP {best['mean_ap']:.4f} ({best['predictor']}_{best['context_source']}_{best['context_summary']}) versus shared D1 {avg(d1ap):.4f} and frozen A1 {refs[0]['mean_ap']:.4f}.",
             "The four predeclared R3/R4 marginal/covariance contexts, all D2/D3/D4 directions, wrong-context controls, angle/cosine and paired 10,000-resample CIs are in the aggregate CSVs. The public tables report whether covariance adds information; no context was added or tuned after inspection.", "",
             "## Causal interpretation", "",
             "The prior patient-specific direction headroom can be genuine while remaining nondeployable. This audit separates representational rank-4 expressivity from prediction using zero-label patient statistics; the three outcomes are not interchangeable.",
             "A direct patient-conditioned low-rank readout is justified only if both fixed rank-4 and D4 predictability gates pass. No adapter or outer test was run.", "",
             f"`{rank_terminal}`", f"`{predict_terminal}`", f"`{overall}`"]
    (ROOT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[FINAL]", rank_terminal, predict_terminal, overall, flush=True)


if __name__ == "__main__":
    main()
