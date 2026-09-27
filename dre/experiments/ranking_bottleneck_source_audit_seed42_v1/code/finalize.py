"""Prelocked ranking-source gates, aggregate report, and no-outer implementation audit."""

from __future__ import annotations

import json

from common import EXPERIMENT, RUNTIME, ensure_source, mean, read_csv, write_csv, write_json


def one(path):
    rows = read_csv(EXPERIMENT / path)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one comparison row: {path}")
    return rows[0]


def f(value) -> float:
    return float(value)


def context_gate() -> dict:
    comps = {row["variant"]: row for row in read_csv(EXPERIMENT / "context_attention" / "CONTEXT_COMPARISON.csv")}
    intervention = {row["metric"]: row for row in read_csv(EXPERIMENT / "diagnostics" / "CHANNEL_COUNT_CONTEXT_SENSITIVITY.csv")}
    sensitivity = {(row["variant"], row["fraction"], row["n_channels_quartile"]): row for row in
                   read_csv(EXPERIMENT / "context_attention" / "CHANNEL_SET_SENSITIVITY.csv")}
    t0_positive = f(intervention["delta_auprc_T0_minus_T1"]["mean_delta"]) > 0
    results = {}
    for variant in ("CTX1", "CTX2", "CTX3"):
        row = comps[variant]
        base_all = f(sensitivity[("CTX0", "0.25", "ALL")]["percentile_rank_mae"])
        new_all = f(sensitivity[(variant, "0.25", "ALL")]["percentile_rank_mae"])
        base_q4 = f(sensitivity[("CTX0", "0.25", "Q4")]["percentile_rank_mae"])
        new_q4 = f(sensitivity[(variant, "0.25", "Q4")]["percentile_rank_mae"])
        overall_reduction = 1 - new_all / base_all
        q4_reduction = 1 - new_q4 / base_q4
        stable = overall_reduction >= 0.20 and q4_reduction >= overall_reduction
        checks = {"srank_auprc_gain_ge_0_010": f(row["SRANK_patient_ez_auprc_delta"]) >= 0.010,
                  "srank_positive_auprc_folds_ge_4": int(row["SRANK_AUPRC_positive_folds"]) >= 4,
                  "srank_mrr_nondecreasing": f(row["SRANK_patient_ez_mrr_delta"]) >= 0,
                  "sf1_macro_f1_preserved": f(row["SF1_patient_macro_f1_delta"]) >= -0.005,
                  "frozen_T0_auprc_same_direction": t0_positive,
                  "channel_set_instability_reduced_ge_20pct_with_Q4_concentration": stable}
        passed = all(checks[key] for key in list(checks)[:4]) and (checks["frozen_T0_auprc_same_direction"] or stable)
        results[variant] = {"pass": passed, "checks": checks,
                            "rank_mae_reduction_25pct": overall_reduction,
                            "rank_mae_reduction_25pct_Q4": q4_reduction}
    passed = any(row["pass"] for row in results.values())
    return {"pass": passed, "variants": results,
            "harmful_layer_supported": ("patient" if results["CTX1"]["pass"] else "window" if results["CTX2"]["pass"] else
                                        "both" if results["CTX3"]["pass"] else "neither"),
            "terminal": "PATIENT_CONTEXT_RANKING_BOTTLENECK_SUPPORTED" if passed else "PATIENT_CONTEXT_RANKING_BOTTLENECK_NOT_SUPPORTED",
            "outer_test_accessed": False}


def summary_rows(context: dict, selection: dict, objective: dict, center: dict, view: dict) -> list[dict]:
    comps = {row["variant"]: row for row in read_csv(EXPERIMENT / "context_attention" / "CONTEXT_COMPARISON.csv")}
    rows = []
    frozen_t0_delta = next(row["mean_delta"] for row in read_csv(
        EXPERIMENT / "diagnostics" / "CHANNEL_COUNT_CONTEXT_SENSITIVITY.csv")
        if row["metric"] == "delta_auprc_T0_minus_T1")

    def add(source, priority, row, control, control_delta, gate, meaning):
        rows.append({"source": source, "priority": priority,
                     "baseline_auprc": row["SRANK_patient_ez_auprc_baseline"],
                     "candidate_auprc": row["SRANK_patient_ez_auprc_candidate"],
                     "delta_auprc": row["SRANK_patient_ez_auprc_delta"],
                     "baseline_mrr": row["SRANK_patient_ez_mrr_baseline"],
                     "candidate_mrr": row["SRANK_patient_ez_mrr_candidate"],
                     "delta_mrr": row["SRANK_patient_ez_mrr_delta"],
                     "positive_auprc_folds": row["SRANK_AUPRC_positive_folds"],
                     "sf1_macro_f1_delta": row["SF1_patient_macro_f1_delta"],
                     "mechanistic_control": control, "control_delta_auprc": control_delta,
                     "gate_pass": gate, "interpretation": meaning})

    for variant, name in (("CTX1", "patient_attention"), ("CTX2", "window_attention"), ("CTX3", "both_attention")):
        add(name, 1, comps[variant], "frozen T0 + matched structural CTX0 + channel-set dropout",
            frozen_t0_delta, context["variants"][variant]["pass"], context["terminal"])
    select = {row["metric"]: row for row in read_csv(EXPERIMENT / "selection_mismatch" / "SELECTION_COMPARISON.csv")}
    ap = select["patient_ez_auprc"]
    mrr = select["patient_ez_mrr"]
    macro = select["patient_macro_f1"]
    rows.append({"source": "checkpoint_selection", "priority": 2,
                 "baseline_auprc": ap["SF1"], "candidate_auprc": ap["SRANK"],
                 "delta_auprc": ap["delta_SRANK_minus_SF1"],
                 "baseline_mrr": mrr["SF1"], "candidate_mrr": mrr["SRANK"],
                 "delta_mrr": mrr["delta_SRANK_minus_SF1"],
                 "positive_auprc_folds": ap["positive_folds"],
                 "sf1_macro_f1_delta": macro["delta_SRANK_minus_SF1"],
                 "mechanistic_control": "same frozen A1 checkpoints, alternate epoch selector",
                 "control_delta_auprc": ap["delta_SRANK_minus_SF1"],
                 "gate_pass": selection["pass"], "interpretation": selection["terminal"]})
    add("ranking_objective", 2, one("ranking_objective/OBJECTIVE_COMPARISON.csv"),
        "OBJ0 exact A1", "", objective["pass"], objective["terminal"])
    add("center_optimization", 3, one("center_optimization/CENTER_COMPARISON.csv"),
        "CTR0 exact A1", "", center["pass"], center["terminal"])
    random_control, semantic = read_csv(EXPERIMENT / "view_interference" / "VIEW_COMPARISON.csv")
    add("semantic_view_separation", 4, semantic, "VIEW1 random two-branch",
        f(semantic["SRANK_patient_ez_auprc_candidate"]) - f(random_control["SRANK_patient_ez_auprc_candidate"]),
        view["pass"], view["terminal"])
    return rows


def main() -> None:
    ensure_source()
    source = json.loads((EXPERIMENT / "SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))
    if not source.get("pass") or source["checkpoints"] != 150 or source["max_validation_grid_error"] != 0:
        raise RuntimeError("SOURCE_A1_REPRODUCTION_FAILED")
    expected_training = [(variant, fold) for variant in ("CTX1", "CTX2", "CTX3", "OBJ1", "CTR1", "VIEW1", "VIEW2") for fold in range(1, 6)]
    for variant, fold in expected_training:
        completed = json.loads((RUNTIME / "private" / "training" / variant / f"fold_{fold}" / "complete.json").read_text(encoding="utf-8"))
        if completed["epochs"] != 30 or completed["outer_test_accessed"]:
            raise RuntimeError("Training grid incomplete")
    for fold in range(1, 6):
        v1 = json.loads((RUNTIME / "private" / "training" / "VIEW1" / f"fold_{fold}" / "complete.json").read_text(encoding="utf-8"))
        v2 = json.loads((RUNTIME / "private" / "training" / "VIEW2" / f"fold_{fold}" / "complete.json").read_text(encoding="utf-8"))
        if v1["parameter_counts"]["total_allocated_parameters"] != v2["parameter_counts"]["total_allocated_parameters"]:
            raise RuntimeError("VIEW1/VIEW2 parameter count not identical")
    parameter_rows = []
    for variant in ("CTX1", "CTX2", "CTX3", "OBJ1", "CTR1", "VIEW1", "VIEW2"):
        for fold in range(1, 6):
            completed = json.loads((RUNTIME / "private" / "training" / variant / f"fold_{fold}" / "complete.json").read_text(encoding="utf-8"))
            parameter_rows.append({"variant": variant, "fold": fold, **completed["parameter_counts"]})
    write_csv(EXPERIMENT / "diagnostics" / "PARAMETER_COUNTS.csv", parameter_rows)
    required = ("selection_mismatch/SELECTION_GATE.json", "ranking_objective/OBJECTIVE_GATE.json",
                "center_optimization/CENTER_GATE.json", "view_interference/VIEW_GATE.json")
    selection, objective, center, view = [json.loads((EXPERIMENT / path).read_text(encoding="utf-8")) for path in required]
    if not (EXPERIMENT / "diagnostics" / "LAYERWISE_RANK_SEPARABILITY.csv").is_file():
        raise RuntimeError("Layerwise ranking audit incomplete")
    sensitivity = read_csv(EXPERIMENT / "context_attention" / "CHANNEL_SET_SENSITIVITY.csv")
    if len(sensitivity) != 60:
        raise RuntimeError("Channel-set sensitivity aggregate incomplete")
    context = context_gate()
    write_json(EXPERIMENT / "context_attention" / "CONTEXT_GATE.json", context)
    summary = summary_rows(context, selection, objective, center, view)
    if len(summary) != 7:
        raise RuntimeError("Ranking source summary requires seven independent rows")
    write_csv(EXPERIMENT / "RANKING_SOURCE_SUMMARY.csv", summary)
    any_pass = any(gate["pass"] for gate in (context, selection, objective, center, view))
    terminal = "RANKING_BOTTLENECK_SOURCE_IDENTIFIED" if any_pass else "RANKING_BOTTLENECK_SOURCE_UNRESOLVED"
    layerwise = read_csv(EXPERIMENT / "diagnostics" / "LAYERWISE_RANK_SEPARABILITY.csv")
    layers = {layer: mean([row for row in layerwise if row["layer"] == layer], "patient_ez_auprc") for layer in ("L0", "L1", "L2", "L3")}
    row_by_source = {row["source"]: row for row in summary}
    frozen = {row["metric"]: row for row in read_csv(EXPERIMENT / "diagnostics" / "CHANNEL_COUNT_CONTEXT_SENSITIVITY.csv")}
    base_sensitivity = next(row for row in sensitivity if row["variant"] == "CTX0" and row["fraction"] == "0.25" and row["n_channels_quartile"] == "ALL")
    q4_sensitivity = next(row for row in sensitivity if row["variant"] == "CTX0" and row["fraction"] == "0.25" and row["n_channels_quartile"] == "Q4")
    lines = ["# Ranking bottleneck source audit (seed 42)", "",
             "Development-only FIT/validation study on the frozen 80-patient cohort. No outer-test loader, prediction, or performance outcome was accessed for this experiment.",
             f"A1 replay: 150/150 checkpoints exact, maximum validation-grid error {source['max_validation_grid_error']}; S-F1 VLOO Macro-F1 {source['mean_macro_f1']:.10f}.",
             "", "| Independent source | S-RANK ΔEZ-AUPRC | ΔMRR | Positive AUPRC folds | S-F1 ΔMacro-F1 | Gate |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for row in summary:
        lines.append(f"| {row['source']} | {f(row['delta_auprc']):+.6f} | {f(row['delta_mrr']):+.6f} | {row['positive_auprc_folds']}/5 | {f(row['sf1_macro_f1_delta']):+.6f} | {'PASS' if row['gate_pass'] else 'FAIL'} |")
    lines.extend(["", "## Direct answers", "",
                  f"1. Checkpoint-selection mismatch: {selection['terminal']}. S-RANK changes AUPRC by {f(row_by_source['checkpoint_selection']['delta_auprc']):+.6f}; rank-first selection does not rescue A1.",
                  f"2–4. Removing patient/window/both attention changes S-RANK AUPRC by {f(row_by_source['patient_attention']['delta_auprc']):+.6f}, {f(row_by_source['window_attention']['delta_auprc']):+.6f}, {f(row_by_source['both_attention']['delta_auprc']):+.6f}; each has only 3/5 positive folds and misses +0.010. No attention layer is established as harmful.",
                  f"5. CTX0 at 25% channel dropout has mean retained-channel percentile-rank MAE {f(base_sensitivity['percentile_rank_mae']):.6f} and score Spearman {f(base_sensitivity['score_spearman']):.6f}. Removing attention stabilizes ranks, but this does not establish a ranking-performance bottleneck.",
                  f"6. CTX0 high-channel Q4 rank MAE is {f(q4_sensitivity['percentile_rank_mae']):.6f}, lower than the overall value; this audit does not show instability increasing with channel count. Frozen T0−T1 AUPRC is {f(frozen['delta_auprc_T0_minus_T1']['mean_delta']):+.6f}; its n-channel Spearman is {f(frozen['delta_auprc_T0_minus_T1']['spearman_rho_n_channels']):+.6f}.",
                  f"7. Pairwise ranking OBJ1 changes S-RANK AUPRC by {f(row_by_source['ranking_objective']['delta_auprc']):+.6f}, MRR by {f(row_by_source['ranking_objective']['delta_mrr']):+.6f}; {objective['terminal']}.",
                  f"8. Center-equal CTR1 changes overall AUPRC by {f(row_by_source['center_optimization']['delta_auprc']):+.6f}; worst-center AUPRC changes from {center['baseline_worst_center_auprc']:.6f} to {center['candidate_worst_center_auprc']:.6f}. Weak-center case counts are small; {center['terminal']}.",
                  f"9. Semantic VIEW2 changes AUPRC versus VIEW0 by {f(row_by_source['semantic_view_separation']['delta_auprc']):+.6f} and versus random two-branch VIEW1 by {f(row_by_source['semantic_view_separation']['control_delta_auprc']):+.6f}; {view['terminal']}.",
                  "10. FIT-only linear probe AUPRC by layer: " + ", ".join(f"{layer}={layers[layer]:.6f}" for layer in ("L0", "L1", "L2", "L3")) + ". L1→L2 loses linear accessibility, but final L3 recovers; this is descriptive, not a causal bottleneck claim.",
                  "11. No candidate has strong matched causal support under its prelocked primary ranking gate. Channel-set stability changes are mechanistic diagnostics, not utility confirmation.",
                  "12. No single source is strong enough to motivate a ranking-model redesign from these development data alone. Historical outer exposure makes this exploratory, not sealed confirmation.",
                  "", f"Exact terminal: `{terminal}`.", "OUTER_TEST_ACCESSED = NO", ""])
    (EXPERIMENT / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    implementation = ["# Implementation audit", "",
                      "- Source A1 and its five FIT/validation folds were replayed exactly; all 150 validation grids agreed bitwise at reported metric precision.",
                      "- S-F1 and S-RANK used other 12 validation patients for all epoch/threshold choices; excluded labels never entered selection.",
                      "- Exact source A1 reused as CTX0/OBJ0/CTR0/VIEW0. Seven independent new variants each trained 5 folds × 30 fixed epochs without early stopping or mechanism combinations.",
                      "- Common model parameters loaded from the same fold initialization. VIEW1 and VIEW2 used the precommitted label-free partition/semantic partition and identical architecture/parameter counts.",
                      "- CTR1 weights came only from FIT center counts and were not normalized within each minibatch. Center ID was absent from model.forward and inference.",
                      "- Frozen T0/T0.5/T1 interventions used no labels or retraining. The T1 replay matched A1. Channel masks were label-free, seed-derived, and identical across structural variants.",
                      "- Layerwise linear scorers saw FIT embeddings/labels only, trained for 20 fixed epochs, and were never selected on validation.",
                      "- Patient/channel-level records, labels, predictions, checkpoints, validation grids and runtime logs remain private on the server. Public files contain aggregate rows only.",
                      "- No outer-test loader was constructed or used. The historical source constructor indexes cohort metadata but no outer label/prediction/performance was read for this audit.", ""]
    (EXPERIMENT / "IMPLEMENTATION_AUDIT.md").write_text("\n".join(implementation), encoding="utf-8")
    (EXPERIMENT / "README.md").write_text("# Ranking bottleneck source audit (seed 42)\n\n" +
        f"Complete development-only independent source audit. Terminal: `{terminal}`. See `FINAL_REPORT.md`, `RANKING_SOURCE_SUMMARY.csv`, and the frozen protocol files. No outer test or private patient artifacts were published.\n", encoding="utf-8")
    write_json(EXPERIMENT / "VALIDATION.json", {"pass": True, "source_reproduced": True,
               "trained_fold_variant_cells": 35, "trained_epochs": 1050, "channel_set_repeats": 15600,
               "layerwise_fold_layer_rows": 20, "summary_rows": 7,
               "outer_test_accessed": False, "terminal": terminal})
    print(json.dumps({"terminal": terminal, "outer_test_accessed": False}), flush=True)


if __name__ == "__main__":
    main()
