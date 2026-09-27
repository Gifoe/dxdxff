"""Aggregate complete independent training variants under both frozen VLOO selectors."""

from __future__ import annotations

import json
from collections import defaultdict

from common import (EXPERIMENT, RUNTIME, assert_no_outer_loader, build_fold, ensure_source,
                    make_experiment, mean, read_csv, source_grid, write_csv, write_json)
from selection import both_protocols
from train_variants import VARIANTS

METRICS = ("patient_ez_auprc", "patient_ez_mrr", "top1_is_ez", "patient_ez_auroc",
           "patient_macro_f1", "patient_ez_f1", "patient_balanced_accuracy", "predicted_ez_fraction")
BASES = {"CTX0", "OBJ0", "CTR0", "VIEW0"}


def grids(variant: str, fold: int) -> list[dict]:
    if variant in BASES:
        return [source_grid(fold, epoch) for epoch in range(1, 31)]
    folder = RUNTIME / "private" / "training" / variant / f"fold_{fold}"
    complete = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    if complete.get("epochs") != 30 or complete.get("outer_test_accessed"):
        raise RuntimeError(f"Incomplete or leaked training cell: {variant} fold={fold}")
    return [json.loads((folder / f"epoch_{epoch:02d}_validation_grid.json").read_text(encoding="utf-8"))
            for epoch in range(1, 31)]


def comparison(base_sf1: list[dict], base_rank: list[dict], candidate_sf1: list[dict],
               candidate_rank: list[dict], name: str) -> dict:
    row = {"variant": name}
    for prefix, baseline, candidate in (("SF1", base_sf1, candidate_sf1), ("SRANK", base_rank, candidate_rank)):
        for metric in METRICS:
            row[f"{prefix}_{metric}_baseline"] = mean(baseline, metric)
            row[f"{prefix}_{metric}_candidate"] = mean(candidate, metric)
            row[f"{prefix}_{metric}_delta"] = mean(candidate, metric) - mean(baseline, metric)
        row[f"{prefix}_AUPRC_positive_folds"] = sum(float(x["patient_ez_auprc"]) > float(b["patient_ez_auprc"])
                                                    for b, x in zip(baseline, candidate, strict=True))
    return row


def center_tables(sf1: dict[str, list[dict]], rank: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], dict]:
    exp = make_experiment()
    center_map = {}
    for split in exp.outer_splits:
        fold, _train, _train_loader, val_loader, test_loader, _norm = build_fold(exp, split, "validation")
        assert_no_outer_loader(test_loader)
        for item in val_loader.dataset.patient_examples:
            center_map[(fold, str(item["subject_id"]))] = str(item["center"])
    out = {}
    for variant in ("CTR0", "CTR1"):
        sf1_patient, rank_patient = [], []
        for fold in range(1, 6):
            sf1_patient.extend(read_csv(RUNTIME / "private" / f"{variant}_fold_{fold}_SF1_PATIENT.csv"))
            rank_patient.extend(read_csv(RUNTIME / "private" / f"{variant}_fold_{fold}_SRANK_PATIENT.csv"))
        if len(sf1_patient) != 65 or len(rank_patient) != 65:
            raise RuntimeError("Center validation patient counts changed")
        sf1_by = {(int(row["fold"]), row["subject_id"]): row for row in sf1_patient}
        rank_by = {(int(row["fold"]), row["subject_id"]): row for row in rank_patient}
        if set(sf1_by) != set(rank_by) or set(sf1_by) != set(center_map):
            raise RuntimeError("Center metadata/selection membership changed")
        grouped = defaultdict(list)
        for key, s in sf1_by.items():
            r = rank_by[key]
            grouped[center_map[key]].append({"subject_id": key[1], "SF1_MacroF1": float(s["patient_macro_f1"]),
                                               "SF1_EZ_F1": float(s["patient_ez_f1"]),
                                               "SRANK_AUPRC": float(r["patient_ez_auprc"]),
                                               "SRANK_AUROC": float(r["patient_ez_auroc"]),
                                               "SRANK_MRR": float(r["patient_ez_mrr"]),
                                               "SRANK_Top1": float(r["top1_is_ez"])})
        rows = []
        for center, items in sorted(grouped.items()):
            row = {"variant": variant, "center": center, "n_cases": len(items),
                   "n_patients": len(set(item["subject_id"] for item in items)), "small_n_caution": len(set(item["subject_id"] for item in items)) < 10}
            row.update({name: mean(items, name) for name in ("SF1_MacroF1", "SF1_EZ_F1", "SRANK_AUPRC", "SRANK_AUROC", "SRANK_MRR", "SRANK_Top1")})
            rows.append(row)
        out[variant] = rows
    return out, center_map


def center_gate(by_center: dict[str, list[dict]], overall: dict) -> dict:
    base = {row["center"]: row for row in by_center["CTR0"]}
    new = {row["center"]: row for row in by_center["CTR1"]}
    if set(base) != set(new):
        raise RuntimeError("Center set changed")
    weak = {key for key in ("lzu", "pediatric") if key in base}
    strong = {key for key in ("hup", "multicenter") if key in base}
    baseline_worst_ap = min(float(row["SRANK_AUPRC"]) for row in base.values())
    candidate_worst_ap = min(float(row["SRANK_AUPRC"]) for row in new.values())
    baseline_worst_f1 = min(float(row["SF1_MacroF1"]) for row in base.values())
    candidate_worst_f1 = min(float(row["SF1_MacroF1"]) for row in new.values())
    baseline_gap = max(float(row["SRANK_AUPRC"]) for row in base.values()) - baseline_worst_ap
    candidate_gap = max(float(row["SRANK_AUPRC"]) for row in new.values()) - candidate_worst_ap
    checks = {"overall_auprc_preserved": overall["SRANK_patient_ez_auprc_delta"] >= -0.005,
              "worst_center_auprc_gain_ge_0_020": candidate_worst_ap - baseline_worst_ap >= 0.020,
              "worst_center_macro_f1_gain_ge_0_010": candidate_worst_f1 - baseline_worst_f1 >= 0.010,
              "center_auprc_gap_reduced_ge_20pct": baseline_gap > 0 and candidate_gap <= 0.8 * baseline_gap,
              "weak_center_ranking_improves": any(float(new[key]["SRANK_AUPRC"]) > float(base[key]["SRANK_AUPRC"]) for key in weak),
              "strong_center_auprc_no_collapse": all(float(new[key]["SRANK_AUPRC"]) >= float(base[key]["SRANK_AUPRC"]) - 0.020 for key in strong),
              "overall_macro_f1_preserved": overall["SF1_patient_macro_f1_delta"] >= -0.005}
    return {"pass": all(checks.values()), "checks": checks, "baseline_worst_center_auprc": baseline_worst_ap,
            "candidate_worst_center_auprc": candidate_worst_ap, "baseline_center_auprc_gap": baseline_gap,
            "candidate_center_auprc_gap": candidate_gap, "small_center_caution": True,
            "terminal": "CENTER_OPTIMIZATION_RANKING_BOTTLENECK_SUPPORTED" if all(checks.values()) else "CENTER_OPTIMIZATION_RANKING_BOTTLENECK_NOT_SUPPORTED",
            "outer_test_accessed": False}


def main() -> None:
    ensure_source()
    source = json.loads((EXPERIMENT / "SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))
    if not source.get("pass"):
        raise RuntimeError("SOURCE_A1_REPRODUCTION_FAILED")
    all_sf1, all_rank = {}, {}
    for variant in (*sorted(BASES), *VARIANTS):
        sf1_rows, rank_rows = [], []
        for fold in range(1, 6):
            s, r = both_protocols(grids(variant, fold), variant, fold)
            sf1_rows.append(s); rank_rows.append(r)
        all_sf1[variant], all_rank[variant] = sf1_rows, rank_rows
        if variant.startswith("CTX"):
            output = EXPERIMENT / "context_attention"
            write_csv(output / f"{variant}_SF1_BY_FOLD.csv", sf1_rows)
            write_csv(output / f"{variant}_SRANK_BY_FOLD.csv", rank_rows)
        elif variant.startswith("OBJ"):
            output = EXPERIMENT / "ranking_objective"
            write_csv(output / f"{variant}_SF1_BY_FOLD.csv", sf1_rows)
            write_csv(output / f"{variant}_SRANK_BY_FOLD.csv", rank_rows)
        elif variant.startswith("CTR"):
            write_csv(EXPERIMENT / "center_optimization" / f"{variant}_BY_FOLD.csv", sf1_rows + rank_rows)
        else:
            write_csv(EXPERIMENT / "view_interference" / f"{variant}_BY_FOLD.csv", sf1_rows + rank_rows)
        print(f"[AGGREGATE] {variant} SRANK AUPRC={mean(rank_rows, 'patient_ez_auprc'):.6f}", flush=True)
    context = [comparison(all_sf1["CTX0"], all_rank["CTX0"], all_sf1[name], all_rank[name], name)
               for name in ("CTX1", "CTX2", "CTX3")]
    write_csv(EXPERIMENT / "context_attention" / "CONTEXT_COMPARISON.csv", context)
    objective = comparison(all_sf1["OBJ0"], all_rank["OBJ0"], all_sf1["OBJ1"], all_rank["OBJ1"], "OBJ1")
    write_csv(EXPERIMENT / "ranking_objective" / "OBJECTIVE_COMPARISON.csv", [objective])
    obj_checks = {"auprc_gain_ge_0_010": objective["SRANK_patient_ez_auprc_delta"] >= 0.010,
                  "auprc_positive_folds_ge_4": objective["SRANK_AUPRC_positive_folds"] >= 4,
                  "mrr_nondecreasing": objective["SRANK_patient_ez_mrr_delta"] >= 0,
                  "top1_nondecreasing": objective["SRANK_top1_is_ez_delta"] >= 0,
                  "macro_f1_preserved": objective["SF1_patient_macro_f1_delta"] >= -0.005}
    obj_pass = all(obj_checks.values())
    write_json(EXPERIMENT / "ranking_objective" / "OBJECTIVE_GATE.json", {"pass": obj_pass, "checks": obj_checks,
               "terminal": "RANKING_OBJECTIVE_MISMATCH_SUPPORTED" if obj_pass else "RANKING_OBJECTIVE_MISMATCH_NOT_SUPPORTED",
               "order_improves_decision_hurts": objective["SRANK_patient_ez_auprc_delta"] > 0 and objective["SF1_patient_macro_f1_delta"] < -0.005,
               "outer_test_accessed": False})
    centers, _ = center_tables(all_sf1, all_rank)
    for name, rows in centers.items():
        write_csv(EXPERIMENT / "center_optimization" / f"{name}_BY_CENTER.csv", rows)
    ctr = comparison(all_sf1["CTR0"], all_rank["CTR0"], all_sf1["CTR1"], all_rank["CTR1"], "CTR1")
    write_csv(EXPERIMENT / "center_optimization" / "CENTER_COMPARISON.csv", [ctr])
    write_json(EXPERIMENT / "center_optimization" / "CENTER_GATE.json", center_gate(centers, ctr))
    views = [comparison(all_sf1["VIEW0"], all_rank["VIEW0"], all_sf1[name], all_rank[name], name)
             for name in ("VIEW1", "VIEW2")]
    write_csv(EXPERIMENT / "view_interference" / "VIEW_COMPARISON.csv", views)
    random_control, semantic = views
    view_checks = {"semantic_vs_source_auprc_ge_0_010": semantic["SRANK_patient_ez_auprc_delta"] >= 0.010,
                   "semantic_vs_random_auprc_ge_0_010": semantic["SRANK_patient_ez_auprc_candidate"] - random_control["SRANK_patient_ez_auprc_candidate"] >= 0.010,
                   "semantic_vs_source_positive_folds_ge_4": semantic["SRANK_AUPRC_positive_folds"] >= 4,
                   "semantic_vs_random_positive_folds_ge_3": sum(float(s["patient_ez_auprc"]) > float(r["patient_ez_auprc"])
                                                                for r, s in zip(all_rank["VIEW1"], all_rank["VIEW2"], strict=True)) >= 3,
                   "mrr_nondecreasing_vs_source": semantic["SRANK_patient_ez_mrr_delta"] >= 0,
                   "mrr_nondecreasing_vs_random": semantic["SRANK_patient_ez_mrr_candidate"] >= random_control["SRANK_patient_ez_mrr_candidate"],
                   "macro_f1_preserved": semantic["SF1_patient_macro_f1_delta"] >= -0.005}
    view_pass = all(view_checks.values())
    write_json(EXPERIMENT / "view_interference" / "VIEW_GATE.json", {"pass": view_pass, "checks": view_checks,
               "terminal": "SHARED_VIEW_INTERFERENCE_SUPPORTED" if view_pass else "SHARED_VIEW_INTERFERENCE_NOT_SUPPORTED",
               "outer_test_accessed": False})
    print(json.dumps({"training_aggregate_complete": True, "objective_gate": obj_pass,
                      "center_gate": center_gate(centers, ctr)["pass"], "view_gate": view_pass}), flush=True)


if __name__ == "__main__":
    main()
