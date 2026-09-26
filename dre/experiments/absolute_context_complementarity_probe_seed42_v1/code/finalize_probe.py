"""Aggregate frozen P0/P1/P2 validation results; never load outer examples."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve()
EXPERIMENT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from run_probe import A1_RUNTIME, LOCK_SHA256, RUNTIME, ResidualHead, sha256  # noqa: E402
sys.path.insert(0, str(HERE.parents[2] / "a1_a2_patient_equal_objective_seed42_v1" / "code"))
from development_metrics import METRICS, THRESHOLDS, finalize_fold  # noqa: E402

DEVELOPMENT = EXPERIMENT / "development"
VARIANTS = ("P0", "P1", "P2")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Empty CSV: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def avg(rows: list[dict], metric: str) -> float:
    return float(np.mean([float(row[metric]) for row in rows]))


def correlation(x: list[float], y: list[float]) -> tuple[float | None, float | None]:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(a) < 3 or not np.isfinite(a).all() or not np.isfinite(b).all() or \
            np.std(a) <= 0 or np.std(b) <= 0:
        return None, None
    return float(pearsonr(a, b).statistic), float(spearmanr(a, b).statistic)


def oracle_threshold(grid: dict, subject_id: str) -> float:
    row = next(patient for patient in grid["patients"] if patient["subject_id"] == subject_id)
    best, selected = None, None
    for i, threshold in enumerate(THRESHOLDS):
        key = (round(float(row["grid"]["patient_macro_f1"][i]), 12),
               round(float(row["grid"]["patient_ez_f1"][i]), 12),
               round(float(row["grid"]["patient_balanced_accuracy"][i]), 12),
               -round(abs(float(threshold) - 0.5), 6))
        if best is None or key > best:
            best, selected = key, i
    return float(THRESHOLDS[selected])


def load_cell(fold: int, epoch: int) -> tuple[dict, Path]:
    cell = RUNTIME / "private" / f"fold_{fold}" / f"epoch_{epoch:02d}"
    payload = torch.load(cell / "representation.pt", map_location="cpu", weights_only=False)
    if payload["lock_sha256"] != LOCK_SHA256 or len(payload["validation"]) != 13:
        raise RuntimeError("Private representation provenance changed")
    return payload, cell


def selected_residual_records(fold: int, variant: str, selected_rows: list[dict]) -> list[dict]:
    by_epoch: dict[int, list[dict]] = {}
    for row in selected_rows:
        by_epoch.setdefault(int(row["selected_epoch"]), []).append(row)
    details = []
    for epoch, rows in by_epoch.items():
        payload, cell = load_cell(fold, epoch)
        probe = torch.load(cell / f"{variant}_probe.pt", map_location="cpu", weights_only=False)
        if probe["lock_sha256"] != LOCK_SHA256 or probe["variant"] != variant or probe["probe_epochs"] != 15:
            raise RuntimeError("Private probe provenance changed")
        validation = {row["subject_id"]: row for row in payload["validation"]}
        first = payload["validation"][0]
        input_dim = first["r"].shape[1] + first["context_norm"].shape[0]
        head = ResidualHead(input_dim)
        head.load_state_dict(probe["model_state_dict"], strict=True)
        head.eval()
        for selected in rows:
            row = validation[selected["subject_id"]]
            context = row["context_norm"] if variant == "P2" else np.zeros_like(row["context_norm"])
            x = np.concatenate([row["r"], np.repeat(context[None, :], len(row["r"]), axis=0)], axis=1)
            with torch.no_grad():
                delta = head(torch.as_tensor(x.astype(np.float32))).numpy()
            base = row["logits"].astype(np.float64)
            final = base + delta.astype(np.float64)
            pair = np.triu_indices(len(base), k=1)
            initial_order = np.sign(base[:, None] - base[None, :])[pair]
            final_order = np.sign(final[:, None] - final[None, :])[pair]
            details.append({"fold": fold, "variant": variant, "subject_id": selected["subject_id"],
                            "base_logits": base, "delta": delta.astype(np.float64),
                            "changed_pairs": int(np.count_nonzero(initial_order != final_order)),
                            "total_pairs": int(len(initial_order))})
    if len(details) != 13:
        raise RuntimeError("Expected 13 selected residual validation patients")
    return details


def summarize_residual(rows: list[dict], variant: str, fold: str | int) -> dict:
    base = np.concatenate([row["base_logits"] for row in rows])
    delta = np.concatenate([row["delta"] for row in rows])
    magnitude = np.abs(delta)
    corr, _ = correlation(base.tolist(), delta.tolist())
    return {"variant": variant, "fold": fold, "n_patients": len(rows), "n_channels": len(delta),
            "mean_delta": float(delta.mean()), "mean_abs_delta": float(magnitude.mean()),
            "delta_std": float(delta.std()), "abs_delta_q10": float(np.quantile(magnitude, 0.10)),
            "abs_delta_median": float(np.median(magnitude)),
            "abs_delta_q90": float(np.quantile(magnitude, 0.90)),
            "fraction_abs_delta_gt_0_10": float(np.mean(magnitude > 0.10)),
            "fraction_abs_delta_gt_0_25": float(np.mean(magnitude > 0.25)),
            "fraction_abs_delta_gt_0_45": float(np.mean(magnitude > 0.45)),
            "saturation_rate_abs_delta_ge_0_49": float(np.mean(magnitude >= 0.49)),
            "corr_A1_logit_delta": corr,
            "fraction_channel_pairs_order_changed": float(sum(row["changed_pairs"] for row in rows) /
                                                          sum(row["total_pairs"] for row in rows))}


def context_diagnostic_rows(p0_selected: dict[int, list[dict]], p0_grids: dict[int, list[dict]]) -> list[dict]:
    private = []
    for fold, rows in p0_selected.items():
        for selection in rows:
            epoch = int(selection["selected_epoch"])
            payload, _ = load_cell(fold, epoch)
            row = next(r for r in payload["validation"] if r["subject_id"] == selection["subject_id"])
            raw = row["context_raw"].astype(float)
            d = len(raw) // 2
            ez_fraction = float(np.mean(row["y_ez"]))
            private.append({"fold": fold, "mu_l2": float(np.linalg.norm(raw[:d])),
                            "log_sigma_mean": float(np.mean(raw[d:])),
                            "log_sigma_l2": float(np.linalg.norm(raw[d:])),
                            "true_EZ_fraction": ez_fraction,
                            "label_using_K_over_channels": ez_fraction,
                            "label_using_oracle_threshold": oracle_threshold(p0_grids[fold][epoch - 1],
                                                                               selection["subject_id"])})
    if len(private) != 65:
        raise RuntimeError("Expected 65 validation context diagnostics")
    rows = []
    for scope in ["pooled"] + list(range(1, 6)):
        group = private if scope == "pooled" else [r for r in private if r["fold"] == scope]
        for statistic in ("mu_l2", "log_sigma_mean", "log_sigma_l2"):
            for target in ("true_EZ_fraction", "label_using_oracle_threshold", "label_using_K_over_channels"):
                pearson, spearman = correlation([r[statistic] for r in group], [r[target] for r in group])
                rows.append({"scope": scope, "n_patients": len(group), "context_statistic": statistic,
                             "label_derived_target": target, "pearson": pearson, "spearman": spearman,
                             "label_using_diagnostic_only": True})
    return rows


def write_report(by: dict, comparison: list[dict], residual: list[dict], context: list[dict],
                 complementarity: dict, readiness: dict) -> None:
    lines = ["# Frozen A1 absolute-context complementarity probe", "",
             "All reported results use the original frozen A1 checkpoints and FIT/validation patients only. No outer-test loader was built, and no outer-test predictions or performance metrics were evaluated. No A1 parameter was trained.", "",
             "| Variant | VLOO Macro-F1 | EZ-F1 | BA | EZ-AUPRC | EZ-AUROC | EZ-MRR | Top1-EZ |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for variant in VARIANTS:
        rows = by[variant]
        lines.append(f"| {variant} | {avg(rows, 'patient_macro_f1'):.6f} | {avg(rows, 'patient_ez_f1'):.6f} | "
                     f"{avg(rows, 'patient_balanced_accuracy'):.6f} | {avg(rows, 'patient_ez_auprc'):.6f} | "
                     f"{avg(rows, 'patient_ez_auroc'):.6f} | {avg(rows, 'patient_ez_mrr'):.6f} | "
                     f"{avg(rows, 'top1_is_ez'):.6f} |")
    lines.extend(["", "| Fold | P1-P0 Macro-F1 | P2-P0 Macro-F1 | P2-P1 Macro-F1 |",
                  "| --- | ---: | ---: | ---: |"])
    for row in comparison:
        lines.append(f"| {row['fold']} | {row['P1_minus_P0_macro_f1']:+.6f} | "
                     f"{row['P2_minus_P0_macro_f1']:+.6f} | {row['P2_minus_P1_macro_f1']:+.6f} |")
    lines.extend(["", f"Mean P1-P0: {np.mean([r['P1_minus_P0_macro_f1'] for r in comparison]):+.6f}; "
                  f"P2-P0: {np.mean([r['P2_minus_P0_macro_f1'] for r in comparison]):+.6f}; "
                  f"P2-P1: {np.mean([r['P2_minus_P1_macro_f1'] for r in comparison]):+.6f}.",
                  f"Positive folds: P2>P0 {complementarity['positive_folds_P2_vs_P0']}/5; "
                  f"P2>P1 {complementarity['positive_folds_P2_vs_P1']}/5.", "",
                  "P2 residual diagnostics on VLOO-selected validation patients:"])
    r2 = next(r for r in residual if r["variant"] == "P2" and r["fold"] == "pooled")
    r1 = next(r for r in residual if r["variant"] == "P1" and r["fold"] == "pooled")
    lines.extend([f"- P1 mean |delta| {r1['mean_abs_delta']:.6f}; P2 mean |delta| {r2['mean_abs_delta']:.6f}; P2 fraction |delta|>0.45 "
                  f"{r2['fraction_abs_delta_gt_0_45']:.6f}; saturation near ±0.5 "
                  f"{r2['saturation_rate_abs_delta_ge_0_49']:.6f}.",
                  f"- Changed channel-pair orderings {r2['fraction_channel_pairs_order_changed']:.6f}; "
                  f"corr(A1 logit, delta) {r2['corr_A1_logit_delta']}.", "",
                  "Absolute-context correlations are **LABEL_USING_DIAGNOSTIC_ONLY**. The 9 pooled and 45 fold-level "
                  "Pearson/Spearman comparisons are in `development/ABSOLUTE_CONTEXT_INFORMATION_DIAGNOSTIC.csv`; "
                  "they were not used for model selection and are exploratory without multiplicity correction. "
                  "True EZ fraction and K*/channel count are identical here, so their correlation rows are duplicates.", "",
                  "Complementarity checks:"])
    lines.extend(f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in complementarity["checks"].items())
    lines.extend(["", "Test-readiness checks:"])
    lines.extend(f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in readiness["checks"].items())
    lines.extend(["", f"P2 APPARENT_FULLVAL mean/worst fold Macro-F1: "
                  f"{readiness['P2_apparent_fullval_mean_macro_f1']:.6f}/"
                  f"{readiness['P2_apparent_fullval_worst_fold_macro_f1']:.6f}.",
                  f"**Terminal: `{readiness['terminal']}`.**", ""])
    if readiness["pass"]:
        lines.append("The validation finding is frozen; no outer test or end-to-end integration was run.")
    else:
        lines.append("The locked development gate failed. Stop this route; no outer test or post-hoc probe adjustment was run.")
    (EXPERIMENT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not os.environ.get("ABS_CONTEXT_RUNTIME") or not RUNTIME.is_absolute():
        raise RuntimeError("Set absolute ABS_CONTEXT_RUNTIME")
    if sha256(EXPERIMENT / "PROTOCOL_LOCK.json") != LOCK_SHA256:
        raise RuntimeError("Probe protocol lock changed")
    lock = json.loads((EXPERIMENT / "PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    source = json.loads((EXPERIMENT / "SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))
    complete = json.loads((RUNTIME / "private" / "PROBE_TRAINING_COMPLETE.json").read_text(encoding="utf-8"))
    if source.get("pass") is not True or complete.get("completed_probe_cells") != 300 or \
            complete.get("lock_sha256") != LOCK_SHA256:
        raise RuntimeError("Source or probe training incomplete")
    head_shapes = None
    matched_parameter_count = None
    for fold in range(1, 6):
        for epoch in range(1, 31):
            cell = RUNTIME / "private" / f"fold_{fold}" / f"epoch_{epoch:02d}"
            pair = [torch.load(cell / f"{variant}_probe.pt", map_location="cpu", weights_only=False)
                    for variant in ("P1", "P2")]
            if any(p["lock_sha256"] != LOCK_SHA256 or p["probe_epochs"] != 15 or
                   p["fold"] != fold or p["base_epoch"] != epoch for p in pair):
                raise RuntimeError("Incomplete or mismatched frozen probe cell")
            if pair[0]["source_checkpoint_sha256"] != pair[1]["source_checkpoint_sha256"]:
                raise RuntimeError("Matched P1/P2 probes used different A1 checkpoints")
            shapes = [{k: tuple(v.shape) for k, v in p["model_state_dict"].items()} for p in pair]
            if shapes[0] != shapes[1] or (head_shapes is not None and shapes[0] != head_shapes):
                raise RuntimeError("P1/P2 parameter count or architecture changed")
            head_shapes = shapes[0]
            matched_parameter_count = sum(v.numel() for v in pair[0]["model_state_dict"].values())
    fold_results, fullval, p0_selected, p0_grids = [], [], {}, {}
    for fold in range(1, 6):
        p0_grids[fold] = [json.loads((A1_RUNTIME / "A1" / f"fold_{fold}" /
                                     f"epoch_{epoch:02d}_validation_grid.json").read_text(encoding="utf-8"))
                          for epoch in range(1, 31)]
        for variant in VARIANTS:
            grids = p0_grids[fold] if variant == "P0" else [
                json.loads((RUNTIME / "private" / f"fold_{fold}" / f"epoch_{epoch:02d}" /
                            f"{variant}_validation_grid.json").read_text(encoding="utf-8"))
                for epoch in range(1, 31)]
            private_csv = RUNTIME / "private" / f"fold_{fold}_{variant}_VLOO_PATIENT.csv"
            aggregate, selected = finalize_fold(grids, variant, fold, private_csv)
            fold_results.append(aggregate)
            fullval.append(selected)
            if variant == "P0":
                with private_csv.open(newline="", encoding="utf-8") as stream:
                    p0_selected[fold] = list(csv.DictReader(stream))
    by = {variant: [r for r in fold_results if r["variant"] == variant] for variant in VARIANTS}
    for fold, reference in enumerate(lock["source_vloo_macro_f1_by_fold"], start=1):
        observed = next(r for r in by["P0"] if r["fold"] == fold)["patient_macro_f1"]
        if abs(observed - reference) > lock["source_tolerance"]:
            raise RuntimeError("SOURCE_A1_REPRODUCTION_FAILED")
    for variant in VARIANTS:
        write_csv(DEVELOPMENT / f"{variant}_VLOO_BY_FOLD.csv", by[variant])
    write_csv(DEVELOPMENT / "FULLVAL_SELECTION.csv", fullval)
    comparison = []
    for fold in range(1, 6):
        rows = {v: next(r for r in by[v] if r["fold"] == fold) for v in VARIANTS}
        comparison.append({"fold": fold, "P0_macro_f1": rows["P0"]["patient_macro_f1"],
                           "P1_macro_f1": rows["P1"]["patient_macro_f1"],
                           "P2_macro_f1": rows["P2"]["patient_macro_f1"],
                           "P1_minus_P0_macro_f1": rows["P1"]["patient_macro_f1"] - rows["P0"]["patient_macro_f1"],
                           "P2_minus_P0_macro_f1": rows["P2"]["patient_macro_f1"] - rows["P0"]["patient_macro_f1"],
                           "P2_minus_P1_macro_f1": rows["P2"]["patient_macro_f1"] - rows["P1"]["patient_macro_f1"]})
    write_csv(DEVELOPMENT / "VLOO_COMPARISON.csv", comparison)
    residual_rows, residual_details = [], {}
    for variant in ("P1", "P2"):
        all_details = []
        for fold in range(1, 6):
            private_csv = RUNTIME / "private" / f"fold_{fold}_{variant}_VLOO_PATIENT.csv"
            with private_csv.open(newline="", encoding="utf-8") as stream:
                selected = list(csv.DictReader(stream))
            details = selected_residual_records(fold, variant, selected)
            all_details.extend(details)
            residual_rows.append(summarize_residual(details, variant, fold))
        residual_rows.append(summarize_residual(all_details, variant, "pooled"))
        residual_details[variant] = all_details
    write_csv(DEVELOPMENT / "RESIDUAL_DIAGNOSTICS.csv", residual_rows)
    context_rows = context_diagnostic_rows(p0_selected, p0_grids)
    write_csv(DEVELOPMENT / "ABSOLUTE_CONTEXT_INFORMATION_DIAGNOSTIC.csv", context_rows)
    p0, p1, p2 = [avg(by[v], "patient_macro_f1") for v in VARIANTS]
    gain20, gain21 = p2 - p0, p2 - p1
    positive20 = sum(r["P2_minus_P0_macro_f1"] > 0 for r in comparison)
    positive21 = sum(r["P2_minus_P1_macro_f1"] > 0 for r in comparison)
    ez_delta = avg(by["P2"], "patient_ez_f1") - avg(by["P0"], "patient_ez_f1")
    auprc_delta = avg(by["P2"], "patient_ez_auprc") - avg(by["P0"], "patient_ez_auprc")
    checks = {"P2_minus_P0_macro_f1_ge_0_010": gain20 >= 0.010 - 1e-12,
              "P2_minus_P1_macro_f1_ge_0_005": gain21 >= 0.005 - 1e-12,
              "positive_P2_vs_P0_folds_ge_4": positive20 >= 4,
              "positive_P2_vs_P1_folds_ge_3": positive21 >= 3,
              "P2_mean_EZ_F1_nondecreasing": ez_delta >= -1e-12,
              "P2_EZ_AUPRC_delta_ge_minus_0_005": auprc_delta >= -0.005 - 1e-12,
              "P2_VLOO_macro_f1_ge_0_640": p2 >= 0.640 - 1e-12}
    complementarity = {"pass": all(checks.values()), "checks": checks,
                       "P0_vloo_macro_f1": p0, "P1_vloo_macro_f1": p1, "P2_vloo_macro_f1": p2,
                       "P2_minus_P0_macro_f1": gain20, "P2_minus_P1_macro_f1": gain21,
                       "P1_minus_P0_macro_f1": p1 - p0,
                       "positive_folds_P2_vs_P0": positive20, "positive_folds_P2_vs_P1": positive21,
                       "P2_minus_P0_EZ_F1": ez_delta, "P2_minus_P0_EZ_AUPRC": auprc_delta,
                       "outer_test_accessed": False}
    write_json(DEVELOPMENT / "COMPLEMENTARITY_GATE.json", complementarity)
    apparent_p2 = [r for r in fullval if r["variant"] == "P2"]
    apparent_mean = avg(apparent_p2, "apparent_patient_macro_f1")
    apparent_worst = min(float(r["apparent_patient_macro_f1"]) for r in apparent_p2)
    r2 = next(r for r in residual_rows if r["variant"] == "P2" and r["fold"] == "pooled")
    readiness_checks = {"source_P0_reproduced": True, "complementarity_gate_pass": complementarity["pass"],
                        "P2_apparent_fullval_mean_ge_0_665": apparent_mean >= 0.665 - 1e-12,
                        "P2_apparent_fullval_worst_fold_ge_0_620": apparent_worst >= 0.620 - 1e-12,
                        "P2_fraction_abs_delta_gt_0_45_lt_0_10": r2["fraction_abs_delta_gt_0_45"] < 0.10,
                        "no_leakage_or_pathology": all(math.isfinite(float(r["mean_abs_delta"])) for r in residual_rows)}
    readiness_pass = all(readiness_checks.values())
    capacity_only = p1 - p0 >= 0.010 - 1e-12 and p2 - p0 >= 0.010 - 1e-12 and gain21 < 0.005 - 1e-12
    terminal = ("ABSOLUTE_CONTEXT_COMPLEMENTARITY_SUPPORTED" if readiness_pass else
                "RESIDUAL_CAPACITY_ONLY_SUPPORTED" if capacity_only else
                "ABSOLUTE_CONTEXT_COMPLEMENTARITY_NOT_SUPPORTED")
    readiness = {"pass": readiness_pass, "terminal": terminal, "checks": readiness_checks,
                 "P2_apparent_fullval_mean_macro_f1": apparent_mean,
                 "P2_apparent_fullval_worst_fold_macro_f1": apparent_worst,
                 "P2_fraction_abs_delta_gt_0_45": r2["fraction_abs_delta_gt_0_45"],
                 "outer_test_accessed": False}
    write_json(DEVELOPMENT / "TEST_READINESS_GATE.json", readiness)
    freeze = {"source_A1_reproduced": True, "source_checkpoints": 150,
              "completed_probe_cells": complete["completed_probe_cells"], "probe_epochs_each": 15,
              "matched_head_input_dim": head_shapes["net.0.weight"][1],
              "matched_head_hidden_dim": head_shapes["net.0.weight"][0],
              "matched_head_parameter_count_each": matched_parameter_count,
              "A1_parameters_frozen": True, "probe_head_only_trained": True,
              "fit_only_context_normalization": True, "P1_P2_matched_architecture_and_initialization": True,
              "validation_probe_epoch_selection": False, "outer_loader_built": False,
              "outer_test_accessed": False, "protocol_lock_sha256": LOCK_SHA256,
              "source_protocol_sha256": lock["source_protocol_sha256"],
              "cache_sha256": lock["input_sha256"]["window_cache"],
              "split_sha256": lock["input_sha256"]["fixed_partition_manifest"]}
    write_json(EXPERIMENT / "FREEZE_AUDIT.json", freeze)
    if readiness_pass:
        write_json(EXPERIMENT / "ABSOLUTE_CONTEXT_COMPLEMENTARITY_SUPPORTED.json",
                   {"P2_minus_P0_macro_f1": gain20, "P2_minus_P1_macro_f1": gain21,
                    "positive_folds_P2_vs_P0": positive20, "positive_folds_P2_vs_P1": positive21,
                    "P2_minus_P0_EZ_F1": ez_delta, "P2_minus_P0_EZ_AUPRC": auprc_delta,
                    "P2_residual_diagnostics": r2, "source_sha256": lock["source_sha256"],
                    "cache_sha256": lock["input_sha256"]["window_cache"],
                    "split_sha256": lock["input_sha256"]["fixed_partition_manifest"],
                    "outer_test_accessed": False})
    write_report(by, comparison, residual_rows, context_rows, complementarity, readiness)
    print(json.dumps({"terminal": terminal, "P0": p0, "P1": p1, "P2": p2,
                      "P2_minus_P1": gain21, "test_readiness": readiness_pass}, indent=2), flush=True)


if __name__ == "__main__":
    main()
