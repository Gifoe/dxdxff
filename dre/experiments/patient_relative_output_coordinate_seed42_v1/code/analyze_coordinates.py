"""Validation-only comparison of fixed raw/centered/robust-z A1 logit coordinates."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr

HERE = Path(__file__).resolve()
EXPERIMENT = HERE.parents[1]
sys.path.insert(0, str(HERE.parents[2] / "a1_a2_patient_equal_objective_seed42_v1" / "code"))
from development_metrics import METRICS, THRESHOLDS, epoch_grid, finalize_fold, patient_grid  # noqa: E402


RUNTIME = Path(os.environ.get("COORD_RUNTIME", ""))
DEVELOPMENT = EXPERIMENT / "development"
VARIANTS = ("C0_RAW", "C1_CENTERED", "C2_ROBUSTZ")
RANK_METRICS = ("patient_ez_auprc", "patient_ez_auroc", "patient_ez_mrr", "top1_is_ez")
EXPECTED_LOCK_SHA256 = "879a585e7a6ae8df3b69faaa6efedc29a02711f5c5a07adcd64fed0b305b5805"


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Empty aggregate CSV: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def average(rows: list[dict], metric: str) -> float:
    return float(np.mean([float(row[metric]) for row in rows]))


def record_for_coordinate(patient: dict, variant: str) -> tuple[dict, dict]:
    logits = np.asarray(patient["logits_nez"], dtype=np.float64)
    if logits.size < 2 or not np.isfinite(logits).all():
        raise RuntimeError("Invalid source validation logits")
    median = float(np.median(logits))
    mad = float(np.median(np.abs(logits - median)))
    scale = max(1.4826 * mad, 1e-6)
    if variant == "C0_RAW":
        transformed = logits
        p = np.asarray(patient["score_nez_core"], dtype=np.float32)
        q = np.asarray(patient["score_ez_core"], dtype=np.float32)
    elif variant == "C1_CENTERED":
        transformed = logits - median
        p = expit(transformed)
        q = 1.0 - p
    elif variant == "C2_ROBUSTZ":
        transformed = (logits - median) / scale
        p = expit(transformed)
        q = 1.0 - p
    else:
        raise ValueError(variant)
    if not np.isfinite(transformed).all() or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise RuntimeError("Nonfinite normalized score")
    labels_nez = np.asarray(patient["labels_nez"], dtype=np.float32)
    labels_ez = np.asarray(patient["labels_ez"], dtype=np.float32)
    record = {"subject_id": patient["subject_id"], "labels": labels_nez, "labels_nez": labels_nez,
              "labels_ez": labels_ez, "score_nez": p, "score_ez": q,
              "channel_mask": np.ones(len(logits), dtype=bool)}
    stats = {"mean": float(np.mean(logits)), "std": float(np.std(logits)), "median": median,
             "mad": mad, "scale": scale, "saturated_probabilities": int(np.sum((p == 0) | (p == 1))),
             "spearman": float(spearmanr(logits, transformed).statistic)}
    return record, stats


def oracle_for_patient(grid: dict, patient_idx: int) -> tuple[float, float]:
    row = grid["patients"][patient_idx]
    best_key, selected = None, None
    for threshold_idx, threshold in enumerate(THRESHOLDS):
        key = (
            round(float(row["grid"]["patient_macro_f1"][threshold_idx]), 12),
            round(float(row["grid"]["patient_ez_f1"][threshold_idx]), 12),
            round(float(row["grid"]["patient_balanced_accuracy"][threshold_idx]), 12),
            -round(abs(float(threshold) - 0.5), 6),
        )
        if best_key is None or key > best_key:
            best_key, selected = key, threshold_idx
    assert selected is not None
    return float(THRESHOLDS[selected]), float(row["grid"]["patient_macro_f1"][selected])


def threshold_stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    counts = Counter(f"{value:.2f}" for value in arr)
    probabilities = np.asarray(list(counts.values()), dtype=float) / len(arr)
    return {"mean": float(arr.mean()), "std": float(arr.std()),
            "iqr": float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25)),
            "mad": float(np.median(np.abs(arr - np.median(arr)))),
            "entropy_bits": float(-(probabilities * np.log2(probabilities)).sum()),
            "unique": len(counts), "distribution_json": json.dumps(dict(sorted(counts.items())))}


def main() -> None:
    if not os.environ.get("COORD_RUNTIME") or not RUNTIME.is_absolute():
        raise RuntimeError("Set absolute COORD_RUNTIME")
    if sha256(EXPERIMENT / "PROTOCOL_LOCK.json") != EXPECTED_LOCK_SHA256:
        raise RuntimeError("Output-coordinate protocol lock changed")
    source = json.loads((EXPERIMENT / "SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))
    if source.get("pass") is not True or source.get("outer_test_accessed") is not False:
        raise RuntimeError("SOURCE_A1_VLOO_REPRODUCTION_FAILED")
    lock = json.loads((EXPERIMENT / "PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    fold_results, fullval, private_rows, oracle_rows, raw_score_rows = [], [], [], [], []
    rank_max = {variant: {metric: 0.0 for metric in RANK_METRICS} for variant in VARIANTS[1:]}
    rank_min_spearman = {variant: 1.0 for variant in VARIANTS[1:]}
    cases = 0
    min_mad = float("inf")
    min_scale = float("inf")
    for fold in range(1, 6):
        epoch_grids = {variant: [] for variant in VARIANTS}
        for epoch in range(1, 31):
            payload = json.loads((RUNTIME / "validation_logits_private" / f"fold_{fold}" /
                                  f"epoch_{epoch:02d}.json").read_text(encoding="utf-8"))
            if payload["epoch"] != epoch or len(payload["patients"]) != 13:
                raise RuntimeError("Private validation logits are incomplete")
            records = {variant: [] for variant in VARIANTS}
            for patient in payload["patients"]:
                generated = {}
                for variant in VARIANTS:
                    record, stats = record_for_coordinate(patient, variant)
                    records[variant].append(record)
                    generated[variant] = (record, stats, patient_grid(record))
                raw_stats = generated["C0_RAW"][1]
                raw_score_rows.append({"fold": fold, "epoch": epoch, **raw_stats})
                min_mad = min(min_mad, raw_stats["mad"])
                min_scale = min(min_scale, raw_stats["scale"])
                for variant in VARIANTS[1:]:
                    stat = generated[variant][1]
                    if not math.isfinite(stat["spearman"]):
                        rank_min_spearman[variant] = -1.0
                    else:
                        rank_min_spearman[variant] = min(rank_min_spearman[variant], stat["spearman"])
                    for metric in RANK_METRICS:
                        delta = abs(generated[variant][2]["fixed"][metric] -
                                    generated["C0_RAW"][2]["fixed"][metric])
                        rank_max[variant][metric] = max(rank_max[variant][metric], float(delta))
                cases += 1
            for variant in VARIANTS:
                epoch_grids[variant].append(epoch_grid(records[variant], epoch))
        # The audit must pass before any C1/C2 candidate selection.
        print(f"[COORD] rebuilt fold={fold} all 30 epochs; audited {cases} patient-epochs", flush=True)
        for variant in VARIANTS:
            aggregate, selected = finalize_fold(epoch_grids[variant], variant, fold,
                                                RUNTIME / "private" / f"fold_{fold}_{variant}_VLOO_PATIENT.csv")
            fold_results.append(aggregate)
            fullval.append(selected)
            selected_by_id = {}
            with (RUNTIME / "private" / f"fold_{fold}_{variant}_VLOO_PATIENT.csv").open(newline="", encoding="utf-8") as stream:
                selected_by_id = {row["subject_id"]: row for row in csv.DictReader(stream)}
            private_rows.extend(selected_by_id.values())
            for subject_id, selected_row in selected_by_id.items():
                selected_epoch = int(selected_row["selected_epoch"])
                patient_rows = epoch_grids[variant][selected_epoch - 1]["patients"]
                patient_idx = next(i for i, row in enumerate(patient_rows) if row["subject_id"] == subject_id)
                oracle_threshold, oracle_f1 = oracle_for_patient(epoch_grids[variant][selected_epoch - 1], patient_idx)
                oracle_rows.append({"variant": variant, "fold": fold, "subject_id": subject_id,
                                    "oracle_threshold": oracle_threshold, "oracle_macro_f1": oracle_f1})
    rank_pass = all(rank_min_spearman[variant] >= lock["rank_invariance"]["spearman_min"] and
                    all(delta <= lock["rank_invariance"]["ranking_metric_delta_max_abs"]
                        for delta in rank_max[variant].values()) for variant in VARIANTS[1:])
    rank_audit = {"pass": rank_pass, "terminal": "RANK_INVARIANCE_PASSED" if rank_pass else
                  "OUTPUT_COORDINATE_RANK_INVARIANCE_FAILED", "patient_epoch_cases": cases,
                  "cases_per_transform": cases, "minimum_spearman": rank_min_spearman,
                  "maximum_absolute_ranking_metric_deltas": rank_max,
                  "tolerance": lock["rank_invariance"], "outer_test_accessed": False}
    write_json(EXPERIMENT / "RANK_INVARIANCE_AUDIT.json", rank_audit)
    if not rank_pass:
        raise RuntimeError("OUTPUT_COORDINATE_RANK_INVARIANCE_FAILED")
    if min_mad <= 0 or not math.isfinite(min_mad) or min_scale <= 0 or not math.isfinite(min_scale):
        score_pathology = True
    else:
        score_pathology = False
    by = {variant: [row for row in fold_results if row["variant"] == variant] for variant in VARIANTS}
    apparent = {variant: [row for row in fullval if row["variant"] == variant] for variant in VARIANTS}
    if abs(average(by["C0_RAW"], "patient_macro_f1") - source["reproduced_mean"]) > 1e-6:
        raise RuntimeError("C0 changed after source reproduction")
    for variant in VARIANTS:
        write_csv(DEVELOPMENT / f"{variant}_VLOO_BY_FOLD.csv", by[variant])
    write_csv(DEVELOPMENT / "FULLVAL_SELECTION.csv", fullval)
    comparison = []
    for fold in range(1, 6):
        c0, c1, c2 = (next(row for row in by[variant] if row["fold"] == fold) for variant in VARIANTS)
        comparison.append({"fold": fold,
                           "C0_macro_f1": c0["patient_macro_f1"],
                           "C1_macro_f1": c1["patient_macro_f1"],
                           "C2_macro_f1": c2["patient_macro_f1"],
                           "C1_minus_C0": c1["patient_macro_f1"] - c0["patient_macro_f1"],
                           "C2_minus_C0": c2["patient_macro_f1"] - c0["patient_macro_f1"],
                           "C2_minus_C1": c2["patient_macro_f1"] - c1["patient_macro_f1"],
                           "C0_ez_f1": c0["patient_ez_f1"],
                           "C1_ez_f1": c1["patient_ez_f1"],
                           "C2_ez_f1": c2["patient_ez_f1"]})
    write_csv(DEVELOPMENT / "VLOO_COMPARISON.csv", comparison)
    stability = []
    for variant in VARIANTS:
        rows = [row for row in private_rows if row["variant"] == variant]
        thresholds = [float(row["selected_threshold"]) for row in rows]
        fold_groups = [[float(row["selected_threshold"]) for row in rows if int(row["fold"]) == fold]
                       for fold in range(1, 6)]
        if len(thresholds) != 65 or any(len(group) != 13 for group in fold_groups):
            raise RuntimeError("VLOO selected threshold count changed")
        stability.append({"variant": variant, "n_patient_fold_selections": 65,
                          **threshold_stats(thresholds),
                          "between_fold_mean_threshold_std": float(np.std([np.mean(group) for group in fold_groups])),
                          "mean_within_fold_patient_threshold_std": float(np.mean([np.std(group) for group in fold_groups]))})
    write_csv(DEVELOPMENT / "THRESHOLD_STABILITY.csv", stability)
    score_diagnostics = []
    for fold in range(1, 6):
        rows = [row for row in raw_score_rows if row["fold"] == fold]
        score_diagnostics.append({"fold": fold, "patient_epoch_cases": len(rows),
                                  "raw_logit_mean_mean": average(rows, "mean"),
                                  "raw_logit_std_mean": average(rows, "std"),
                                  "raw_logit_median_mean": average(rows, "median"),
                                  "between_patient_epoch_median_std": float(np.std([row["median"] for row in rows])),
                                  "raw_logit_mad_mean": average(rows, "mad"),
                                  "raw_logit_mad_min": min(row["mad"] for row in rows)})
    write_csv(DEVELOPMENT / "SCORE_DISTRIBUTION_DIAGNOSTICS.csv", score_diagnostics)
    oracle_diagnostics = []
    for variant in VARIANTS:
        rows = [row for row in oracle_rows if row["variant"] == variant]
        if len(rows) != 65:
            raise RuntimeError("Oracle diagnostic patient-fold count changed")
        stats = threshold_stats([row["oracle_threshold"] for row in rows])
        oracle_diagnostics.append({"variant": variant, "scope": "all_five_validation_folds",
                                   "label_using_diagnostic_only": True, "n_patient_fold_cases": 65,
                                   "oracle_threshold_mean": stats["mean"], "oracle_threshold_std": stats["std"],
                                   "oracle_threshold_iqr": stats["iqr"], "oracle_threshold_mad": stats["mad"],
                                   "oracle_macro_f1_mean": average(rows, "oracle_macro_f1")})
    write_csv(DEVELOPMENT / "ORACLE_THRESHOLD_DISPERSION_DIAGNOSTIC.csv", oracle_diagnostics)
    c0_mean = average(by["C0_RAW"], "patient_macro_f1")
    c1_mean = average(by["C1_CENTERED"], "patient_macro_f1")
    c2_mean = average(by["C2_ROBUSTZ"], "patient_macro_f1")
    preferred = "C1_CENTERED" if c1_mean >= c2_mean + 0.005 - 1e-12 else "C2_ROBUSTZ"
    candidate_rows = by[preferred]
    positive = sum(next(row for row in candidate_rows if row["fold"] == fold)["patient_macro_f1"] >
                   next(row for row in by["C0_RAW"] if row["fold"] == fold)["patient_macro_f1"] for fold in range(1, 6))
    ez_nondecreasing = average(candidate_rows, "patient_ez_f1") >= average(by["C0_RAW"], "patient_ez_f1") - 1e-12
    candidate_checks = {"gain_ge_0_010": average(candidate_rows, "patient_macro_f1") - c0_mean >= 0.010 - 1e-12,
                        "positive_folds_ge_4": positive >= 4,
                        "mean_ez_f1_nondecreasing": ez_nondecreasing}
    retained = all(candidate_checks.values())
    candidate = {"preferred_by_locked_C1_vs_C2_rule": preferred, "retained": retained,
                 "C0_mean_macro_f1": c0_mean, "C1_mean_macro_f1": c1_mean, "C2_mean_macro_f1": c2_mean,
                 "preferred_gain_vs_C0": average(candidate_rows, "patient_macro_f1") - c0_mean,
                 "positive_folds": positive, "checks": candidate_checks,
                 "selection_uses_oracle_labels": False, "outer_test_accessed": False}
    write_json(DEVELOPMENT / "CANDIDATE_SELECTION.json", candidate)
    apparent_mean = average(apparent[preferred], "apparent_patient_macro_f1")
    apparent_worst = min(float(row["apparent_patient_macro_f1"]) for row in apparent[preferred])
    checks = {"source_C0_reproduced": source["pass"] is True,
              "normalized_candidate_retained": retained,
              "candidate_vloo_macro_f1_ge_0_640": average(candidate_rows, "patient_macro_f1") >= 0.640 - 1e-12,
              "gain_vs_C0_ge_0_010": candidate_checks["gain_ge_0_010"],
              "positive_folds_ge_4": candidate_checks["positive_folds_ge_4"],
              "mean_ez_f1_nondecreasing": ez_nondecreasing,
              "apparent_fullval_mean_ge_0_665": apparent_mean >= 0.665 - 1e-12,
              "apparent_fullval_worst_fold_ge_0_620": apparent_worst >= 0.620 - 1e-12,
              "ranking_invariant": rank_pass,
              "no_score_pathology": not score_pathology}
    gate_pass = all(checks.values())
    terminal = ("CENTERED_OUTPUT_COORDINATE_SELECTED" if preferred == "C1_CENTERED" else
                "ROBUSTZ_OUTPUT_COORDINATE_SELECTED") if gate_pass else "OUTPUT_COORDINATE_DEVELOPMENT_GATE_FAILED"
    gate = {"pass": gate_pass, "terminal": terminal, "preferred_coordinate": preferred,
            "checks": checks, "candidate_vloo_macro_f1": average(candidate_rows, "patient_macro_f1"),
            "candidate_gain_vs_C0": candidate["preferred_gain_vs_C0"], "positive_folds": positive,
            "candidate_apparent_fullval_mean_macro_f1": apparent_mean,
            "candidate_apparent_fullval_worst_fold_macro_f1": apparent_worst,
            "minimum_raw_mad": min_mad, "minimum_robust_scale": min_scale,
            "outer_test_accessed": False}
    write_json(DEVELOPMENT / "DEVELOPMENT_GATE.json", gate)
    if gate_pass:
        # Hash only the validation-selected frozen source A1 checkpoints; no test inference.
        source_runtime = Path(os.environ.get("A1_A2_RUNTIME", ""))
        if not source_runtime.is_absolute():
            raise RuntimeError("Need private A1 runtime to freeze selected checkpoints")
        manifest = {"coordinate": preferred, "epsilon": 1e-6, "source_protocol_sha256": lock["source_protocol_sha256"],
                    "cache_sha256": lock["input_sha256"]["window_cache"],
                    "split_sha256": lock["input_sha256"]["fixed_partition_manifest"],
                    "outer_test_accessed": False, "folds": []}
        for row in apparent[preferred]:
            checkpoint = source_runtime / "A1" / f"fold_{row['fold']}" / f"epoch_{row['selected_epoch']:02d}.pt"
            manifest["folds"].append({"fold": row["fold"], "epoch": row["selected_epoch"],
                                      "threshold": row["selected_threshold"], "checkpoint_sha256": sha256(checkpoint)})
        write_json(EXPERIMENT / "FROZEN_OUTPUT_COORDINATE_MANIFEST.json", manifest)
    write_report(by, comparison, fullval, stability, oracle_diagnostics, source, rank_audit, candidate, gate)
    print(json.dumps({"terminal": terminal, "preferred_coordinate": preferred,
                      "candidate_vloo_macro_f1": gate["candidate_vloo_macro_f1"],
                      "development_gate_pass": gate_pass}, indent=2), flush=True)


def write_report(by, comparison, fullval, stability, oracle, source, rank, candidate, gate) -> None:
    lines = ["# Patient-relative output coordinate: seed-42 validation-only analysis", "",
             "All results use frozen A1 checkpoints and validation patients only. No new model was trained and no current outer-test result was read.",
             f"Source A1 VLOO reproduction: {'PASS' if source['pass'] else 'FAIL'}; "
             f"mean {source['reproduced_mean']:.10f}; max fold error "
             f"{max(row['absolute_error'] for row in source['folds']):.2e}.",
             f"Ranking invariance: {'PASS' if rank['pass'] else 'FAIL'} across {rank['patient_epoch_cases']} patient-epochs per transform.", "",
             "| Coordinate | VLOO Macro-F1 | EZ-F1 | BA | EZ-AUPRC | EZ-AUROC | EZ-MRR | Top-1 EZ | Pred. EZ fraction |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for variant in VARIANTS:
        rows = by[variant]
        lines.append(f"| {variant} | {average(rows, 'patient_macro_f1'):.6f} | "
                     f"{average(rows, 'patient_ez_f1'):.6f} | {average(rows, 'patient_balanced_accuracy'):.6f} | "
                     f"{average(rows, 'patient_ez_auprc'):.6f} | {average(rows, 'patient_ez_auroc'):.6f} | "
                     f"{average(rows, 'patient_ez_mrr'):.6f} | {average(rows, 'top1_is_ez'):.6f} | "
                     f"{average(rows, 'predicted_ez_fraction'):.6f} |")
    lines.extend(["", "| Fold | C0 Macro-F1 | C1 Macro-F1 | C2 Macro-F1 | C1−C0 | C2−C0 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in comparison:
        lines.append(f"| {row['fold']} | {row['C0_macro_f1']:.6f} | {row['C1_macro_f1']:.6f} | "
                     f"{row['C2_macro_f1']:.6f} | {row['C1_minus_C0']:+.6f} | {row['C2_minus_C0']:+.6f} |")
    lines.extend(["", f"Preferred by the locked C1-vs-C2 rule: {candidate['preferred_by_locked_C1_vs_C2_rule']}; "
                  f"retained vs C0: {candidate['retained']}; gain {candidate['preferred_gain_vs_C0']:+.6f}; "
                  f"positive folds {candidate['positive_folds']}/5.",
                  f"Candidate APPARENT_FULLVAL mean/worst-fold Macro-F1: "
                  f"{gate['candidate_apparent_fullval_mean_macro_f1']:.6f}/"
                  f"{gate['candidate_apparent_fullval_worst_fold_macro_f1']:.6f}.", "",
                  "| Coordinate | VLOO threshold std | IQR | entropy (bits) | unique | oracle std | oracle IQR | oracle Macro-F1 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for s, o in zip(stability, oracle, strict=True):
        lines.append(f"| {s['variant']} | {s['std']:.6f} | {s['iqr']:.6f} | {s['entropy_bits']:.6f} | "
                     f"{s['unique']} | {o['oracle_threshold_std']:.6f} | {o['oracle_threshold_iqr']:.6f} | "
                     f"{o['oracle_macro_f1_mean']:.6f} |")
    lines.extend(["", "Oracle thresholds are **LABEL_USING_DIAGNOSTIC_ONLY** at each patient's VLOO-selected epoch; "
                  "they were not used to choose an epoch, coordinate, global threshold, or gate outcome. "
                  "A finite 19-point grid can change each patient's oracle Macro-F1 after a monotone score transform, "
                  "because it samples different cut points in raw-logit space.", "",
                  "Strong-gate checks:"])
    lines.extend(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in gate["checks"].items())
    lines.extend(["", f"**Terminal: `{gate['terminal']}`.**", ""])
    if gate["pass"]:
        lines.append("The coordinate/checkpoint/threshold manifest is frozen. No outer evaluation was run; explicit authorization is still required.")
    else:
        lines.append("Stop at validation. Do not read current outer test or tune another transformation after this result.")
    lines.extend(["", "Threshold distribution, full-validation selection, raw score-statistic aggregates, and "
                  "oracle diagnostic aggregates are in `development/`. Patient-level data and logits remain private."])
    (EXPERIMENT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
