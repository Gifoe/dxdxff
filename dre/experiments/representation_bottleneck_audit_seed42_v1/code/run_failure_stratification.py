"""Diagnostic-only stratification of frozen A1 VLOO validation cases."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, rankdata, spearmanr

HERE = Path(__file__).resolve()
EXPERIMENT = HERE.parents[1]
sys.path.insert(0, str(HERE.parents[2] / "r1_hlv_ictal_dynamics_seed42_v1" / "code"))
from run_matched import SOURCE_ROOT, build_fold, install_interleaved_hlv_view, make_args  # noqa: E402
sys.path.insert(0, str(SOURCE_ROOT))
import exp_ez_hybrid as core  # noqa: E402
from reproduce_source import RUNTIME, ensure_source  # noqa: E402

OUTPUT = EXPERIMENT / "audit_e_failures"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Empty diagnostic table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_cases() -> list[dict]:
    install_interleaved_hlv_view()
    args = make_args("R0", RUNTIME)
    exp = core.Exp_EZHybridLocalization(args)
    if len(exp.patient_index) != 80 or len(exp.outer_splits) != 5:
        raise RuntimeError("Frozen cohort changed")
    cases = []
    for split in exp.outer_splits:
        fold, _fit, _train_loader, val_loader, test_loader, _normalizer = build_fold(exp, split, "validation")
        if test_loader is not None:
            raise RuntimeError("Outer-test loader constructed")
        metadata = {str(item["subject_id"]): item for item in val_loader.dataset.patient_examples}
        selected = read_csv(RUNTIME / "private" / f"fold_{fold}_A1_VLOO_PATIENT.csv")
        cached_epochs: dict[int, dict] = {}
        for record in selected:
            subject = record["subject_id"]
            epoch = int(record["selected_epoch"])
            if epoch not in cached_epochs:
                path = RUNTIME / "private" / f"fold_{fold}" / f"epoch_{epoch:02d}" / "representation.pt"
                if not path.exists():
                    raise RuntimeError(f"Selected frozen representation not cached: fold={fold}, epoch={epoch}")
                payload = torch.load(path, map_location="cpu", weights_only=False)
                cached_epochs[epoch] = {row["subject_id"]: row for row in payload["validation"]}
            row = cached_epochs[epoch][subject]
            meta = metadata[subject]
            threshold = float(record["selected_threshold"])
            score = row["score_nez"].astype(np.float64)
            if not len(score) or len(score) != len(row["y_ez"]):
                raise RuntimeError("Frozen diagnostic score/label alignment changed")
            eps = 1e-12
            entropy = -score * np.log(score + eps) - (1 - score) * np.log(1 - score + eps)
            windows = [np.asarray(mask, dtype=bool) for mask in meta["window_mask"]]
            actual_windows = int(sum(mask.sum() for mask in windows))
            slots = int(sum(len(mask) for mask in windows))
            missing_window_fraction = 1 - actual_windows / max(slots, 1)
            seizures = len(meta["b0_features"])
            center = str(meta.get("center", "unknown"))
            modality = str(meta.get("recording_modality", "unknown"))
            source_dataset = subject.split(":", 1)[0] if ":" in subject else "unknown"
            case = {"subject_id": subject, "fold": fold, "center": center,
                    "recording_modality": modality, "source_dataset": source_dataset,
                    "n_channels": len(score), "n_seizures": seizures, "valid_window_count": actual_windows,
                    "missing_invalid_window_fraction": missing_window_fraction,
                    "true_ez_fraction": float(np.mean(row["y_ez"])),
                    "selected_epoch": epoch, "selected_threshold": threshold,
                    "mean_prediction_entropy": float(np.mean(entropy)),
                    "median_probability_margin": float(np.median(np.abs(score - threshold)))}
            for field in ("patient_macro_f1", "patient_ez_f1", "patient_nez_f1", "patient_balanced_accuracy",
                          "patient_ez_auprc", "patient_ez_auroc", "patient_ez_mrr", "top1_is_ez", "predicted_ez_fraction"):
                case[field] = float(record[field])
            cases.append(case)
    if len(cases) != 65:
        raise RuntimeError(f"Expected 65 excluded-validation cases, got {len(cases)}")
    return cases


def correlations(cases: list[dict]) -> list[dict]:
    y = np.asarray([row["patient_macro_f1"] for row in cases], dtype=np.float64)
    predictors = ("patient_ez_auprc", "patient_ez_mrr", "patient_ez_auroc", "n_seizures", "n_channels",
                  "true_ez_fraction", "valid_window_count", "missing_invalid_window_fraction",
                  "mean_prediction_entropy", "median_probability_margin")
    rows = []
    for name in predictors:
        x = np.asarray([row[name] for row in cases], dtype=np.float64)
        if np.std(x) == 0 or np.std(y) == 0:
            pearson = spearman = None
        else:
            pearson = float(pearsonr(x, y).statistic)
            spearman = float(spearmanr(x, y).statistic)
        rows.append({"predictor": name, "n_cases": len(cases), "pearson_r": pearson if pearson is not None else "",
                     "spearman_rho": spearman if spearman is not None else "",
                     "diagnostic_only": True})
    return rows


def center_bootstrap(cases: list[dict], seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    output = []
    for center in sorted(set(row["center"] for row in cases)):
        subset = [row for row in cases if row["center"] == center]
        subjects = sorted(set(row["subject_id"] for row in subset))
        by_subject = {subject: [row["patient_macro_f1"] for row in subset if row["subject_id"] == subject] for subject in subjects}
        draws = np.empty(10000, dtype=np.float64)
        for j in range(10000):
            sample = rng.choice(subjects, size=len(subjects), replace=True)
            draws[j] = np.mean([value for subject in sample for value in by_subject[subject]])
        values = np.asarray([row["patient_macro_f1"] for row in subset])
        output.append({"center": center, "n_patients": len(subjects), "n_cases": len(subset),
                       "macro_f1_mean": float(values.mean()), "macro_f1_median": float(np.median(values)),
                       "bootstrap_95ci_low": float(np.quantile(draws, 0.025)),
                       "bootstrap_95ci_high": float(np.quantile(draws, 0.975)),
                       "small_n_caution": len(subjects) < 5})
    return output


def stratify(cases: list[dict]) -> None:
    ranking_fields = ("patient_ez_auprc", "patient_ez_auroc", "patient_ez_mrr")
    for field in ranking_fields:
        ranks = (rankdata([row[field] for row in cases], method="average") - 1) / max(len(cases) - 1, 1)
        for row, rank in zip(cases, ranks, strict=True):
            row[f"{field}_percentile"] = float(rank)
    for row in cases:
        row["ranking_score"] = float(np.mean([row[f"{field}_percentile"] for field in ranking_fields]))
    median_rank = float(np.median([row["ranking_score"] for row in cases]))
    median_f1 = float(np.median([row["patient_macro_f1"] for row in cases]))
    for row in cases:
        poor_rank = row["ranking_score"] < median_rank
        poor_f1 = row["patient_macro_f1"] < median_f1
        row["failure_type"] = "Type-BOTH" if poor_rank and poor_f1 else "Type-RANK" if poor_rank else "Type-DECISION" if poor_f1 else "Type-NEITHER"
    type_rows = []
    for kind in ("Type-RANK", "Type-DECISION", "Type-BOTH", "Type-NEITHER"):
        selected = [row for row in cases if row["failure_type"] == kind]
        type_rows.append({"failure_type": kind, "n_cases": len(selected),
                          "mean_macro_f1": float(np.mean([row["patient_macro_f1"] for row in selected])) if selected else "",
                          "mean_EZ_AUPRC": float(np.mean([row["patient_ez_auprc"] for row in selected])) if selected else "",
                          "pooled_ranking_median": median_rank, "pooled_macro_f1_median": median_f1})
    write_csv(OUTPUT / "FAILURE_TYPE_SUMMARY.csv", type_rows)


def grouped(cases: list[dict], key_fn, group_order: list[str], group_name: str) -> list[dict]:
    output = []
    for group in group_order:
        selected = [row for row in cases if key_fn(row) == group]
        if not selected:
            continue
        output.append({group_name: group, "n_cases": len(selected), "n_patients": len(set(row["subject_id"] for row in selected)),
                       "macro_f1_mean": float(np.mean([row["patient_macro_f1"] for row in selected])),
                       "macro_f1_median": float(np.median([row["patient_macro_f1"] for row in selected])),
                       "EZ_AUPRC_mean": float(np.mean([row["patient_ez_auprc"] for row in selected]))})
    return output


def main() -> None:
    ensure_source()
    source = json.loads((EXPERIMENT / "SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))
    if not source.get("pass"):
        raise RuntimeError("SOURCE_A1_REPRODUCTION_FAILED")
    cases = collect_cases()
    stratify(cases)
    write_csv(RUNTIME / "private" / "PATIENT_FAILURE_AGGREGATES_PRIVATE.csv", cases)
    write_csv(OUTPUT / "FAILURE_CORRELATIONS.csv", correlations(cases))
    centers = center_bootstrap(cases)
    write_csv(OUTPUT / "FAILURE_BY_CENTER.csv", centers)
    write_csv(OUTPUT / "FAILURE_BY_SEIZURE_COUNT.csv", grouped(cases,
              lambda row: "1" if row["n_seizures"] == 1 else "2" if row["n_seizures"] == 2 else "3+",
              ["1", "2", "3+"], "seizure_count_group"))
    quantiles = np.quantile([row["true_ez_fraction"] for row in cases], [0.25, 0.5, 0.75])
    write_csv(OUTPUT / "FAILURE_BY_EZ_FRACTION.csv", grouped(cases,
              lambda row: f"Q{int(np.searchsorted(quantiles, row['true_ez_fraction'], side='left')) + 1}",
              ["Q1", "Q2", "Q3", "Q4"], "true_EZ_fraction_quartile"))
    corr = {row["predictor"]: row for row in correlations(cases)}
    lines = ["# Frozen A1 validation failure stratification", "",
             "Diagnostic only: 65 VLOO excluded-validation patient-fold cases. No outer test was run.",
             "Types use pooled medians of an equal-weight AUPRC/AUROC/MRR percentile score and Macro-F1. Type-NEITHER is the nonfailure remainder.",
             "No multiple-hypothesis significance claims are made. Small centers are not interpreted as effects.", "",
             "| Predictor | Pearson r vs Macro-F1 | Spearman rho |", "| --- | ---: | ---: |"]
    for row in corr.values():
        lines.append(f"| {row['predictor']} | {row['pearson_r']} | {row['spearman_rho']} |")
    lines.extend(["", "Missing/invalid-window fractions follow the cache window masks; if all recorded masks are valid, this variable has no variance and cannot diagnose signal quality.",
                  "Center, seizure-count and EZ-burden aggregates are in the accompanying CSVs. Patient IDs, labels and individual predictions remain private.",
                  "", "Terminal: `FAILURE_STRATIFICATION_COMPLETE`."])
    (OUTPUT / "FAILURE_STRATIFICATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"terminal": "FAILURE_STRATIFICATION_COMPLETE", "cases": len(cases),
                      "centers": len(centers), "outer_test_accessed": False}), flush=True)


if __name__ == "__main__":
    main()
