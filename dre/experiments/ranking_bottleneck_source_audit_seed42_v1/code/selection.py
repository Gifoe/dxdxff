"""Two predeclared excluded-patient VLOO selectors over frozen validation grids."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import (EXPERIMENT, METRICS, RUNTIME, THRESHOLDS, ensure_source, finalize_fold,
                    mean, source_grid, write_csv, write_json)


def metric_cube(grids: list[dict]) -> tuple[list[str], dict[str, np.ndarray]]:
    if [item["epoch"] for item in grids] != list(range(1, 31)):
        raise RuntimeError("Selection requires all 30 epochs")
    ids = [item["subject_id"] for item in grids[0]["patients"]]
    if len(ids) != 13 or any([item["subject_id"] for item in grid["patients"]] != ids for grid in grids):
        raise RuntimeError("Validation membership changed")
    cube = {metric: np.stack([np.stack([
        np.asarray(item["grid"][metric] if metric in item["grid"] else
                   [item["fixed"][metric]] * len(THRESHOLDS), dtype=float)
        for item in grid["patients"]], axis=1) for grid in grids], axis=0) for metric in METRICS}
    if any(value.shape != (30, 19, 13) or not np.isfinite(value).all() for value in cube.values()):
        raise RuntimeError("Invalid frozen metric cube")
    return ids, cube


def choose_rank(cube: dict[str, np.ndarray], retained: list[int]) -> tuple[int, int]:
    best_epoch, best_key = 0, None
    for epoch in range(30):
        key = tuple(round(float(cube[metric][epoch, 0, retained].mean()), 12) for metric in
                    ("patient_ez_auprc", "patient_ez_mrr", "patient_ez_auroc")) + (-epoch,)
        if best_key is None or key > best_key:
            best_epoch, best_key = epoch, key
    best_threshold, best_key = 0, None
    for threshold in range(19):
        key = (tuple(round(float(cube[metric][best_epoch, threshold, retained].mean()), 12) for metric in
                     ("patient_macro_f1", "patient_ez_f1", "patient_balanced_accuracy")) +
               (-round(abs(float(THRESHOLDS[threshold]) - 0.5), 6),))
        if best_key is None or key > best_key:
            best_threshold, best_key = threshold, key
    return best_epoch, best_threshold


def srank_fold(grids: list[dict], variant: str, fold: int, private_csv: Path) -> dict:
    ids, cube = metric_cube(grids)
    rows = []
    for excluded, subject_id in enumerate(ids):
        epoch, threshold = choose_rank(cube, [i for i in range(13) if i != excluded])
        row = {"subject_id": subject_id, "variant": variant, "fold": fold,
               "selected_epoch": epoch + 1, "selected_threshold": float(THRESHOLDS[threshold])}
        row.update({metric: float(cube[metric][epoch, threshold, excluded]) for metric in METRICS})
        rows.append(row)
    write_csv(private_csv, rows)
    public = {"variant": variant, "fold": fold, "n_patients": 13}
    public.update({metric: mean(rows, metric) for metric in METRICS})
    epochs = [row["selected_epoch"] for row in rows]
    public["mean_selected_epoch"] = float(np.mean(epochs))
    public["median_selected_epoch"] = float(np.median(epochs))
    public["epoch_distribution_json"] = json.dumps({str(epoch): epochs.count(epoch) for epoch in sorted(set(epochs))})
    return public


def both_protocols(grids: list[dict], variant: str, fold: int) -> tuple[dict, dict]:
    sf1, _ = finalize_fold(grids, variant, fold, RUNTIME / "private" / f"{variant}_fold_{fold}_SF1_PATIENT.csv")
    srank = srank_fold(grids, variant, fold, RUNTIME / "private" / f"{variant}_fold_{fold}_SRANK_PATIENT.csv")
    sf1["protocol"] = "S-F1"
    srank["protocol"] = "S-RANK"
    return sf1, srank


def main() -> None:
    ensure_source()
    reproduction = json.loads((EXPERIMENT / "SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))
    if not reproduction.get("pass"):
        raise RuntimeError("SOURCE_A1_REPRODUCTION_FAILED")
    sf1_rows, srank_rows = [], []
    for fold in range(1, 6):
        sf1, srank = both_protocols([source_grid(fold, epoch) for epoch in range(1, 31)], "A1", fold)
        sf1_rows.append(sf1)
        srank_rows.append(srank)
    folder = EXPERIMENT / "selection_mismatch"
    write_csv(folder / "A1_SF1_BY_FOLD.csv", sf1_rows)
    write_csv(folder / "A1_SRANK_BY_FOLD.csv", srank_rows)
    metrics = ("patient_ez_auprc", "patient_ez_mrr", "top1_is_ez", "patient_ez_auroc", "patient_macro_f1", "patient_ez_f1")
    comparison = [{"metric": metric, "SF1": mean(sf1_rows, metric), "SRANK": mean(srank_rows, metric),
                   "delta_SRANK_minus_SF1": mean(srank_rows, metric) - mean(sf1_rows, metric),
                   "positive_folds": sum(float(r[metric]) > float(s[metric]) for s, r in zip(sf1_rows, srank_rows, strict=True))}
                  for metric in metrics]
    write_csv(folder / "SELECTION_COMPARISON.csv", comparison)
    by_metric = {row["metric"]: row for row in comparison}
    checks = {"auprc_gain_ge_0_010": by_metric["patient_ez_auprc"]["delta_SRANK_minus_SF1"] >= 0.010,
              "auprc_positive_folds_ge_4": by_metric["patient_ez_auprc"]["positive_folds"] >= 4,
              "macro_f1_preserved": by_metric["patient_macro_f1"]["delta_SRANK_minus_SF1"] >= -0.005}
    passed = all(checks.values())
    write_json(folder / "SELECTION_GATE.json", {"pass": passed, "checks": checks,
               "terminal": "CHECKPOINT_SELECTION_MISMATCH_SUPPORTED" if passed else "CHECKPOINT_SELECTION_MISMATCH_NOT_SUPPORTED",
               "outer_test_accessed": False})
    print(json.dumps({"checks": checks, "sf1_auprc": by_metric["patient_ez_auprc"]["SF1"],
                      "srank_auprc": by_metric["patient_ez_auprc"]["SRANK"]}), flush=True)


if __name__ == "__main__":
    main()
