"""Patient metrics and locked VLOO checkpoint/threshold selection; no outer input."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


THRESHOLDS = np.linspace(0.05, 0.95, 19, dtype=np.float32)
METRICS = (
    "patient_macro_f1", "patient_ez_f1", "patient_nez_f1", "patient_balanced_accuracy",
    "patient_ez_auprc", "patient_ez_auroc", "patient_ez_mrr", "top1_is_ez",
    "predicted_ez_fraction",
)


def patient_grid(record: dict) -> dict:
    valid = np.asarray(record["channel_mask"], dtype=bool)
    yez = np.asarray(record["labels_ez"], dtype=np.float64)[valid]
    ynez = np.asarray(record.get("labels_nez", record["labels"]), dtype=np.float64)[valid]
    snez = np.asarray(record["score_nez"], dtype=np.float64)[valid]
    sez = np.asarray(record["score_ez"], dtype=np.float64)[valid]
    if valid.sum() == 0 or set(np.unique(yez)) != {0.0, 1.0} or not np.array_equal(1 - yez, ynez):
        raise RuntimeError("Validation patient lacks both classes or has inconsistent label semantics")
    if not np.isfinite(snez).all() or not np.isfinite(sez).all():
        raise RuntimeError("Nonfinite validation score")
    pred_nez = snez[None, :] >= THRESHOLDS.astype(np.float64)[:, None]
    true_nez = ynez.astype(bool)[None, :]
    tp_nez = (pred_nez & true_nez).sum(axis=1)
    fp_nez = (pred_nez & ~true_nez).sum(axis=1)
    fn_nez = (~pred_nez & true_nez).sum(axis=1)
    tn_nez = (~pred_nez & ~true_nez).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        nez_f1 = np.divide(2 * tp_nez, 2 * tp_nez + fp_nez + fn_nez,
                           out=np.zeros(len(THRESHOLDS)), where=(2 * tp_nez + fp_nez + fn_nez) > 0)
        ez_f1 = np.divide(2 * tn_nez, 2 * tn_nez + fn_nez + fp_nez,
                          out=np.zeros(len(THRESHOLDS)), where=(2 * tn_nez + fn_nez + fp_nez) > 0)
    balanced = 0.5 * (tp_nez / (tp_nez + fn_nez) + tn_nez / (tn_nez + fp_nez))
    macro = 0.5 * (nez_f1 + ez_f1)
    order = np.argsort(sez)[::-1]
    positive_ranks = np.flatnonzero(yez[order] == 1)
    mrr = 1.0 / (int(positive_ranks[0]) + 1) if positive_ranks.size else 0.0
    fixed = {
        "patient_ez_auprc": float(average_precision_score(yez, sez)),
        "patient_ez_auroc": float(roc_auc_score(yez, sez)),
        "patient_ez_mrr": float(mrr),
        "top1_is_ez": float(yez[int(np.argmax(sez))] == 1.0),
    }
    return {
        "subject_id": str(record["subject_id"]),
        "n_channels": int(valid.sum()),
        "grid": {
            "patient_macro_f1": macro.tolist(),
            "patient_ez_f1": ez_f1.tolist(),
            "patient_nez_f1": nez_f1.tolist(),
            "patient_balanced_accuracy": balanced.tolist(),
            "predicted_ez_fraction": ((~pred_nez).sum(axis=1) / int(valid.sum())).tolist(),
        },
        "fixed": fixed,
    }


def epoch_grid(records: list[dict], epoch: int) -> dict:
    patients = [patient_grid(record) for record in records]
    if len(patients) != 13 or len({row["subject_id"] for row in patients}) != 13:
        raise RuntimeError("Expected 13 distinct validation patients")
    return {"epoch": epoch, "patients": sorted(patients, key=lambda row: row["subject_id"])}


def _candidate_key(metric_array: dict[str, np.ndarray], indices: list[int], epoch_idx: int, threshold_idx: int) -> tuple:
    def avg(metric: str) -> float:
        return round(float(metric_array[metric][epoch_idx, threshold_idx, indices].mean()), 12)
    return (
        avg("patient_macro_f1"), avg("patient_ez_f1"), avg("patient_balanced_accuracy"),
        -round(abs(float(THRESHOLDS[threshold_idx]) - 0.5), 6), -epoch_idx,
    )


def choose(metric_array: dict[str, np.ndarray], indices: list[int]) -> tuple[int, int]:
    best, selected = None, None
    for epoch_idx in range(30):
        for threshold_idx in range(len(THRESHOLDS)):
            key = _candidate_key(metric_array, indices, epoch_idx, threshold_idx)
            if best is None or key > best:
                best, selected = key, (epoch_idx, threshold_idx)
    assert selected is not None
    return selected


def finalize_fold(epoch_payloads: list[dict], variant: str, fold: int, private_csv: Path) -> tuple[dict, dict]:
    if [payload["epoch"] for payload in epoch_payloads] != list(range(1, 31)):
        raise RuntimeError("Cannot select checkpoints without all 30 validation epochs")
    ids = [row["subject_id"] for row in epoch_payloads[0]["patients"]]
    if any([row["subject_id"] for row in payload["patients"]] != ids for payload in epoch_payloads):
        raise RuntimeError("Validation patient membership/order changed across epochs")
    metric_array = {
        metric: np.stack([np.stack([
            np.asarray(row["grid"][metric] if metric in row["grid"] else
                       [row["fixed"][metric]] * len(THRESHOLDS), dtype=float)
            for row in payload["patients"]], axis=1)
            for payload in epoch_payloads], axis=0)
        for metric in METRICS
    }
    if any(array.shape != (30, 19, 13) or not np.isfinite(array).all() for array in metric_array.values()):
        raise RuntimeError("Malformed validation metric grid")
    selected = []
    for patient_idx, subject_id in enumerate(ids):
        epoch_idx, threshold_idx = choose(metric_array, [i for i in range(13) if i != patient_idx])
        row = {"subject_id": subject_id, "variant": variant, "fold": fold,
               "selected_epoch": epoch_idx + 1, "selected_threshold": float(THRESHOLDS[threshold_idx])}
        row.update({metric: float(metric_array[metric][epoch_idx, threshold_idx, patient_idx]) for metric in METRICS})
        selected.append(row)
    private_csv.parent.mkdir(parents=True, exist_ok=True)
    with private_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    public_fold = {"variant": variant, "fold": fold, "n_patients": 13}
    public_fold.update({metric: float(np.mean([row[metric] for row in selected])) for metric in METRICS})
    epochs = np.asarray([row["selected_epoch"] for row in selected])
    public_fold["mean_selected_epoch"] = float(epochs.mean())
    public_fold["median_selected_epoch"] = float(np.median(epochs))
    threshold_counts = {f"{float(threshold):.2f}": sum(abs(row["selected_threshold"] - float(threshold)) < 1e-7
                                                  for row in selected) for threshold in THRESHOLDS}
    public_fold["threshold_distribution_json"] = json.dumps({key: count for key, count in threshold_counts.items() if count})
    epoch_idx, threshold_idx = choose(metric_array, list(range(13)))
    fullval = {"variant": variant, "fold": fold, "selected_epoch": epoch_idx + 1,
               "selected_threshold": float(THRESHOLDS[threshold_idx]), "n_patients": 13}
    fullval.update({f"apparent_{metric}": float(metric_array[metric][epoch_idx, threshold_idx, :].mean()) for metric in METRICS})
    return public_fold, fullval
