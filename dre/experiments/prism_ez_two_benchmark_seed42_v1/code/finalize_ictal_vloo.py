"""Create the historical A1-compatible, validation-only Ictal VLOO summary.

This program reads only per-epoch inner-validation predictions written by
``train_prism.py``.  It never opens the manifest's ``test`` identities, raw EEG,
or an outer-result directory.  For each validation target, epoch and threshold
are selected using the other twelve patients in that frozen fold, then the
target is scored on the same 20 deterministic fixed-query cells used by the
historical A1 development reference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import logit
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score


THRESHOLDS = np.linspace(0.05, 0.95, 19, dtype=np.float64)
METRICS = ("auroc", "ap", "macro_f1", "ez_f1", "balanced_accuracy", "mrr", "top1")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def fixed_query(n_channels: int, fold: int, patient: str, repetition: int) -> np.ndarray:
    if n_channels < 4:
        raise RuntimeError("Historical fixed-query unit requires at least four channels")
    permutation = np.random.default_rng(stable_seed(42, fold, patient, repetition, "split")).permutation(n_channels)
    return permutation[n_channels // 2:]


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if len(labels) != len(scores) or not len(labels) or not np.isfinite(scores).all():
        raise RuntimeError("Invalid validation prediction array")
    if len(np.unique(labels)) != 2:
        raise RuntimeError("A historical VLOO target lacks both Ictal classes")
    order = np.argsort(-scores, kind="stable")
    positive_rank = int(np.flatnonzero(labels[order] == 1)[0])
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "mrr": float(1.0 / (positive_rank + 1)),
        "top1": float(labels[order[0]]),
    }


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = scores >= threshold
    return {
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "ez_f1": float(f1_score(labels, prediction, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
    }


def query_metrics(labels: np.ndarray, margin: np.ndarray) -> dict[str, float]:
    prediction = margin > 0.0
    result = {
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "ez_f1": float(f1_score(labels, prediction, pos_label=1, zero_division=0)),
        "balanced_accuracy": (float(balanced_accuracy_score(labels, prediction))
                              if len(np.unique(labels)) == 2 else float("nan")),
    }
    if len(np.unique(labels)) < 2:
        result.update({name: float("nan") for name in ("auroc", "ap", "mrr", "top1")})
        return result
    result.update(ranking_metrics(labels, margin))
    return result


def canonical_patient(values, channel_order: list[str] | None = None) -> dict[str, list]:
    """Mean repeated records while retaining historical source channel order."""
    if isinstance(values, dict) and set(("labels", "scores")) <= set(values):
        labels = np.asarray(values["labels"], dtype=np.int8)
        scores = np.asarray(values["scores"], dtype=np.float64)
    elif isinstance(values, list):
        by_channel, label = defaultdict(list), {}
        for record in values:
            for channel, current_label, score in zip(record["channel"], record["label"], record["score"]):
                if int(current_label) < 0:
                    continue
                if channel in label and label[channel] != int(current_label):
                    raise RuntimeError("Conflicting ictal channel labels in validation grid")
                label[channel] = int(current_label)
                by_channel[channel].append(float(score))
        channels = ([channel for channel in channel_order if channel in by_channel]
                    if channel_order is not None else list(by_channel))
        if len(channels) != len(by_channel) or set(channels) != set(by_channel):
            raise RuntimeError("Validation rows differ from frozen source channel order")
        labels = np.asarray([label[channel] for channel in channels], dtype=np.int8)
        scores = np.asarray([np.mean(by_channel[channel]) for channel in channels], dtype=np.float64)
    else:
        raise RuntimeError("Unsupported private validation prediction structure")
    ranking_metrics(labels, scores)
    return {"labels": labels.tolist(), "scores": scores.tolist()}


def _patient_hash(value: str, size: int = 20) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:size]


def frozen_channel_order(feature_root: Path, patient: str) -> tuple[list[str], str]:
    paths = sorted((feature_root / "ictal").glob(f"{_patient_hash(patient)}_*.npz"))
    if len(paths) != 1:
        raise RuntimeError("Expected one frozen Ictal token group per validation patient")
    with np.load(paths[0], allow_pickle=False) as item:
        if str(item["patient"]) != patient:
            raise RuntimeError("Frozen token patient hash collision")
        names = [str(value) for value in item["channel_names"]]
    if len(names) != len(set(names)):
        raise RuntimeError("Frozen source channel order contains duplicates")
    return names, digest(paths[0])


def load_grid(work: Path, fold: int, protocol_sha: str,
              feature_root: Path) -> tuple[list[dict[str, dict]], list[str], dict[str, str]]:
    selection = json.loads((work / "PRISM_VALIDATION_SELECTION.json").read_text(encoding="utf-8"))
    if selection.get("status") != "TRAIN_VALIDATION_COMPLETE" or selection.get("test_accessed"):
        raise RuntimeError(f"Fold {fold} is not a completed validation-only run")
    if selection.get("protocol_sha256") != protocol_sha:
        raise RuntimeError(f"Fold {fold} protocol mismatch")
    completed = int(selection["early_stopping"]["completed_epochs"])
    paths = [work / f"validation_epoch_{epoch:02d}_private.json" for epoch in range(1, completed + 1)]
    if not paths or any(not path.is_file() for path in paths):
        raise RuntimeError(f"Fold {fold} has an incomplete validation-prediction grid")
    raw_grid, hashes = [], {}
    for epoch, path in enumerate(paths, 1):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("epoch", -1)) != epoch:
            raise RuntimeError(f"Fold {fold} validation epoch order changed")
        private = payload.get("private", {})
        if len(private) != 13:
            raise RuntimeError(f"Fold {fold} must have 13 validation patients for historical VLOO")
        raw_grid.append(private)
        hashes[path.name] = digest(path)
    patient_ids = sorted(raw_grid[0])
    if any(sorted(snapshot) != patient_ids for snapshot in raw_grid):
        raise RuntimeError(f"Fold {fold} validation membership changed across epochs")
    orders = {}
    for index, patient in enumerate(patient_ids):
        orders[patient], order_hash = frozen_channel_order(feature_root, patient)
        hashes[f"channel_order_source_{index:02d}"] = order_hash
    grid = [{patient: canonical_patient(snapshot[patient], orders[patient])
             for patient in patient_ids} for snapshot in raw_grid]
    for snapshot in grid:
        for patient in patient_ids:
            values = snapshot[patient]
            ranking_metrics(np.asarray(values["labels"], dtype=np.int8),
                            np.asarray(values["scores"], dtype=np.float64))
    return grid, patient_ids, hashes


def choose_excluding(grid: list[dict[str, dict]], patient_ids: list[str], target: str) -> tuple[int, float]:
    other = [patient for patient in patient_ids if patient != target]
    ranked_epochs = []
    for epoch_index, snapshot in enumerate(grid):
        values = [ranking_metrics(np.asarray(snapshot[patient]["labels"], dtype=np.int8),
                                  np.asarray(snapshot[patient]["scores"], dtype=np.float64)) for patient in other]
        key = tuple(round(float(np.mean([value[name] for value in values])), 12)
                    for name in ("auroc", "ap", "mrr")) + (-epoch_index,)
        ranked_epochs.append((key, epoch_index))
    epoch_index = max(ranked_epochs)[1]
    snapshot = grid[epoch_index]
    candidates = []
    for index, threshold in enumerate(THRESHOLDS):
        values = [threshold_metrics(np.asarray(snapshot[patient]["labels"], dtype=np.int8),
                                    np.asarray(snapshot[patient]["scores"], dtype=np.float64), float(threshold))
                  for patient in other]
        key = tuple(round(float(np.mean([value[name] for value in values])), 12)
                    for name in ("macro_f1", "ez_f1", "balanced_accuracy"))
        key += (-round(abs(float(threshold) - 0.5), 6), -index)
        candidates.append((key, float(threshold)))
    return epoch_index, max(candidates)[1]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_sha = digest(args.protocol)
    freezes, private_selection, private_queries, summary = {}, [], [], []
    for fold in range(1, 6):
        grid, patient_ids, hashes = load_grid(args.runtime / "ictal" / f"fold{fold}", fold,
                                              protocol_sha, args.feature_cache)
        freezes[f"fold{fold}"] = hashes
        rows = []
        for patient in patient_ids:
            epoch_index, threshold = choose_excluding(grid, patient_ids, patient)
            values = grid[epoch_index][patient]
            labels = np.asarray(values["labels"], dtype=np.int8)
            scores = np.asarray(values["scores"], dtype=np.float64)
            margin = logit(np.clip(scores, 1e-8, 1.0 - 1e-8)) - logit(threshold)
            private_selection.append({"fold": fold, "patient_private": patient, "selected_epoch": epoch_index + 1,
                                      "selected_threshold": threshold})
            for repetition in range(20):
                query = fixed_query(len(labels), fold, patient, repetition)
                rows.append({"fold": fold, "patient_private": patient, "repetition": repetition,
                             **query_metrics(labels[query], margin[query])})
        if len(rows) != 260:
            raise RuntimeError(f"Fold {fold} did not generate 13 x 20 VLOO query cells")
        private_queries.extend(rows)
        selected_rows = [row for row in private_selection if row["fold"] == fold]
        summary.append({"benchmark": "Ictal", "fold": fold, "validation_patients": len(patient_ids),
                        "query_cells": len(rows), "mean_selected_epoch": float(np.mean([row["selected_epoch"] for row in selected_rows]),
                        ), "mean_selected_threshold": float(np.mean([row["selected_threshold"] for row in selected_rows]),
                        ), **{name: float(np.nanmean([row[name] for row in rows])) for name in METRICS}})
    if len(private_queries) != 1300 or len({row["patient_private"] for row in private_queries}) != 47:
        raise RuntimeError("Historical 65-target x 20-query structure was not reproduced")
    public = {"benchmark": "Ictal", "scope": "validation-only historical 65-target fixed-query VLOO",
              "matched_cells": 65, "query_repetitions": 20, "unique_patients": 47,
              **{name: float(np.nanmean([row[name] for row in private_queries])) for name in METRICS}}
    freeze = {"status": "ICTAL_VALIDATION_SCORE_FREEZE_COMPLETE", "protocol_sha256": protocol_sha,
              "validation_prediction_hashes": freezes, "matched_cells": 65, "query_repetitions": 20,
              "target_labels_used_for_epoch_or_threshold_selection": False,
              "outer_test_accessed": False}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ICTAL_VALIDATION_SCORE_FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    # Private patient identifiers and cell-level predictions remain under runtime, not Git.
    private_root = args.runtime / "ictal" / "private_vloo"
    write_csv(private_root / "VLOO_SELECTION_PRIVATE.csv", private_selection)
    write_csv(private_root / "VLOO_QUERY_METRICS_PRIVATE.csv", private_queries)
    write_csv(args.output / "ICTAL_VALIDATION.csv", [*summary, public])
    print(json.dumps({"status": "ICTAL_VLOO_VALIDATION_COMPLETE", "summary": public,
                      "outer_test_accessed": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
