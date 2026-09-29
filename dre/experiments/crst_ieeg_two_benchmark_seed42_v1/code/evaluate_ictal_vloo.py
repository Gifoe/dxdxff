"""Frozen 65-cell/47-ID fixed-query VLOO evaluation of CRST scores.

No model retraining or target-label checkpoint/threshold selection occurs.
Private per-patient epoch scores and query rows remain outside the Git tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np
from scipy.special import logit
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)

from crst_metrics import patient_metrics


METRICS = ("ap", "auc", "mrr", "top1", "ndcg", "macro_f1", "ez_f1", "ba")
THRESHOLDS = np.linspace(0.05, 0.95, 19)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(*parts):
    data = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little") % (2**32)


def fixed_query(n, fold, patient, rep):
    if n < 4:
        raise RuntimeError("Too few channels for historical fixed query")
    perm = np.random.default_rng(stable_seed(42, fold, patient, rep, "split")).permutation(n)
    return perm[n // 2:]


def query_metrics(y, margin):
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(margin, dtype=np.float64)
    if len(y) != len(score) or not np.isfinite(score).all():
        raise RuntimeError("Invalid fixed-query inputs")
    pred = score > 0
    out = {"macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
           "ez_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
           "ba": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else float("nan")}
    if len(np.unique(y)) < 2:
        out.update({key: float("nan") for key in ("ap", "auc", "mrr", "top1", "ndcg")})
        return out
    order = np.argsort(-score, kind="stable")
    positive_ranks = np.flatnonzero(y[order] == 1)
    discounts = np.log2(np.arange(2, len(y) + 2))
    out.update({"ap": float(average_precision_score(y, score)),
                "auc": float(roc_auc_score(y, score)),
                "mrr": float(1 / (1 + positive_ranks[0])),
                "top1": float(y[order[0]]),
                "ndcg": float(np.sum(y[order] / discounts) /
                              np.sum(np.sort(y)[::-1] / discounts))})
    return out


def load_grid(work: Path, fold: int, variant: str, protocol_sha: str, train_sha: str):
    complete = json.loads((work / "supervised_complete.json").read_text(encoding="utf-8"))
    paths = [work / f"epoch_{epoch:02d}_validation_private.json"
             for epoch in range(1, int(complete["last_epoch"]) + 1)]
    if not paths or any(not p.is_file() for p in paths):
        raise RuntimeError("Incomplete frozen validation score grid")
    files = {p.name: digest(p) for p in paths}
    grid = []
    for epoch, path in enumerate(paths, 1):
        obj = json.loads(path.read_text(encoding="utf-8"))
        if (obj["fold"] != fold or obj["variant"] != variant or obj["epoch"] != epoch or
                obj["protocol_sha"] != protocol_sha or obj["train_sha"] != train_sha):
            raise RuntimeError("Epoch validation score provenance mismatch")
        if len(obj["patient_scores_private"]) != 13:
            raise RuntimeError("Historical validation role count not 13")
        grid.append(obj["patient_scores_private"])
    ids = sorted(grid[0])
    if any(sorted(item) != ids for item in grid):
        raise RuntimeError("Validation patient identity changed across epochs")
    return grid, ids, files


def threshold_score(other, snapshot, tau):
    vals = []
    for patient in other:
        row = snapshot[patient]
        y, p = np.asarray(row["labels"], int), np.asarray(row["scores"], float)
        pred = p >= tau
        macro = f1_score(y, pred, average="macro", zero_division=0)
        positive = f1_score(y, pred, pos_label=1, zero_division=0)
        ba = balanced_accuracy_score(y, pred)
        vals.append((macro, positive, ba))
    return tuple(round(float(x), 12) for x in np.mean(vals, axis=0))


def choose_excluding(grid, ids, target):
    other = [patient for patient in ids if patient != target]
    candidates = []
    for epoch_idx, snapshot in enumerate(grid):
        values = [patient_metrics(snapshot[patient]["labels"],
                                  snapshot[patient]["scores"]) for patient in other]
        rank = tuple(round(float(np.mean([row[key] for row in values])), 12)
                     for key in ("auroc", "ap", "mrr")) + (-epoch_idx,)
        candidates.append((rank, epoch_idx))
    epoch_idx = max(candidates)[1]
    snapshot = grid[epoch_idx]
    thresholds = [(threshold_score(other, snapshot, float(tau)) +
                   (-round(abs(float(tau) - 0.5), 6), -idx), float(tau))
                  for idx, tau in enumerate(THRESHOLDS)]
    tau = max(thresholds)[1]
    return epoch_idx, tau


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--training-lock", type=Path, required=True)
    p.add_argument("--historical-queries", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    protocol_sha, train_sha = digest(args.protocol), digest(args.training_lock)
    snapshots = {}
    files = {}
    for variant in ("CRST-0", "CRST-FULL"):
        for fold in range(1, 6):
            work = args.runtime / "ictal" / f"fold{fold}" / variant
            grid, ids, hashes = load_grid(work, fold, variant, protocol_sha, train_sha)
            snapshots[(variant, fold)] = grid, ids
            files[f"{variant}/fold{fold}"] = hashes
    freeze = {"protocol_sha": protocol_sha, "train_sha": train_sha,
              "private_score_file_hashes": files,
              "target_labels_used_for_checkpoint_or_threshold_selection": False,
              "test_waveforms_used": False}
    freeze_path = args.runtime / "ICTAL_SCORE_FREEZE_BEFORE_VLOO.json"
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    query_rows, selection_rows = [], []
    for variant in ("CRST-0", "CRST-FULL"):
        for fold in range(1, 6):
            grid, ids = snapshots[(variant, fold)]
            for patient in ids:
                epoch_idx, tau = choose_excluding(grid, ids, patient)
                row = grid[epoch_idx][patient]
                y = np.asarray(row["labels"], dtype=np.int8)
                score = np.asarray(row["scores"], dtype=np.float64)
                margin = logit(np.clip(score, 1e-8, 1-1e-8)) - logit(tau)
                selection_rows.append({"variant": variant, "fold": fold,
                                       "patient_private": patient, "epoch": epoch_idx+1,
                                       "threshold": tau})
                for rep in range(20):
                    query = fixed_query(len(y), fold, patient, rep)
                    query_rows.append({"variant": variant, "fold": fold,
                                       "patient_private": patient, "rep": rep,
                                       **query_metrics(y[query], margin[query])})
            print(json.dumps({"variant": variant, "fold": fold,
                              "validation_targets": len(ids), "query_repetitions": 20}), flush=True)
    unique = {row["patient_private"] for row in query_rows}
    if len(query_rows) != 2600 or len(unique) != 47:
        raise RuntimeError("Historical fixed-query 65x20/47-ID structure not reproduced")
    with args.historical_queries.open("rb") as f:
        history = pickle.load(f)
    baseline = {(int(row["fold"]), str(row["sid"]), int(row["rep"])): row
                for row in history if row["model"] == "I0_A1"}
    if len(baseline) != 1300 or set(baseline) != {
            (row["fold"], row["patient_private"], row["rep"]) for row in query_rows}:
        raise RuntimeError("Historical A1 matched query identities differ")
    if abs(float(np.nanmean([row["ap"] for row in baseline.values()])) -
           0.5767434626151353) > 1e-9:
        raise RuntimeError("Historical A1 fixed-query AP replay failed")
    private_dir = args.runtime / "ictal" / "private_evaluation"
    write_csv(private_dir / "QUERY_ROWS_PRIVATE.csv", query_rows)
    write_csv(private_dir / "VLOO_SELECTION_PRIVATE.csv", selection_rows)
    baseline_row = {"model": "A1", "benchmark": "Ictal", "matched_cells": 65,
                    "query_repetitions": 1300, "unique_patient_ids": 47}
    baseline_row.update({metric: float(np.nanmean([row[metric] for row in baseline.values()]))
                         for metric in METRICS})
    summary = [baseline_row]
    for variant in ("CRST-0", "CRST-FULL"):
        part = [row for row in query_rows if row["variant"] == variant]
        result = {"model": variant, "benchmark": "Ictal", "matched_cells": 65,
                  "query_repetitions": 1300, "unique_patient_ids": 47}
        result.update({metric: float(np.nanmean([row[metric] for row in part]))
                       for metric in METRICS})
        summary.append(result)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "ICTAL_PRIMARY_METRICS.csv", summary)
    query_map = {(row["variant"], row["fold"], row["patient_private"], row["rep"]): row
                 for row in query_rows}
    ids = sorted(unique)
    draws = np.random.default_rng(42).integers(0, len(ids), size=(10000, len(ids)))
    counts = np.zeros((10000, len(ids)), dtype=np.int16)
    np.add.at(counts, (np.arange(10000)[:, None], draws), 1)
    counts = counts.astype(np.float32)
    by_id = {patient: i for i, patient in enumerate(ids)}
    bootstrap = []
    for candidate, reference in (("CRST-0", "A1"), ("CRST-FULL", "A1"),
                                 ("CRST-FULL", "CRST-0")):
        for metric in METRICS:
            numerator = np.zeros(len(ids))
            denominator = np.zeros(len(ids))
            for fold, patient, rep in baseline:
                current = query_map[(candidate, fold, patient, rep)][metric]
                prior = (baseline[(fold, patient, rep)][metric] if reference == "A1" else
                         query_map[(reference, fold, patient, rep)][metric])
                if np.isfinite(current) and np.isfinite(prior):
                    idx = by_id[patient]
                    numerator[idx] += current - prior
                    denominator[idx] += 1
            values = (counts @ numerator) / (counts @ denominator).clip(min=1)
            bootstrap.append({"comparison": f"{candidate}-{reference}", "metric": metric,
                              "delta": float(numerator.sum() / denominator.sum()),
                              "ci_lower": float(np.quantile(values, .025)),
                              "ci_upper": float(np.quantile(values, .975)),
                              "patient_clusters": 47, "draws": 10000, "seed": 42})
    write_csv(args.output / "ICTAL_BOOTSTRAP.csv", bootstrap)
    (args.output / "ICTAL_SCORE_FREEZE_AUDIT.json").write_text(json.dumps({
        "pass": True, "private_freeze_sha256": digest(freeze_path),
        "private_query_sha256": digest(private_dir / "QUERY_ROWS_PRIVATE.csv"),
        "historical_a1_query_sha256": digest(args.historical_queries),
        "matched_cells": 65, "query_repetitions": 1300, "unique_patient_ids": 47,
        "target_labels_used_for_checkpoint_or_threshold_selection": False,
        "outer_test_waveforms_accessed": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ICTAL_VLOO_COMPLETE", "summary": summary,
                      "test_accessed": False}), flush=True)


if __name__ == "__main__":
    main()
