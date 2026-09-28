"""Frozen 65-cell/47-ID matched-query I1 vs historical I0 evaluation.

Private per-target selection/query records never leave runtime. The source
historical A1 split, target labels, fixed queries and metric definitions are
imported rather than reimplemented.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np

from gate_tf_ictal_stage2 import sha256
from train_tf_ictal_stage1 import BASE_LOCK_SHA, AMENDMENT_SHA

METRICS = ("ap", "auc", "mrr", "top1", "ndcg", "macro_f1", "ez_f1", "ba")


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"Empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def historical_modules(source):
    sys.path.insert(0, str(source / "a1_a2_patient_equal_objective_seed42_v1/code"))
    sys.path.insert(0, str(source / "active_fewshot_patient_calibration_seed42_v1/code"))
    import common as afc
    import run as afr
    from development_metrics import patient_grid, THRESHOLDS
    return afc, afr, patient_grid, THRESHOLDS


def score_path(runtime, fold, stage, epoch):
    return runtime / f"fold_{fold}" / f"stage{stage}" / f"epoch_{epoch:02d}_SCORES_PRIVATE.pkl"


def check(a):
    if sha256(a.lock) != BASE_LOCK_SHA or sha256(a.amendment) != AMENDMENT_SHA:
        raise RuntimeError("I1 protocol/amendment changed")
    gate = json.loads(a.gate_audit.read_text(encoding="utf-8"))
    if gate["protocol_sha256"] != BASE_LOCK_SHA or gate["amendment_sha256"] != AMENDMENT_SHA:
        raise RuntimeError("Stage-2 gate audit mismatch")
    private = a.runtime / "private/ICTAL_STAGE2_GATE_PRIVATE.csv"
    if sha256(private) != gate["private_gate_sha256"]:
        raise RuntimeError("Private gate changed")
    return gate, list(csv.DictReader(private.open(encoding="utf-8")))


def freeze(a, gate):
    files = {}
    for fold in range(1, 6):
        for stage in (1, 2):
            if stage == 2 and fold not in gate["eligible_folds"]:
                continue
            summary = a.runtime / f"fold_{fold}" / f"stage{stage}" / "summary.json"
            value = json.loads(summary.read_text(encoding="utf-8"))
            if value.get(f"stage{stage}_complete") is not True or value["epochs"] != 15:
                raise RuntimeError(f"Incomplete I1 fold={fold} stage={stage}")
            for epoch in range(1, 16):
                path = score_path(a.runtime, fold, stage, epoch)
                files[str(path.relative_to(a.runtime))] = sha256(path)
    if len(files) != 15 * (5 + len(gate["eligible_folds"])):
        raise RuntimeError("Incomplete I1 score-only grid")
    payload = {"protocol_sha256": BASE_LOCK_SHA, "amendment_sha256": AMENDMENT_SHA,
               "stage2_gate_private_sha256": gate["private_gate_sha256"],
               "n_score_files": len(files), "files": files,
               "target_labels_indexed_during_score_freeze": False,
               "historical_loader_materialized_validation_labels": True}
    private = a.runtime / "private/ICTAL_SCORE_FREEZE_BEFORE_TARGET_SELECTION.json"
    private.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "ICTAL_SCORE_FREEZE_AUDIT.json").write_text(json.dumps({
        "pass": True, "score_files": len(files), "private_manifest_sha256": sha256(private),
        "target_labels_indexed_during_score_freeze": False}, indent=2) + "\n", encoding="utf-8")
    print(f"I1_SCORE_FREEZE_PASS {len(files)}", flush=True)


def labels(history, fold, row):
    path = history / "private" / f"fold_{fold}" / \
        f"epoch_{int(row['selected_epoch']):02d}_representations.pkl"
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload["fold"] != fold or payload["epoch"] != int(row["selected_epoch"]):
        raise RuntimeError("Historical A1 representation provenance mismatch")
    return np.asarray(payload["val"][row["subject_id"]]["y"], dtype=np.int8), \
        payload["val"][row["subject_id"]]


def evaluate(a, gate, gate_rows):
    manifest = a.runtime / "private/ICTAL_SCORE_FREEZE_BEFORE_TARGET_SELECTION.json"
    frozen = json.loads(manifest.read_text(encoding="utf-8"))
    if frozen["protocol_sha256"] != BASE_LOCK_SHA or frozen["amendment_sha256"] != AMENDMENT_SHA:
        raise RuntimeError("Score freeze missing or wrong")
    if frozen["stage2_gate_private_sha256"] != gate["private_gate_sha256"]:
        raise RuntimeError("Score freeze/gate mismatch")
    afc, afr, patient_grid, thresholds = historical_modules(a.source)
    gate_map = {(int(row["fold"]), row["target_private"]): row["stage2_eligible"] == "True"
                for row in gate_rows}
    private_rows = []
    selections = []
    for fold in range(1, 6):
        history_rows = list(csv.DictReader((a.history / "private" / f"fold_{fold}" /
                                            "A1_VLOO_PRIVATE.csv").open(encoding="utf-8")))
        if len(history_rows) != 13:
            raise RuntimeError("Historical A1 VLOO cell count changed")
        names = sorted(row["subject_id"] for row in history_rows)
        historical = {row["subject_id"]: row for row in history_rows}
        labels_by_name = {}
        item_by_name = {}
        for name in names:
            labels_by_name[name], item_by_name[name] = labels(a.history, fold, historical[name])
        candidates = []
        for stage in (1, 2):
            if stage == 2 and fold not in gate["eligible_folds"]:
                continue
            for epoch in range(1, 16):
                path = score_path(a.runtime, fold, stage, epoch)
                if sha256(path) != frozen["files"].get(str(path.relative_to(a.runtime))):
                    raise RuntimeError("Frozen I1 score changed")
                with path.open("rb") as handle:
                    snap = pickle.load(handle)
                if sorted(snap) != names:
                    raise RuntimeError("I1 validation patient membership changed")
                metric_by_name = {}
                for name in names:
                    y = labels_by_name[name]
                    score = snap[name]
                    if len(y) != score["n_channels"]:
                        raise RuntimeError("I1 score/label channel alignment mismatch")
                    metric_by_name[name] = patient_grid({
                        "subject_id": name, "channel_mask": np.ones(len(y), bool),
                        "labels_ez": y, "labels_nez": 1-y, "labels": 1-y,
                        "score_ez": score["score_ez"], "score_nez": score["score_nez"]})
                candidates.append((stage, epoch, snap, metric_by_name))
        for name in names:
            prior = historical[name]
            y = labels_by_name[name]
            item = item_by_name[name]
            tau0 = float(prior["selected_threshold"])
            base_margin = afr.margin_for(item, tau0)
            afr.verify_b0(item, tau0, base_margin)
            for rep in range(20):
                _, query = afc.split_indices(len(y), 42, fold, name, rep)
                private_rows.append({"fold": fold, "sid": name, "rep": rep, "model": "I0_A1",
                                     **afc.query_metrics(y[query], base_margin[query])})
            other = [patient for patient in names if patient != name]
            choice = None
            for ordinal, (stage, epoch, snap, grid) in enumerate(candidates):
                if stage == 2 and not gate_map[(fold, name)]:
                    continue
                for ti, tau in enumerate(thresholds):
                    def mean_metric(metric):
                        return round(float(np.mean([grid[p]["grid"][metric][ti]
                                                    for p in other])), 12)
                    key = (mean_metric("patient_macro_f1"), mean_metric("patient_ez_f1"),
                           mean_metric("patient_balanced_accuracy"),
                           -round(abs(float(tau)-0.5), 6), -ordinal)
                    if choice is None or key > choice[0]:
                        choice = (key, stage, epoch, ti, snap[name], grid[name])
            if choice is None:
                raise RuntimeError("No eligible I1 candidate for target")
            _, stage, epoch, ti, score, selected_grid = choice
            tau = float(thresholds[ti])
            logit = np.asarray(score["logit_nez"], dtype=np.float64)
            margin = math.log(tau/(1-tau)) - logit
            if len(margin) != len(y) or not np.isfinite(margin).all():
                raise RuntimeError("Invalid I1 target score")
            for rep in range(20):
                _, query = afc.split_indices(len(y), 42, fold, name, rep)
                private_rows.append({"fold": fold, "sid": name, "rep": rep, "model": "I1_A1_TF",
                                     **afc.query_metrics(y[query], margin[query])})
            selections.append({"fold": fold, "target_private": name, "selected_stage": stage,
                               "selected_epoch": epoch, "selected_threshold": tau,
                               "stage2_eligible": gate_map[(fold, name)],
                               "target_labels_used_for_selection": False})
        print(f"I1 fixed query fold={fold} cells=13", flush=True)
    if len(private_rows) != 2600 or len({r["sid"] for r in private_rows}) != 47:
        raise RuntimeError("I1 fixed-query denominator differs from historical 65x20/47-ID")
    private_dir = a.runtime / "private"
    write_csv(private_dir / "ICTAL_SELECTION_PRIVATE.csv", selections)
    with (private_dir / "ICTAL_QUERY_ROWS_PRIVATE.pkl").open("wb") as handle:
        pickle.dump(private_rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
    exact = {(r["fold"], r["sid"], r["rep"], r["model"]): r for r in private_rows}
    ids = sorted({r["sid"] for r in private_rows})
    draws = np.random.default_rng(42).integers(0, len(ids), size=(10000, len(ids)))
    counts = np.zeros((10000, len(ids)), dtype=np.int16)
    np.add.at(counts, (np.arange(10000)[:, None], draws), 1)
    counts = counts.astype(np.float32)
    index = {sid: i for i, sid in enumerate(ids)}
    def statistic(metric):
        numerator = np.zeros(len(ids))
        denominator = np.zeros(len(ids))
        for r in private_rows:
            if r["model"] != "I1_A1_TF":
                continue
            base = exact[(r["fold"], r["sid"], r["rep"], "I0_A1")]
            difference = r[metric] - base[metric]
            if np.isfinite(difference):
                idx = index[r["sid"]]
                numerator[idx] += difference
                denominator[idx] += 1
        resamples = (counts @ numerator) / (counts @ denominator)
        return {"metric": metric, "delta": float(numerator.sum()/denominator.sum()),
                "ci_lower": float(np.quantile(resamples, .025)),
                "ci_upper": float(np.quantile(resamples, .975)),
                "patient_clusters": 47, "draws": 10000, "seed": 42}
    bootstrap = [statistic(metric) for metric in METRICS]
    summary = []
    for model in ("I0_A1", "I1_A1_TF"):
        part = [r for r in private_rows if r["model"] == model]
        row = {"model": model, "matched_cells": 65, "query_repetitions": 1300,
               "unique_patient_ids": 47}
        row.update({metric: float(np.nanmean([r[metric] for r in part])) for metric in METRICS})
        summary.append(row)
    if abs(summary[0]["ap"] - 0.5767434626151353) > 1e-9:
        raise RuntimeError("Historical A1 fixed-query AP replay failed")
    a.output.mkdir(parents=True, exist_ok=True)
    write_csv(a.output / "ICTAL_PRIMARY_METRICS.csv", summary)
    write_csv(a.output / "ICTAL_PATIENT_BOOTSTRAP.csv", bootstrap)
    public_selection = []
    for fold in range(1, 6):
        part = [r for r in selections if r["fold"] == fold]
        public_selection.append({"fold": fold, "selected_stage1": sum(r["selected_stage"] == 1 for r in part),
                                 "selected_stage2": sum(r["selected_stage"] == 2 for r in part),
                                 "mean_selected_epoch": float(np.mean([r["selected_epoch"] for r in part]))})
    write_csv(a.output / "ICTAL_CHECKPOINT_SELECTION.csv", public_selection)
    print(json.dumps({"I0_ap": summary[0]["ap"], "I1_ap": summary[1]["ap"],
                      "delta_ap_ci": next(r for r in bootstrap if r["metric"] == "ap")},
                     sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("freeze", "evaluate"), required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--gate-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    gate, rows = check(a)
    if a.phase == "freeze":
        freeze(a, gate)
    else:
        os.environ["SRGI_RUNTIME"] = str(a.history)
        evaluate(a, gate, rows)


if __name__ == "__main__":
    main()
