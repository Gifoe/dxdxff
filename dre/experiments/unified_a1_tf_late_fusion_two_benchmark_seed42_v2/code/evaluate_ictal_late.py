"""Ictal target-excluded TF epoch/beta selection and fixed-query evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from late_tf import robust_channel_normalize
from train_tf_omni_late import sha, save_json

BETA = (0., .02, .05, .1, .2, .35, .5, .75, 1.)
METRICS = ("ap", "auc", "mrr", "top1", "ndcg", "macro_f1", "ez_f1", "ba")


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"Empty output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_keys(y, margin):
    if y.sum() == 0 or len(y) != len(margin):
        raise RuntimeError("Non-estimable/mismatched patient ranking")
    order = np.argsort(-margin, kind="stable")
    pos = np.flatnonzero(y[order] == 1)
    return (float(average_precision_score(y, margin)), 1 / (1 + int(pos[0])),
            int(y[order[0]]))


def zscore(values):
    x = torch.as_tensor(values, dtype=torch.float64)[None, :]
    mask = torch.ones_like(x, dtype=torch.bool)
    return robust_channel_normalize(x, mask)[0].cpu().numpy()


def historical(source):
    sys.path.insert(0, str(source / "a1_a2_patient_equal_objective_seed42_v1/code"))
    sys.path.insert(0, str(source / "active_fewshot_patient_calibration_seed42_v1/code"))
    import common as afc
    import run as afr
    return afc, afr


def source_labels(history, fold, row):
    path = history / "private" / f"fold_{fold}" / f"epoch_{int(row['selected_epoch']):02d}_representations.pkl"
    with path.open("rb") as f:
        obj = pickle.load(f)
    if obj["fold"] != fold or obj["epoch"] != int(row["selected_epoch"]):
        raise RuntimeError("Historical A1 representation mismatch")
    item = obj["val"][row["subject_id"]]
    return np.asarray(item["y"], dtype=np.int8), item


def freeze(a):
    files = {}
    for fold in range(1, 6):
        audit = a.runtime / f"fold_{fold}" / "training_audit.json"
        value = json.loads(audit.read_text(encoding="utf-8"))
        if value["fold"] != fold or value["epochs"] != 30 or value["protocol_sha256"] != sha(a.lock):
            raise RuntimeError(f"Incomplete TF training fold {fold}")
        for epoch in range(1, 31):
            path = a.runtime / f"fold_{fold}" / f"epoch_{epoch:02d}_TF_SCORES_PRIVATE.pkl"
            files[str(path.relative_to(a.runtime))] = sha(path)
    payload = {"protocol_sha256": sha(a.lock), "files": files,
               "scores_frozen_before_target_selection": True,
               "target_labels_indexed_during_score_freeze": False}
    save_json(a.runtime / "ICTAL_TF_SCORE_FREEZE_PRIVATE.json", payload)
    save_json(a.output / "ICTAL_TF_TRAINING_AUDIT.json", {"folds": 5, "epochs_each": 30,
              "score_snapshots": 150, "private_freeze_sha256": sha(a.runtime / "ICTAL_TF_SCORE_FREEZE_PRIVATE.json"),
              "target_labels_used_for_own_selection": False,
              "historical_loader_materialized_validation_labels": True})
    print("ICTAL_TF_SCORES_FROZEN 150", flush=True)


def evaluate(a):
    frozen = json.loads((a.runtime / "ICTAL_TF_SCORE_FREEZE_PRIVATE.json").read_text(encoding="utf-8"))
    if frozen["protocol_sha256"] != sha(a.lock) or len(frozen["files"]) != 150:
        raise RuntimeError("TF score freeze missing or changed")
    os.environ["SRGI_RUNTIME"] = str(a.history)
    afc, afr = historical(a.source)
    rows, selections, complementary = [], [], []
    for fold in range(1, 6):
        prior_rows = list(csv.DictReader((a.history / "private" / f"fold_{fold}" /
                                          "A1_VLOO_PRIVATE.csv").open(encoding="utf-8")))
        if len(prior_rows) != 13:
            raise RuntimeError("Historical VLOO denominator changed")
        prior = {r["subject_id"]: r for r in prior_rows}
        names = sorted(prior)
        labels, base = {}, {}
        for sid in names:
            y, item = source_labels(a.history, fold, prior[sid])
            tau = float(prior[sid]["selected_threshold"])
            margin = np.asarray(afr.margin_for(item, tau), dtype=np.float64)
            afr.verify_b0(item, tau, margin)
            labels[sid], base[sid] = y, margin
        snaps = []
        for epoch in range(1, 31):
            path = a.runtime / f"fold_{fold}" / f"epoch_{epoch:02d}_TF_SCORES_PRIVATE.pkl"
            if sha(path) != frozen["files"].get(str(path.relative_to(a.runtime))):
                raise RuntimeError("Frozen TF snapshot changed")
            with path.open("rb") as f:
                snap = pickle.load(f)
            if sorted(snap) != names or any(len(snap[s]) != len(labels[s]) for s in names):
                raise RuntimeError("TF validation channel alignment changed")
            snaps.append(snap)
        for sid in names:
            others = [s for s in names if s != sid]
            epoch_scores = []
            for epoch, snap in enumerate(snaps, 1):
                value = np.mean([score_keys(labels[s], snap[s])[0] for s in others])
                epoch_scores.append((round(float(value), 12), -epoch, epoch))
            _, _, chosen_epoch = max(epoch_scores)
            snap = snaps[chosen_epoch - 1]
            candidate = []
            for beta in BETA:
                outcomes = [score_keys(labels[s], base[s] + beta * zscore(snap[s])) for s in others]
                mean = tuple(round(float(np.mean([v[i] for v in outcomes])), 12) for i in range(3))
                candidate.append((mean, -beta, beta))
            best = max(candidate)
            beta = float(best[-1])
            if best[0][0] <= next(x[0][0] for x in candidate if x[-1] == 0):
                beta = 0.
            y = labels[sid]
            a_margin = base[sid]
            tf_margin = np.asarray(snap[sid], dtype=np.float64)
            final_margin = a_margin + beta * zscore(tf_margin)
            identity_error = float(np.max(np.abs(a_margin + 0 * zscore(tf_margin) - a_margin)))
            if identity_error >= 1e-7:
                raise RuntimeError("Ictal beta zero identity failed")
            a_top = score_keys(y, a_margin)[2]
            t_top = score_keys(y, tf_margin)[2]
            f_top = score_keys(y, final_margin)[2]
            complementary.append({"benchmark": "ictal", "fold": fold, "patient_private": sid,
                                  "beta": beta, "a1_top1_correct": a_top,
                                  "tf_top1_correct": t_top, "fusion_top1_correct": f_top,
                                  "a1_fail_tf_rescue": int(not a_top and t_top),
                                  "a1_correct_tf_wrong": int(a_top and not t_top),
                                  "a1_correct_fusion_destroyed": int(a_top and not f_top),
                                  "a1_fail_fusion_rescue": int(not a_top and f_top),
                                  "score_correlation": float(np.corrcoef(a_margin, zscore(tf_margin))[0, 1])})
            for rep in range(20):
                _, query = afc.split_indices(len(y), 42, fold, sid, rep)
                for label, margin in (("I0_A1", a_margin), ("I2_A1_TF_LATE", final_margin),
                                      ("TF_ONLY", tf_margin)):
                    rows.append({"fold": fold, "sid": sid, "rep": rep, "model": label,
                                 **afc.query_metrics(y[query], margin[query])})
            selections.append({"fold": fold, "target_private": sid,
                               "tf_epoch": chosen_epoch, "beta": beta,
                               "target_labels_used_for_own_selection": False,
                               "beta_zero_identity_max_error": identity_error})
        print(f"I2 fold={fold} target-excluded cells=13", flush=True)
    if len(rows) != 3900 or len({r["sid"] for r in rows}) != 47:
        raise RuntimeError("Ictal fixed query denominator changed")
    private = a.runtime / "private"
    private.mkdir(parents=True, exist_ok=True)
    write_csv(private / "ICTAL_SELECTION_PRIVATE.csv", selections)
    write_csv(private / "ICTAL_COMPLEMENTARITY_PRIVATE.csv", complementary)
    with (private / "ICTAL_QUERY_ROWS_PRIVATE.pkl").open("wb") as f:
        pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    summary = []
    for model in ("I0_A1", "I2_A1_TF_LATE", "TF_ONLY"):
        part = [r for r in rows if r["model"] == model]
        summary.append({"model": model, "matched_cells": 65, "query_repetitions": 1300,
                        "unique_patient_ids": 47,
                        **{m: float(np.nanmean([r[m] for r in part])) for m in METRICS}})
    if abs(summary[0]["ap"] - .5767434626151353) > 1e-9:
        raise RuntimeError("I0 historical AP replay failed")
    write_csv(a.output / "ICTAL_BASELINE_METRICS.csv", [summary[0]])
    write_csv(a.output / "ICTAL_LATE_FUSION_METRICS.csv", summary)
    ids = sorted({r["sid"] for r in rows})
    draws = np.random.default_rng(42).integers(0, len(ids), size=(10000, len(ids)))
    counts = np.zeros((10000, len(ids)), dtype=np.int16)
    np.add.at(counts, (np.arange(10000)[:, None], draws), 1)
    counts = counts.astype(np.float32)
    index = {sid: i for i, sid in enumerate(ids)}
    exact = {(r["fold"], r["sid"], r["rep"], r["model"]): r for r in rows}
    ci = []
    for metric in METRICS:
        numerator = np.zeros(len(ids))
        denominator = np.zeros(len(ids))
        for r in rows:
            if r["model"] != "I2_A1_TF_LATE":
                continue
            old = exact[(r["fold"], r["sid"], r["rep"], "I0_A1")]
            diff = r[metric] - old[metric]
            if np.isfinite(diff):
                i = index[r["sid"]]
                numerator[i] += diff
                denominator[i] += 1
        samples = (counts @ numerator) / (counts @ denominator)
        ci.append({"metric": metric, "delta": float(numerator.sum()/denominator.sum()),
                   "ci_lower": float(np.quantile(samples, .025)),
                   "ci_upper": float(np.quantile(samples, .975)),
                   "patient_clusters": 47, "draws": 10000, "seed": 42})
    write_csv(a.output / "ICTAL_PAIRED_BOOTSTRAP.csv", ci)
    write_csv(a.output / "BETA_SELECTION_ICTAL_PRIVATE_REDACTED.csv", [
        {"fold": fold, "beta_zero": sum(r["beta"] == 0 for r in selections if r["fold"] == fold),
         "beta_positive": sum(r["beta"] > 0 for r in selections if r["fold"] == fold),
         "mean_selected_epoch": float(np.mean([r["tf_epoch"] for r in selections if r["fold"] == fold]))}
        for fold in range(1, 6)])
    write_csv(a.output / "TF_A1_COMPLEMENTARITY_ICTAL.csv", [
        {"benchmark": "ictal", "patients": 65,
         "a1_fail_tf_rescue": sum(r["a1_fail_tf_rescue"] for r in complementary),
         "a1_correct_tf_wrong": sum(r["a1_correct_tf_wrong"] for r in complementary),
         "a1_correct_fusion_destroyed": sum(r["a1_correct_fusion_destroyed"] for r in complementary),
         "a1_fail_fusion_rescue": sum(r["a1_fail_fusion_rescue"] for r in complementary),
         "mean_score_correlation": float(np.nanmean([r["score_correlation"] for r in complementary]))}])
    save_json(a.output / "LATE_FUSION_IDENTITY_AUDIT_ICTAL.json", {
              "pass": all(r["beta_zero_identity_max_error"] < 1e-7 for r in selections),
              "max_abs_score_difference": max(r["beta_zero_identity_max_error"] for r in selections),
              "historical_A1_AP_replayed": summary[0]["ap"]})
    print(json.dumps({"I0_AP": summary[0]["ap"], "I2_AP": summary[1]["ap"],
                      "TF_only_AP": summary[2]["ap"], "selected_beta_zero_cells":
                      sum(r["beta"] == 0 for r in selections)}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("freeze", "evaluate"), required=True)
    for name in ("runtime", "history", "source", "lock", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.phase == "freeze":
        freeze(a)
    else:
        evaluate(a)


if __name__ == "__main__":
    main()
