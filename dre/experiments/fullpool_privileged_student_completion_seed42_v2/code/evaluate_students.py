"""Evaluate only after all FIT models and B=0 target scores are hash-frozen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np

from student_core import LOCK_SHA, VARIANTS, atomic_json, imports, sha

METRICS = ("ap", "auc", "mrr", "top1", "ndcg", "macro_f1", "ez_f1", "ba", "predicted_ez_fraction")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("Refusing empty aggregate CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def avg(values) -> float | None:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def bootstrap(values: np.ndarray, rows: list[dict], ids: list[str], counts: np.ndarray) -> dict:
    index = {sid: i for i, sid in enumerate(ids)}
    numerator = np.zeros(len(ids), dtype=np.float64)
    denominator = np.zeros(len(ids), dtype=np.float64)
    for value, row in zip(values, rows):
        if np.isfinite(value):
            at = index[row["sid"]]
            numerator[at] += value
            denominator[at] += 1
    if denominator.sum() == 0:
        raise RuntimeError("No estimable paired metric")
    draw_n = counts @ numerator
    draw_d = counts @ denominator
    draws = np.divide(draw_n, draw_d, out=np.full(len(draw_d), np.nan), where=draw_d > 0)
    good = draws[np.isfinite(draws)]
    if len(good) < .99 * len(draws):
        raise RuntimeError("Degenerate patient-cluster bootstrap")
    return {"mean": float(numerator.sum() / denominator.sum()),
            "ci_low": float(np.quantile(good, .025)),
            "ci_high": float(np.quantile(good, .975)),
            "n_estimable": int(denominator.sum())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if sha(a.output / "PROTOCOL_LOCK.json") != LOCK_SHA:
        raise RuntimeError("v2 protocol lock changed")
    freeze = json.loads((a.output / "TARGET_SCORE_FREEZE_AUDIT.json").read_text(encoding="utf-8"))
    private_path = a.runtime / "private" / "TARGET_SCORE_AND_CHECKPOINT_MANIFEST.json"
    if (not freeze["pass"] or freeze["lock_sha"] != LOCK_SHA or
            sha(private_path) != freeze["private_manifest_sha256"]):
        raise RuntimeError("Target scores/checkpoints were not frozen before label access")
    manifest = json.loads(private_path.read_text(encoding="utf-8"))["files"]
    variants = list(VARIANTS[:5]) + ([VARIANTS[5]] if freeze["n_variants"] == 6 else [])
    if freeze["n_variants"] != len(variants):
        raise RuntimeError("Model grid changed after score freeze")
    _, v1, _, _, _, _ = imports()
    all_models = ["D0_ORIGINAL_A1"] + variants
    rows = []
    for fold in range(1, 6):
        selected = v1.afc.read_csv(v1.afc.PRIOR_RUNTIME / "private" / f"fold_{fold}" /
                                   "A1_VLOO_PRIVATE.csv")
        if len(selected) != 13:
            raise RuntimeError("A1 VLOO target membership changed")
        source = {}
        for choice in selected:
            sid = choice["subject_id"]
            epoch, threshold = int(choice["selected_epoch"]), float(choice["selected_threshold"])
            item = v1.afc.load_representation(fold, epoch)["val"][sid]
            y = np.asarray(item["y"], dtype=np.int8)
            margin = np.asarray(v1.afr.margin_for(item, threshold), dtype=np.float64)
            v1.afr.verify_b0(item, threshold, margin)
            source[sid] = y
            for rep in range(20):
                _, query = v1.afc.split_indices(len(y), 42, fold, sid, rep)
                rows.append({"fold": fold, "sid": sid, "rep": rep, "model": "D0_ORIGINAL_A1",
                             **v1.afc.query_metrics(y[query], margin[query])})
        for variant in variants:
            path = a.runtime / "private" / "target_scores" / variant / f"fold_{fold}.pkl"
            key = str(path.relative_to(a.runtime)).replace("\\", "/")
            if sha(path) != manifest[key]:
                raise RuntimeError("Student score file changed after label unlock")
            with path.open("rb") as stream:
                scores = pickle.load(stream)
            if set(scores) != set(source):
                raise RuntimeError("Student target membership mismatch")
            for sid, item in scores.items():
                y = source[sid]
                margin = -np.asarray(item["logit_nez"], dtype=np.float64)
                if len(y) != item["n_channels"] or len(margin) != len(y) or not np.isfinite(margin).all():
                    raise RuntimeError("Student target channel count/score mismatch")
                for rep in range(20):
                    _, query = v1.afc.split_indices(len(y), 42, fold, sid, rep)
                    rows.append({"fold": fold, "sid": sid, "rep": rep, "model": variant,
                                 **v1.afc.query_metrics(y[query], margin[query])})
            print(f"[TARGET_EVAL] {variant} fold={fold} patients=13", flush=True)
    expected = 65 * 20 * len(all_models)
    if len(rows) != expected or len({r["sid"] for r in rows}) != 47:
        raise RuntimeError("Fixed-query target coverage mismatch")
    private_rows = a.runtime / "private" / "STUDENT_FIXED_QUERY_ROWS_PRIVATE.pkl"
    with private_rows.open("wb") as stream:
        pickle.dump(rows, stream, protocol=5)
    by_key = {(r["fold"], r["sid"], r["rep"], r["model"]): r for r in rows}
    if len(by_key) != len(rows):
        raise RuntimeError("Duplicate fixed-query record")
    ids = sorted({r["sid"] for r in rows})
    sample = np.random.default_rng(42).integers(0, len(ids), size=(10000, len(ids)))
    counts = np.zeros((10000, len(ids)), dtype=np.int16)
    np.add.at(counts, (np.arange(10000)[:, None], sample), 1)
    counts = counts.astype(np.float32)
    matrix, pairs, folds, headroom = [], [], [], []
    for model in all_models:
        part = [r for r in rows if r["model"] == model]
        if len(part) != 1300:
            raise RuntimeError("Model missing fixed-query repetitions")
        row = {"model": model, "n_cells": 65, "n_unique_patient_ids": 47,
               "n_repetitions": 1300, "n_estimable_ap": sum(np.isfinite(r["ap"]) for r in part)}
        row.update({metric: avg(r[metric] for r in part) for metric in METRICS})
        matrix.append(row)
        for fold in range(1, 6):
            sub = [r for r in part if r["fold"] == fold]
            folds.append({"model": model, "fold": fold, "ap": avg(r["ap"] for r in sub),
                          "mrr": avg(r["mrr"] for r in sub), "top1": avg(r["top1"] for r in sub),
                          "n_cells": 13})
    model_matrix = {r["model"]: r for r in matrix}
    if abs(model_matrix["D0_ORIGINAL_A1"]["ap"] - 0.5767434626151353) > 1e-9:
        raise RuntimeError("Exact A1 fixed-query AP replay failed")
    for model in variants:
        part = [r for r in rows if r["model"] == model]
        for reference in ("D0_CONTINUED_HARDLABEL", "D0_ORIGINAL_A1"):
            if model == reference:
                continue
            stats = {}
            for metric in ("ap", "mrr", "top1"):
                delta = np.asarray([r[metric] - by_key[(r["fold"], r["sid"], r["rep"], reference)][metric]
                                    for r in part], dtype=np.float64)
                stats[metric] = bootstrap(delta, part, ids, counts)
            fold_signs = [avg(r["ap"] - by_key[(fold, r["sid"], r["rep"], reference)]["ap"]
                              for r in part if r["fold"] == fold) for fold in range(1, 6)]
            pairs.append({"model": model, "reference": reference,
                          "delta_ap": stats["ap"]["mean"], "ap_ci_low": stats["ap"]["ci_low"],
                          "ap_ci_high": stats["ap"]["ci_high"],
                          "delta_mrr": stats["mrr"]["mean"], "mrr_ci_low": stats["mrr"]["ci_low"],
                          "mrr_ci_high": stats["mrr"]["ci_high"],
                          "delta_top1": stats["top1"]["mean"], "top1_ci_low": stats["top1"]["ci_low"],
                          "top1_ci_high": stats["top1"]["ci_high"],
                          "positive_folds": sum(value is not None and value > 0 for value in fold_signs),
                          "n_patient_clusters": 47, "bootstrap_resamples": 10000, "seed": 42})
        headroom.append({"model": model, "ap": model_matrix[model]["ap"],
                         "teacher_added_headroom_transfer_percent": 100 *
                         (model_matrix[model]["ap"] - 0.5767434626) / (0.6924644868 - 0.5767434626)})
    # Define target patient strata only from predetermined original A1 AP.
    baseline = [r for r in rows if r["model"] == "D0_ORIGINAL_A1"]
    sid_ap = {sid: avg(r["ap"] for r in baseline if r["sid"] == sid) for sid in ids}
    ordered = sorted(ids, key=lambda sid: (sid_ap[sid], sid))
    strata = {sid: ("poor" if i < 16 else "medium" if i < 32 else "strong")
              for i, sid in enumerate(ordered)}
    target_strata = []
    for model in variants:
        part = [r for r in rows if r["model"] == model]
        for name in ("poor", "medium", "strong"):
            sub = [r for r in part if strata[r["sid"]] == name]
            target_strata.append({"model": model, "original_A1_AP_stratum": name,
                                  "n_unique_patients": len({r["sid"] for r in sub}),
                                  "ap": avg(r["ap"] for r in sub),
                                  "delta_ap_vs_D0plus": avg(r["ap"] - by_key[(r["fold"], r["sid"], r["rep"], "D0_CONTINUED_HARDLABEL")]["ap"] for r in sub),
                                  "delta_ap_vs_A1": avg(r["ap"] - by_key[(r["fold"], r["sid"], r["rep"], "D0_ORIGINAL_A1")]["ap"] for r in sub)})
    write_csv(a.output / "STUDENT_VARIANT_MATRIX.csv", matrix)
    write_csv(a.output / "PATIENT_CLUSTER_BOOTSTRAP.csv", pairs)
    write_csv(a.output / "FOLD_CONSISTENCY.csv", folds)
    write_csv(a.output / "HEADROOM_TRANSFER.csv", headroom)
    write_csv(a.output / "TARGET_FAILURE_STRATIFIED_ANALYSIS.csv", target_strata)
    by_pair = {(r["model"], r["reference"]): r for r in pairs}
    best = max((model_matrix[model] for model in variants), key=lambda r: r["ap"])
    eligible = []
    for model in variants[1:]:
        p = by_pair[(model, "D0_CONTINUED_HARDLABEL")]
        mrr_ok = p["delta_mrr"] > -.005 or p["mrr_ci_high"] >= 0
        top1_ok = p["delta_top1"] > -.01 or p["top1_ci_high"] >= 0
        if p["delta_ap"] >= .010 and p["ap_ci_low"] > 0 and p["positive_folds"] >= 4 and mrr_ok and top1_ok:
            eligible.append(model)
    pbest = by_pair.get((best["model"], "D0_CONTINUED_HARDLABEL"))
    significant = pbest is not None and pbest["ap_ci_low"] > 0 and pbest["positive_folds"] >= 4
    if best["ap"] >= .6052912572:
        terminal = "PRIVILEGED_TEACHER_SUCCESSFULLY_DISTILLED_TO_B8_LEVEL_ZERO_SHOT"
    elif best["ap"] >= .5996322682:
        terminal = "PRIVILEGED_DISTILLATION_RECOVERS_CURRENT_B8"
    elif best["ap"] >= .590 and significant:
        terminal = "PRIVILEGED_PATIENT_GEOMETRY_PARTIALLY_TRANSFERS"
    elif best["ap"] > model_matrix["D0_ORIGINAL_A1"]["ap"] and not significant:
        terminal = "APPARENT_GAIN_EXPLAINED_BY_CONTINUED_TRAINING"
    else:
        terminal = "STRONG_PRIVILEGED_SIGNAL_NOT_ZEROSHOT_DISTILLABLE"
    gates = {"lock_sha": LOCK_SHA, "PRIVILEGED_DISTILLATION_SUPPORTED": bool(eligible),
             "supported_variants": eligible, "STUDENT_AP_059_REACHED": best["ap"] >= .590,
             "STUDENT_REACHES_CURRENT_B8": best["ap"] >= .5996322682,
             "STUDENT_REACHES_BEST_B8": best["ap"] >= .6052912572 and significant and
             pbest["delta_mrr"] > -.005 and pbest["delta_top1"] > -.01,
             "best_student": best["model"], "best_student_ap": best["ap"],
             "terminal": terminal, "outer_test_accessed": False,
             "student_target_used_for_training_or_selection": False}
    atomic_json(a.output / "STUDENT_GATES.json", gates)
    print(f"STUDENT_EVALUATION_COMPLETE best={best['model']} ap={best['ap']:.6f} terminal={terminal}", flush=True)


if __name__ == "__main__":
    main()
