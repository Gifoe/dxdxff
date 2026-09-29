"""10k paired 96-patient bootstrap against exact frozen Omni comparators.

Reads private per-(EDF,channel) predictions.  Publishes only aggregate CIs.
Baseline F1 at 0.5 is explicitly diagnostic; AUROC is threshold-free.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from crst_metrics import patient_metrics


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def normal_edf(value):
    return str(value).replace("\\", "/")


def packed(rows, kind):
    result = {}
    for row in rows:
        if kind == "CRST":
            key = (row["patient_private"], normal_edf(row["edf_private"]),
                   row["channel_private"])
            value = int(row["label"]), float(row["score"])
        elif kind == "RawCNN":
            key = (row["patient"], normal_edf(row["edf"]), row["channel"])
            value = int(row["y_true"]), float(row["pathological_score"])
        else:
            key = (row["patient"], normal_edf(row["edf"]), row["channel"])
            value = int(row["pathology"]), float(row["score_pathology"])
        if key in result:
            raise RuntimeError(f"Duplicate {kind} official EDF-channel key")
        result[key] = value
    return result


def weighted_pool_distributions(y, score, patient_idx, counts):
    """Exact weighted ROC/AP for each patient bootstrap draw, tied scores grouped."""
    order = np.argsort(-score, kind="stable")
    yy, pp, ii = y[order], score[order], patient_idx[order]
    first = np.r_[0, np.flatnonzero(np.diff(pp)) + 1]
    auroc = np.empty(len(counts), np.float64)
    ap = np.empty(len(counts), np.float64)
    for offset in range(0, len(counts), 128):
        block = counts[offset:offset+128]
        weight = block[:, ii]
        gp = np.add.reduceat(weight * (yy == 1), first, axis=1)
        gn = np.add.reduceat(weight * (yy == 0), first, axis=1)
        ptotal, ntotal = gp.sum(axis=1), gn.sum(axis=1)
        cum_p, cum_n = gp.cumsum(axis=1), gn.cumsum(axis=1)
        favorable_neg = ntotal[:, None] - cum_n + 0.5 * gn
        area = (gp * favorable_neg).sum(axis=1) / np.maximum(1, ptotal * ntotal)
        precision = cum_p / np.maximum(1, cum_p + cum_n)
        average = (precision * gp).sum(axis=1) / np.maximum(1, ptotal)
        invalid = (ptotal == 0) | (ntotal == 0)
        area[invalid] = np.nan
        average[ptotal == 0] = np.nan
        auroc[offset:offset+len(block)] = area
        ap[offset:offset+len(block)] = average
    return auroc, ap


def fixed_f1_distribution(y, score, patient_idx, counts, threshold):
    pred = score >= threshold
    n = counts.shape[1]
    tp = np.bincount(patient_idx, weights=(y == 1) & pred, minlength=n)
    fp = np.bincount(patient_idx, weights=(y == 0) & pred, minlength=n)
    tn = np.bincount(patient_idx, weights=(y == 0) & ~pred, minlength=n)
    fn = np.bincount(patient_idx, weights=(y == 1) & ~pred, minlength=n)
    tp, fp, tn, fn = (counts @ item for item in (tp, fp, tn, fn))
    f1p = 2*tp / np.maximum(1, 2*tp+fp+fn)
    f1n = 2*tn / np.maximum(1, 2*tn+fp+fn)
    return 0.5*(f1p+f1n)


def patient_equal_distributions(keys, packed_rows, ids, counts):
    by_patient_channel = defaultdict(lambda: defaultdict(list))
    for key in keys:
        patient, _, channel = key
        by_patient_channel[patient][channel].append(packed_rows[key])
    metrics = []
    for patient in ids:
        pairs = []
        for channel, observations in by_patient_channel[patient].items():
            labels = {label for label, _ in observations}
            if len(labels) != 1:
                continue
            pairs.append((observations[0][0], float(np.mean([p for _, p in observations]))))
        metrics.append(patient_metrics([a for a, _ in pairs], [b for _, b in pairs]))
    result = {}
    for metric in ("auroc", "ap", "mrr", "top1"):
        values = np.array([row[metric] for row in metrics], float)
        finite = np.isfinite(values)
        numer = counts @ np.where(finite, values, 0)
        denom = counts @ finite.astype(float)
        result["patient_equal_" + metric] = numer / np.maximum(1, denom)
        result["patient_equal_" + metric][denom == 0] = np.nan
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crst0", type=Path, required=True)
    p.add_argument("--full", type=Path, required=True)
    p.add_argument("--raw-cnn", type=Path, required=True)
    p.add_argument("--a1", type=Path, required=True)
    p.add_argument("--primary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    maps = {"CRST-0": packed(read(args.crst0), "CRST"),
            "CRST-FULL": packed(read(args.full), "CRST"),
            "RawCNN": packed(read(args.raw_cnn), "RawCNN"),
            "A1-v2": packed(read(args.a1), "A1")}
    keys = sorted(maps["CRST-FULL"])
    if len(keys) != 8104 or any(set(m) != set(keys) for m in maps.values()):
        raise RuntimeError("Frozen comparator EDF-channel key sets differ")
    y = np.array([maps["CRST-FULL"][key][0] for key in keys], int)
    for name, comparator in maps.items():
        if any(comparator[key][0] != yy for key, yy in zip(keys, y)):
            raise RuntimeError(f"Frozen {name} official labels differ")
    if int(y.sum()) != 807 or len(y) != 8104:
        raise RuntimeError("Official 807 pathological / 8104 labels changed")
    ids = sorted({key[0] for key in keys})
    if len(ids) != 96:
        raise RuntimeError("Frozen comparator patient count not 96")
    by_id = {name: i for i, name in enumerate(ids)}
    patient_idx = np.array([by_id[key[0]] for key in keys], int)
    scores = {name: np.array([m[key][1] for key in keys], float)
              for name, m in maps.items()}
    if abs(roc_auc_score(y, scores["RawCNN"]) - 0.7987673466324111) > 1e-8:
        raise RuntimeError("Frozen Raw CNN AUROC replay differs")
    if abs(roc_auc_score(y, scores["A1-v2"]) - 0.6886896025407396) > 1e-8:
        raise RuntimeError("Frozen A1-v2 AUROC replay differs")
    with args.primary.open(newline="", encoding="utf-8") as f:
        primary = {row["model"]: row for row in csv.DictReader(f)}
    thresholds = {"CRST-0": float(primary["CRST-0"]["threshold"]),
                  "CRST-FULL": float(primary["CRST-FULL"]["threshold"]),
                  "RawCNN": 0.5, "A1-v2": 0.515586256980896}
    rng = np.random.default_rng(42)
    draw_idx = rng.integers(0, len(ids), size=(10000, len(ids)))
    counts = np.zeros((10000, len(ids)), dtype=np.float32)
    np.add.at(counts, (np.arange(10000)[:, None], draw_idx), 1)
    distributions = {}
    point_values = {}
    for name, score in scores.items():
        auc, ap = weighted_pool_distributions(y, score, patient_idx, counts)
        result = {"official_pooled_auroc": auc, "official_pooled_ap": ap,
                  "macro_f1": fixed_f1_distribution(y, score, patient_idx,
                                                      counts, thresholds[name])}
        result.update(patient_equal_distributions(keys, maps[name], ids, counts))
        distributions[name] = result
        unit = np.ones((1, len(ids)), dtype=np.float32)
        unit_auc, unit_ap = weighted_pool_distributions(y, score, patient_idx, unit)
        point = {"official_pooled_auroc": float(unit_auc[0]),
                 "official_pooled_ap": float(unit_ap[0]),
                 "macro_f1": float(fixed_f1_distribution(y, score, patient_idx,
                                                          unit, thresholds[name])[0])}
        point.update({key: float(value[0]) for key, value in
                      patient_equal_distributions(keys, maps[name], ids, unit).items()})
        point_values[name] = point
    rows = []
    for candidate, reference in (("CRST-FULL", "CRST-0"),
                                 ("CRST-FULL", "RawCNN"),
                                 ("CRST-FULL", "A1-v2")):
        for metric in distributions[candidate]:
            delta = distributions[candidate][metric] - distributions[reference][metric]
            rows.append({"comparison": f"{candidate}-{reference}", "metric": metric,
                         "point_delta": point_values[candidate][metric] -
                         point_values[reference][metric],
                         "ci_lower": float(np.nanquantile(delta, 0.025)),
                         "ci_upper": float(np.nanquantile(delta, 0.975)),
                         "positive_draw_fraction": float(np.nanmean(delta > 0)),
                         "draws": 10000, "patient_clusters": 96, "seed": 42,
                         "reference_f1_threshold_origin":
                         "fixed_0.5_diagnostic" if reference == "RawCNN" else
                         "frozen_prior_threshold" if reference == "A1-v2" else
                         "frozen_inner_validation"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "OMNI_PAIRED_BOOTSTRAP_COMPLETE",
                      "comparisons": 3, "metrics": len(distributions["CRST-FULL"]),
                      "draws": 10000, "matched_edf_channels": len(keys),
                      "patient_clusters": 96}), flush=True)


if __name__ == "__main__":
    main()
