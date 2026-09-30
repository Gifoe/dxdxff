"""Aggregate an interrupted frozen Omni test from hash-verified private scores.

This program deliberately has no PyTorch, EDF, HDF5, or waveform imports.  It
only validates the existing post-freeze patient prediction cache and computes
the predeclared aggregate metrics.  It cannot run a model or alter a score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, roc_auc_score)


MODELS = ("RawCNN", "PC-CNN", "PC_no_physiology", "PC_no_context",
          "PC_all_disabled")
HISTORICAL_COHORT_SHA256 = "e10241ce0e823ced7dd262ed6eda4eeb0ffdbe52082aa4ec771590e253ffaf00"
CENTER_DISPLAY = {"hup": "HUP", "openieeg": "Open-iEEG",
                  "sourcesink": "SourceSink", "zurich": "Zurich"}


def digest(path: Path) -> str:
    hashed = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hashed.update(block)
    return hashed.hexdigest()


def prediction_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def table(path: Path, rows) -> None:
    names = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def flat(private, patients):
    y, score, group = [], [], []
    for index, patient in enumerate(patients):
        current = private[patient]
        y.extend(current["labels"])
        score.extend(current["scores"])
        group.extend([index] * len(current["labels"]))
    return np.asarray(y, dtype=np.int8), np.asarray(score, dtype=float), np.asarray(group, dtype=np.int16)


def patient_ranking(row):
    grouped, labels, conflict = defaultdict(list), {}, set()
    for channel, label, score in zip(row["channel"], row["labels"], row["scores"]):
        if channel in labels and labels[channel] != label:
            conflict.add(channel)
        labels[channel] = int(label)
        grouped[channel].append(float(score))
    names = sorted(set(grouped) - conflict)
    if not names:
        raise RuntimeError("No consistent patient-channel label")
    y = np.asarray([labels[name] for name in names], dtype=np.int8)
    score = np.asarray([np.mean(grouped[name]) for name in names], dtype=float)
    if len(np.unique(y)) < 2:
        return {"auroc": None, "ap": None, "mrr": None, "top1": None}
    order = np.argsort(-score, kind="stable")
    positions = np.flatnonzero(y[order] == 1)
    return {"auroc": float(roc_auc_score(y, score)),
            "ap": float(average_precision_score(y, score)),
            "mrr": float(1 / (1 + positions[0])), "top1": float(y[order[0]])}


def scored(private, patients, threshold):
    y, score, _ = flat(private, patients)
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    ranks = [patient_ranking(private[patient]) for patient in patients]
    values = {key: [row[key] for row in ranks if row[key] is not None]
              for key in ("auroc", "ap", "mrr", "top1")}
    return {"edf_channel_units": len(y), "patients": len(patients),
            "positive_units": int(y.sum()),
            "auroc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
            "pooled_ap": float(average_precision_score(y, score)) if y.sum() else None,
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "pathological_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)) if y.sum() else None,
            "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "accuracy": float((tp + tn) / len(y)), "tp": int(tp), "fp": int(fp),
            "tn": int(tn), "fn": int(fn),
            "patient_equal_auroc": float(np.mean(values["auroc"])) if values["auroc"] else None,
            "patient_ap": float(np.mean(values["ap"])) if values["ap"] else None,
            "mrr": float(np.mean(values["mrr"])) if values["mrr"] else None,
            "top1": float(np.mean(values["top1"])) if values["top1"] else None}


def bootstrap_paired(raw_private, pc_private, patients, raw_threshold, pc_threshold):
    ry, rs, group = flat(raw_private, patients)
    py, ps, other_group = flat(pc_private, patients)
    if not np.array_equal(ry, py) or not np.array_equal(group, other_group):
        raise RuntimeError("Frozen EDF-channel paired alignment differs")
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(patients), size=(10000, len(patients)))
    deltas = {key: np.empty(len(draws)) for key in
              ("auroc", "ap", "macro_f1", "pathological_f1", "sensitivity",
               "specificity", "mrr", "top1")}
    rankings = {"RawCNN": [patient_ranking(raw_private[p]) for p in patients],
                "PC-CNN": [patient_ranking(pc_private[p]) for p in patients]}
    rank_delta = {metric: np.asarray([
        left[metric] - right[metric] if left[metric] is not None and right[metric] is not None else np.nan
        for left, right in zip(rankings["PC-CNN"], rankings["RawCNN"])], dtype=float)
        for metric in ("mrr", "top1")}
    for index, sampled in enumerate(draws):
        counts = np.bincount(sampled, minlength=len(patients))
        weights, active = counts[group], counts[group] > 0
        if len(np.unique(ry[active])) < 2:
            deltas["auroc"][index] = deltas["ap"][index] = np.nan
        else:
            deltas["auroc"][index] = roc_auc_score(py, ps, sample_weight=weights) - roc_auc_score(ry, rs, sample_weight=weights)
            deltas["ap"][index] = average_precision_score(py, ps, sample_weight=weights) - average_precision_score(ry, rs, sample_weight=weights)
        confusion = {}
        for name, score, threshold in (("RawCNN", rs, raw_threshold), ("PC-CNN", ps, pc_threshold)):
            tn, fp, fn, tp = confusion_matrix(ry, score >= threshold, labels=[0, 1], sample_weight=weights).ravel()
            p_f1, n_f1 = 2 * tp / max(2 * tp + fp + fn, 1), 2 * tn / max(2 * tn + fp + fn, 1)
            confusion[name] = {"macro_f1": (p_f1 + n_f1) / 2, "pathological_f1": p_f1,
                               "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
                               "specificity": tn / (tn + fp) if tn + fp else np.nan}
        for metric in confusion["RawCNN"]:
            deltas[metric][index] = confusion["PC-CNN"][metric] - confusion["RawCNN"][metric]
        for metric in ("mrr", "top1"):
            deltas[metric][index] = np.nanmean(rank_delta[metric][sampled])
    return [{"candidate": "PC-CNN", "reference": "RawCNN", "metric": key,
             "delta_mean_bootstrap": float(np.nanmean(value)),
             "ci_low": float(np.nanquantile(value, 0.025)),
             "ci_high": float(np.nanquantile(value, 0.975)), "draws": 10000,
             "cluster_unit": "patient"} for key, value in deltas.items()]


def load_private(cache, binding, checkpoint_by_model):
    private, patients = {}, None
    for name in MODELS:
        rows = []
        for ordinal in range(96):
            path = cache / f"{name}_{ordinal:03d}.json"
            if not path.is_file():
                raise RuntimeError(f"Missing frozen patient prediction {path.name}")
            row = json.loads(path.read_text(encoding="utf-8"))
            expected = {**binding, "checkpoint_sha256": checkpoint_by_model[name],
                        "model": name, "ordinal": ordinal, "cohort_patients": 96}
            if any(row.get(key) != value for key, value in expected.items()) or \
                    row.get("prediction_sha256") != prediction_digest(row.get("prediction")):
                raise RuntimeError(f"Frozen prediction provenance mismatch {path.name}")
            rows.append((row["patient"], row["prediction"]))
        current_patients = [patient for patient, _ in rows]
        if len(set(current_patients)) != 96:
            raise RuntimeError(f"Duplicate frozen patients for {name}")
        if patients is None:
            patients = current_patients
        elif patients != current_patients:
            raise RuntimeError("Model patient-order mismatch in frozen predictions")
        private[name] = dict(rows)
    if len(list(cache.glob("*.json"))) != len(MODELS) * 96:
        raise RuntimeError("Unexpected frozen prediction cache file count")
    return private, patients


def patient_centers(official_split, cohort_audit, patients):
    if digest(cohort_audit) != HISTORICAL_COHORT_SHA256:
        raise RuntimeError("Historical supervised cohort audit changed")
    with cohort_audit.open(newline="", encoding="utf-8-sig") as stream:
        audited = {row["edf"]: row["patient"] for row in csv.DictReader(stream)
                   if row["official_split"] == "test" and float(row["official_labeled_channels"]) > 0}
    if len(audited) != 174 or len(set(audited.values())) != 96:
        raise RuntimeError("Historical supervised cohort membership changed")
    with official_split.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    centers = {}
    for row in rows:
        if row["edf_name"] in audited and row["patient_name"] == audited[row["edf_name"]]:
            name = CENTER_DISPLAY.get(row["dataset"].casefold())
            if name is None:
                raise RuntimeError("Unexpected Omni center")
            if row["patient_name"] in centers and centers[row["patient_name"]] != name:
                raise RuntimeError("Patient spans centers")
            centers[row["patient_name"]] = name
    if set(centers) != set(patients):
        raise RuntimeError("Frozen patient/cache cohort mismatch")
    return centers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--official-cnn", type=Path, required=True)
    parser.add_argument("--official-split", type=Path, required=True)
    parser.add_argument("--cohort-audit", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze, protocol = json.loads(args.freeze.read_text(encoding="utf-8")), json.loads(args.protocol.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_OFFICIAL_TEST" or not freeze.get("model_frozen_before_final_test") or freeze.get("protocol_sha256") != digest(args.protocol):
        raise RuntimeError("Aggregation requires the pre-test freeze")
    selected = freeze["benchmark_checkpoints"]["omni/fold1"]
    for name in ("RawCNN", "PC-CNN"):
        if digest(Path(selected[name]["private_path"])) != selected[name]["sha256"]:
            raise RuntimeError("Selected checkpoint changed after freeze")
    binding = {"freeze_sha256": digest(args.freeze), "protocol_sha256": digest(args.protocol),
               "official_cnn_sha256": digest(args.official_cnn)}
    cache = args.runtime / "omni" / "FROZEN_TEST_PATIENT_CACHE_PRIVATE"
    checkpoint_by_model = {"RawCNN": selected["RawCNN"]["sha256"],
                           **{name: selected["PC-CNN"]["sha256"] for name in MODELS[1:]}}
    private, patients = load_private(cache, binding, checkpoint_by_model)
    centers = patient_centers(args.official_split, args.cohort_audit, patients)
    args.output.mkdir(parents=True, exist_ok=True)
    thresholds = {name: selected[name]["validation_threshold"]["threshold"] for name in ("RawCNN", "PC-CNN")}
    existing_threshold = args.output / "OMNI_THRESHOLD_SELECTION.csv"
    threshold_rows = [{"model": name, **value["validation_threshold"]} for name, value in selected.items()]
    if existing_threshold.is_file() and existing_threshold.read_text(encoding="utf-8") != "":
        with existing_threshold.open(newline="", encoding="utf-8") as stream:
            if list(csv.DictReader(stream)) != [{key: str(value) for key, value in row.items()} for row in threshold_rows]:
                raise RuntimeError("Existing frozen threshold table differs")
    else:
        table(existing_threshold, threshold_rows)
    totals = []
    for name in MODELS:
        tau = thresholds["RawCNN"] if name == "RawCNN" else thresholds["PC-CNN"]
        totals.append({"benchmark": "Omni", "model": name, "threshold": tau, **scored(private[name], patients, tau)})
    table(args.output / "OMNI_RAWCNN_METRICS.csv", [totals[0]])
    table(args.output / "OMNI_PCCNN_METRICS.csv", [totals[1]])
    table(args.output / "OMNI_INTERVENTION.csv", totals)
    center_rows = []
    for center in ("HUP", "Open-iEEG", "SourceSink", "Zurich"):
        subset = [patient for patient in patients if centers[patient] == center]
        for name in ("RawCNN", "PC-CNN"):
            metrics = scored(private[name], subset, thresholds[name])
            if not any(sum(private[name][patient]["labels"]) for patient in subset):
                for key in ("auroc", "pooled_ap", "pathological_f1", "sensitivity"):
                    metrics[key] = "not_estimable"
            center_rows.append({"center": center, "model": name, "reason": "", **metrics})
    table(args.output / "OMNI_CENTER_METRICS.csv", center_rows)
    table(args.output / "OMNI_BOOTSTRAP.csv", bootstrap_paired(private["RawCNN"], private["PC-CNN"], patients, thresholds["RawCNN"], thresholds["PC-CNN"]))
    status = {"status": "ONE_FROZEN_OFFICIAL_TEST_PASS_COMPLETE", "patients": len(patients),
              "edf_channels": totals[0]["edf_channel_units"], "test_used_for_tuning": False,
              "historical_test_previously_viewed": True,
              "aggregation_source": "hash_verified_private_frozen_predictions"}
    temporary = args.output / "OMNI_OFFICIAL_TEST_STATUS.json.tmp"
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output / "OMNI_OFFICIAL_TEST_STATUS.json")
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
