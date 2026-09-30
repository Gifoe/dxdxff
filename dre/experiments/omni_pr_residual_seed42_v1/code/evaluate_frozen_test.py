"""One frozen official TEST evaluation from cached CNN embeddings/logits."""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erf
from sklearn.metrics import average_precision_score, roc_auc_score

from common import (atomic_json, binary_metrics, center_name, mean_logit_score,
                    percentile_summary, ranking_metrics, relative_features,
                    score_from_segment_logits, sha256, table)


MODELS = ("FrozenCNN", "ABS-ONLY", "PR-CNN")


class NumpyResidualHead:
    """Inference-only export of the frozen PyTorch residual head."""
    def __init__(self, archive, name):
        key = lambda field: np.asarray(archive[f"{name}::{field}"], dtype=np.float32)
        self.ln_weight = key("network.0.weight")
        self.ln_bias = key("network.0.bias")
        self.fc1_weight = key("network.1.weight")
        self.fc1_bias = key("network.1.bias")
        self.fc2_weight = key("network.4.weight")
        self.fc2_bias = key("network.4.bias")

    def __call__(self, features):
        value = np.asarray(features, dtype=np.float32)
        mean = value.mean(axis=1, keepdims=True)
        variance = ((value - mean) ** 2).mean(axis=1, keepdims=True)
        value = (value - mean) / np.sqrt(variance + np.float32(1e-5))
        value = value * self.ln_weight + self.ln_bias
        value = value @ self.fc1_weight.T + self.fc1_bias
        value = np.float32(0.5) * value * (np.float32(1.0) + erf(value / np.float32(np.sqrt(2.0))))
        value = value @ self.fc2_weight.T + self.fc2_bias
        return (np.float32(0.5) * np.tanh(np.tanh(value))).reshape(-1)


def load_records(cache: Path):
    records = []
    paths = sorted(cache.glob("*.npz"))
    for index, path in enumerate(paths, start=1):
        marker = path.with_suffix(".json")
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8"))["output_sha256"] != sha256(path):
            raise RuntimeError("Private TEST embedding cache marker mismatch")
        with np.load(path, allow_pickle=False) as source:
            offsets = np.asarray(source["segment_offsets"], dtype=np.int64)
            flat = np.asarray(source["segment_logits"], dtype=np.float32)
            embedding = np.asarray(source["embeddings"], dtype=np.float32)
            difference, rank, fallback = relative_features(embedding)
            records.append({
                "patient": str(source["patient"]), "edf": str(source["edf"]),
                "channel": np.asarray(source["channel_names"]).astype(str),
                "label": np.asarray(source["pathological_labels"], dtype=np.int8),
                "embedding": embedding, "difference": difference, "rank": rank,
                "segment_logits": [flat[offsets[i]:offsets[i + 1]].copy()
                                   for i in range(len(offsets) - 1)],
                "fallback": fallback,
            })
        if index % 25 == 0 or index == len(paths):
            print(f"loaded_test_embeddings={index}/{len(paths)}", flush=True)
    return records


def input_features(record, variant, permutation=None):
    zeros = np.zeros_like(record["embedding"])
    if variant == "ABS-ONLY" or variant == "relative-zero":
        return np.concatenate([record["embedding"], zeros, zeros], axis=1)
    if variant == "PR-CNN":
        return np.concatenate([record["embedding"], record["difference"], record["rank"]], axis=1)
    if variant == "shuffle":
        return np.concatenate([record["embedding"], record["difference"][permutation],
                               record["rank"][permutation]], axis=1)
    raise ValueError(variant)


def predict(records, head=None, variant="FrozenCNN", rng=None):
    rows, deltas = [], []
    started = time.perf_counter()
    for record in records:
        if variant == "FrozenCNN":
            delta = np.zeros(len(record["channel"]), dtype=np.float32)
        else:
            permutation = rng.permutation(len(record["channel"])) if variant == "shuffle" else None
            delta = head(input_features(record, variant, permutation))
        normal = score_from_segment_logits(record["segment_logits"], delta)
        literal = mean_logit_score(record["segment_logits"], delta)
        deltas.extend(delta.tolist())
        for channel, label, score, diagnostic, change in zip(
                record["channel"], record["label"], normal, literal, delta):
            if label in (0, 1):
                rows.append({"patient": record["patient"], "edf": record["edf"],
                             "channel": channel, "y": int(label),
                             "score": float(1 - score),
                             "mean_logit_score": float(1 - diagnostic),
                             "delta": float(change)})
    return rows, np.asarray(deltas, dtype=np.float64), time.perf_counter() - started


def all_metrics(rows, threshold):
    frame = pd.DataFrame(rows)
    y = frame.y.to_numpy(dtype=np.int8)
    score = frame.score.to_numpy(dtype=np.float64)
    return {"edf_channel_units": len(frame), "patients": int(frame.patient.nunique()),
            "pathological_units": int(y.sum()), "threshold": float(threshold),
            **binary_metrics(y, score, threshold), **ranking_metrics(frame)}


def patient_metric_rows(predictions, thresholds):
    public = []
    for model, rows in predictions.items():
        frame = pd.DataFrame(rows)
        values = defaultdict(list)
        for _, group in frame.groupby("patient", sort=False):
            y, score = group.y.to_numpy(dtype=np.int8), group.score.to_numpy(dtype=float)
            metric = binary_metrics(y, score, thresholds[model])
            if metric["auroc"] is not None: values["auroc"].append(metric["auroc"])
            if np.any(y == 1): values["ap"].append(float(average_precision_score(y, score)))
            values["macro_f1"].append(metric["macro_f1"])
        for metric, current in values.items():
            summary = percentile_summary(np.asarray(current))
            public.append({"model": model, "metric": metric, "estimable_patients": len(current), **summary})
    return public


def bootstrap(predictions, thresholds, draws=10000):
    frames = {name: pd.DataFrame(rows) for name, rows in predictions.items()}
    patients = sorted(set(frames["FrozenCNN"].patient))
    if any(set(frame.patient) != set(patients) for frame in frames.values()):
        raise RuntimeError("Paired patient coverage differs")
    indices = {name: {patient: np.flatnonzero(frame.patient.to_numpy() == patient)
                      for patient in patients} for name, frame in frames.items()}
    rng = np.random.default_rng(42)
    comparisons = (("PR-CNN", "FrozenCNN"), ("PR-CNN", "ABS-ONLY"))
    values = {(left, right, metric): [] for left, right in comparisons
              for metric in ("auroc", "ap", "macro_f1", "pathological_f1", "sensitivity", "specificity")}
    for draw in range(draws):
        sampled = rng.integers(0, len(patients), size=len(patients))
        metric_by_model = {}
        for name, frame in frames.items():
            take = np.concatenate([indices[name][patients[index]] for index in sampled])
            y = frame.y.to_numpy(dtype=np.int8)[take]
            score = frame.score.to_numpy(dtype=float)[take]
            metric = binary_metrics(y, score, thresholds[name])
            metric_by_model[name] = {**metric, "ap": float(average_precision_score(y, score))}
        for left, right in comparisons:
            for metric in ("auroc", "ap", "macro_f1", "pathological_f1", "sensitivity", "specificity"):
                values[(left, right, metric)].append(metric_by_model[left][metric] - metric_by_model[right][metric])
        if (draw + 1) % 1000 == 0 or draw + 1 == draws:
            print(f"patient_bootstrap={draw + 1}/{draws}", flush=True)
    rows = []
    for (left, right, metric), current in values.items():
        current = np.asarray(current, dtype=float)
        observed_left = all_metrics(predictions[left], thresholds[left])
        observed_right = all_metrics(predictions[right], thresholds[right])
        key = "ap" if metric == "ap" else metric
        if metric == "ap":
            observed = float(average_precision_score(
                [row["y"] for row in predictions[left]], [row["score"] for row in predictions[left]])) - float(average_precision_score(
                [row["y"] for row in predictions[right]], [row["score"] for row in predictions[right]]))
        else:
            observed = observed_left[key] - observed_right[key]
        rows.append({"comparison": f"{left} - {right}", "metric": metric,
                     "observed_delta": observed, "ci_low": float(np.percentile(current, 2.5)),
                     "ci_high": float(np.percentile(current, 97.5)),
                     "pr_delta_gt_zero": float(np.mean(current > 0)),
                     "two_sided_p": float(min(1.0, 2 * min(np.mean(current <= 0), np.mean(current >= 0)))),
                     "draws": draws, "seed": 42, "cluster": "patient"})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--test-audit", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--official-split", type=Path, required=True)
    parser.add_argument("--numpy-heads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.protocol.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    audit = json.loads(args.test_audit.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_OFFICIAL_TEST" or not freeze.get("model_frozen_before_final_test") or \
            freeze.get("protocol_sha256") != sha256(args.protocol):
        raise RuntimeError("Missing valid pre-test freeze")
    if audit.get("status") != "COMPLETE" or not audit.get("baseline_replay_pass"):
        raise RuntimeError("BASELINE_REPLAY_FAILED")
    if sha256(args.official_split) != lock["official_split_sha256"]:
        raise RuntimeError("Official split changed")
    records = load_records(args.test_cache)
    if len(records) != 237:
        raise RuntimeError("Expected 237 frozen TEST EDFs")
    with np.load(args.numpy_heads, allow_pickle=False) as archive:
        if str(archive["freeze_sha256"]) != sha256(args.freeze):
            raise RuntimeError("NumPy head export is not bound to this freeze")
        heads = {name: NumpyResidualHead(archive, name) for name in ("ABS-ONLY", "PR-CNN")}
    thresholds = {name: float(freeze["thresholds"][name]["threshold"]) for name in MODELS}
    predictions, delta_by_model, seconds = {}, {}, {}
    predictions["FrozenCNN"], delta_by_model["FrozenCNN"], seconds["FrozenCNN"] = predict(records)
    for name in ("ABS-ONLY", "PR-CNN"):
        predictions[name], delta_by_model[name], seconds[name] = predict(records, heads[name], name)
    # The baseline hard gate is repeated on the live embedding cache.
    baseline_auc = all_metrics(predictions["FrozenCNN"], thresholds["FrozenCNN"])["auroc"]
    if abs(baseline_auc - lock["expected_baseline_auroc"]) >= lock["baseline_absolute_tolerance"]:
        raise RuntimeError("BASELINE_REPLAY_FAILED")
    args.output.mkdir(parents=True, exist_ok=True)
    test_rows = []
    metric_by_model = {}
    for name in MODELS:
        metric = all_metrics(predictions[name], thresholds[name])
        metric_by_model[name] = metric
        delta_summary = percentile_summary(delta_by_model[name])
        literal_auc = float(roc_auc_score([row["y"] for row in predictions[name]],
                                          [row["mean_logit_score"] for row in predictions[name]]))
        test_rows.append({"model": name, **metric,
                          "delta_auroc_vs_frozen_cnn": metric["auroc"] - metric_by_model["FrozenCNN"]["auroc"],
                          "mean_logit_aggregation_diagnostic_auroc": literal_auc,
                          "residual_mean": delta_summary["mean"], "residual_std": delta_summary["std"],
                          "residual_p5": delta_summary["p5"], "residual_p25": delta_summary["p25"],
                          "residual_p50": delta_summary["p50"], "residual_p75": delta_summary["p75"],
                          "residual_p95": delta_summary["p95"],
                          "residual_saturation_fraction": float(np.mean(np.abs(delta_by_model[name]) >= 0.49)),
                          "inference_seconds_from_frozen_cache": seconds[name]})
    # Recompute deltas now that the baseline row exists.
    for row in test_rows:
        row["delta_auroc_vs_frozen_cnn"] = row["auroc"] - test_rows[0]["auroc"]
    table(args.output / "TEST_METRICS.csv", test_rows)
    table(args.output / "PATIENT_METRICS.csv", patient_metric_rows(predictions, thresholds))
    table(args.output / "PAIRED_BOOTSTRAP.csv", bootstrap(predictions, thresholds))

    split = pd.read_csv(args.official_split)
    center_by_edf = {str(row.edf_name): center_name(row.dataset) for row in split.itertuples(index=False)
                     if str(row.split) == "test"}
    center_rows = []
    for center in ("HUP", "Open-iEEG", "SourceSink", "Zurich"):
        for name in MODELS:
            subset = [row for row in predictions[name] if center_by_edf.get(row["edf"]) == center]
            if not subset:
                raise RuntimeError(f"No test rows for center {center}")
            metric = all_metrics(subset, thresholds[name])
            if not any(row["y"] for row in subset):
                for key in ("auroc", "ap", "pathological_f1", "sensitivity", "patient_equal_ap", "mrr", "top1"):
                    metric[key] = "not_estimable"
            baseline_center = None
            if name != "FrozenCNN":
                frozen_subset = [row for row in predictions["FrozenCNN"] if center_by_edf.get(row["edf"]) == center]
                frozen_metric = all_metrics(frozen_subset, thresholds["FrozenCNN"])
                baseline_center = None if metric["auroc"] == "not_estimable" else metric["auroc"] - frozen_metric["auroc"]
            center_rows.append({"center": center, "model": name, **metric,
                                "delta_auroc_vs_frozen_cnn": baseline_center})
    table(args.output / "CENTER_METRICS.csv", center_rows)

    relative_zero, zero_delta, zero_seconds = predict(records, heads["PR-CNN"], "relative-zero")
    intervention = [
        {"intervention": "FrozenCNN", **metric_by_model["FrozenCNN"]},
        {"intervention": "ABS-ONLY retrained", **metric_by_model["ABS-ONLY"]},
        {"intervention": "PR-CNN full", **metric_by_model["PR-CNN"]},
        {"intervention": "PR-CNN relative inputs zeroed", **all_metrics(relative_zero, thresholds["PR-CNN"])},
    ]
    table(args.output / "ABLATION_SUMMARY.csv", intervention)

    shuffle_rows = []
    rng = np.random.default_rng(42)
    for repeat in range(100):
        shuffled, _, _ = predict(records, heads["PR-CNN"], "shuffle", rng)
        metric = all_metrics(shuffled, thresholds["PR-CNN"])
        shuffle_rows.append({"repeat": repeat, "seed": 42, **metric,
                             "delta_auroc_vs_full": metric["auroc"] - metric_by_model["PR-CNN"]["auroc"]})
        if (repeat + 1) % 10 == 0:
            print(f"patient_shuffle_control={repeat + 1}/100", flush=True)
    table(args.output / "PATIENT_SHUFFLE_CONTROL.csv", shuffle_rows)
    status = {
        "status": "ONE_FROZEN_OFFICIAL_TEST_PASS_COMPLETE",
        "model_frozen_before_final_test": True,
        "final_heldout_accessed": True,
        "final_test_used_for_tuning": False,
        "test_rows": len(predictions["FrozenCNN"]),
        "patients": len(set(row["patient"] for row in predictions["FrozenCNN"])),
        "baseline_auroc": baseline_auc,
        "fallback_edfs": sum(record["fallback"] for record in records),
        "historical_test_previously_viewed": True,
        "individual_predictions_kept_private": True,
    }
    atomic_json(args.output / "FINAL_TEST_STATUS.json", status)
    print(json.dumps(status, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
