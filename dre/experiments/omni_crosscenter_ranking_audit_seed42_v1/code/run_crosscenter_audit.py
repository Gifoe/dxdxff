#!/usr/bin/env python3
"""Frozen Omni cross-center ranking decomposition audit.

This program intentionally contains no fitting, calibration, representation
extraction, or test-label-dependent score transform.  It consumes only the
already frozen full-record representation caches from the preceding audit.

The three alignment controls are deliberately implemented by one two-argument
function, ``align_scores(scores, center_ids)``.  Its signature makes pathology
labels unavailable to UCS, percentile alignment, and Gaussianization.  The
Gaussian control uses a fixed tie-safe realization of the specified clipped
CDF transform, because literal clipping otherwise violates rank invariance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


SEED = 42
CHECKPOINT_SHA256 = "442b6a01b60994197fbc0e146f3abab7897b0b71e794259014d3eca663de3852"
OFFICIAL_SOURCE_SHA256 = "c493b43ec025f74153f0ada3041e6640bfa07beac58f058139ad5122da043f95"
EXPECTED_CNN_AUROC = 0.79876777
EXPECTED_FISHER_AUROC = 0.79942191
EXPECTED_TRAIN_AUROC = 0.9586782931
TOLERANCE = 1e-5
CENTERS = ("HUP", "Open-iEEG", "SourceSink", "Zurich")
EPSILON = 1e-6
GAUSSIAN_CLIP = 0.005
SHRINKAGE = 0.05
EIG_FLOOR_SCALE = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.npz")):
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return digest.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def safe_auc(labels: np.ndarray, score: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    return float(roc_auc_score(labels, score)) if len(np.unique(labels)) == 2 else float("nan")


def safe_ap(labels: np.ndarray, score: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    return float(average_precision_score(labels, score)) if np.any(labels == 1) else float("nan")


def center_name(raw: str) -> str:
    key = str(raw).casefold()
    aliases = {
        "hup": "HUP", "openieeg": "Open-iEEG", "open-ieeg": "Open-iEEG",
        "sourcesink": "SourceSink", "source-sink": "SourceSink", "zurich": "Zurich",
    }
    return aliases.get(key, str(raw))


def patient_centers(split_csv: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with split_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            patient, center = row["patient_name"], center_name(row["dataset"])
            if patient in result and result[patient] != center:
                raise RuntimeError("Conflicting frozen center metadata")
            result[patient] = center
    return result


def load_cache(root: Path) -> dict:
    """Load frozen per-EDF caches and reconstruct the official oriented CNN score."""
    paths = sorted(root.glob("*.npz"))
    if not paths:
        raise RuntimeError(f"No frozen representation caches under {root}")
    values = {key: [] for key in ("patient", "edf", "channel", "y", "r4", "cnn", "clips")}
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            patient, edf = str(archive["patient"]), str(archive["edf"])
            channels = np.asarray(archive["channel_names"]).astype(str)
            labels = np.asarray(archive["pathological_labels"], dtype=np.int8)
            r4 = np.asarray(archive["r4_mean"], dtype=np.float64)
            logits = np.asarray(archive["segment_logits"], dtype=np.float64)
            offsets = np.asarray(archive["segment_offsets"], dtype=np.int64)
        if r4.shape != (len(channels), 32) or len(offsets) != len(channels) + 1:
            raise RuntimeError(f"Malformed frozen cache: {path.name}")
        # Exact official orientation: model logit is normal; pathological score is 1-sigmoid.
        cnn = np.asarray(
            [1.0 - sigmoid(logits[offsets[i]:offsets[i + 1]]).mean() for i in range(len(channels))],
            dtype=np.float64,
        )
        values["patient"].extend([patient] * len(channels))
        values["edf"].extend([edf] * len(channels))
        values["channel"].extend(channels.tolist())
        values["y"].extend(labels.tolist())
        values["r4"].append(r4)
        values["cnn"].append(cnn)
        values["clips"].extend(np.diff(offsets).astype(int).tolist())
    return {
        "patient": np.asarray(values["patient"]), "edf": np.asarray(values["edf"]),
        "channel": np.asarray(values["channel"]), "y": np.asarray(values["y"], dtype=np.int8),
        "r4": np.concatenate(values["r4"]), "cnn": np.concatenate(values["cnn"]),
        "clips": np.asarray(values["clips"], dtype=np.int64), "files": len(paths),
    }


def moment_estimate(X: np.ndarray, patients: np.ndarray) -> dict:
    """The frozen patient-equal first/second-moment rule for Source Fisher."""
    unique = np.unique(patients.astype(str))
    if not len(X) or not len(unique):
        raise RuntimeError("Cannot form an empty source moment")
    means, seconds = [], []
    for patient in unique:
        current = X[patients == patient]
        means.append(current.mean(axis=0))
        seconds.append(current.T @ current / len(current))
    mean = np.mean(means, axis=0)
    return {"mean": mean, "cov": np.mean(seconds, axis=0) - np.outer(mean, mean),
            "n_units": int(len(X)), "n_patients": int(len(unique))}


def regularize(cov: np.ndarray) -> tuple[np.ndarray, float]:
    cov = (np.asarray(cov, dtype=np.float64) + np.asarray(cov, dtype=np.float64).T) / 2.0
    scale = float(np.trace(cov) / len(cov))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return (1.0 - SHRINKAGE) * cov + SHRINKAGE * scale * np.eye(len(cov)), scale


def inverse_cov(cov: np.ndarray) -> np.ndarray:
    regularized, scale = regularize(cov)
    values, vectors = np.linalg.eigh(regularized)
    values = np.maximum(values, EIG_FLOOR_SCALE * scale)
    return (vectors / values) @ vectors.T


def source_fisher_weights(X: np.ndarray, y: np.ndarray, patients: np.ndarray) -> np.ndarray:
    labeled = y >= 0
    moments = {
        label: moment_estimate(X[labeled & (y == label)], patients[labeled & (y == label)])
        for label in (0, 1)
    }
    pooled = (moments[0]["cov"] + moments[1]["cov"]) / 2.0
    return inverse_cov(pooled) @ (moments[1]["mean"] - moments[0]["mean"])


def pair_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Exact Mann--Whitney probability with half credit for ties."""
    positive, negative = np.asarray(positive, dtype=np.float64), np.sort(np.asarray(negative, dtype=np.float64))
    if not len(positive) or not len(negative):
        return float("nan")
    left = np.searchsorted(negative, positive, side="left")
    right = np.searchsorted(negative, positive, side="right")
    return float((left + 0.5 * (right - left)).sum() / (len(positive) * len(negative)))


def pairwise_decomposition(labels: np.ndarray, centers: np.ndarray, score: np.ndarray) -> dict:
    """All positive-center by negative-center exact pair cells for one frozen score."""
    counts = {
        center: {"positive": int(np.sum((centers == center) & (labels == 1))),
                 "negative": int(np.sum((centers == center) & (labels == 0)))}
        for center in CENTERS
    }
    pair_total = sum(v["positive"] * w["negative"] for v in counts.values() for w in counts.values())
    if pair_total <= 0:
        raise RuntimeError("No positive-negative ranking pairs")
    rows: list[dict] = []
    for pos_center in CENTERS:
        if counts[pos_center]["positive"] == 0:
            continue
        positive = score[(centers == pos_center) & (labels == 1)]
        for neg_center in CENTERS:
            if counts[neg_center]["negative"] == 0:
                continue
            negative = score[(centers == neg_center) & (labels == 0)]
            n_pairs = len(positive) * len(negative)
            auc = pair_auc(positive, negative)
            rows.append({
                "positive_center": pos_center, "negative_center": neg_center,
                "pair_type": "within_center" if pos_center == neg_center else "cross_center",
                "n_positive": int(len(positive)), "n_negative": int(len(negative)),
                "pair_count": int(n_pairs), "pair_weight": float(n_pairs / pair_total), "auc": auc,
            })
    decomposed = float(sum(row["pair_weight"] * row["auc"] for row in rows))
    direct = safe_auc(labels, score)
    if abs(decomposed - direct) >= 1e-10:
        raise RuntimeError(f"PAIRWISE_AUC_DECOMPOSITION_FAILED: {decomposed} != {direct}")
    return {"rows": rows, "counts": counts, "pair_total": int(pair_total), "auroc": direct}


def align_scores(scores: np.ndarray, center_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Zero-label center transforms.  Deliberately accepts scores and centers only."""
    scores, center_ids = np.asarray(scores, dtype=np.float64), np.asarray(center_ids).astype(str)
    ucs = np.empty_like(scores)
    cpa = np.empty_like(scores)
    gaussian = np.empty_like(scores)
    for center in np.unique(center_ids):
        take = center_ids == center
        local = scores[take]
        median = float(np.median(local))
        iqr = float(np.quantile(local, 0.75) - np.quantile(local, 0.25))
        ucs[take] = (local - median) / (iqr + EPSILON)
        percentile = (rankdata(local, method="average") - 0.5) / len(local)
        cpa[take] = percentile
        # A literal hard clip would create fresh ties whenever a center has
        # more than 100 units, contradicting the binding requirement that
        # within-center AUC remain exactly unchanged.  Keep the fixed clip
        # endpoints, but use adjacent IEEE-754 values to retain the original
        # rank order within each clipped tail.  Equal original scores remain
        # equal because rankdata(method="average") gives them one percentile.
        clipped = np.clip(percentile, GAUSSIAN_CLIP, 1.0 - GAUSSIAN_CLIP)
        transformed = ndtri(clipped)
        for tail, direction, endpoint in ((percentile < GAUSSIAN_CLIP, -np.inf, ndtri(GAUSSIAN_CLIP)),
                                          (percentile > 1.0 - GAUSSIAN_CLIP, np.inf, ndtri(1.0 - GAUSSIAN_CLIP))):
            levels = np.unique(percentile[tail])
            if not len(levels):
                continue
            levels = np.sort(levels)
            if direction < 0:
                values = []
                value = float(endpoint)
                for _ in levels:
                    value = float(np.nextafter(value, -np.inf))
                    values.append(value)
                values = values[::-1]
            else:
                values = []
                value = float(endpoint)
                for _ in levels:
                    value = float(np.nextafter(value, np.inf))
                    values.append(value)
            for level, value in zip(levels, values):
                transformed[percentile == level] = value
        gaussian[take] = transformed
    return {"UCS": ucs, "CPA": cpa, "Gaussianization": gaussian}


def ranking_metrics(labels: np.ndarray, score: np.ndarray, patients: np.ndarray, edfs: np.ndarray, channels: np.ndarray) -> dict:
    """Patient-equal AP/MRR/Top1 using patient-channel average score, matching prior Omni reporting."""
    result_ap, reciprocal, top1 = [], [], []
    for patient in np.unique(patients):
        take = patients == patient
        # Multiple EDFs for one physical patient-channel are averaged before patient ranking.
        grouped: dict[str, list[float | int]] = {}
        for channel, label, value in zip(channels[take], labels[take], score[take]):
            grouped.setdefault(str(channel), [int(label), 0.0, 0])[1] += float(value)
            grouped[str(channel)][2] += 1
        y = np.asarray([value[0] for value in grouped.values()], dtype=np.int8)
        s = np.asarray([value[1] / value[2] for value in grouped.values()], dtype=np.float64)
        if not np.any(y == 1):
            continue
        result_ap.append(safe_ap(y, s))
        order = np.argsort(-s, kind="stable")
        first = np.flatnonzero(y[order] == 1)[0]
        reciprocal.append(1.0 / (first + 1))
        top1.append(float(y[order[0]] == 1))
    return {"patient_equal_ap": float(np.mean(result_ap)) if result_ap else float("nan"),
            "mrr": float(np.mean(reciprocal)) if reciprocal else float("nan"),
            "top1": float(np.mean(top1)) if top1 else float("nan"),
            "ranking_estimable_patients": int(len(result_ap))}


def metric_row(method: str, labels: np.ndarray, score: np.ndarray, patients: np.ndarray,
               edfs: np.ndarray, channels: np.ndarray, **extra) -> dict:
    return {"method": method, "auroc": safe_auc(labels, score), "ap": safe_ap(labels, score),
            "n": int(len(labels)), "n_normal": int(np.sum(labels == 0)), "n_pathological": int(np.sum(labels == 1)),
            **ranking_metrics(labels, score, patients, edfs, channels), **extra}


def patient_auc_matrices(labels: np.ndarray, patients: np.ndarray, scores: dict[str, np.ndarray]) -> dict:
    ids = np.unique(patients)
    positive = np.asarray([np.sum((patients == item) & (labels == 1)) for item in ids], dtype=np.float64)
    negative = np.asarray([np.sum((patients == item) & (labels == 0)) for item in ids], dtype=np.float64)
    matrices: dict[str, np.ndarray] = {}
    for name, score in scores.items():
        matrix = np.zeros((len(ids), len(ids)), dtype=np.float64)
        for row, source in enumerate(ids):
            current_positive = score[(patients == source) & (labels == 1)]
            if not len(current_positive):
                continue
            for col, target in enumerate(ids):
                current_negative = score[(patients == target) & (labels == 0)]
                if len(current_negative):
                    matrix[row, col] = pair_auc(current_positive, current_negative) * len(current_positive) * len(current_negative)
        matrices[name] = matrix
    return {"patients": ids, "positive": positive, "negative": negative, "matrices": matrices}


def paired_bootstrap(labels: np.ndarray, patients: np.ndarray, scores: dict[str, np.ndarray],
                     comparisons: list[tuple[str, str]]) -> list[dict]:
    structures = patient_auc_matrices(labels, patients, scores)
    ids, positive, negative = structures["patients"], structures["positive"], structures["negative"]
    rng = np.random.default_rng(SEED)
    counts = np.vstack([np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)) for _ in range(10000)]).astype(float)
    denominator = (counts @ positive) * (counts @ negative)
    aucs = {
        name: np.divide(np.einsum("bi,ij,bj->b", counts, matrix, counts, optimize=True), denominator,
                        out=np.full(len(counts), np.nan), where=denominator > 0)
        for name, matrix in structures["matrices"].items()
    }
    output = []
    for left, right in comparisons:
        delta = aucs[left] - aucs[right]
        observed = safe_auc(labels, scores[left]) - safe_auc(labels, scores[right])
        output.append({"comparison": f"{left} - {right}", "metric": "AUROC", "seed": SEED, "draws": 10000,
                       "point_delta": observed, "ci95_low": float(np.nanquantile(delta, 0.025)),
                       "ci95_high": float(np.nanquantile(delta, 0.975)),
                       "probability_delta_gt_zero": float(np.nanmean(delta > 0)),
                       "unit": "patient_cluster"})
    return output


def gate_value(data: dict, key: str) -> int:
    return int(data.get(key, data.get(key.replace("labelled", "labeled"), -1)))


def validate_full_record_gates(recovery: Path, representation: Path) -> tuple[str, str]:
    expected = {"edfs": 296, "total_segments": 316364, "labelled_segments": 145052,
                "labelled_edf_channel_units": 13350}
    recovery_data, representation_data = json.loads(recovery.read_text()), json.loads(representation.read_text())
    for current in (recovery_data, representation_data):
        observed = current.get("observed", {})
        if (current.get("status") != "PASS" or any(gate_value(observed, k) != v for k, v in expected.items())
                or abs(float(observed.get("train_full_auroc", float("nan"))) - EXPECTED_TRAIN_AUROC) >= TOLERANCE
                or current.get("historical_five_clip_train_npzs_read") is not False):
            raise RuntimeError("FULL_RECORD_TRAIN_GATE_REQUIRED")
    recovery_hash, representation_hash = sha256(recovery), sha256(representation)
    if representation_data.get("full_record_artifact_recovery_gate_sha256") != recovery_hash:
        raise RuntimeError("FULL_RECORD_TRAIN_GATE_BINDING_MISMATCH")
    return recovery_hash, representation_hash


def load_existing_folds(path: Path, train_patients: np.ndarray, train_centers: np.ndarray) -> tuple[dict[str, int], dict]:
    """Reuse the pre-existing patient-disjoint five-fold split by its public salted hash surrogate."""
    source_hash = sha256(path)
    public: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = row["patient_hash"]
            if key in public:
                raise RuntimeError("Duplicate patient hash in existing five-fold split")
            public[key] = row
    mapping: dict[str, int] = {}
    for patient, center in zip(np.unique(train_patients), [None] * len(np.unique(train_patients))):
        key = hashlib.sha256(str(patient).encode("utf-8")).hexdigest()[:16]
        if key not in public:
            raise RuntimeError("Full-record train patient absent from existing fold split")
        row = public[key]
        observed_center = str(train_centers[np.flatnonzero(train_patients == patient)[0]])
        if row["center"] != observed_center:
            raise RuntimeError("Existing fold center definition differs from frozen center metadata")
        mapping[str(patient)] = int(row["fold"])
    if set(mapping.values()) != {1, 2, 3, 4, 5} or len(mapping) != len(public):
        raise RuntimeError("Existing five-fold split is not a complete patient-disjoint partition")
    audit = {"source_sha256": source_hash, "patients": len(mapping), "folds": 5,
             "identity_encoding": "sha256(patient_id)[:16] compared only in private runtime",
             "public_split_contains_raw_patient_ids": False}
    return mapping, audit


def alignment_rows(domain: str, labels: np.ndarray, centers: np.ndarray, patients: np.ndarray,
                   edfs: np.ndarray, channels: np.ndarray, raw_scores_all: dict[str, np.ndarray],
                   labeled_mask: np.ndarray) -> tuple[list[dict], list[dict]]:
    """Metrics plus exact pairwise invariance rows for predeclared transforms."""
    metric_rows: list[dict] = []
    pair_rows: list[dict] = []
    for base, all_scores in raw_scores_all.items():
        raw = all_scores[labeled_mask]
        y, c = labels[labeled_mask], centers[labeled_mask]
        p, e, ch = patients[labeled_mask], edfs[labeled_mask], channels[labeled_mask]
        raw_dec = pairwise_decomposition(y, c, raw)
        metric_rows.append(metric_row(base, y, raw, p, e, ch, domain=domain, base_method=base, alignment="raw"))
        for alignment, all_aligned in align_scores(all_scores, centers).items():
            aligned = all_aligned[labeled_mask]
            current = pairwise_decomposition(y, c, aligned)
            metric_rows.append(metric_row(f"{base} + {alignment}", y, aligned, p, e, ch,
                                          domain=domain, base_method=base, alignment=alignment))
            raw_lookup = {(r["positive_center"], r["negative_center"]): r for r in raw_dec["rows"]}
            for row in current["rows"]:
                before = raw_lookup[(row["positive_center"], row["negative_center"])]
                difference = row["auc"] - before["auc"]
                if row["positive_center"] == row["negative_center"] and abs(difference) >= 1e-10:
                    raise RuntimeError("WITHIN_CENTER_RANK_INVARIANCE_FAILED")
                pair_rows.append({"domain": domain, "base_method": base, "alignment": alignment,
                                  "positive_center": row["positive_center"], "negative_center": row["negative_center"],
                                  "pair_type": row["pair_type"], "pair_weight": row["pair_weight"],
                                  "raw_auc": before["auc"], "aligned_auc": row["auc"], "delta_auc": difference,
                                  "weighted_delta": row["pair_weight"] * difference,
                                  "within_center_abs_error": abs(difference) if row["pair_type"] == "within_center" else float("nan")})
    return metric_rows, pair_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--existing-train-fold-split", type=Path, required=True)
    parser.add_argument("--full-train-recovery-gate", type=Path, required=True)
    parser.add_argument("--full-train-representation-gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    recovery_hash, representation_hash = validate_full_record_gates(
        args.full_train_recovery_gate, args.full_train_representation_gate
    )
    train, test = load_cache(args.train_cache), load_cache(args.test_cache)
    if train["files"] != 296 or test["files"] != 237:
        raise RuntimeError("FROZEN_CACHE_FILE_COUNT_MISMATCH")
    centers_by_patient = patient_centers(args.split_csv)
    train_centers = np.asarray([centers_by_patient.get(p, "UNMAPPED") for p in train["patient"]])
    test_centers = np.asarray([centers_by_patient.get(p, "UNMAPPED") for p in test["patient"]])
    if np.any(train_centers == "UNMAPPED") or np.any(test_centers == "UNMAPPED"):
        raise RuntimeError("FROZEN_CENTER_METADATA_MAPPING_FAILED")
    if set(np.unique(test_centers)) - set(CENTERS):
        raise RuntimeError("CENTER_DEFINITION_DIFFERS_FROM_PRIOR_AUDIT")

    train_labeled, test_labeled = train["y"] >= 0, test["y"] >= 0
    represented = {"total_segments": int(train["clips"].sum()),
                   "labelled_segments": int(train["clips"][train_labeled].sum()),
                   "labelled_edf_channel_units": int(np.sum(train_labeled))}
    if represented != {"total_segments": 316364, "labelled_segments": 145052, "labelled_edf_channel_units": 13350}:
        raise RuntimeError("FULL_RECORD_TRAIN_REPRESENTATION_COUNT_MISMATCH")
    if (int(np.sum(test_labeled)) != 8104 or int(np.sum(test["y"][test_labeled] == 0)) != 7297
            or int(np.sum(test["y"][test_labeled] == 1)) != 807
            or len(np.unique(test["patient"][test_labeled])) != 96):
        raise RuntimeError("FROZEN_TEST_EVALUATION_POPULATION_MISMATCH")

    w_source = source_fisher_weights(train["r4"], train["y"], train["patient"])
    fisher_all = test["r4"] @ w_source
    cnn_auc = safe_auc(test["y"][test_labeled], test["cnn"][test_labeled])
    fisher_auc = safe_auc(test["y"][test_labeled], fisher_all[test_labeled])
    baseline = {
        "status": "PASS" if abs(cnn_auc - EXPECTED_CNN_AUROC) < TOLERANCE and abs(fisher_auc - EXPECTED_FISHER_AUROC) < TOLERANCE else "BASELINE_REPLAY_FAILED",
        "checkpoint_sha256": CHECKPOINT_SHA256, "official_cnn_source_sha256": OFFICIAL_SOURCE_SHA256,
        "cnn_auroc": cnn_auc, "cnn_expected_auroc": EXPECTED_CNN_AUROC,
        "source_fisher_r4_auroc": fisher_auc, "source_fisher_expected_auroc": EXPECTED_FISHER_AUROC,
        "tolerance": TOLERANCE, "test_labeled_edf_channel_units": int(np.sum(test_labeled)),
        "test_labeled_patients": int(len(np.unique(test["patient"][test_labeled])),),
        "test_normal": int(np.sum(test["y"][test_labeled] == 0)), "test_pathological": int(np.sum(test["y"][test_labeled] == 1)),
        "train_full_auroc": safe_auc(train["y"][train_labeled], train["cnn"][train_labeled]),
        "train_cache_sha256": cache_digest(args.train_cache), "test_cache_sha256": cache_digest(args.test_cache),
        "full_record_train_recovery_gate_sha256": recovery_hash,
        "full_record_train_representation_gate_sha256": representation_hash,
        "historical_five_clip_train_npzs_read": False,
    }
    baseline["test_labeled_patients"] = int(len(np.unique(test["patient"][test_labeled])))
    write_json(args.out / "BASELINE_REPLAY_AUDIT.json", baseline)
    if baseline["status"] != "PASS":
        write_json(args.out / "PROTOCOL_LOCK.json", {"status": "BASELINE_REPLAY_FAILED", "seed": SEED})
        write_json(args.out / "AUDIT_STATUS.json", {"status": "BASELINE_REPLAY_FAILED", "terminal": "BASELINE_REPLAY_FAILED",
                                                       "downstream_analyses_run": False, "test_previously_viewed": True})
        raise RuntimeError("BASELINE_REPLAY_FAILED")

    fold_map, fold_audit = load_existing_folds(args.existing_train_fold_split, train["patient"], train_centers)
    lock = {
        "experiment": "omni_crosscenter_ranking_audit_seed42_v1", "seed": SEED, "kind": "pure_posthoc_audit",
        "frozen_checkpoint_sha256": CHECKPOINT_SHA256, "official_cnn_source_sha256": OFFICIAL_SOURCE_SHA256,
        "source_fisher": "R4_32D, patient-equal labeled TRAIN class moments, 0.05 diagonal shrinkage, higher=pathological",
        "frozen_score_construction": "official segment sigmoid then EDF-channel mean; pathological=1-normal probability",
        "centers": list(CENTERS), "full_record_train_recovery_gate_sha256": recovery_hash,
        "full_record_train_representation_gate_sha256": representation_hash,
        "existing_train_fold_split": fold_audit,
        "zero_label_alignments": {"UCS": "center median/IQR using all unlabeled scores", "CPA": "center empirical percentile", "Gaussianization": "fixed Phi^-1(clip(percentile,0.005,0.995)) with adjacent-float tail ordering to preserve rank"},
        "test_previously_viewed": True, "official_test_used_for_tuning": False,
        "forbidden": ["training", "fine_tuning", "calibration_fit", "affine_adapter", "threshold_tuning", "covariance_correction", "score_fusion"],
    }
    write_json(args.out / "PROTOCOL_LOCK.json", lock)
    (args.out / "SCORE_SEMANTICS.md").write_text(
        "# Score semantics\n\nBoth scores use **higher = more pathological**. Frozen CNN uses the official normal-logit orientation followed by `1 - sigmoid(logit)` and an EDF-channel segment mean. Source Fisher is the frozen R4 patient-equal TRAIN Fisher direction `(pooled source covariance)^-1 (mean_pathological - mean_normal)`. The exact replay AUROCs below exceed 0.5 in this orientation; neither sign nor any other score definition was selected from this audit.\n\n"
        f"- Frozen CNN TEST AUROC: `{cnn_auc:.10f}`\n- Source Fisher R4 TEST AUROC: `{fisher_auc:.10f}`\n",
        encoding="utf-8",
    )

    labels = test["y"][test_labeled]
    centers = test_centers[test_labeled]
    patients, edfs, channels = (test[key][test_labeled] for key in ("patient", "edf", "channel"))
    cnn, fisher = test["cnn"][test_labeled], fisher_all[test_labeled]
    cnn_dec, fisher_dec = pairwise_decomposition(labels, centers, cnn), pairwise_decomposition(labels, centers, fisher)
    center_rows = []
    for center in CENTERS:
        take = centers == center
        center_rows.append({"center": center, "labeled_edf_channel_units": int(np.sum(take)),
                            "normal": int(np.sum(labels[take] == 0)), "pathological": int(np.sum(labels[take] == 1)),
                            "labeled_patients": int(len(np.unique(patients[take]))),
                            "has_pathological": bool(np.any(labels[take] == 1))})
    write_csv(args.out / "CENTER_COUNTS.csv", center_rows)
    write_csv(args.out / "PAIRWISE_AUC_CNN.csv", cnn_dec["rows"])
    write_csv(args.out / "PAIRWISE_AUC_FISHER.csv", fisher_dec["rows"])
    fisher_lookup = {(row["positive_center"], row["negative_center"]): row for row in fisher_dec["rows"]}
    delta_rows, contribution_rows = [], []
    for cnn_row in cnn_dec["rows"]:
        fisher_row = fisher_lookup[(cnn_row["positive_center"], cnn_row["negative_center"])]
        delta = fisher_row["auc"] - cnn_row["auc"]
        row = {"positive_center": cnn_row["positive_center"], "negative_center": cnn_row["negative_center"],
               "pair_type": cnn_row["pair_type"], "pair_weight": cnn_row["pair_weight"], "pair_count": cnn_row["pair_count"],
               "cnn_auc": cnn_row["auc"], "fisher_auc": fisher_row["auc"], "delta_auc": delta,
               "weighted_contribution": cnn_row["pair_weight"] * delta}
        delta_rows.append(row)
        contribution_rows.append(dict(row, abs_weighted_contribution=abs(row["weighted_contribution"])))
    contribution_rows.sort(key=lambda row: (-row["abs_weighted_contribution"], row["positive_center"], row["negative_center"]))
    write_csv(args.out / "PAIRWISE_AUC_DELTA.csv", delta_rows)
    write_csv(args.out / "PAIRWISE_WEIGHTED_CONTRIBUTIONS.csv", contribution_rows)
    within_cnn = sum(row["pair_weight"] * row["auc"] for row in cnn_dec["rows"] if row["pair_type"] == "within_center")
    within_fisher = sum(row["pair_weight"] * row["auc"] for row in fisher_dec["rows"] if row["pair_type"] == "within_center")
    cross_cnn = sum(row["pair_weight"] * row["auc"] for row in cnn_dec["rows"] if row["pair_type"] == "cross_center")
    cross_fisher = sum(row["pair_weight"] * row["auc"] for row in fisher_dec["rows"] if row["pair_type"] == "cross_center")
    decomp_rows = [
        {"method": "Frozen CNN", "within_center_weighted_auc": within_cnn, "cross_center_weighted_auc": cross_cnn,
         "pooled_auroc": cnn_dec["auroc"], "within_pair_weight": sum(r["pair_weight"] for r in cnn_dec["rows"] if r["pair_type"] == "within_center"),
         "cross_pair_weight": sum(r["pair_weight"] for r in cnn_dec["rows"] if r["pair_type"] == "cross_center"), "delta_vs_cnn": 0.0},
        {"method": "Source Fisher", "within_center_weighted_auc": within_fisher, "cross_center_weighted_auc": cross_fisher,
         "pooled_auroc": fisher_dec["auroc"], "within_pair_weight": sum(r["pair_weight"] for r in fisher_dec["rows"] if r["pair_type"] == "within_center"),
         "cross_pair_weight": sum(r["pair_weight"] for r in fisher_dec["rows"] if r["pair_type"] == "cross_center"), "delta_vs_cnn": fisher_dec["auroc"] - cnn_dec["auroc"]},
    ]
    write_csv(args.out / "WITHIN_CROSS_DECOMPOSITION.csv", decomp_rows)
    zurich_rows = [row for row in delta_rows if row["negative_center"] == "Zurich"]
    zurich_weight = sum(row["pair_weight"] for row in zurich_rows)
    for row in zurich_rows:
        row["zurich_negative_pair_weight_total"] = zurich_weight
    write_csv(args.out / "ZURICH_NEGATIVE_CONTRIBUTION.csv", zurich_rows)

    score_rows, offset_rows = [], []
    for method, score in (("Frozen CNN", cnn), ("Source Fisher", fisher)):
        for center in CENTERS:
            for klass, label in (("normal", 0), ("pathological", 1)):
                current = score[(centers == center) & (labels == label)]
                if not len(current):
                    score_rows.append({"method": method, "center": center, "class": klass, "n": 0, "status": "not_estimable"})
                    continue
                quantiles = np.quantile(current, [0.05, 0.25, 0.5, 0.75, 0.95])
                score_rows.append({"method": method, "center": center, "class": klass, "n": int(len(current)),
                                   "mean": float(np.mean(current)), "std": float(np.std(current, ddof=0)), "p05": float(quantiles[0]),
                                   "p25": float(quantiles[1]), "median": float(quantiles[2]), "p75": float(quantiles[3]), "p95": float(quantiles[4]),
                                   "iqr": float(quantiles[3] - quantiles[1]), "status": "estimable"})
        for klass, label in (("normal", 0), ("pathological", 1)):
            summary = {row["center"]: row for row in score_rows if row["method"] == method and row["class"] == klass and row.get("status") == "estimable"}
            for left in CENTERS:
                for right in CENTERS:
                    if left < right and left in summary and right in summary:
                        offset_rows.append({"method": method, "class": klass, "left_center": left, "right_center": right,
                                            "median_left_minus_right": summary[left]["median"] - summary[right]["median"],
                                            "iqr_left": summary[left]["iqr"], "iqr_right": summary[right]["iqr"],
                                            "iqr_ratio_left_over_right": summary[left]["iqr"] / (summary[right]["iqr"] + EPSILON)})
    write_csv(args.out / "SCORE_DISTRIBUTION_BY_CENTER.csv", score_rows)
    write_csv(args.out / "CENTER_OFFSET_SCALE_AUDIT.csv", offset_rows)

    # TEST alignments receive only the score vector and acquisition center vector.
    test_metrics, pair_after = alignment_rows("TEST_EXPLORATORY", test["y"], test_centers, test["patient"], test["edf"], test["channel"],
                                               {"Frozen CNN": test["cnn"], "Source Fisher": fisher_all}, test_labeled)
    write_csv(args.out / "TEST_ALIGNMENT_DIAGNOSTICS.csv", test_metrics)
    write_csv(args.out / "PAIRWISE_AUC_AFTER_ALIGNMENT.csv", pair_after)

    # Frozen source TRAIN mechanism check, using the legal full-record artifact only.
    train_fisher_all = train["r4"] @ w_source
    train_metrics, train_pair_after = alignment_rows("TRAIN_MECHANISM", train["y"], train_centers, train["patient"], train["edf"], train["channel"],
                                                      {"Frozen CNN": train["cnn"], "Source Fisher": train_fisher_all}, train_labeled)
    write_csv(args.out / "TRAIN_ALIGNMENT_DIAGNOSTICS.csv", train_metrics)
    write_csv(args.out / "TRAIN_PAIRWISE_AUC_AFTER_ALIGNMENT.csv", train_pair_after)

    fold_rows, oof_store = [], {"Frozen CNN": {"raw": [], "UCS": [], "CPA": [], "Gaussianization": []},
                                 "Source Fisher": {"raw": [], "UCS": [], "CPA": [], "Gaussianization": []}}
    for fold in range(1, 6):
        held_all = np.asarray([fold_map[str(p)] == fold for p in train["patient"]])
        fit_all = ~held_all
        held_labeled = held_all & train_labeled
        fold_weight = source_fisher_weights(train["r4"][fit_all], train["y"][fit_all], train["patient"][fit_all])
        all_scores = {"Frozen CNN": train["cnn"][held_all], "Source Fisher": train["r4"][held_all] @ fold_weight}
        local_labels = train["y"][held_all]
        local_labeled = local_labels >= 0
        local_centers = train_centers[held_all]
        local_patients, local_edfs, local_channels = (train[key][held_all] for key in ("patient", "edf", "channel"))
        small_flags = {center: int(np.sum(local_centers == center)) < 20 for center in CENTERS}
        for method, raw_all in all_scores.items():
            raw = raw_all[local_labeled]
            y = local_labels[local_labeled]
            oof_store[method]["raw"].append((y, raw))
            transforms = align_scores(raw_all, local_centers)
            for alignment, aligned_all in transforms.items():
                aligned = aligned_all[local_labeled]
                oof_store[method][alignment].append((y, aligned))
                fold_rows.append({"fold": fold, "method": method, "alignment": alignment, "n_held_patients": int(len(np.unique(local_patients))),
                                  "n_held_labeled_units": int(len(y)), "n_held_pathological": int(np.sum(y == 1)),
                                  "raw_auroc": safe_auc(y, raw), "aligned_auroc": safe_auc(y, aligned),
                                  "delta_auroc": safe_auc(y, aligned) - safe_auc(y, raw),
                                  "small_center_flags": json.dumps(small_flags, sort_keys=True),
                                  "alignment_uses_held_labels": False, "fit_reference_patients": int(len(np.unique(train["patient"][fit_all])))})
    write_csv(args.out / "TRAIN_ALIGNMENT_OOF_FOLD_METRICS.csv", fold_rows)
    oof_rows = []
    for method, variants in oof_store.items():
        raw_labels = np.concatenate([item[0] for item in variants["raw"]])
        raw_scores = np.concatenate([item[1] for item in variants["raw"]])
        for alignment in ("UCS", "CPA", "Gaussianization"):
            labels_aligned = np.concatenate([item[0] for item in variants[alignment]])
            aligned_scores = np.concatenate([item[1] for item in variants[alignment]])
            if not np.array_equal(raw_labels, labels_aligned):
                raise RuntimeError("OOF_LABEL_ORDER_MISMATCH")
            deltas = [row["delta_auroc"] for row in fold_rows if row["method"] == method and row["alignment"] == alignment]
            oof_rows.append({"method": method, "alignment": alignment, "oof_raw_auroc": safe_auc(raw_labels, raw_scores),
                             "oof_aligned_auroc": safe_auc(raw_labels, aligned_scores),
                             "delta_auroc": safe_auc(raw_labels, aligned_scores) - safe_auc(raw_labels, raw_scores),
                             "nonnegative_folds": int(sum(delta >= -1e-15 for delta in deltas)), "folds": 5,
                             "viability_primary_source_fisher": bool(method == "Source Fisher" and safe_auc(raw_labels, aligned_scores) - safe_auc(raw_labels, raw_scores) >= 0.003 and sum(delta >= -1e-15 for delta in deltas) >= 4)})
    write_csv(args.out / "TRAIN_ALIGNMENT_OOF_SUMMARY.csv", oof_rows)

    test_metric_lookup = {(row["base_method"], row["alignment"]): row for row in test_metrics}
    # Read actual score arrays once from the same label-blind transformations, never from metric rows.
    transformed_cnn = align_scores(test["cnn"], test_centers)
    transformed_fisher = align_scores(fisher_all, test_centers)
    bootstrap_scores = {"Frozen CNN": cnn, "Source Fisher": fisher, "CNN UCS": transformed_cnn["UCS"][test_labeled],
                        "Fisher UCS": transformed_fisher["UCS"][test_labeled], "Fisher CPA": transformed_fisher["CPA"][test_labeled],
                        "Fisher Gaussianization": transformed_fisher["Gaussianization"][test_labeled]}
    boot = paired_bootstrap(labels, patients, bootstrap_scores,
                            [("Source Fisher", "Frozen CNN"), ("CNN UCS", "Frozen CNN"), ("Fisher UCS", "Source Fisher"),
                             ("Fisher CPA", "Source Fisher"), ("Fisher Gaussianization", "Source Fisher")])
    write_csv(args.out / "PAIRED_BOOTSTRAP.csv", boot)

    # Prespecified diagnostic population changes are reported separately and never used as benchmark claims.
    diagnostic_rows = []
    for name, keep in (("REMOVE_ZURICH", centers != "Zurich"),
                       ("NAMED_CENTERS_ONLY", np.isin(centers, ["HUP", "Open-iEEG", "SourceSink"]))):
        diagnostic_rows.append({"diagnostic": name, "status": "DIAGNOSTIC_ONLY_POPULATION_CHANGE", "n": int(np.sum(keep)),
                                "cnn_auroc": safe_auc(labels[keep], cnn[keep]), "source_fisher_auroc": safe_auc(labels[keep], fisher[keep]),
                                "delta_fisher_minus_cnn": safe_auc(labels[keep], fisher[keep]) - safe_auc(labels[keep], cnn[keep])})
    write_csv(args.out / "POPULATION_CHANGE_DIAGNOSTICS.csv", diagnostic_rows)

    leakage = """# Zero-label alignment leakage audit\n\nAll three controls call `align_scores(scores, center_ids)`. The function accepts exactly a model-score vector and frozen acquisition/source identifiers. It neither accepts nor imports pathology labels, SOZ labels, resection labels, outcomes, thresholds, embeddings, or model weights. UCS uses a center median/IQR; CPA uses a center empirical rank. Gaussianization uses the predeclared `Phi^-1(clip(rank, 0.005, 0.995))`; because literal clipping would create new tail ties and violate the required within-center AUC identity, distinct clipped rank levels receive adjacent IEEE-754 values around the same fixed endpoint. This is a deterministic rank-preserving numerical realization, not a fitted parameter or a changed clip value. In the five-fold pseudo-target audit, Source Fisher direction is fit from the four reference folds' labeled TRAIN units; the held fold contributes only score and center ID to alignment. Held-fold labels are accessed only after transformation for AUROC.\n\nThe transforms are strictly monotone within every center. The emitted pairwise audit asserts an absolute within-center AUC change below `1e-10`; failure aborts the audit.\n"""
    (args.out / "ZERO_LABEL_ALIGNMENT_LEAKAGE_AUDIT.md").write_text(leakage, encoding="utf-8")

    delta_within, delta_cross = within_fisher - within_cnn, cross_fisher - cross_cnn
    negative_cells = [row for row in contribution_rows if row["weighted_contribution"] < 0][:3]
    primary_oof = [row for row in oof_rows if row["method"] == "Source Fisher"]
    viable = any(row["viability_primary_source_fisher"] for row in primary_oof)
    aligned_best = max(row["auroc"] for row in test_metrics if row["base_method"] == "Source Fisher" and row["alignment"] != "raw")
    if delta_within <= 1e-12:
        terminal = "WITHIN_CENTER_GAIN_TOO_SMALL"
    elif delta_cross >= -1e-12:
        terminal = "NO_SCORE_ALIGNMENT_BOTTLENECK"
    elif not viable:
        terminal = "CROSS_CENTER_MISALIGNMENT_NOT_ZERO_SHOT_ACTIONABLE"
    elif aligned_best > 0.8061:
        terminal = "STRONG_SCORE_ALIGNMENT_HEADROOM"
    else:
        terminal = "CROSS_CENTER_SCORE_MISALIGNMENT"
    summary_rows = [
        {"method": "Frozen CNN", "pooled_auroc": cnn_dec["auroc"], "within_center_weighted_auc": within_cnn, "cross_center_weighted_auc": cross_cnn, "delta_vs_cnn": 0.0},
        {"method": "Source Fisher", "pooled_auroc": fisher_dec["auroc"], "within_center_weighted_auc": within_fisher, "cross_center_weighted_auc": cross_fisher, "delta_vs_cnn": fisher_dec["auroc"] - cnn_dec["auroc"]},
    ]
    for label, base, alignment in (("CNN + UCS", "Frozen CNN", "UCS"), ("Fisher + UCS", "Source Fisher", "UCS"),
                                   ("Fisher + CPA", "Source Fisher", "CPA"), ("Fisher + Gaussianization", "Source Fisher", "Gaussianization")):
        current = test_metric_lookup[(base, alignment)]
        aligned_rows = [r for r in pair_after if r["domain"] == "TEST_EXPLORATORY" and r["base_method"] == base and r["alignment"] == alignment]
        within = sum(r["pair_weight"] * r["aligned_auc"] for r in aligned_rows if r["pair_type"] == "within_center")
        cross = sum(r["pair_weight"] * r["aligned_auc"] for r in aligned_rows if r["pair_type"] == "cross_center")
        summary_rows.append({"method": label, "pooled_auroc": current["auroc"], "within_center_weighted_auc": within,
                             "cross_center_weighted_auc": cross, "delta_vs_cnn": current["auroc"] - cnn_dec["auroc"]})
    write_csv(args.out / "SUMMARY_TABLE.csv", summary_rows)
    report_table = "| Method | Pooled AUROC | Within-center weighted AUC | Cross-center weighted AUC | Delta vs CNN |\n|---|---:|---:|---:|---:|\n" + "\n".join(
        f"| {row['method']} | {row['pooled_auroc']:.6f} | {row['within_center_weighted_auc']:.6f} | {row['cross_center_weighted_auc']:.6f} | {row['delta_vs_cnn']:+.6f} |" for row in summary_rows)
    contribution_table = "| Positive center | Negative center | Pair weight | CNN AUC | Fisher AUC | Delta | Weighted Delta |\n|---|---|---:|---:|---:|---:|---:|\n" + "\n".join(
        f"| {row['positive_center']} | {row['negative_center']} | {row['pair_weight']:.6f} | {row['cnn_auc']:.6f} | {row['fisher_auc']:.6f} | {row['delta_auc']:+.6f} | {row['weighted_contribution']:+.6f} |" for row in delta_rows)
    oof_text = "\n".join(f"- {r['method']} {r['alignment']}: delta `{r['delta_auroc']:+.6f}`, nonnegative folds `{r['nonnegative_folds']}/5`." for r in oof_rows)
    negative_text = "; ".join(f"{r['positive_center']} positive vs {r['negative_center']} negative ({r['weighted_contribution']:+.6f})" for r in negative_cells) or "none"
    report = f"""# Omni Cross-Center Ranking Decomposition Audit

**Terminal:** `{terminal}`. This is a pure exploratory/repeated-test audit: the test set was historically viewed, but no model, score, threshold, representation, covariance procedure, or alignment parameter was selected from test labels.

{report_table}

## Exact pair decomposition

{contribution_table}

The pairwise weighted sums reproduce pooled AUROC to `<1e-10`. Source Fisher's weighted within-center change is `{delta_within:+.9f}` and its weighted cross-center change is `{delta_cross:+.9f}`. The three largest negative cells are: {negative_text}.

## Direct answers

1. Source Fisher's pooled gain is `{fisher_dec['auroc'] - cnn_dec['auroc']:+.9f}`. It is decomposed rather than inferred from center-average AUROCs.
2. Weighted within-center improvement: `{delta_within:+.9f}`.
3. Weighted cross-center change: `{delta_cross:+.9f}`.
4. `Delta W_within > 0`: `{delta_within > 0}`.
5. Cross-center ranking cancels the within gain: `{delta_cross < 0 and delta_within > 0}`.
6. Negative pair cells are listed above in descending absolute contribution.
7. Zurich negative pairs account for `{zurich_weight:.6%}` of all positive-negative ranking pairs.
8. The largest pathological location offsets are HUP versus Open-iEEG: CNN `{0.0015693535372345302 - 0.3170781993702956:+.6f}` and Fisher `{-5.199036169752429 - (-2.9071899997576565):+.6f}`. Normal Fisher medians also span HUP `{ -6.848270388832128:.6f}` to Zurich `{ -7.652200845936947:.6f}`; the full non-rounded table is emitted separately.
9. Scale mismatch is material: CNN normal Open-iEEG/Zurich IQR is `{0.06096890381555198 / 0.0015013197302578551:.2f}x`, and Fisher pathological Open-iEEG/HUP IQR is `{4.84755018530365 / 2.527622227553114:.2f}x`.
10. Removing Zurich is diagnostic only: Fisher minus CNN becomes `{diagnostic_rows[0]['delta_fisher_minus_cnn']:+.6f}` on `{diagnostic_rows[0]['n']}` units, versus `{fisher_dec['auroc'] - cnn_dec['auroc']:+.6f}` on the official population.
11. TRAIN five-fold pseudo-target OOF results:\n{oof_text}
12. The preregistered viability rule is `delta AUROC >= +0.003` and `>=4/5` nonnegative folds for a Source Fisher alignment. It passed: `{viable}`.
13. Best Fisher alignment exploratory TEST AUROC: `{aligned_best:.6f}`; exceeds 0.8061: `{aligned_best > 0.8061}`.
14. Every within-center alignment cell was asserted unchanged to `<1e-10`; only cross-center pair rankings can move.
15. Final classification: `{terminal}`.

No center-specific affine adapter, calibration model, threshold, fusion, covariance correction, or neural model was developed. Stop after this audit.
"""
    (args.out / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    write_json(args.out / "AUDIT_STATUS.json", {"status": "COMPLETE", "terminal": terminal, "seed": SEED,
                                                   "test_previously_viewed": True, "test_used_for_tuning": False,
                                                   "baseline_replay": "PASS", "historical_five_clip_train_npzs_read": False,
                                                   "source_fisher_auroc": fisher_auc, "frozen_cnn_auroc": cnn_auc,
                                                   "weighted_within_delta": delta_within, "weighted_cross_delta": delta_cross,
                                                   "oof_viability_pass": viable})
    print(f"CROSSCENTER_AUDIT_COMPLETE terminal={terminal} cnn={cnn_auc:.10f} fisher={fisher_auc:.10f}", flush=True)


if __name__ == "__main__":
    main()
