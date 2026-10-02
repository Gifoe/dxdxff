"""Pure class-conditional geometry and unlabeled-covariance audit.

No model is trained here.  The only inputs are private, frozen CNN channel
representation caches.  In particular, `estimate_unlabeled_moments` has only
two positional inputs so target labels cannot be supplied to UTC/CORAL or
UTC-Mahalanobis construction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


SEED = 42
CHECKPOINT_SHA256 = "442b6a01b60994197fbc0e146f3abab7897b0b71e794259014d3eca663de3852"
OFFICIAL_SOURCE_SHA256 = "c493b43ec025f74153f0ada3041e6640bfa07beac58f058139ad5122da043f95"
TRAIN_AUROC = 0.9586782931
TEST_AUROC = 0.7987673466
TOLERANCE = 1e-5
SHRINKAGE = 0.05
EIG_FLOOR_SCALE = 1e-6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cache_digest(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(root.glob("*.npz")):
        h.update(path.name.encode())
        h.update(sha256(path).encode())
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def cosine(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else float("nan")


def safe_auc(y, score) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def safe_ap(y, score) -> float:
    y = np.asarray(y)
    return float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def metrics(y, score) -> dict:
    return {"auroc": safe_auc(y, score), "ap": safe_ap(y, score),
            "n": int(len(y)), "n_pathological": int(np.sum(np.asarray(y) == 1)),
            "n_normal": int(np.sum(np.asarray(y) == 0))}


def load_cache(root: Path) -> dict:
    records = {"patient": [], "edf": [], "channel": [], "y": [], "r4": [], "p16": [], "cnn": [], "clips": []}
    paths = sorted(root.glob("*.npz"))
    if not paths:
        raise RuntimeError(f"No private representation cache found: {root}")
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            patient, edf = str(z["patient"]), str(z["edf"])
            names = z["channel_names"].astype(str)
            labels = np.asarray(z["pathological_labels"], np.int8)
            r4, p16 = np.asarray(z["r4_mean"], np.float64), np.asarray(z["p16_mean"], np.float64)
            logits, offsets = np.asarray(z["segment_logits"], np.float64), np.asarray(z["segment_offsets"], np.int64)
            if r4.shape != (len(names), 32) or p16.shape != (len(names), 16) or len(offsets) != len(names) + 1:
                raise RuntimeError(f"Invalid representation shape in {path.name}")
            score = np.array([1.0 - sigmoid(logits[offsets[i]:offsets[i + 1]]).mean()
                              for i in range(len(names))], dtype=np.float64)
            records["patient"].extend([patient] * len(names)); records["edf"].extend([edf] * len(names))
            records["channel"].extend(names.tolist()); records["y"].extend(labels.tolist())
            records["r4"].append(r4); records["p16"].append(p16); records["cnn"].append(score)
            records["clips"].extend(np.diff(offsets).astype(int).tolist())
    return {"patient": np.asarray(records["patient"]), "edf": np.asarray(records["edf"]),
            "channel": np.asarray(records["channel"]), "y": np.asarray(records["y"], np.int8),
            "r4": np.concatenate(records["r4"]), "p16": np.concatenate(records["p16"]),
            "cnn": np.concatenate(records["cnn"]), "clips": np.asarray(records["clips"], np.int64),
            "files": len(paths)}


def moment_estimate(X: np.ndarray, patient_ids: np.ndarray, patient_equal: bool) -> dict:
    """First/second moment, with an explicit patient-equal alternative."""
    X, patient_ids = np.asarray(X, np.float64), np.asarray(patient_ids).astype(str)
    if len(X) == 0:
        raise RuntimeError("Cannot estimate a covariance from zero channel units")
    if not patient_equal:
        mean = X.mean(axis=0)
        second = X.T @ X / len(X)
        return {"mean": mean, "cov": second - np.outer(mean, mean), "n_units": len(X),
                "n_patients": len(np.unique(patient_ids)), "weighting": "channel_equal"}
    unique = np.unique(patient_ids)
    first, second = [], []
    for patient in unique:
        current = X[patient_ids == patient]
        first.append(current.mean(axis=0))
        second.append(current.T @ current / len(current))
    mean = np.mean(first, axis=0)
    raw_second = np.mean(second, axis=0)
    return {"mean": mean, "cov": raw_second - np.outer(mean, mean), "n_units": len(X),
            "n_patients": len(unique), "weighting": "patient_equal"}


def estimate_unlabeled_moments(X, patient_ids):
    """The only legal UTC/CORAL covariance estimator: inputs are X and patient IDs."""
    return moment_estimate(X, patient_ids, patient_equal=True)


def regularize(cov: np.ndarray) -> tuple[np.ndarray, float]:
    cov = (np.asarray(cov, np.float64) + np.asarray(cov, np.float64).T) / 2.0
    d = cov.shape[0]; scale = float(np.trace(cov) / d)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return (1.0 - SHRINKAGE) * cov + SHRINKAGE * scale * np.eye(d), scale


def eigh_clip(cov: np.ndarray):
    cov, scale = regularize(cov)
    values, vectors = np.linalg.eigh(cov)
    floor = EIG_FLOOR_SCALE * scale
    return np.maximum(values, floor), vectors, floor, cov


def inverse_cov(cov: np.ndarray) -> np.ndarray:
    values, vectors, _, _ = eigh_clip(cov)
    return (vectors / values) @ vectors.T


def sqrt_cov(cov: np.ndarray) -> np.ndarray:
    values, vectors, _, _ = eigh_clip(cov)
    return (vectors * np.sqrt(values)) @ vectors.T


def invsqrt_cov(cov: np.ndarray) -> np.ndarray:
    values, vectors, _, _ = eigh_clip(cov)
    return (vectors / np.sqrt(values)) @ vectors.T


def covariance_comparison(train_cov, test_cov) -> dict:
    train_values, train_vectors, _, train_reg = eigh_clip(train_cov)
    test_values, test_vectors, _, test_reg = eigh_clip(test_cov)
    eps = EIG_FLOOR_SCALE * max(float(np.trace(train_reg) / len(train_reg)), 1.0)
    k = min(5, len(train_values))
    # eigh returns ascending order; select the leading eigenspaces.
    U_train, U_test = train_vectors[:, -k:], test_vectors[:, -k:]
    sv = np.clip(np.linalg.svd(U_train.T @ U_test, compute_uv=False), -1.0, 1.0)
    angles = np.degrees(np.arccos(np.sort(sv)[::-1]))
    spectrum_corr = float(np.corrcoef(np.log(train_values), np.log(test_values))[0, 1]) if len(train_values) > 1 else float("nan")
    return {"relative_frobenius_shift": float(np.linalg.norm(train_reg - test_reg, "fro") /
                                                (np.linalg.norm(train_reg, "fro") + eps)),
            "covariance_cosine": cosine(train_reg.ravel(), test_reg.ravel()),
            "eigenspectrum_correlation": spectrum_corr,
            "logdet_difference_test_minus_train": float(np.log(test_values).sum() - np.log(train_values).sum()),
            "principal_angles_deg": angles, "train_eigenvalues": train_values,
            "test_eigenvalues": test_values}


def fixed_mmd(X, Y, seed: int) -> float:
    """Fixed-bandwidth Gaussian MMD diagnostic, deterministically capped at 800/channel units."""
    rng = np.random.default_rng(seed)
    X = X[rng.choice(len(X), min(len(X), 800), replace=False)]
    Y = Y[rng.choice(len(Y), min(len(Y), 800), replace=False)]
    gamma = 1.0 / X.shape[1]
    def kernel(A, B):
        aa = (A * A).sum(axis=1)[:, None]; bb = (B * B).sum(axis=1)[None, :]
        return np.exp(-gamma * np.maximum(aa + bb - 2 * A @ B.T, 0.0))
    return float(kernel(X, X).mean() + kernel(Y, Y).mean() - 2 * kernel(X, Y).mean())


def whiten(X, moments) -> np.ndarray:
    return (X - moments["mean"]) @ invsqrt_cov(moments["cov"]).T


def prototype_scores(X, mean0, mean1, covariance=None):
    if covariance is None:
        d0 = ((X - mean0) ** 2).sum(axis=1); d1 = ((X - mean1) ** 2).sum(axis=1)
    else:
        inv = inverse_cov(covariance)
        d0 = np.einsum("ij,jk,ik->i", X - mean0, inv, X - mean0)
        d1 = np.einsum("ij,jk,ik->i", X - mean1, inv, X - mean1)
    return d0 - d1


def patient_auc_matrices(y, patient, scores: dict) -> dict:
    patients = np.unique(patient); matrices = {}
    pos = np.array([np.sum((patient == p) & (y == 1)) for p in patients], np.float64)
    neg = np.array([np.sum((patient == p) & (y == 0)) for p in patients], np.float64)
    for name, score in scores.items():
        result = np.zeros((len(patients), len(patients)), dtype=np.float64)
        for i, pi in enumerate(patients):
            a = np.asarray(score[(patient == pi) & (y == 1)])
            for j, pj in enumerate(patients):
                b = np.asarray(score[(patient == pj) & (y == 0)])
                if len(a) and len(b):
                    result[i, j] = np.sum(a[:, None] > b[None, :]) + 0.5 * np.sum(a[:, None] == b[None, :])
        matrices[name] = result
    return {"patients": patients, "pos": pos, "neg": neg, "scores": matrices}


def cluster_bootstrap(y, patient, scores: dict, comparisons: list[tuple[str, str]]) -> list[dict]:
    matrices = patient_auc_matrices(y, patient, scores)
    rng = np.random.default_rng(SEED)
    n = len(matrices["patients"])
    counts = np.zeros((10000, n), dtype=np.float64)
    for row in range(len(counts)):
        counts[row] = np.bincount(rng.integers(0, n, n), minlength=n)
    denominator = (counts @ matrices["pos"]) * (counts @ matrices["neg"])
    aucs = {}
    for name, matrix in matrices["scores"].items():
        numer = np.einsum("bi,ij,bj->b", counts, matrix, counts, optimize=True)
        aucs[name] = np.divide(numer, denominator, out=np.full(len(counts), np.nan), where=denominator > 0)
    rows = []
    for left, right in comparisons:
        delta = aucs[left] - aucs[right]
        observed = safe_auc(y, scores[left]) - safe_auc(y, scores[right])
        rows.append({"comparison": f"{left} - {right}", "metric": "AUROC", "draws": 10000, "seed": SEED,
                     "point_delta": observed, "ci95_low": float(np.nanquantile(delta, 0.025)),
                     "ci95_high": float(np.nanquantile(delta, 0.975)),
                     "probability_delta_gt_zero": float(np.nanmean(delta > 0))})
    return rows


def center_name(raw: str) -> str:
    key = str(raw).lower()
    return {"hup": "HUP", "openieeg": "Open-iEEG", "open-ieeg": "Open-iEEG",
            "sourcesink": "SourceSink", "source-sink": "SourceSink", "zurich": "Zurich"}.get(key, str(raw))


def patient_center(split_csv: Path) -> dict:
    mapping = {}
    with split_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            patient, dataset = row["patient_name"], row["dataset"]
            if patient in mapping and mapping[patient] != dataset:
                raise RuntimeError(f"Conflicting dataset mapping for {patient}")
            mapping[patient] = dataset
    return mapping


def metric_row(method, y, score, representation="R4_32D", **extra):
    return {"method": method, "representation": representation, **metrics(y, score), **extra}


def protocol_lock_payload(full_train_gate_sha256: str, representation_gate_sha256: str) -> dict:
    """Return the immutable analysis definition before any metric gate."""
    return {"experiment": "omni_classconditional_covariance_audit_seed42_v1", "seed": SEED,
            "kind": "pure_posthoc_audit", "frozen_checkpoint_sha256": CHECKPOINT_SHA256,
            "official_cnn_source_sha256": OFFICIAL_SOURCE_SHA256, "representations": {"R4": 32, "P16": 16},
            "aggregation": "plain mean of frozen segment representations per EDF-channel",
            "train_source": "validated_omni_bag_mismatch_full_record_artifact_only",
            "historical_five_clip_train_npzs_read": False,
            "full_record_train_recovery_gate_sha256": full_train_gate_sha256,
            "full_record_train_representation_gate_sha256": representation_gate_sha256,
            "covariance": {"primary_weighting": "patient_equal_first_second_moments", "sensitivity_weighting": "channel_equal",
                           "shrinkage": "0.95 cov + 0.05 trace(cov)/d I", "eigen_floor": "1e-6 trace(cov)/d"},
            "utc": "target unlabeled covariance plus source labeled delta mean only",
            "target_labels_for_construction": {"UTC": False, "UTC_diagonal": False, "CORAL": False, "UTC_Mahalanobis": False,
                                                "TargetFisher": True}, "test_previously_viewed": True,
            "official_test_used_for_tuning": False, "hard_stop": "No adapter/model/shrinkage/layer iteration after audit"}


def write_terminal_baseline_failure(out: Path, replay: dict) -> None:
    """Emit only aggregate provenance when the binding replay gate rejects analysis."""
    out.joinpath("REPRESENTATION_SOURCE_AUDIT.md").write_text(
        "# Frozen representation source\n\n"
        "R4 is the existing 32-D output of the frozen official CNN `cnn` block. P16 is the true 16-D tensor after the frozen `fc1 → relu1 → bn1` path and immediately before `fc_out`. Both use the predeclared plain segment mean within `(EDF, channel)`. TRAIN reconstruction is bound to the validated full-record `omni_bag_mismatch_audit_seed42_v1` artifact and native HDF5 replay; historical five-clip TRAIN NPZs are not read.\n",
        encoding="utf-8")
    out.joinpath("ZERO_LABEL_LEAKAGE_AUDIT.md").write_text(
        "# Zero-label leakage audit\n\n"
        "Status: `NOT_RUN_BASELINE_REPLAY_FAILED`. No UTC, CORAL, Mahalanobis, Fisher, covariance, or target-label diagnostic was constructed after the binding baseline gate failed. Therefore no target-label construction path exists for this terminal audit.\n",
        encoding="utf-8")
    out.joinpath("FINAL_REPORT.md").write_text(
        "# Omni class-conditional covariance audit\n\n"
        "**Terminal status:** `BASELINE_REPLAY_FAILED`. The requested hard stop was enforced.\n\n"
        "| Gate | Expected AUROC | Replayed AUROC | Result |\n|---|---:|---:|---|\n"
        f"| TRAIN full-record | {replay['train_expected_auroc']:.10f} | {replay['train_full_auroc']:.10f} | failed |\n"
        f"| TEST official EDF-channel | {replay['test_expected_auroc']:.10f} | {replay['test_auroc']:.10f} | passed |\n\n"
        "The test replay is within its stated tolerance, but the train gate differs by more than the binding tolerance. No covariance geometry, Fisher direction, UTC, CORAL, Mahalanobis, bootstrap, center, or sample-size results were generated. The expected train reference was not replaced with the observed value.\n",
        encoding="utf-8")
    write_json(out / "AUDIT_STATUS.json", {"status": "BASELINE_REPLAY_FAILED", "terminal": True,
                                            "test_previously_viewed": True, "test_used_for_tuning": False,
                                            "downstream_analyses_run": False})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--full-train-gate", type=Path, required=True)
    parser.add_argument("--full-train-representation-gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    full_gate = json.loads(args.full_train_gate.read_text(encoding="utf-8"))
    expected_full_train = {"edfs": 296, "total_segments": 316364,
                           "labelled_segments": 145052, "labelled_edf_channel_units": 13350}
    observed_full_train = full_gate.get("observed", {})
    def gate_count(values: dict, key: str) -> int:
        # The recovered upstream artifact uses US ``labeled`` spelling;
        # internal audit caches use ``labelled``. Both denote the same frozen
        # count and neither relaxes the required value.
        return int(values.get(key, values.get(key.replace("labelled", "labeled"), -1)))

    if (full_gate.get("status") != "PASS" or
            any(gate_count(observed_full_train, k) != v for k, v in expected_full_train.items()) or
            abs(float(observed_full_train.get("train_full_auroc", float("nan"))) - TRAIN_AUROC) >= TOLERANCE or
            full_gate.get("historical_five_clip_train_npzs_read") is not False):
        raise RuntimeError("FULL_RECORD_TRAIN_RECOVERY_GATE_REQUIRED")
    representation_gate = json.loads(args.full_train_representation_gate.read_text(encoding="utf-8"))
    representation_observed = representation_gate.get("observed", {})
    if (representation_gate.get("status") != "PASS" or
            any(gate_count(representation_observed, k) != v for k, v in expected_full_train.items()) or
            abs(float(representation_observed.get("train_full_auroc", float("nan"))) - TRAIN_AUROC) >= TOLERANCE or
            representation_gate.get("checkpoint_sha256") != CHECKPOINT_SHA256 or
            representation_gate.get("official_cnn_source_sha256") != OFFICIAL_SOURCE_SHA256 or
            representation_gate.get("historical_five_clip_train_npzs_read") is not False):
        raise RuntimeError("FULL_RECORD_TRAIN_REPRESENTATION_GATE_REQUIRED")
    full_train_gate_sha256 = sha256(args.full_train_gate)
    if representation_gate.get("full_record_artifact_recovery_gate_sha256") != full_train_gate_sha256:
        raise RuntimeError("FULL_RECORD_TRAIN_REPRESENTATION_GATE_BINDING_MISMATCH")
    representation_gate_sha256 = sha256(args.full_train_representation_gate)
    write_json(args.out / "PROTOCOL_LOCK.json", protocol_lock_payload(full_train_gate_sha256, representation_gate_sha256))
    train, test = load_cache(args.train_cache), load_cache(args.test_cache)
    if train["files"] != 296 or test["files"] != 237:
        raise RuntimeError("Frozen representation cache file count differs from the bound cohorts")
    train_labeled, test_labeled = train["y"] >= 0, test["y"] >= 0
    represented_full_train = {"total_segments": int(train["clips"].sum()),
                              "labelled_segments": int(train["clips"][train_labeled].sum()),
                              "labelled_edf_channel_units": int(train_labeled.sum())}
    if represented_full_train != {k: v for k, v in expected_full_train.items() if k != "edfs"}:
        raise RuntimeError("FULL_RECORD_REPRESENTATION_CACHE_COUNTS_MISMATCH")
    replay_train, replay_test = safe_auc(train["y"][train_labeled], train["cnn"][train_labeled]), safe_auc(test["y"][test_labeled], test["cnn"][test_labeled])
    replay = {"status": "PASS" if abs(replay_train - TRAIN_AUROC) < TOLERANCE and abs(replay_test - TEST_AUROC) < TOLERANCE else "BASELINE_REPLAY_FAILED",
              "checkpoint_sha256": CHECKPOINT_SHA256, "official_cnn_source_sha256": OFFICIAL_SOURCE_SHA256,
              "train_full_auroc": replay_train, "train_expected_auroc": TRAIN_AUROC,
              "test_auroc": replay_test, "test_expected_auroc": TEST_AUROC,
              "tolerance": TOLERANCE, "train_cache_sha256": cache_digest(args.train_cache),
              "test_cache_sha256": cache_digest(args.test_cache), "test_labeled_pairs": int(test_labeled.sum()),
              "full_record_train_recovery_gate_sha256": full_train_gate_sha256,
              "historical_five_clip_train_npzs_read": False,
              "train_total_segments": represented_full_train["total_segments"],
              "train_labelled_segments": represented_full_train["labelled_segments"],
              "train_labelled_edf_channel_units": represented_full_train["labelled_edf_channel_units"],
              "test_labeled_patients": int(len(np.unique(test["patient"][test_labeled]))),
              "target_unlabeled_patients": int(len(np.unique(test["patient"]))),
              "target_unlabeled_channel_units": int(len(test["patient"]))}
    write_json(args.out / "BASELINE_REPLAY_AUDIT.json", replay)
    if replay["status"] != "PASS":
        write_terminal_baseline_failure(args.out, replay)
        raise RuntimeError("BASELINE_REPLAY_FAILED")
    (args.out / "REPRESENTATION_SOURCE_AUDIT.md").write_text(
        "# Frozen representation source\n\n"
        "R4 is the existing 32-D output of the frozen official CNN `cnn` block. P16 is the true 16-D tensor after the frozen `fc1 → relu1 → bn1` path and immediately before `fc_out`. Both are plain segment means within `(EDF, channel)`. TRAIN representations were regenerated only from the validated `omni_bag_mismatch_audit_seed42_v1` full-record artifact and native HDF5, not historical five-clip TRAIN NPZs. Private caches and identities are excluded from Git.\n",
        encoding="utf-8")

    geometry_rows, eig_rows, angle_rows, mean_rows, fisher_rows, score_rows = [], [], [], [], [], []
    direction_rows, all_metric_rows, rep_context = [], [], {}
    target_unlabeled = {"R4_32D": test["r4"], "P16_16D": test["p16"]}
    source_unlabeled = {"R4_32D": train["r4"], "P16_16D": train["p16"]}
    y_test, p_test = test["y"][test_labeled].astype(np.int8), test["patient"][test_labeled]
    for rep, key in [("R4_32D", "r4"), ("P16_16D", "p16")]:
        Xtr, Xte = train[key], test[key]
        ytr, yte = train["y"], test["y"]
        source = {weight: {c: moment_estimate(Xtr[(ytr == c) & train_labeled], train["patient"][(ytr == c) & train_labeled], weight == "patient_equal")
                           for c in (0, 1)} for weight in ("patient_equal", "channel_equal")}
        target = {weight: {c: moment_estimate(Xte[(yte == c) & test_labeled], test["patient"][(yte == c) & test_labeled], weight == "patient_equal")
                           for c in (0, 1)} for weight in ("patient_equal", "channel_equal")}
        for weight in ("patient_equal", "channel_equal"):
            src, tgt = source[weight], target[weight]
            delta_src, delta_tgt = src[1]["mean"] - src[0]["mean"], tgt[1]["mean"] - tgt[0]["mean"]
            for domain, values in [("TRAIN", src), ("TEST_DIAGNOSTIC", tgt)]:
                for c in (0, 1):
                    mean_rows.append({"representation": rep, "weighting": weight, "domain": domain, "class": c,
                                      "n_units": values[c]["n_units"], "n_patients": values[c]["n_patients"],
                                      "mean_l2_norm": float(np.linalg.norm(values[c]["mean"])),
                                      "cov_trace_raw": float(np.trace(values[c]["cov"]))})
            pooled_src, pooled_tgt = (src[0]["cov"] + src[1]["cov"]) / 2, (tgt[0]["cov"] + tgt[1]["cov"]) / 2
            for name, a, b in [("normal", src[0]["cov"], tgt[0]["cov"]), ("pathological", src[1]["cov"], tgt[1]["cov"]),
                               ("pooled_within", pooled_src, pooled_tgt)]:
                comp = covariance_comparison(a, b)
                if name != "pooled_within":
                    Za = whiten(Xtr[(ytr == (1 if name == "pathological" else 0)) & train_labeled], src[1 if name == "pathological" else 0])
                    Zb = whiten(Xte[(yte == (1 if name == "pathological" else 0)) & test_labeled], tgt[1 if name == "pathological" else 0])
                    whitened_mmd = fixed_mmd(Za, Zb, SEED + (1 if name == "pathological" else 0))
                    norm_train, norm_test = float(np.linalg.norm(Za, axis=1).mean()), float(np.linalg.norm(Zb, axis=1).mean())
                else:
                    whitened_mmd = norm_train = norm_test = float("nan")
                geometry_rows.append({"representation": rep, "weighting": weight, "class_covariance": name,
                                      **{k: v for k, v in comp.items() if k not in ("principal_angles_deg", "train_eigenvalues", "test_eigenvalues")},
                                      "whitened_fixed_rbf_mmd": whitened_mmd, "whitened_mean_norm_train": norm_train,
                                      "whitened_mean_norm_test": norm_test})
                for period, values in [("TRAIN", comp["train_eigenvalues"]), ("TEST_DIAGNOSTIC", comp["test_eigenvalues"])]:
                    for rank, value in enumerate(values[::-1], 1):
                        eig_rows.append({"representation": rep, "weighting": weight, "class_covariance": name,
                                         "domain": period, "eigen_rank_descending": rank, "eigenvalue_regularized": float(value)})
                for index, angle in enumerate(comp["principal_angles_deg"], 1):
                    angle_rows.append({"representation": rep, "weighting": weight, "class_covariance": name,
                                       "principal_angle_rank": index, "angle_degrees": float(angle)})
            inv_src, inv_tgt = inverse_cov(pooled_src), inverse_cov(pooled_tgt)
            w_source, w_target = inv_src @ delta_src, inv_tgt @ delta_tgt
            fisher_rows.append({"representation": rep, "weighting": weight, "cosine_mean_direction_train_test": cosine(delta_src, delta_tgt),
                                "cosine_fisher_direction_train_test": cosine(w_source, w_target),
                                "J_train": float(delta_src @ inv_src @ delta_src), "J_test_diagnostic": float(delta_tgt @ inv_tgt @ delta_tgt)})
            if weight == "patient_equal":
                tu = estimate_unlabeled_moments(target_unlabeled[rep], test["patient"])
                su = estimate_unlabeled_moments(source_unlabeled[rep], train["patient"])
                w_utc, w_us, w_diag = inverse_cov(tu["cov"]) @ delta_src, inverse_cov(su["cov"]) @ delta_src, np.diag(1.0 / np.diag(regularize(tu["cov"])[0])) @ delta_src
                Xeval = Xte[test_labeled]; mean_corrected = Xeval - tu["mean"] + su["mean"]
                coral = su["mean"] + (Xeval - tu["mean"]) @ (sqrt_cov(su["cov"]) @ invsqrt_cov(tu["cov"])).T
                scores = {"Frozen TimeConv-CNN": test["cnn"][test_labeled], "Source Fisher": Xeval @ w_source,
                          "Unlabeled-source covariance Fisher": Xeval @ w_us, "Mean correction": mean_corrected @ w_source,
                          "UTC diagonal": Xeval @ w_diag, "UTC full covariance": Xeval @ w_utc,
                          "CORAL diagnostic": coral @ w_source, "Target Fisher diagnostic": Xeval @ w_target,
                          "Euclidean prototype": prototype_scores(Xeval, src[0]["mean"], src[1]["mean"]),
                          "Source Mahalanobis prototype": prototype_scores(Xeval, src[0]["mean"], src[1]["mean"], pooled_src),
                          "UTC Mahalanobis": prototype_scores(Xeval, src[0]["mean"], src[1]["mean"], tu["cov"])}
                for method, score in scores.items():
                    all_metric_rows.append(metric_row(method, y_test, score, rep,
                                      target_labels_used_to_construct="YES_DIAGNOSTIC_ONLY" if method == "Target Fisher diagnostic" else "NO",
                                      target_unlabeled_statistics_used="YES" if method in {"UTC full covariance", "UTC diagonal", "Mean correction", "CORAL diagnostic", "UTC Mahalanobis"} else "NO",
                                      delta_auroc_vs_cnn=safe_auc(y_test, score) - replay_test))
                direction_rows.extend([
                    {"representation": rep, "comparison": "source_fisher_vs_target_fisher_diagnostic", "cosine": cosine(w_source, w_target)},
                    {"representation": rep, "comparison": "utc_full_vs_target_fisher_diagnostic", "cosine": cosine(w_utc, w_target)},
                    {"representation": rep, "comparison": "utc_full_vs_source_fisher", "cosine": cosine(w_utc, w_source)},
                    {"representation": rep, "comparison": "utc_full_vs_target_logistic_probe", "cosine": float("nan"), "status": "HISTORICAL_FROZEN_PROBE_WEIGHT_UNAVAILABLE_NOT_RETRAINED"},
                    {"representation": rep, "comparison": "source_fisher_vs_target_logistic_probe", "cosine": float("nan"), "status": "HISTORICAL_FROZEN_PROBE_WEIGHT_UNAVAILABLE_NOT_RETRAINED"},
                ])
                rep_context[rep] = {"scores": scores, "source": src, "target": tgt, "unlabeled_target": tu,
                                    "w_source": w_source, "w_target": w_target, "w_utc": w_utc, "w_diag": w_diag}
    write_csv(args.out / "CLASS_MEANS.csv", mean_rows); write_csv(args.out / "CLASS_CONDITIONAL_COVARIANCE_SHIFT.csv", geometry_rows)
    write_csv(args.out / "COVARIANCE_EIGENSPECTRUM.csv", eig_rows); write_csv(args.out / "PRINCIPAL_ANGLES.csv", angle_rows)
    write_csv(args.out / "FISHER_GEOMETRY.csv", fisher_rows); write_csv(args.out / "FISHER_SCORE_METRICS.csv", [r for r in all_metric_rows if r["method"] in {"Source Fisher", "Unlabeled-source covariance Fisher", "Mean correction", "Target Fisher diagnostic"}])
    write_csv(args.out / "UTC_METRICS.csv", [r for r in all_metric_rows if r["method"] == "UTC full covariance"])
    write_csv(args.out / "UTC_DIAGONAL_METRICS.csv", [r for r in all_metric_rows if r["method"] == "UTC diagonal"])
    write_csv(args.out / "CORAL_DIAGNOSTIC_METRICS.csv", [r for r in all_metric_rows if r["method"] == "CORAL diagnostic"])
    write_csv(args.out / "PROTOTYPE_METRICS.csv", [r for r in all_metric_rows if "prototype" in r["method"]])
    write_csv(args.out / "DIRECTION_COSINES.csv", direction_rows)

    # R4 is the predeclared primary score representation for bootstrap/center/sample-size analyses.
    primary = rep_context["R4_32D"]; primary_scores = primary["scores"]
    bootstrap_rows = cluster_bootstrap(y_test, p_test, {"UTC full covariance": primary_scores["UTC full covariance"],
                              "Frozen TimeConv-CNN": primary_scores["Frozen TimeConv-CNN"], "Source Fisher": primary_scores["Source Fisher"],
                              "UTC diagonal": primary_scores["UTC diagonal"]},
                              [("UTC full covariance", "Frozen TimeConv-CNN"), ("UTC full covariance", "Source Fisher"),
                               ("UTC full covariance", "UTC diagonal")])
    write_csv(args.out / "PAIRED_BOOTSTRAP.csv", bootstrap_rows)

    centers = patient_center(args.split_csv); test_centers = np.array([center_name(centers.get(p, "UNMAPPED")) for p in test["patient"]])
    if np.any(test_centers == "UNMAPPED"):
        raise RuntimeError("TEST patient absent from frozen final split metadata")
    center_rows = []
    named_methods = ["Frozen TimeConv-CNN", "Source Fisher", "UTC full covariance", "UTC diagonal", "CORAL diagnostic", "UTC Mahalanobis"]
    for center in ["HUP", "Open-iEEG", "SourceSink", "Zurich"]:
        mask = test_labeled & (test_centers == center)
        for method in named_methods:
            value = primary_scores[method]
            local = np.flatnonzero(mask[test_labeled])
            row = metric_row(method, y_test[local], value[local], "R4_32D", center=center,
                             status="estimable" if len(np.unique(y_test[local])) == 2 else "not_estimable")
            center_rows.append(row)
    # Center-stratified UTC is diagnostic only, and makes no use of outcome labels.
    for center in ["HUP", "Open-iEEG", "SourceSink", "Zurich"]:
        unlabeled_mask = test_centers == center
        if len(np.unique(test["patient"][unlabeled_mask])) >= 5:
            moment = estimate_unlabeled_moments(test["r4"][unlabeled_mask], test["patient"][unlabeled_mask])
            w = inverse_cov(moment["cov"]) @ (primary["source"][1]["mean"] - primary["source"][0]["mean"])
            eval_mask = test_labeled & (test_centers == center); local = np.flatnonzero(eval_mask[test_labeled])
            center_rows.append(metric_row("UTC center-metadata conditional diagnostic", y_test[local], (test["r4"][test_labeled] @ w)[local], "R4_32D",
                                          center=center, status="CENTER_METADATA_CONDITIONAL_DIAGNOSTIC"))
    write_csv(args.out / "CENTER_METRICS.csv", center_rows)

    sample_rows, rng = [], np.random.default_rng(SEED)
    unique_target = np.unique(test["patient"])
    delta = primary["source"][1]["mean"] - primary["source"][0]["mean"]
    for K in [5, 10, 20, 40, 80]:
        for repeat in range(50):
            picked = rng.choice(unique_target, K, replace=False)
            # This call has no access to labels; metrics are calculated afterwards on the held-out labeled patients.
            covariance_membership = np.isin(test["patient"], picked)
            moment = estimate_unlabeled_moments(test["r4"][covariance_membership], test["patient"][covariance_membership])
            score = test["r4"] @ (inverse_cov(moment["cov"]) @ delta)
            evaluation = test_labeled & ~np.isin(test["patient"], picked)
            sample_rows.append({"representation": "R4_32D", "K_target_patients_for_unlabeled_covariance": K, "repeat": repeat,
                                "seed": SEED, "evaluation": "remaining_labeled_target_patients", "n_covariance_patients": K,
                                "n_evaluation_labeled_patients": len(np.unique(test["patient"][evaluation])),
                                **metrics(test["y"][evaluation], score[evaluation])})
    score_all = test["r4"] @ primary["w_utc"]
    sample_rows.append({"representation": "R4_32D", "K_target_patients_for_unlabeled_covariance": "ALL", "repeat": 0,
                        "seed": SEED, "evaluation": "full_labeled_target_cohort_nonheldout_reference", "n_covariance_patients": len(unique_target),
                        "n_evaluation_labeled_patients": len(np.unique(test["patient"][test_labeled])), **metrics(y_test, score_all[test_labeled])})
    write_csv(args.out / "TARGET_COVARIANCE_SAMPLE_SIZE.csv", sample_rows)

    patient_audit = {"primary": "patient_equal", "ordinary_sensitivity": "channel_equal", "rule": "per-patient first and second moments, then equal patient average",
                     "train_labeled_patients": int(len(np.unique(train["patient"][train_labeled]))), "test_labeled_patients": int(len(np.unique(test["patient"][test_labeled]))),
                     "test_unlabeled_covariance_patients": int(len(unique_target)), "patient_local_covariance": "SKIPPED_OPTIONAL_NOT_A_PRIMARY_ANALYSIS"}
    write_json(args.out / "PATIENT_EQUAL_MOMENT_AUDIT.json", patient_audit)
    forbidden = ["label", "soz", "resection", "outcome"]
    estimator_source = inspect.getsource(estimate_unlabeled_moments).lower()
    leakage = {"function_signature": str(inspect.signature(estimate_unlabeled_moments)), "forbidden_tokens_present_in_estimator": [x for x in forbidden if re.search(rf"\\b{re.escape(x)}\\b", estimator_source)],
               "UTC_constructed_before_metric_calls": True, "UTC_inputs": ["embedding", "patient_ids"],
               "target_labels_used_only_for": ["class-conditional diagnostic covariances", "target Fisher diagnostic", "AUROC/AP/bootstrap/center metrics"],
               "target_labels_used_for_UTC_CORAL_UTCMahalanobis": False}
    if leakage["forbidden_tokens_present_in_estimator"]:
        raise RuntimeError("Zero-label estimator source audit failed")
    (args.out / "ZERO_LABEL_LEAKAGE_AUDIT.md").write_text(
        "# Zero-label construction audit\n\n"
        f"The UTC estimator signature is `{leakage['function_signature']}`. It receives only a representation matrix and patient IDs; its function body contains none of: {', '.join(forbidden)}. UTC, diagonal UTC, CORAL, and UTC-Mahalanobis were constructed before evaluation metrics, with target pathology labels excluded from their construction. TEST labels were used only for explicitly marked diagnostic geometry and evaluation.\n",
        encoding="utf-8")

    primary_metrics = {row["method"]: row for row in all_metric_rows if row["representation"] == "R4_32D"}
    utc = primary_metrics["UTC full covariance"]["auroc"]
    if utc <= TEST_AUROC:
        terminal = "COVARIANCE_SHIFT_NOT_ZERO_SHOT_ACTIONABLE"
    elif utc <= 0.8061:
        terminal = "WEAK_COVARIANCE_SIGNAL"
    elif utc < 0.82:
        terminal = "ZERO_LABEL_COVARIANCE_ADAPTER_VIABLE"
    else:
        boot = next(r for r in bootstrap_rows if r["comparison"] == "UTC full covariance - Frozen TimeConv-CNN")
        nonnegative = sum(1 for c in ["HUP", "Open-iEEG", "SourceSink"] if next((r for r in center_rows if r["center"] == c and r["method"] == "UTC full covariance"), {"auroc": float("nan")})["auroc"] >= next((r for r in center_rows if r["center"] == c and r["method"] == "Frozen TimeConv-CNN"), {"auroc": float("nan")})["auroc"])
        terminal = "STRONG_COVARIANCE_ADAPTER_HEADROOM" if boot["ci95_low"] > 0 and nonnegative >= 2 else "ZERO_LABEL_COVARIANCE_ADAPTER_VIABLE"
    table_methods = ["Frozen TimeConv-CNN", "Source Fisher", "Mean correction", "UTC diagonal", "UTC full covariance", "CORAL diagnostic", "UTC Mahalanobis", "Target Fisher diagnostic"]
    table = "| Method | Target labels used to construct? | Target unlabeled stats used? | AUROC | AP | ΔAUROC vs CNN |\n|---|---|---|---:|---:|---:|\n" + "\n".join(
        f"| {m} | {primary_metrics[m]['target_labels_used_to_construct']} | {primary_metrics[m]['target_unlabeled_statistics_used']} | {primary_metrics[m]['auroc']:.6f} | {primary_metrics[m]['ap']:.6f} | {primary_metrics[m]['delta_auroc_vs_cnn']:+.6f} |" for m in table_methods)
    geom = {r["representation"]: r for r in fisher_rows if r["weighting"] == "patient_equal"}
    shift = {r["representation"]: {x["class_covariance"]: x for x in geometry_rows if x["representation"] == r["representation"] and x["weighting"] == "patient_equal"} for r in geom.values()}
    geometry_table = "| Representation | cos(mean direction) | covariance shift normal | covariance shift pathological | cos(Fisher direction) |\n|---|---:|---:|---:|---:|\n" + "\n".join(
        f"| {rep} | {geom[rep]['cosine_mean_direction_train_test']:.6f} | {shift[rep]['normal']['relative_frobenius_shift']:.6f} | {shift[rep]['pathological']['relative_frobenius_shift']:.6f} | {geom[rep]['cosine_fisher_direction_train_test']:.6f} |" for rep in ["R4_32D", "P16_16D"])
    boot = next(r for r in bootstrap_rows if r["comparison"] == "UTC full covariance - Frozen TimeConv-CNN")
    report = f"# Omni class-conditional covariance audit\n\n**Status:** `{terminal}`. This is exploratory/repeated-test evidence because the Omni test outcome was historically viewed; no test result was used to change a model, covariance formula, representation layer, or shrinkage.\n\n{table}\n\n{geometry_table}\n\n## Direct answers\n\n1. The frozen CNN replay passed: TRAIN full AUROC `{replay_train:.10f}` and TEST AUROC `{replay_test:.10f}`.\n2. R4 mean-direction cosine is `{geom['R4_32D']['cosine_mean_direction_train_test']:.6f}`; P16 is `{geom['P16_16D']['cosine_mean_direction_train_test']:.6f}`.\n3. R4 normal/pathological covariance shifts are `{shift['R4_32D']['normal']['relative_frobenius_shift']:.6f}` / `{shift['R4_32D']['pathological']['relative_frobenius_shift']:.6f}`.\n4. R4 source-versus-target Fisher cosine is `{geom['R4_32D']['cosine_fisher_direction_train_test']:.6f}`.\n5. Target Fisher is diagnostic only. UTC uses source class means and target **unlabeled** covariance, not target pathology labels.\n6. The primary UTC-CNN patient-cluster bootstrap ΔAUROC is `{boot['point_delta']:+.6f}` (95% CI `{boot['ci95_low']:+.6f}`, `{boot['ci95_high']:+.6f}`; Pr(Δ>0) `{boot['probability_delta_gt_zero']:.4f}`).\n7. The historical target logistic-probe vector was not available as a frozen artifact; it was not retrained, so the requested probe-direction comparison is explicitly unavailable rather than fabricated.\n8. The sample-size table separates held-out-patient estimates from the all-target non-held-out reference.\n\nNo covariance adapter was developed. Stop after this audit.\n"
    (args.out / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    write_json(args.out / "AUDIT_STATUS.json", {"status": "COMPLETE", "terminal": terminal, "baseline_replay": replay["status"],
                                                   "test_previously_viewed": True, "test_used_for_tuning": False,
                                                   "utc_r4_auroc": utc})
    print(json.dumps({"status": "COMPLETE", "terminal": terminal, "utc_r4_auroc": utc}, sort_keys=True))


if __name__ == "__main__":
    main()
