#!/usr/bin/env python3
"""Frozen five-fold A1-physiology complementarity audit for Omni Task 2.

The program never trains/replays TimeConv.  It verifies the private frozen OOF
baseline first, extracts only the historical A1 9D descriptors from the
existing 60-second record cache, then trains the predeclared F9/F36 heads on
four patient folds and writes public aggregates plus private predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score


SEED = 42
EPOCHS = 30
RATE = 1000
WINDOW_SECONDS = 2
WINDOW_SAMPLES = RATE * WINDOW_SECONDS
N_WINDOWS = 30
POS_WEIGHT = 2.0
BASELINE_AUROC_EXPECTED = 0.7621630659358641
BASELINE_AUROC_TOLERANCE = 1e-10
FEATURE_NAMES = (
    "log_bp_delta", "log_bp_theta", "log_bp_beta", "log_bp_low_gamma",
    "log_bp_high_gamma", "rms", "variance", "line_length_per_sec", "spectral_entropy",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import historical source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_spectral_import_stubs() -> None:
    """Satisfy unused run-level imports while executing the exact feature source.

    ``compute_spectral_channel_features`` itself only needs NumPy/SciPy.  The
    historical module also imports graph/EDF helpers for a separate full-run
    extraction API.  Stubbing those *unused* names avoids modifying or
    reimplementing the historical descriptor function.
    """
    graph = ModuleType("graph_channel")
    graph.GRAPH_FEATURE_NAMES = ()
    graph.compute_graph_node_features = lambda *args, **kwargs: None
    graph.compute_thresholded_abs_pearson_adjacency = lambda *args, **kwargs: None
    sys.modules.setdefault("graph_channel", graph)
    preprocessing = ModuleType("module2_preprocessing")
    preprocessing.load_and_preprocess_edf = lambda *args, **kwargs: None
    sys.modules.setdefault("module2_preprocessing", preprocessing)
    labels = ModuleType("module3_labels_metadata")
    labels.parse_channel_labels = lambda *args, **kwargs: None
    sys.modules.setdefault("module3_labels_metadata", labels)
    windows = ModuleType("module4_time_windows")
    windows.create_sliding_onset_ictal_sample = lambda *args, **kwargs: None
    windows.has_valid_ictal_bounds = lambda *args, **kwargs: False
    sys.modules.setdefault("module4_time_windows", windows)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def patient_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if set(labels.tolist()) != {0, 1}:
        raise RuntimeError("Both-class patient guarantee failed")
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    first_positive = int(np.flatnonzero(ranked == 1)[0])
    discount = 1.0 / np.log2(np.arange(2, len(labels) + 2, dtype=float))
    idcg = float((np.sort(labels)[::-1] * discount).sum())
    prediction = (scores >= 0.5).astype(np.int8)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "mrr": float(1.0 / (first_positive + 1)),
        "top1": float(ranked[0] == 1),
        "ndcg": float((ranked * discount).sum() / idcg),
        "macro_f1_0_5": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "pathological_f1_0_5": float(f1_score(labels, prediction, pos_label=1, zero_division=0)),
        "normal_f1_0_5": float(f1_score(labels, prediction, pos_label=0, zero_division=0)),
        "balanced_accuracy_0_5": float(balanced_accuracy_score(labels, prediction)),
    }


def means(rows: list[dict], keys: tuple[str, ...]) -> dict:
    return {f"{key}_mean": float(np.mean([float(row[key]) for row in rows])) for key in keys}


def pair_stats(labels: np.ndarray, tc: np.ndarray, feature: np.ndarray) -> dict:
    positive = labels == 1
    negative = labels == 0
    tc_ok = tc[positive, None] > tc[None, negative]
    feat_ok = feature[positive, None] > feature[None, negative]
    rescue = (~tc_ok) & feat_ok
    destroy = tc_ok & (~feat_ok)
    return {
        "pair_count": int(tc_ok.size),
        "tc_correct_count": int(tc_ok.sum()), "feature_correct_count": int(feat_ok.sum()),
        "rescue_count": int(rescue.sum()), "destroy_count": int(destroy.sum()),
        "tc_wrong_count": int((~tc_ok).sum()), "tc_right_count": int(tc_ok.sum()),
        "oracle_correct_count": int((tc_ok | feat_ok).sum()),
    }


def rates(stat: dict) -> dict:
    def div(top: int, bottom: int) -> float:
        return float(top / bottom) if bottom else float("nan")
    rescue = div(stat["rescue_count"], stat["tc_wrong_count"])
    destroy = div(stat["destroy_count"], stat["tc_right_count"])
    return {
        "tc_pair_accuracy": div(stat["tc_correct_count"], stat["pair_count"]),
        "feature_pair_accuracy": div(stat["feature_correct_count"], stat["pair_count"]),
        "rescue_rate": rescue, "destroy_rate": destroy,
        "net_complementarity": rescue - destroy if np.isfinite(rescue) and np.isfinite(destroy) else float("nan"),
        "oracle_pair_accuracy": div(stat["oracle_correct_count"], stat["pair_count"]),
    }


class FeatureHead(torch.nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(dimension, 32), torch.nn.GELU(), torch.nn.LayerNorm(32), torch.nn.Linear(32, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def extract_feature_cache(records: Path, cache: Path, spectral_source: Path, comparison_source: Path) -> tuple[dict, list[dict]]:
    """Reuse historical functions; cache only derived private descriptor values."""
    install_spectral_import_stubs()
    spectral = module_from_path("historical_a1_ez_features", spectral_source)
    # The original dataset module imports ``ez_features`` by name.  Supplying
    # the exact historical module preserves its feature-name contract.
    sys.modules["ez_features"] = spectral
    comparison = module_from_path("historical_a1_dataset", comparison_source)
    source_names = tuple(spectral.BASE_SPECTRAL_FEATURE_NAMES)
    indices = [source_names.index(name) for name in FEATURE_NAMES]
    comparison_args = SimpleNamespace(
        feature_view="interictal_comparison", self_compare_eps=1e-5, interictal_min_windows=10,
        include_inter_abs=True, include_inter_delta=True, include_inter_zdelta=True, include_inter_ratio=True,
        include_inter_percentile=False, interictal_missing_indicators=False,
    )
    cache.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict] = []
    for record in sorted(records.glob("edf_*.npz")):
        destination = cache / record.name
        if destination.is_file():
            with np.load(destination, allow_pickle=False) as saved:
                required = {"f9", "f36", "labels", "channel_names", "patient_name", "record_sha256"}
                if required <= set(saved.files) and str(saved["record_sha256"].item()) == sha256(record):
                    audit_rows.append({"record_cache": record.name, "reused": True, "channels": int(saved["labels"].shape[0])})
                    continue
            raise RuntimeError("Descriptor cache provenance mismatch")
        with np.load(record, allow_pickle=False) as item:
            waves = item["waveforms"].astype(np.float32)
            labels = item["labels"].astype(np.int8)
            names = item["channel_names"].astype(str)
            patient = str(item["patient_name"].item())
            rate = float(item["sampling_rate_hz"].item())
        if waves.ndim != 2 or waves.shape[1] != RATE * 60 or rate != RATE:
            raise RuntimeError("Unexpected frozen record geometry")
        per_window = []
        for window in range(N_WINDOWS):
            view = waves[:, window * WINDOW_SAMPLES:(window + 1) * WINDOW_SAMPLES]
            full = spectral.compute_spectral_channel_features(view, sfreq=RATE)
            per_window.append(full[:, indices])
        absolute = np.stack(per_window, axis=0).astype(np.float32)
        relative = comparison._interictal_comparison_features(
            absolute, np.arange(N_WINDOWS, dtype=np.float32), {}, comparison_args,
        )
        if relative.shape != (N_WINDOWS, waves.shape[0], 36):
            raise RuntimeError(f"Historical interictal feature shape mismatch: {relative.shape}")
        f9 = absolute.mean(axis=0, dtype=np.float64).astype(np.float32)
        f36 = relative.mean(axis=0, dtype=np.float64).astype(np.float32)
        if not np.isfinite(f9).all() or not np.isfinite(f36).all():
            raise RuntimeError("Non-finite historical descriptor")
        temporary = destination.with_suffix(".npz.partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, f9=f9, f36=f36, labels=labels, channel_names=names,
                                patient_name=np.asarray(patient), record_sha256=np.asarray(sha256(record)))
        os.replace(temporary, destination)
        audit_rows.append({"record_cache": record.name, "reused": False, "channels": int(len(labels))})
    provenance = {
        "status": "PASS", "records": len(audit_rows), "windows_per_record": N_WINDOWS,
        "window_seconds": WINDOW_SECONDS, "aggregation": "arithmetic_mean_over_windows_then_edf_occurrences",
        "spectral_source_sha256": sha256(spectral_source), "comparison_source_sha256": sha256(comparison_source),
        "interictal_reference": "historical fallback: robust median/MAD over the 30 windows of each EDF-channel",
        "interictal_percentile_or_missing_indicator_included": False,
    }
    return provenance, audit_rows


def aggregate_features(cache: Path, manifest: pd.DataFrame) -> dict[str, dict]:
    expected = set(manifest.patient_name.astype(str))
    bucket: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray, int]]]] = defaultdict(lambda: defaultdict(list))
    for file in sorted(cache.glob("edf_*.npz")):
        with np.load(file, allow_pickle=False) as item:
            patient = str(item["patient_name"].item())
            if patient not in expected:
                continue
            names, labels = item["channel_names"].astype(str), item["labels"].astype(np.int8)
            f9, f36 = item["f9"].astype(np.float32), item["f36"].astype(np.float32)
        for name, label, nine, thirtysix in zip(names, labels, f9, f36):
            bucket[patient][str(name)].append((nine, thirtysix, int(label)))
    output: dict[str, dict] = {}
    for patient in expected:
        channels = bucket[patient]
        if not channels:
            raise RuntimeError("Manifest patient absent from descriptor cache")
        names = sorted(channels)
        rows9, rows36, labels = [], [], []
        for name in names:
            entries = channels[name]
            known = {label for _, _, label in entries if label >= 0}
            if len(known) > 1:
                raise RuntimeError("STOP_LABEL_CONFLICT in descriptor aggregation")
            rows9.append(np.mean([entry[0] for entry in entries], axis=0))
            rows36.append(np.mean([entry[1] for entry in entries], axis=0))
            labels.append(next(iter(known)) if known else -1)
        output[patient] = {"channels": names, "labels": np.asarray(labels, dtype=np.int8),
                           "f9": np.asarray(rows9, dtype=np.float32), "f36": np.asarray(rows36, dtype=np.float32)}
    return output


def load_baseline(private_predictions: Path, manifest: pd.DataFrame) -> dict[str, dict]:
    data = pd.read_csv(private_predictions)
    data = data.loc[data.variant.eq("baseline")].copy()
    expected = set(manifest.patient_name.astype(str))
    result: dict[str, dict] = {}
    for patient, frame in data.groupby("patient_name", sort=True):
        patient = str(patient)
        if patient not in expected:
            continue
        frame = frame.sort_values("channel_name")
        if frame.channel_name.duplicated().any():
            raise RuntimeError("Duplicate frozen baseline patient-channel prediction")
        result[patient] = {"channels": frame.channel_name.astype(str).tolist(),
                           "labels": frame.label.to_numpy(dtype=np.int8),
                           "scores": frame.pathological_score.to_numpy(dtype=np.float64)}
    if set(result) != expected:
        raise RuntimeError("Frozen baseline OOF patient coverage mismatch")
    return result


def align_patient(features: dict, baseline: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    fmap = {name: index for index, name in enumerate(features["channels"])}
    ids = []
    for name, label in zip(baseline["channels"], baseline["labels"]):
        if name not in fmap:
            raise RuntimeError("Baseline channel missing from descriptor cache")
        feature_label = int(features["labels"][fmap[name]])
        if feature_label >= 0 and int(label) != feature_label:
            raise RuntimeError("Baseline/descriptor label mismatch")
        ids.append(fmap[name])
    return (features["f9"][ids], features["f36"][ids], baseline["labels"].astype(np.int8),
            baseline["scores"].astype(np.float64), list(baseline["channels"]))


def fit_fold(data: dict[str, dict], train_patients: list[str], test_patients: list[str], dimension: int, fold: int,
             device: torch.device) -> tuple[dict[str, np.ndarray], dict]:
    key = "f9" if dimension == 9 else "f36"
    train_values = np.concatenate([data[p][key][data[p]["labels"] >= 0] for p in train_patients], axis=0).astype(np.float64)
    mean = train_values.mean(axis=0)
    std = np.clip(train_values.std(axis=0), 1e-6, None)
    set_seeds(SEED + fold * 100 + dimension)
    model = FeatureHead(dimension).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    rng = np.random.default_rng(SEED + fold * 1000 + dimension)
    model.train()
    for _ in range(EPOCHS):
        for patient in rng.permutation(train_patients).tolist():
            sample = data[patient]
            valid = sample["labels"] >= 0
            values = torch.as_tensor((sample[key][valid] - mean) / std, dtype=torch.float32, device=device)
            targets = torch.as_tensor(sample["labels"][valid], dtype=torch.float32, device=device)
            logits = model(values)
            weight = torch.where(targets > 0, POS_WEIGHT, 1.0)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=weight, reduction="mean")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    prediction: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for patient in test_patients:
            values = torch.as_tensor((data[patient][key] - mean) / std, dtype=torch.float32, device=device)
            prediction[patient] = torch.sigmoid(model(values)).detach().cpu().numpy().astype(np.float64)
    return prediction, {"fold": fold, "feature_set": f"F{dimension}", "dimension": dimension,
                        "training_patients": len(train_patients), "training_labeled_channels": int(len(train_values)),
                        "mean_sha256": hashlib.sha256(mean.astype(np.float64).tobytes()).hexdigest(),
                        "std_sha256": hashlib.sha256(std.astype(np.float64).tobytes()).hexdigest(),
                        "population_statistics_fit_on_test_fold": False, "epochs": EPOCHS}


def summarize_complement(rows: list[dict], scope: str, model: str) -> list[dict]:
    micro = defaultdict(int)
    for row in rows:
        for key in ("pair_count", "tc_correct_count", "feature_correct_count", "rescue_count", "destroy_count", "tc_wrong_count", "tc_right_count", "oracle_correct_count"):
            micro[key] += int(row[key])
    result = [{"scope": scope, "model": model, "weighting": "micro_pair_weighted", **micro, **rates(micro)}]
    finite = lambda name: np.asarray([float(row[name]) for row in rows if np.isfinite(float(row[name]))], dtype=float)
    patient = {"scope": scope, "model": model, "weighting": "patient_equal", "n_patients": len(rows),
               "pair_count": int(sum(int(row["pair_count"]) for row in rows))}
    for key in ("tc_pair_accuracy", "feature_pair_accuracy", "rescue_rate", "destroy_rate", "net_complementarity", "oracle_pair_accuracy"):
        values = finite(key)
        patient[key] = float(values.mean()) if len(values) else float("nan")
        patient[f"n_{key}"] = int(len(values))
    result.append(patient)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--feature-source", type=Path, required=True)
    parser.add_argument("--comparison-source", type=Path, required=True)
    parser.add_argument("--private-runtime", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    required = {"patient_name", "center", "fold"}
    if not required <= set(manifest):
        raise RuntimeError("Invalid frozen manifest")
    if manifest.patient_name.duplicated().any() or len(manifest) != 124 or not set(manifest.fold.unique()) == {1, 2, 3, 4, 5}:
        raise RuntimeError("STOP_PROTOCOL_INVALID manifest")
    if not args.records.is_dir() or len(list(args.records.glob("edf_*.npz"))) != 137:
        raise RuntimeError("Record cache provenance/count gate failed")
    baseline = load_baseline(args.baseline_predictions, manifest)
    baseline_rows = []
    for patient, record in baseline.items():
        valid = record["labels"] >= 0
        baseline_rows.append({"patient_name": patient, **patient_metrics(record["labels"][valid], record["scores"][valid])})
    baseline_auroc = float(np.mean([row["auroc"] for row in baseline_rows]))
    if abs(baseline_auroc - BASELINE_AUROC_EXPECTED) > BASELINE_AUROC_TOLERANCE:
        raise RuntimeError("STOP_BASELINE_OOF_REPLAY_FAILED")
    args.public_output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.public_output / "BASELINE_REPLAY_AUDIT.json", {
        "status": "PASS", "patients": len(baseline_rows), "patient_equal_auroc": baseline_auroc,
        "expected_patient_equal_auroc": BASELINE_AUROC_EXPECTED, "tolerance": BASELINE_AUROC_TOLERANCE,
        "baseline_predictions_retrained_or_replayed": False,
    })
    protocol = {
        "status": "LOCKED_BEFORE_FEATURE_CLASSIFIER_TRAINING", "seed": SEED, "manifest_sha256": sha256(args.manifest),
        "records": 137, "patients": 124, "epochs": EPOCHS, "optimizer": "AdamW", "learning_rate": 1e-4,
        "weight_decay": 1e-3, "patient_equal_weighted_bce": True, "pathological_weight": POS_WEIGHT,
        "fold_safe_population_normalization": "mean/std fit on four training folds only",
        "feature_temporal_aggregation": "mean windows then mean EDF occurrences", "fusion_betas_predeclared": [0.1, 0.25, 0.5],
        "feature_source_sha256": sha256(args.feature_source), "comparison_source_sha256": sha256(args.comparison_source),
        "baseline_oof_patient_equal_auroc": baseline_auroc,
    }
    atomic_json(args.public_output / "PROTOCOL_LOCK.json", protocol)
    audit_md = f"""# A1 feature source audit\n\n- Spectral source: `{args.feature_source.name}`; function `compute_spectral_channel_features`; SHA-256 `{sha256(args.feature_source)}`.\n- Interictal comparison source: `{args.comparison_source.name}`; function `_interictal_comparison_features`; SHA-256 `{sha256(args.comparison_source)}`.\n- Selected ABS descriptor order: `{', '.join(FEATURE_NAMES)}`.\n- Historical bands: delta 1-4 Hz, theta 4-8 Hz, beta 13-30 Hz, low-gamma 30-80 Hz, high-gamma 80-150 Hz.\n- Historical spectral implementation: Welch (`nperseg=min(samples, round(2*fs))`, 50% overlap, no detrending), integration by trapezoid, and `log1p` for all band powers.\n- Numerical constants: RMS uses `+1e-8`; entropy normalizes PSD with floor `1e-8` and uses `log(p+1e-8)/log(n_bins+1e-8)`; interictal robust reference uses median and `max(1.4826*MAD, 1e-5)`.\n- Line length is `sum(abs(diff(signal)))/duration_seconds`.\n- F36 exactly requests historical `[ABS, DELTA, ZDELTA, LOGR]`; percentile and missing-indicator extensions are explicitly disabled.  With no external interictal baseline supplied for an Omni clip, the historical function's fallback uses its 30 2-second windows as the robust reference.\n"""
    (args.public_output / "A1_FEATURE_SOURCE_AUDIT.md").write_text(audit_md, encoding="utf-8")
    cache = args.private_runtime / "feature_records"
    extraction, record_audit = extract_feature_cache(args.records, cache, args.feature_source, args.comparison_source)
    atomic_json(args.public_output / "A1_FEATURE_EXTRACTION_AUDIT.json", extraction)
    features = aggregate_features(cache, manifest)
    data: dict[str, dict] = {}
    finite_rows = []
    for patient in manifest.patient_name.astype(str):
        f9, f36, labels, tc, channels = align_patient(features[patient], baseline[patient])
        data[patient] = {"f9": f9, "f36": f36, "labels": labels, "tc": tc, "channels": channels}
        finite_rows.append({"patient": "redacted", "feature_set": "F9", "n_channels": len(labels), "n_values": int(f9.size), "n_nonfinite": int((~np.isfinite(f9)).sum())})
        finite_rows.append({"patient": "redacted", "feature_set": "F36", "n_channels": len(labels), "n_values": int(f36.size), "n_nonfinite": int((~np.isfinite(f36)).sum())})
    finite_summary = []
    for feature_set in ("F9", "F36"):
        part = [row for row in finite_rows if row["feature_set"] == feature_set]
        finite_summary.append({"feature_set": feature_set, "patients": len(part), "n_values": sum(row["n_values"] for row in part),
                               "n_nonfinite": sum(row["n_nonfinite"] for row in part), "status": "PASS"})
    write_rows(args.public_output / "A1_FEATURE_FINITE_VALUE_AUDIT.csv", finite_summary)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores: dict[str, dict[str, np.ndarray]] = {"TimeConv": {patient: data[patient]["tc"] for patient in data}, "F9": {}, "F36": {}}
    normalization = []
    for fold in range(1, 6):
        test = manifest.loc[manifest.fold.eq(fold), "patient_name"].astype(str).tolist()
        train = manifest.loc[~manifest.fold.eq(fold), "patient_name"].astype(str).tolist()
        for dimension in (9, 36):
            predicted, norm = fit_fold(data, train, test, dimension, fold, device)
            scores[f"F{dimension}"].update(predicted)
            normalization.append(norm)
    write_rows(args.public_output / "FOLD_NORMALIZATION_AUDIT.csv", normalization)
    private = args.private_runtime / "private_predictions"
    private.mkdir(parents=True, exist_ok=True)
    prediction_rows, metric_rows, pair_rows = [], [], []
    for row in manifest.itertuples(index=False):
        patient, center, fold = str(row.patient_name), str(row.center), int(row.fold)
        valid = data[patient]["labels"] >= 0
        labels = data[patient]["labels"][valid]
        for model in ("TimeConv", "F9", "F36"):
            score = scores[model][patient][valid]
            metric_rows.append({"patient_name": patient, "center": center, "fold": fold, "model": model, **patient_metrics(labels, score)})
            for channel, label, value in zip(np.asarray(data[patient]["channels"])[valid], labels, score):
                prediction_rows.append({"patient_name": patient, "center": center, "fold": fold, "channel_name": str(channel), "label": int(label), "model": model, "score": float(value)})
        for model in ("F9", "F36"):
            stat = pair_stats(labels, scores["TimeConv"][patient][valid], scores[model][patient][valid])
            pair_rows.append({"patient_name": patient, "center": center, "fold": fold, "model": model, **stat, **rates(stat)})
    write_rows(private / "OOF_PATIENT_CHANNEL_PREDICTIONS.csv", prediction_rows)
    write_rows(private / "OOF_PATIENT_METRICS.csv", metric_rows)
    write_rows(private / "PATIENT_PAIR_COMPLEMENTARITY.csv", pair_rows)
    # Public feature metrics (patient-level and center-level), never patient rows.
    public_metrics = []
    for model in ("TimeConv", "F9", "F36"):
        subset = [row for row in metric_rows if row["model"] == model]
        public_metrics.append({"scope": "overall", "center": "ALL", "model": model, "n_patients": len(subset), **means(subset, ("auroc", "ap", "mrr", "top1", "ndcg", "macro_f1_0_5", "pathological_f1_0_5", "normal_f1_0_5", "balanced_accuracy_0_5"))})
        for center in sorted({str(row.center) for row in manifest.itertuples(index=False)}):
            part = [row for row in subset if row["center"] == center]
            public_metrics.append({"scope": "center", "center": center, "model": model, "n_patients": len(part), **means(part, ("auroc", "ap", "mrr", "top1", "ndcg"))})
    write_rows(args.public_output / "FEATURE_ONLY_OOF_METRICS.csv", public_metrics)
    summaries = []
    centers = []
    for model in ("F9", "F36"):
        part = [row for row in pair_rows if row["model"] == model]
        summaries.extend(summarize_complement(part, "overall", model))
        for center in sorted({row["center"] for row in part}):
            centers.extend(summarize_complement([row for row in part if row["center"] == center], center, model))
    write_rows(args.public_output / "PAIR_COMPLEMENTARITY_SUMMARY.csv", summaries)
    write_rows(args.public_output / "CENTERWISE_COMPLEMENTARITY.csv", centers)
    # Pooled secondary diagnostic.
    pooled_rows = []
    all_labels = np.concatenate([data[p]["labels"][data[p]["labels"] >= 0] for p in data])
    all_tc = np.concatenate([scores["TimeConv"][p][data[p]["labels"] >= 0] for p in data])
    for model in ("F9", "F36"):
        all_feature = np.concatenate([scores[model][p][data[p]["labels"] >= 0] for p in data])
        stat = pair_stats(all_labels, all_tc, all_feature)
        pooled_rows.append({"model": model, "n_units": int(len(all_labels)), "timeconv_pooled_auroc": float(roc_auc_score(all_labels, all_tc)),
                            "feature_pooled_auroc": float(roc_auc_score(all_labels, all_feature)), **stat, **rates(stat)})
    write_rows(args.public_output / "POOLED_COMPLEMENTARITY.csv", pooled_rows)
    # No beta is selected.  Every predeclared numeric beta is reported.
    fusion_rows = []
    for model in ("F9", "F36"):
        for beta in (0.1, 0.25, 0.5):
            metrics = []
            for patient in manifest.patient_name.astype(str):
                valid = data[patient]["labels"] >= 0
                mixed = (1.0 - beta) * scores["TimeConv"][patient][valid] + beta * scores[model][patient][valid]
                metrics.append(patient_metrics(data[patient]["labels"][valid], mixed))
            fusion_rows.append({"feature_model": model, "beta": beta, "selection": "predeclared_no_test_optimization", "n_patients": len(metrics), **means(metrics, ("auroc", "ap", "mrr", "top1", "ndcg")), "delta_auroc_vs_timeconv": float(np.mean([row["auroc"] for row in metrics]) - baseline_auroc)})
    write_rows(args.public_output / "LEGAL_FUSION_DIAGNOSTIC.csv", fusion_rows)
    oracle_rows = []
    for model in ("F9", "F36"):
        part = [row for row in pair_rows if row["model"] == model]
        total = defaultdict(int)
        for row in part:
            for key in ("pair_count", "tc_correct_count", "feature_correct_count", "oracle_correct_count"):
                total[key] += int(row[key])
        oracle_rows.append({"model": model, "status": "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE", "pair_count": total["pair_count"],
                            "timeconv_pair_accuracy": total["tc_correct_count"] / total["pair_count"],
                            "feature_pair_accuracy": total["feature_correct_count"] / total["pair_count"],
                            "oracle_pair_accuracy": total["oracle_correct_count"] / total["pair_count"],
                            "oracle_headroom_over_timeconv": (total["oracle_correct_count"] - total["tc_correct_count"]) / total["pair_count"]})
    write_rows(args.public_output / "ORACLE_COMPLEMENTARITY_DIAGNOSTIC.csv", oracle_rows)
    # Compact report contains no patient or channel records.
    result_map = {(row["scope"], row["center"], row["model"]): row for row in public_metrics}
    tc, f9, f36 = (result_map[("overall", "ALL", name)] for name in ("TimeConv", "F9", "F36"))
    comp = {row["model"]: row for row in summaries if row["weighting"] == "patient_equal"}
    best_fusion = max(fusion_rows, key=lambda row: row["delta_auroc_vs_timeconv"])
    # A fixed beta grid is an OOF diagnostic, not a nested train-side selection.
    # The formal plugin gate therefore requires the rescue threshold as well as a
    # separately selectable fusion gain; this audit cannot promote a grid maximum.
    best_rescue = max(comp["F9"]["rescue_rate"], comp["F36"]["rescue_rate"])
    terminal = "A1_FEATURES_NOT_COMPLEMENTARY" if best_rescue <= 0.50 else "A1_FEATURES_WEAKLY_COMPLEMENTARY_BUT_PLUGIN_GATE_NOT_MET"
    report = f"""# A1 physiological-feature complementarity audit\n\n## Result\n\nTerminal: `{terminal}`.  This is a fixed 124-patient, seed-42, five-fold OOF audit.  TimeConv was not retrained.  F9/F36 use only the historical A1 physiology descriptors and simple fold-safe 2-layer heads.\n\n| Model | Patient-equal AUROC | AP | MRR | Top1 | NDCG |\n|---|---:|---:|---:|---:|---:|\n| TimeConv | {tc['auroc_mean']:.4f} | {tc['ap_mean']:.4f} | {tc['mrr_mean']:.4f} | {tc['top1_mean']:.4f} | {tc['ndcg_mean']:.4f} |\n| F9 | {f9['auroc_mean']:.4f} | {f9['ap_mean']:.4f} | {f9['mrr_mean']:.4f} | {f9['top1_mean']:.4f} | {f9['ndcg_mean']:.4f} |\n| F36 | {f36['auroc_mean']:.4f} | {f36['ap_mean']:.4f} | {f36['mrr_mean']:.4f} | {f36['top1_mean']:.4f} | {f36['ndcg_mean']:.4f} |\n\nThe frozen TimeConv replay is {baseline_auroc:.4f}, matching the prior 0.7622 audit.  F9 rescue/destroy are {comp['F9']['rescue_rate']:.4f}/{comp['F9']['destroy_rate']:.4f}; F36 rescue/destroy are {comp['F36']['rescue_rate']:.4f}/{comp['F36']['destroy_rate']:.4f}.  Patient-equal net complementarity is {comp['F9']['net_complementarity']:.4f} for F9 and {comp['F36']['net_complementarity']:.4f} for F36.\n\nAll numeric beta diagnostics were predeclared (0.1, 0.25, 0.5); none was selected on held-out labels.  The largest reported diagnostic delta is {best_fusion['delta_auroc_vs_timeconv']:+.4f} ({best_fusion['feature_model']}, beta={best_fusion['beta']}).  Oracle results are diagnostic only and not deployable.\n\nThis does not repeat PC-CNN: that earlier result mixed a weaker RawCNN, physiology, patient context, and later encoder/classifier updates.  It does not repeat A1-TF, which tested an A1 plus learned time-frequency residual direction.  Here, patient context is absent by design, following the preceding matched TimeConv context result (0.7622 to 0.7630).\n\nNo raw signals, feature cache, patient/channel predictions, checkpoints, or runtime logs are public.\n"""
    (args.public_output / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    atomic_json(args.public_output / "AUDIT_STATUS.json", {"status": "COMPLETE", "terminal": terminal, "patients": 124, "baseline_replay_auroc": baseline_auroc, "feature_cache_records": extraction["records"], "device": str(device), "test_labels_used_for_model_selection": False})


if __name__ == "__main__":
    main()
