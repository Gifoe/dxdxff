"""Ictal-only PRCD-EZ development runner.

There is intentionally no outer-test or Omni argument.  Private caches and
predictions stay below --runtime; only the final compact aggregates are copied
to the experiment directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from prcd_core import (BIAS_QUANTILES, GROUPS, KERNELS, POOLED_DIM, RECORD_DIM,
                       SEED, dictionary_feature_name, dictionary_record,
                       fine_spectrum, fit_biases, fixed_band_features,
                       make_dictionary, patient_coordinates, pool_records,
                       stable_seed)

HERE = Path(__file__).resolve()
EXPERIMENT = HERE.parents[1]
A1_FOLDS = EXPERIMENT / "HISTORICAL_A1_FOLD_REFERENCE.csv"
THRESHOLDS = np.linspace(.05, .95, 19)
HISTORICAL = {"auroc": .746382, "ap": .576743, "macro_f1": .620810,
              "mrr": .740038, "top1": .654771}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def patient_hash(patient: str) -> str:
    return hashlib.sha256(patient.encode()).hexdigest()[:20]


def read_manifest(path: Path) -> dict[int, dict[str, list[str]]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    folds = {}
    for fold in range(1, 6):
        grouped = defaultdict(list)
        for row in rows:
            if int(row["outer_fold"]) == fold:
                grouped[row["split_role"]].append(str(row["subject_id"]))
        if set(grouped) != {"fit", "validation", "test"} or len(set(sum(grouped.values(), []))) != 80:
            raise RuntimeError(f"invalid frozen fold {fold}")
        folds[fold] = {key: sorted(value) for key, value in grouped.items()}
    return folds


def _prepare_one(source: str, destination: str, protocol_sha: str) -> dict:
    source_path, destination_path = Path(source), Path(destination)
    started = time.perf_counter()
    if destination_path.exists():
        with np.load(destination_path, allow_pickle=False) as saved:
            if str(saved["protocol_sha256"]) != protocol_sha:
                raise RuntimeError("base token protocol mismatch")
            return {"status": "REUSED", "patient_hash": destination_path.stem, "seconds": 0.0,
                    "records": int(saved["values"].shape[0]), "channels": int(saved["values"].shape[1])}
    with np.load(source_path, allow_pickle=False) as sample:
        patient = str(sample["patient"]); fs = float(sample["sampling_rate_hz"])
        descriptors = sample["descriptors"].astype(np.float32)
        windows = sample["window_mask"].astype(bool)
        presence = sample["channel_present"].astype(bool)
        raw = sample["waveforms"]
        starts, lengths = sample["valid_start"], sample["valid_samples"]
        labels = sample["labels"].astype(np.int8)
        names = sample["channel_names"].astype("U")
        if fs != 250 or descriptors.shape[-2:] != (59, 36):
            raise RuntimeError("validated A1 cache contract changed")
        values, masks, biomarker_records = [], [], []
        for record in range(len(lengths)):
            begin, length = int(starts[record]), int(lengths[record])
            wave = raw[record, :, begin:begin + length].astype(np.float32)
            window_mask = windows[record] & presence[record, :, None]
            spectrum, spectrum_mask = fine_spectrum(wave, fs, window_mask)
            descriptor_mask = np.broadcast_to(window_mask[..., None], descriptors[record].shape)
            current = np.concatenate((np.where(descriptor_mask, descriptors[record], 0), spectrum), axis=-1)
            current_mask = np.concatenate((descriptor_mask, spectrum_mask), axis=-1)
            values.append(current.astype(np.float32)); masks.append(current_mask)
            burden, pac, aec = fixed_band_features(wave, fs, presence[record], window_mask)
            biomarker_records.append(np.concatenate((burden, pac, aec), axis=1))
        values = np.stack(values); masks = np.stack(masks); biomarker_records = np.stack(biomarker_records)
        biomarkers = np.zeros((len(labels), 30), dtype=np.float32)
        valid_channels = presence.any(axis=0)
        for channel in np.flatnonzero(valid_channels):
            biomarkers[channel] = biomarker_records[presence[:, channel], channel].mean(axis=0)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, protocol_sha256=np.asarray(protocol_sha), patient=np.asarray(patient),
                            values=values, mask=masks, presence=presence, labels=labels,
                            channel_names=names, biomarkers=biomarkers)
        temporary.replace(destination_path)
    return {"status": "CREATED", "patient_hash": destination_path.stem,
            "seconds": time.perf_counter() - started, "records": len(values), "channels": len(labels)}


def prepare_all(args) -> None:
    protocol_sha = digest(args.protocol)
    sources = sorted(args.ictal_cache.glob("patient_*.npz"))
    if len(sources) != 80:
        raise RuntimeError("expected exactly 80 frozen Ictal patient caches")
    destination = args.runtime / "base_tokens"
    jobs = [(str(path), str(destination / path.name), protocol_sha) for path in sources]
    started = time.perf_counter(); rows = []
    workers = max(1, min(args.workers, 8))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_prepare_one, *job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            row = future.result(); rows.append(row)
            print(json.dumps({"phase": "base_tokens", "done": done, "total": len(jobs), **row}), flush=True)
    atomic_json(args.runtime / "BASE_TOKEN_STATUS.json", {
        "status": "COMPLETE", "patients": 80, "protocol_sha256": protocol_sha,
        "wall_seconds": time.perf_counter() - started, "workers": workers,
        "sum_worker_seconds": sum(row["seconds"] for row in rows), "test_accessed": False})


def speed_gate(args) -> None:
    protocol_sha = digest(args.protocol)
    source = sorted(args.ictal_cache.glob("patient_*.npz"))[0]
    destination = args.runtime / "speed_gate" / source.name
    if destination.exists():
        destination.unlink()
    prep = _prepare_one(str(source), str(destination), protocol_sha)
    with np.load(destination, allow_pickle=False) as item:
        values, mask, presence = item["values"], item["mask"], item["presence"]
    # A real-patient fold-like pass uses fit-only style scaling and dictionary bias fitting.
    median = np.nanmedian(np.where(mask, values, np.nan), axis=(0, 1, 2))
    q25 = np.nanpercentile(np.where(mask, values, np.nan), 25, axis=(0, 1, 2))
    q75 = np.nanpercentile(np.where(mask, values, np.nan), 75, axis=(0, 1, 2))
    scaled = np.clip((values - median) / np.maximum(q75 - q25, 1e-5), -8, 8).astype(np.float32)
    scaled[~mask] = 0
    sequences = scaled.reshape(-1, 59, 52)[presence.reshape(-1)]
    sequence_masks = mask.reshape(-1, 59, 52)[presence.reshape(-1)]
    dictionary = make_dictionary(); started = time.perf_counter()
    sample = sequences[:min(64, len(sequences))]; sample_mask = sequence_masks[:len(sample)]
    biases = fit_biases(sample, sample_mask, dictionary)
    record_features = []
    offset = 0
    for record in range(len(values)):
        count = len(values[record]); current = dictionary_record(scaled[record], mask[record], dictionary, biases)
        record_features.append(current); offset += count
    elapsed = time.perf_counter() - started
    payload = {"status": "PASS" if prep["seconds"] + elapsed <= args.max_seconds else "FAIL",
               "real_patient": True, "records": prep["records"], "channels": prep["channels"],
               "base_feature_seconds": prep["seconds"], "dictionary_seconds": elapsed,
               "total_seconds": prep["seconds"] + elapsed, "maximum_allowed_seconds": args.max_seconds,
               "test_accessed": False}
    atomic_json(args.runtime / "SPEED_GATE.json", payload)
    print(json.dumps(payload, indent=2), flush=True)
    if payload["status"] != "PASS":
        raise RuntimeError("real-patient speed gate failed; full folds are blocked")


def token_path(runtime: Path, patient: str) -> Path:
    path = runtime / "base_tokens" / f"patient_{patient_hash(patient)}.npz"
    if not path.is_file():
        raise RuntimeError("base token missing")
    return path


def fit_scaler(runtime: Path, patients: list[str], protocol_sha: str, output: Path) -> tuple[np.ndarray, np.ndarray]:
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload["protocol_sha256"] != protocol_sha or payload["fit_patient_hash"] != hashlib.sha256("|".join(patients).encode()).hexdigest():
            raise RuntimeError("scaler resume provenance mismatch")
        return np.asarray(payload["median"], dtype=np.float32), np.asarray(payload["iqr"], dtype=np.float32)
    pieces = []
    for patient in patients:
        with np.load(token_path(runtime, patient), allow_pickle=False) as item:
            values, mask, presence = item["values"], item["mask"], item["presence"]
            rows = values.reshape(-1, 52)[mask.reshape(-1, 52).any(axis=1)]
            row_mask = mask.reshape(-1, 52)[mask.reshape(-1, 52).any(axis=1)]
            pieces.append(np.where(row_mask, rows, np.nan).astype(np.float32))
    all_values = np.concatenate(pieces)
    median = np.nanmedian(all_values, axis=0).astype(np.float32)
    q25, q75 = np.nanpercentile(all_values, (25, 75), axis=0)
    iqr = np.maximum(q75 - q25, 1e-5).astype(np.float32)
    atomic_json(output, {"protocol_sha256": protocol_sha,
                         "fit_patient_hash": hashlib.sha256("|".join(patients).encode()).hexdigest(),
                         "fit_patients": len(patients), "median": median.tolist(), "iqr": iqr.tolist(),
                         "validation_or_test_used": False})
    return median, iqr


def normalized_tokens(runtime: Path, patient: str, median: np.ndarray, iqr: np.ndarray):
    with np.load(token_path(runtime, patient), allow_pickle=False) as item:
        values, mask = item["values"].astype(np.float32), item["mask"].astype(bool)
        scaled = np.clip((values - median) / iqr, -8, 8).astype(np.float32); scaled[~mask] = 0
        return scaled, mask, item["presence"].astype(bool), item["labels"].astype(np.int8), \
            item["channel_names"].astype("U"), item["biomarkers"].astype(np.float32)


def fit_fold_biases(runtime: Path, patients: list[str], median: np.ndarray, iqr: np.ndarray,
                    dictionary, output: Path) -> np.ndarray:
    if output.exists():
        return np.load(output, allow_pickle=False)["biases"].astype(np.float32)
    candidates = []
    for patient in patients:
        values, mask, presence, *_ = normalized_tokens(runtime, patient, median, iqr)
        for record, channel in zip(*np.nonzero(presence)):
            key = stable_seed(SEED, "bias_sample", patient, record, channel)
            candidates.append((key, values[record, channel], mask[record, channel]))
    chosen = sorted(candidates, key=lambda row: row[0])[:512]
    sample = np.stack([row[1] for row in chosen]); sample_mask = np.stack([row[2] for row in chosen])
    biases = fit_biases(sample, sample_mask, dictionary)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, biases=biases, sample_sequences=np.asarray(len(chosen)),
                        fit_patient_hash=np.asarray(hashlib.sha256("|".join(patients).encode()).hexdigest()))
    temporary.replace(output)
    return biases


def extract_fold_patient(runtime: Path, fold: int, patient: str, median: np.ndarray, iqr: np.ndarray,
                         biases: np.ndarray, dictionary, protocol_sha: str) -> Path:
    output = runtime / "folds" / f"fold{fold}" / "features" / f"patient_{patient_hash(patient)}.npz"
    if output.exists():
        return output
    values, mask, presence, labels, names, biomarkers = normalized_tokens(runtime, patient, median, iqr)
    record_features = np.stack([dictionary_record(values[r], mask[r], dictionary, biases) for r in range(len(values))])
    absolute = pool_records(record_features, presence)
    valid = presence.any(axis=0) & (labels >= 0)
    coordinate = patient_coordinates(absolute, valid)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, protocol_sha256=np.asarray(protocol_sha), patient=np.asarray(patient),
                        absolute=absolute, coordinate=coordinate, biomarkers=biomarkers,
                        valid=valid, labels=labels, channel_names=names)
    temporary.replace(output)
    return output


def load_fold_patient(runtime: Path, fold: int, patient: str) -> dict:
    path = runtime / "folds" / f"fold{fold}" / "features" / f"patient_{patient_hash(patient)}.npz"
    with np.load(path, allow_pickle=False) as item:
        return {key: item[key].copy() for key in ("absolute", "coordinate", "biomarkers", "valid", "labels", "channel_names")}


def balanced_relevance(patients: dict[str, dict], key: str) -> np.ndarray:
    contrasts = []
    for item in patients.values():
        x, y, valid = item[key], item["labels"], item["valid"]
        use = valid & (y >= 0)
        if (y[use] == 1).any() and (y[use] == 0).any():
            contrasts.append(x[use & (y == 1)].mean(0) - x[use & (y == 0)].mean(0))
    delta = np.stack(contrasts).astype(np.float32)
    return (np.abs(delta.mean(0)) / np.maximum(delta.std(0, ddof=1) / np.sqrt(len(delta)), 1e-6)).astype(np.float32)


def select_mrmr(patients: dict[str, dict], key: str, k: int, variant: str) -> tuple[np.ndarray, np.ndarray]:
    relevance = balanced_relevance(patients, key)
    pool_size = min(1024, len(relevance))
    candidate = np.argsort(-relevance, kind="stable")[:pool_size]
    sampled = []
    for patient, item in sorted(patients.items()):
        y, valid = item["labels"], item["valid"]
        ids = []
        for label in (0, 1):
            choices = np.flatnonzero(valid & (y == label))
            choices = sorted(choices, key=lambda index: stable_seed(SEED, "redundancy", patient, int(index)))[:8]
            ids.extend(choices)
        if ids:
            sampled.append(item[key][ids][:, candidate])
    x = np.concatenate(sampled).astype(np.float32)
    x -= x.mean(0); x /= np.maximum(x.std(0), 1e-6)
    correlation = np.abs((x.T @ x) / max(len(x) - 1, 1)).astype(np.float32)
    rel = relevance[candidate]; rel = rel / max(float(rel.max()), 1e-8)
    selected_local = [int(np.argmax(rel))]
    available = np.ones(pool_size, dtype=bool); available[selected_local[0]] = False
    redundancy = correlation[:, selected_local[0]].copy()
    for count in range(1, k):
        score = rel - redundancy / count
        score[~available] = -np.inf
        chosen = int(np.argmax(score)); selected_local.append(chosen); available[chosen] = False
        redundancy += correlation[:, chosen]
    selected = candidate[np.asarray(selected_local)]
    return selected.astype(np.int32), relevance[selected]


def training_matrix(patients: dict[str, dict], key: str, selected: np.ndarray,
                    include_biomarkers: bool):
    xs, ys, weights, owner = [], [], [], []
    for patient, item in sorted(patients.items()):
        use = np.flatnonzero(item["valid"] & (item["labels"] >= 0))
        current = item[key][use][:, selected]
        if include_biomarkers:
            current = np.concatenate((current, item["biomarkers"][use]), axis=1)
        labels = item["labels"][use].astype(np.float32)
        xs.append(current); ys.append(labels); owner.extend([patient] * len(use))
        weights.append(np.full(len(use), 1.0 / max(len(use), 1), dtype=np.float32))
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)
    pos_weight = float(np.clip(math.sqrt((y == 0).sum() / max((y == 1).sum(), 1)), 1, 4))
    w *= np.where(y > 0, pos_weight, 1.0); w /= w.mean()
    return x.astype(np.float32), y.astype(np.float32), w.astype(np.float32), owner, pos_weight


def validation_matrix(item: dict, key: str, selected: np.ndarray, include_biomarkers: bool):
    use = np.flatnonzero(item["valid"] & (item["labels"] >= 0))
    x = item[key][use][:, selected]
    if include_biomarkers:
        x = np.concatenate((x, item["biomarkers"][use]), axis=1)
    return x.astype(np.float32), item["labels"][use].astype(np.int8), item["channel_names"][use]


class TinyMLP(torch.nn.Module):
    def __init__(self, dimensions: int):
        super().__init__()
        self.network = torch.nn.Sequential(torch.nn.Linear(dimensions, 32), torch.nn.GELU(),
                                           torch.nn.Dropout(.1), torch.nn.Linear(32, 1))
    def forward(self, x):
        return self.network(x).squeeze(-1)


def _validation_predictions(model, scaler, validation: dict[str, dict], key: str,
                            selected: np.ndarray, include_biomarkers: bool, torch_model: bool):
    result = {}
    if torch_model:
        model.eval()
    with torch.inference_mode() if torch_model else _nullcontext():
        for patient, item in validation.items():
            x, y, names = validation_matrix(item, key, selected, include_biomarkers)
            x = scaler.transform(x).astype(np.float32)
            if torch_model:
                score = torch.sigmoid(model(torch.from_numpy(x))).numpy()
            else:
                score = model.predict_proba(x)[:, 1]
            result[patient] = {"labels": y.tolist(), "scores": score.astype(float).tolist(),
                               "channel_names": names.tolist()}
    return result


class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *args): return False


def train_candidates(fold: int, variant: str, fit: dict[str, dict], validation: dict[str, dict],
                     selected_by_k: dict[int, np.ndarray], output: Path) -> dict:
    key = "absolute" if variant == "B1_CD" else "coordinate"
    include_biomarkers = variant == "B2_PRCD"
    snapshots, configs, timing = {}, [], []
    for k, selected in selected_by_k.items():
        x, y, weights, owners, positive_weight = training_matrix(fit, key, selected, include_biomarkers)
        scaler = StandardScaler().fit(x, sample_weight=weights)
        transformed = scaler.transform(x).astype(np.float32)
        # H0: the only six predeclared elastic-net settings.
        for C in (.1, 1.0, 10.0):
            for l1 in (0.0, .5):
                name = f"K{k}_H0_C{C:g}_L1{l1:g}"
                started = time.perf_counter()
                model = LogisticRegression(penalty="elasticnet", solver="saga", C=C, l1_ratio=l1,
                                           max_iter=400, tol=1e-3, random_state=SEED, n_jobs=1)
                model.fit(transformed, y.astype(int), sample_weight=weights)
                predictions = _validation_predictions(model, scaler, validation, key, selected,
                                                       include_biomarkers, False)
                snapshots[name] = [predictions]
                parameters = int(model.coef_.size + 1)
                configs.append({"name": name, "variant": variant, "fold": fold, "k": k, "head": "H0",
                                "C": C, "l1_ratio": l1, "parameters": parameters,
                                "positive_weight": positive_weight, "epochs": 1})
                timing.append({"fold": fold, "variant": variant, "config": name,
                               "phase": "classifier_training", "seconds": time.perf_counter() - started})
        # H1: train all 40 epochs once; target-excluded patience is applied in finalization.
        name = f"K{k}_H1"
        started = time.perf_counter()
        torch.manual_seed(SEED + fold * 100 + (1 if variant == "B1_CD" else 2) + k)
        model = TinyMLP(transformed.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        tx, ty, tw = map(torch.from_numpy, (transformed, y, weights))
        epoch_predictions = []
        for epoch in range(1, 41):
            model.train(); optimizer.zero_grad(set_to_none=True)
            logits = model(tx)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, ty, weight=tw)
            loss.backward(); optimizer.step()
            epoch_predictions.append(_validation_predictions(model, scaler, validation, key, selected,
                                                              include_biomarkers, True))
        snapshots[name] = epoch_predictions
        parameters = int(sum(parameter.numel() for parameter in model.parameters()))
        if parameters >= 10000:
            raise RuntimeError("tiny MLP exceeded locked 10K budget")
        configs.append({"name": name, "variant": variant, "fold": fold, "k": k, "head": "H1",
                        "C": "", "l1_ratio": "", "parameters": parameters,
                        "positive_weight": positive_weight, "epochs": 40})
        timing.append({"fold": fold, "variant": variant, "config": name,
                       "phase": "classifier_training", "seconds": time.perf_counter() - started})
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / f"{variant}_PREDICTIONS_PRIVATE.json", snapshots)
    write_csv(output / f"{variant}_CONFIGS.csv", configs)
    return {"configs": configs, "timing": timing}


def run_fold(args) -> None:
    if args.fold not in range(1, 6):
        raise RuntimeError("fold must be 1..5")
    speed = json.loads((args.runtime / "SPEED_GATE.json").read_text(encoding="utf-8"))
    base = json.loads((args.runtime / "BASE_TOKEN_STATUS.json").read_text(encoding="utf-8"))
    if speed["status"] != "PASS" or base["status"] != "COMPLETE":
        raise RuntimeError("speed/base-token gate not complete")
    protocol_sha = digest(args.protocol)
    folds = read_manifest(args.manifest); split = folds[args.fold]
    if set(split["fit"]) & set(split["validation"]):
        raise RuntimeError("fit/validation overlap")
    work = args.runtime / "folds" / f"fold{args.fold}"
    marker = work / "FOLD_STATUS.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")).get("status") == "COMPLETE":
        print(json.dumps({"status": "REUSED_COMPLETE", "fold": args.fold}), flush=True); return
    timings = []; phase = time.perf_counter()
    median, iqr = fit_scaler(args.runtime, split["fit"], protocol_sha, work / "SCALER.json")
    timings.append({"fold": args.fold, "phase": "scaler", "seconds": time.perf_counter() - phase})
    dictionary = make_dictionary(); phase = time.perf_counter()
    biases = fit_fold_biases(args.runtime, split["fit"], median, iqr, dictionary, work / "BIASES.npz")
    timings.append({"fold": args.fold, "phase": "dictionary_bias", "seconds": time.perf_counter() - phase})
    phase = time.perf_counter()
    for done, patient in enumerate(split["fit"] + split["validation"], 1):
        extract_fold_patient(args.runtime, args.fold, patient, median, iqr, biases, dictionary, protocol_sha)
        if done % 10 == 0:
            print(json.dumps({"phase": "fold_features", "fold": args.fold, "done": done,
                              "total": len(split["fit"]) + len(split["validation"])}), flush=True)
    timings.append({"fold": args.fold, "phase": "dictionary_extraction", "seconds": time.perf_counter() - phase})
    fit = {patient: load_fold_patient(args.runtime, args.fold, patient) for patient in split["fit"]}
    validation = {patient: load_fold_patient(args.runtime, args.fold, patient) for patient in split["validation"]}
    selection_rows = []
    selections = {}
    phase = time.perf_counter()
    for variant, key in (("B1_CD", "absolute"), ("B2_PRCD", "coordinate")):
        selections[variant] = {}
        for k in (128, 256):
            selected, relevance = select_mrmr(fit, key, k, variant)
            selections[variant][k] = selected
            for rank, (index, value) in enumerate(zip(selected, relevance), 1):
                coordinate = "A" if key == "absolute" else ("A", "D", "R")[int(index) // POOLED_DIM]
                base_index = int(index) % POOLED_DIM
                selection_rows.append({"fold": args.fold, "variant": variant, "k": k, "rank": rank,
                                       "feature_index": int(index), "relevance": float(value),
                                       **dictionary_feature_name(base_index, coordinate)})
    timings.append({"fold": args.fold, "phase": "feature_selection", "seconds": time.perf_counter() - phase})
    write_csv(work / "FEATURE_SELECTION_PRIVATE.csv", selection_rows)
    training = []
    for variant in ("B1_CD", "B2_PRCD"):
        result = train_candidates(args.fold, variant, fit, validation, selections[variant], work)
        training.extend(result["timing"])
    timings.extend(training)
    write_csv(work / "RUNTIME.csv", timings)
    atomic_json(marker, {"status": "COMPLETE", "fold": args.fold, "fit_patients": len(split["fit"]),
                         "validation_patients": len(split["validation"]), "test_patients_opened": 0,
                         "protocol_sha256": protocol_sha, "test_accessed": False})
    print(json.dumps({"status": "FOLD_COMPLETE", "fold": args.fold}, indent=2), flush=True)


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if len(np.unique(labels)) != 2:
        return {key: float("nan") for key in ("auroc", "ap", "mrr", "top1")}
    order = np.argsort(-scores, kind="stable")
    sorted_labels, sorted_scores = labels[order], scores[order]
    positive_rank = int(np.flatnonzero(sorted_labels == 1)[0])
    # Exact empirical AUROC with average ranks for score ties.
    ascending = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    position = 0
    while position < len(scores):
        end = position + 1
        while end < len(scores) and scores[ascending[end]] == scores[ascending[position]]:
            end += 1
        ranks[ascending[position:end]] = (position + 1 + end) / 2.0
        position = end
    npos, nneg = int((labels == 1).sum()), int((labels == 0).sum())
    auroc = (ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    # Exact non-interpolated AP: each tied threshold contributes its positive
    # recall increment times cumulative precision at that threshold.
    tp = fp = 0; ap = 0.0; position = 0
    while position < len(scores):
        end = position + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[position]:
            end += 1
        positive_group = int((sorted_labels[position:end] == 1).sum())
        tp += positive_group; fp += (end - position - positive_group)
        ap += (positive_group / npos) * (tp / (tp + fp))
        position = end
    return {"auroc": float(auroc),
            "ap": float(ap),
            "mrr": float(1 / (positive_rank + 1)), "top1": float(labels[order[0]])}


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = scores >= threshold
    # Historical fixed-query cells can contain only one true class.  Reuse
    # sklearn's inferred-label semantics exactly for those cells; the direct
    # two-class formula below is equivalent when both classes are present.
    if len(np.unique(labels)) < 2:
        return {"macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
                "ez_f1": float(f1_score(labels, prediction, pos_label=1, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(labels, prediction))}
    tp = int(((labels == 1) & prediction).sum()); fn = int(((labels == 1) & ~prediction).sum())
    tn = int(((labels == 0) & ~prediction).sum()); fp = int(((labels == 0) & prediction).sum())
    ez_denom, nez_denom = 2 * tp + fp + fn, 2 * tn + fp + fn
    ez_f1 = 2 * tp / ez_denom if ez_denom else 0.0
    nez_f1 = 2 * tn / nez_denom if nez_denom else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {"macro_f1": float((ez_f1 + nez_f1) / 2), "ez_f1": float(ez_f1),
            "balanced_accuracy": float((sensitivity + specificity) / 2)}


def ranking_table(snapshots: list[dict], patients: list[str]) -> np.ndarray:
    table = np.empty((len(snapshots), len(patients), 3), dtype=np.float64)
    for epoch, snapshot in enumerate(snapshots):
        for index, patient in enumerate(patients):
            row = ranking_metrics(np.asarray(snapshot[patient]["labels"]), np.asarray(snapshot[patient]["scores"]))
            table[epoch, index] = (row["auroc"], row["ap"], row["mrr"])
    return table


def choose_epoch_excluding(table: np.ndarray, target_index: int) -> int:
    if len(table) == 1:
        return 0
    best_key, best_epoch, stale = None, 0, 0
    for epoch in range(len(table)):
        keep = np.arange(table.shape[1]) != target_index
        key = tuple(round(float(value), 12) for value in np.nanmean(table[epoch, keep], axis=0))
        if best_key is None or key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
        else:
            stale += 1
            if stale >= 6:
                break
    return best_epoch


def threshold_table(snapshot: dict, patients: list[str]) -> np.ndarray:
    table = np.empty((len(patients), len(THRESHOLDS), 3), dtype=np.float64)
    for patient_index, patient in enumerate(patients):
        labels = np.asarray(snapshot[patient]["labels"], dtype=np.int8)
        scores = np.asarray(snapshot[patient]["scores"], dtype=np.float64)
        prediction = scores[:, None] >= THRESHOLDS[None, :]
        positive = labels[:, None] == 1
        negative = ~positive
        tp = (prediction & positive).sum(0); fp = (prediction & negative).sum(0)
        fn = ((~prediction) & positive).sum(0); tn = ((~prediction) & negative).sum(0)
        ez_f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) > 0)
        nez_f1 = np.divide(2 * tn, 2 * tn + fp + fn, out=np.zeros_like(tn, dtype=float), where=(2 * tn + fp + fn) > 0)
        sensitivity = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
        specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=(tn + fp) > 0)
        table[patient_index] = np.stack(((ez_f1 + nez_f1) / 2, ez_f1,
                                         (sensitivity + specificity) / 2), axis=1)
    return table


def choose_threshold_excluding(table: np.ndarray, target_index: int) -> float:
    mean = (table.sum(axis=0) - table[target_index]) / max(len(table) - 1, 1)
    candidates = []
    for index, threshold in enumerate(THRESHOLDS):
        key = tuple(round(float(value), 12) for value in mean[index])
        key += (-round(abs(float(threshold) - .5), 6), -index)
        candidates.append((key, float(threshold)))
    return max(candidates)[1]


def fixed_query(n_channels: int, fold: int, patient: str, repetition: int) -> np.ndarray:
    permutation = np.random.default_rng(stable_seed(SEED, fold, patient, repetition, "split")).permutation(n_channels)
    return permutation[n_channels // 2:]


def query_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    result = threshold_metrics(labels, scores, threshold)
    result.update(ranking_metrics(labels, scores))
    return result


def evaluate_config(runtime: Path, variant: str, config: str) -> tuple[list[dict], list[dict]]:
    cells, selections = [], []
    for fold in range(1, 6):
        work = runtime / "folds" / f"fold{fold}"
        snapshots = json.loads((work / f"{variant}_PREDICTIONS_PRIVATE.json").read_text(encoding="utf-8"))[config]
        patients = sorted(snapshots[0])
        ranks = ranking_table(snapshots, patients)
        threshold_cache = {}
        for patient_index, patient in enumerate(patients):
            epoch = choose_epoch_excluding(ranks, patient_index)
            snapshot = snapshots[epoch]
            if epoch not in threshold_cache:
                threshold_cache[epoch] = threshold_table(snapshot, patients)
            threshold = choose_threshold_excluding(threshold_cache[epoch], patient_index)
            labels = np.asarray(snapshot[patient]["labels"], dtype=np.int8)
            scores = np.asarray(snapshot[patient]["scores"], dtype=np.float64)
            selections.append({"fold": fold, "patient_private": patient, "selected_epoch": epoch + 1,
                               "selected_threshold": threshold})
            for repetition in range(20):
                query = fixed_query(len(labels), fold, patient, repetition)
                cells.append({"fold": fold, "patient_private": patient, "repetition": repetition,
                              **query_metrics(labels[query], scores[query], threshold)})
    return cells, selections


def aggregate(cells: list[dict]) -> dict[str, float]:
    return {name: float(np.nanmean([row[name] for row in cells]))
            for name in ("auroc", "ap", "macro_f1", "ez_f1", "balanced_accuracy", "mrr", "top1")}


def finalize(args) -> None:
    protocol_sha = digest(args.protocol)
    for fold in range(1, 6):
        status = json.loads((args.runtime / "folds" / f"fold{fold}" / "FOLD_STATUS.json").read_text(encoding="utf-8"))
        if status.get("status") != "COMPLETE" or status.get("protocol_sha256") != protocol_sha:
            raise RuntimeError("five completed protocol-matched folds required")
    all_configs = defaultdict(list)
    config_meta = {}
    for fold in range(1, 6):
        work = args.runtime / "folds" / f"fold{fold}"
        for variant in ("B1_CD", "B2_PRCD"):
            with (work / f"{variant}_CONFIGS.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                all_configs[variant].append(row["name"]); config_meta[(variant, row["name"])] = row
    candidates, private = [], {}
    for variant in ("B1_CD", "B2_PRCD"):
        common = sorted(name for name, count in __import__("collections").Counter(all_configs[variant]).items() if count == 5)
        for config in common:
            cells, selections = evaluate_config(args.runtime, variant, config)
            metric = aggregate(cells); meta = config_meta[(variant, config)]
            row = {"variant": variant, "config": config, "head": meta["head"], "k": int(meta["k"]),
                   "parameters": int(meta["parameters"]), **metric}
            candidates.append(row); private[(variant, config)] = (cells, selections)
    selected = {}
    for variant in ("B1_CD", "B2_PRCD"):
        rows = [row for row in candidates if row["variant"] == variant]
        selected[variant] = max(rows, key=lambda row: (round(row["auroc"], 12), round(row["ap"], 12),
                                                       round(row["mrr"], 12), -row["parameters"]))
    private_root = args.runtime / "private_final"
    for variant, row in selected.items():
        cells, selections = private[(variant, row["config"])]
        write_csv(private_root / f"{variant}_QUERY_CELLS_PRIVATE.csv", cells)
        write_csv(private_root / f"{variant}_SELECTION_PRIVATE.csv", selections)
    # Compact fold metrics for selected architectures.
    fold_rows = []
    a1 = {}
    with A1_FOLDS.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream): a1[int(row["fold"])] = float(row["patient_ez_auroc"])
    for variant, selection in selected.items():
        cells = private[(variant, selection["config"])][0]
        for fold in range(1, 6):
            metric = aggregate([row for row in cells if row["fold"] == fold])
            fold_rows.append({"variant": variant, "fold": fold, **metric,
                              "historical_A1_fold_auroc": a1[fold],
                              "auroc_delta_vs_A1_fold": metric["auroc"] - a1[fold]})
    b1, b2 = selected["B1_CD"], selected["B2_PRCD"]
    positive_folds = sum(row["auroc_delta_vs_A1_fold"] > 0 for row in fold_rows if row["variant"] == "B2_PRCD")
    checks = {"auroc_ge_0_751382": b2["auroc"] >= .751382,
              "ap_ge_0_576743": b2["ap"] >= .576743,
              "positive_folds_ge_3": positive_folds >= 3,
              "mrr_ge_0_730038": b2["mrr"] >= .730038,
              "macro_f1_ge_0_605810": b2["macro_f1"] >= .605810}
    passed = all(checks.values())
    terminal = ("ICTAL_STRONG_PASS" if b2["auroc"] >= .77 and b2["ap"] >= .585 else "ICTAL_MINIMUM_PASS") if passed else "STOP_ICTAL_GATE_FAILED"
    gate = {"status": terminal, "pass": passed, "checks": checks, "positive_folds": positive_folds,
            "selected_B1": b1, "selected_B2": b2, "historical_A1": HISTORICAL,
            "omni_authorized": passed, "outer_test_accessed": False, "test_used_for_tuning": False}
    # Public feature selection audit and runtime.
    feature_rows, runtime_rows = [], []
    for fold in range(1, 6):
        work = args.runtime / "folds" / f"fold{fold}"
        with (work / "FEATURE_SELECTION_PRIVATE.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for variant, selection in selected.items():
            feature_rows.extend(row for row in rows if row["variant"] == variant and int(row["k"]) == selection["k"])
        with (work / "RUNTIME.csv").open(newline="", encoding="utf-8") as stream:
            runtime_rows.extend(csv.DictReader(stream))
    write_csv(EXPERIMENT / "ICTAL_B1_CD_METRICS.csv", [b1])
    write_csv(EXPERIMENT / "ICTAL_B2_PRCD_METRICS.csv", [b2])
    write_csv(EXPERIMENT / "ICTAL_FOLD_METRICS.csv", fold_rows)
    write_csv(EXPERIMENT / "ICTAL_FEATURE_SELECTION_AUDIT.csv", feature_rows)
    write_csv(EXPERIMENT / "RUNTIME_AUDIT.csv", runtime_rows)
    atomic_json(EXPERIMENT / "ICTAL_GATE.json", gate)
    parameter_rows = [{"model": "H0", "k": k, "input_dimensions_B1": k,
                       "input_dimensions_B2": k + 30, "max_trainable_parameters": k + 31,
                       "budget_lt_25000": True} for k in (128, 256)]
    parameter_rows += [{"model": "H1", "k": k, "input_dimensions_B1": k,
                        "input_dimensions_B2": k + 30,
                        "max_trainable_parameters": (k + 30) * 32 + 32 + 32 + 1,
                        "budget_lt_25000": (k + 30) * 32 + 65 < 25000} for k in (128, 256)]
    write_csv(EXPERIMENT / "MODEL_PARAMETER_AUDIT.csv", parameter_rows)
    scenario = ("A" if b1["auroc"] > HISTORICAL["auroc"] and b2["auroc"] > b1["auroc"] else
                "B" if b1["auroc"] > HISTORICAL["auroc"] else
                "C" if b2["auroc"] > HISTORICAL["auroc"] else "D")
    lines = ["# PRCD-EZ Ictal development", "",
             "Validation-only historical fixed-query VLOO; no outer-test record was opened.", "",
             "| Model | Config | Params | AUROC | AP | Macro-F1 | MRR | Top1 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
             f"| Historical A1 | fixed reference | historical | {HISTORICAL['auroc']:.6f} | {HISTORICAL['ap']:.6f} | {HISTORICAL['macro_f1']:.6f} | {HISTORICAL['mrr']:.6f} | {HISTORICAL['top1']:.6f} |",
             f"| CD-EZ | {b1['config']} | {b1['parameters']} | {b1['auroc']:.6f} | {b1['ap']:.6f} | {b1['macro_f1']:.6f} | {b1['mrr']:.6f} | {b1['top1']:.6f} |",
             f"| PRCD-EZ | {b2['config']} | {b2['parameters']} | {b2['auroc']:.6f} | {b2['ap']:.6f} | {b2['macro_f1']:.6f} | {b2['mrr']:.6f} | {b2['top1']:.6f} |", "",
             f"PRCD−A1 AUROC: {b2['auroc'] - HISTORICAL['auroc']:+.6f}; AP: {b2['ap'] - HISTORICAL['ap']:+.6f}.",
             f"PRCD−CD AUROC: {b2['auroc'] - b1['auroc']:+.6f}.",
             f"PRCD exceeded historical A1 fold AUROC in {positive_folds}/5 folds.", "",
             "Gate checks:"]
    lines += [f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in checks.items()]
    lines += ["", f"Scientific scenario: **{scenario}**.",
              "Scenario D means the fixed-dictionary route should stop." if scenario == "D" else
              "See selected-feature audit for coordinate, view, dilation-group proxy and statistic frequencies.",
              "", f"**Terminal: `{terminal}`.**", ""]
    (EXPERIMENT / "ICTAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (EXPERIMENT / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    failure = EXPERIMENT / "ICTAL_FAILURE_REPORT.md"
    if not passed:
        failure.write_text("\n".join(lines + ["Omni was not started because the locked Ictal gate failed.", ""]), encoding="utf-8")
    elif failure.exists():
        failure.unlink()
    print(json.dumps(gate, indent=2), flush=True)


def diagnose_fold(args) -> None:
    """Deterministically replay the selected H1 and run inference-only interventions."""
    gate = json.loads((EXPERIMENT / "ICTAL_GATE.json").read_text(encoding="utf-8")) if (EXPERIMENT / "ICTAL_GATE.json").exists() else None
    # On the server the compact gate lives beside the uploaded protocol.
    server_gate = args.protocol.parent / "ICTAL_GATE.json"
    if server_gate.exists(): gate = json.loads(server_gate.read_text(encoding="utf-8"))
    if not gate or gate["selected_B2"]["config"] != "K128_H1":
        raise RuntimeError("diagnostic replay currently requires the frozen selected K128_H1")
    folds = read_manifest(args.manifest); split = folds[args.fold]
    work = args.runtime / "folds" / f"fold{args.fold}"
    rows = list(csv.DictReader((work / "FEATURE_SELECTION_PRIVATE.csv").open(newline="", encoding="utf-8")))
    chosen = sorted((row for row in rows if row["variant"] == "B2_PRCD" and int(row["k"]) == 128),
                    key=lambda row: int(row["rank"]))
    selected = np.asarray([int(row["feature_index"]) for row in chosen], dtype=np.int32)
    fit = {patient: load_fold_patient(args.runtime, args.fold, patient) for patient in split["fit"]}
    validation = {patient: load_fold_patient(args.runtime, args.fold, patient) for patient in split["validation"]}
    x, y, weights, owners, positive_weight = training_matrix(fit, "coordinate", selected, True)
    scaler = StandardScaler().fit(x, sample_weight=weights)
    transformed = scaler.transform(x).astype(np.float32)
    torch.manual_seed(SEED + args.fold * 100 + 2 + 128)
    model = TinyMLP(transformed.shape[1]); optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    tx, ty, tw = map(torch.from_numpy, (transformed, y, weights))
    frozen = json.loads((work / "B2_PRCD_PREDICTIONS_PRIVATE.json").read_text(encoding="utf-8"))["K128_H1"]
    diagnostic = {mode: [] for mode in ("full_replay", "dictionary_shuffle", "relative_removal", "biomarker_removal")}
    max_difference = 0.0
    raw_validation = {}
    for patient, item in validation.items():
        vx, vy, names = validation_matrix(item, "coordinate", selected, True)
        raw_validation[patient] = (vx, vy, names)
    for epoch in range(40):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(tx), ty, weight=tw)
        loss.backward(); optimizer.step(); model.eval()
        epoch_rows = {mode: {} for mode in diagnostic}
        with torch.inference_mode():
            for patient, (raw_x, labels, names) in raw_validation.items():
                variants = {"full_replay": raw_x.copy(), "dictionary_shuffle": raw_x.copy(),
                            "relative_removal": raw_x.copy(), "biomarker_removal": raw_x.copy()}
                permutation = np.random.default_rng(stable_seed(SEED, "dictionary_shuffle", args.fold, patient)).permutation(len(raw_x))
                variants["dictionary_shuffle"][:, :128] = raw_x[permutation, :128]
                relative_columns = np.flatnonzero(selected >= POOLED_DIM)
                variants["relative_removal"][:, relative_columns] = 0.0
                variants["biomarker_removal"][:, 128:] = 0.0
                for mode, current in variants.items():
                    score = torch.sigmoid(model(torch.from_numpy(scaler.transform(current).astype(np.float32)))).numpy()
                    epoch_rows[mode][patient] = {"labels": labels.tolist(), "scores": score.astype(float).tolist(),
                                                 "channel_names": names.tolist()}
                reference = np.asarray(frozen[epoch][patient]["scores"], dtype=np.float64)
                max_difference = max(max_difference, float(np.max(np.abs(reference - np.asarray(epoch_rows["full_replay"][patient]["scores"])))))
        for mode in diagnostic: diagnostic[mode].append(epoch_rows[mode])
    if max_difference >= 1e-6:
        raise RuntimeError(f"diagnostic replay did not reproduce frozen predictions: {max_difference}")
    atomic_json(work / "B2_DIAGNOSTICS_PRIVATE.json", diagnostic)
    atomic_json(work / "B2_DIAGNOSTIC_REPLAY_AUDIT.json", {"status": "PASS", "fold": args.fold,
                "selected_config": "K128_H1", "max_abs_prediction_difference": max_difference,
                "epochs": 40, "thresholds_reoptimized": False, "test_accessed": False})
    print(json.dumps({"status": "DIAGNOSTIC_FOLD_COMPLETE", "fold": args.fold,
                      "max_abs_prediction_difference": max_difference}), flush=True)


def finalize_diagnostics(args) -> None:
    selection_rows = {}
    with (args.runtime / "private_final" / "B2_PRCD_SELECTION_PRIVATE.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream): selection_rows[(int(row["fold"]), row["patient_private"])] = row
    modes = ("full_replay", "dictionary_shuffle", "relative_removal", "biomarker_removal")
    cells = {mode: [] for mode in modes}; audits = []
    for fold in range(1, 6):
        work = args.runtime / "folds" / f"fold{fold}"
        audit = json.loads((work / "B2_DIAGNOSTIC_REPLAY_AUDIT.json").read_text(encoding="utf-8")); audits.append(audit)
        diagnostic = json.loads((work / "B2_DIAGNOSTICS_PRIVATE.json").read_text(encoding="utf-8"))
        patients = sorted(diagnostic["full_replay"][0])
        for patient in patients:
            selected_row = selection_rows[(fold, patient)]
            epoch, threshold = int(selected_row["selected_epoch"]) - 1, float(selected_row["selected_threshold"])
            for mode in modes:
                payload = diagnostic[mode][epoch][patient]
                labels = np.asarray(payload["labels"], dtype=np.int8); scores = np.asarray(payload["scores"], dtype=float)
                for repetition in range(20):
                    query = fixed_query(len(labels), fold, patient, repetition)
                    cells[mode].append({"fold": fold, "patient_private": patient, "repetition": repetition,
                                        **query_metrics(labels[query], scores[query], threshold)})
    rows = []
    full = aggregate(cells["full_replay"])
    for mode in modes:
        metric = aggregate(cells[mode])
        rows.append({"intervention": mode, **metric,
                     "delta_auroc_vs_full": metric["auroc"] - full["auroc"],
                     "delta_ap_vs_full": metric["ap"] - full["ap"],
                     "delta_macro_f1_vs_full": metric["macro_f1"] - full["macro_f1"],
                     "fixed_selected_epoch_and_threshold": True, "retrained_ablation": False})
    # Full replay must also match the already-published compact aggregate.
    gate = json.loads((args.protocol.parent / "ICTAL_GATE.json").read_text(encoding="utf-8"))
    if abs(full["auroc"] - gate["selected_B2"]["auroc"]) >= 1e-10 or any(a["status"] != "PASS" for a in audits):
        raise RuntimeError("aggregate diagnostic replay differs from frozen B2")
    write_csv(args.protocol.parent / "ICTAL_INTERVENTION.csv", rows)
    atomic_json(args.protocol.parent / "DIAGNOSTIC_REPLAY_AUDIT.json", {
        "status": "PASS", "fold_max_abs_prediction_difference": [a["max_abs_prediction_difference"] for a in audits],
        "thresholds_reoptimized": False, "test_accessed": False})
    print(json.dumps({"status": "DIAGNOSTICS_COMPLETE", "rows": rows}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("speed-gate", "prepare-all", "run-fold", "finalize",
                                             "diagnose-fold", "finalize-diagnostics"))
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--ictal-cache", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--workers", type=int, default=max(1, min((os.cpu_count() or 2) // 2, 8)))
    parser.add_argument("--max-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if digest(args.protocol) != hashlib.sha256(args.protocol.read_bytes()).hexdigest():
        raise RuntimeError("protocol digest failure")
    if args.manifest and digest(args.manifest) != json.loads(args.protocol.read_text())["ictal_fold_manifest_sha256"]:
        raise RuntimeError("frozen manifest hash mismatch")
    if args.command == "speed-gate": speed_gate(args)
    elif args.command == "prepare-all": prepare_all(args)
    elif args.command == "run-fold": run_fold(args)
    elif args.command == "finalize": finalize(args)
    elif args.command == "diagnose-fold": diagnose_fold(args)
    else: finalize_diagnostics(args)


if __name__ == "__main__":
    main()
