"""Validation-only Ictal SCM-EZ training, VLOO selection and diagnostics.

The executable deliberately has no outer-test or Omni entry point.  It writes
private checkpoint/prediction material only below ``--runtime``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.special import logit
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

from scm_core import (N_BINS, N_STATES, SCMEZ, SEED, RobustScaler, aggregate_states,
                      fit_robust_scaler, parameter_count, scm_matrices, stable_seed,
                      visible_bins)


HERE = Path(__file__).resolve()
EXPERIMENT = HERE.parents[1]
THRESHOLDS = np.linspace(0.05, 0.95, 19, dtype=np.float64)
HISTORICAL = {"auroc": 0.746382, "ap": 0.576743, "macro_f1": 0.620810,
              "mrr": 0.740038, "top1": 0.654771}
METRICS = ("auroc", "ap", "macro_f1", "ez_f1", "balanced_accuracy", "mrr", "top1")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value, *, allow_nan: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=allow_nan) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def read_manifest(path: Path) -> dict[int, dict[str, list[str]]]:
    result = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for fold in range(1, 6):
        group = defaultdict(list)
        for row in rows:
            if int(row["outer_fold"]) == fold:
                group[row["split_role"]].append(str(row["subject_id"]))
        if set(group) != {"fit", "validation", "test"} or len(group["validation"]) != 13 or \
                sum(map(len, group.values())) != 80 or len(set(sum(group.values(), []))) != 80:
            raise RuntimeError(f"frozen fold {fold} membership changed")
        result[fold] = {key: sorted(value) for key, value in group.items()}
    return result


def patient_path(cache: Path, patient: str) -> Path:
    token = hashlib.sha256(patient.encode()).hexdigest()[:20]
    path = Path(cache) / f"patient_{token}.npz"
    if not path.is_file() or not path.with_suffix(".json").is_file():
        raise RuntimeError("private patient spectral cache missing")
    return path


def load_raw_patient(cache: Path, patient: str) -> dict:
    with np.load(patient_path(cache, patient), allow_pickle=False) as item:
        if str(item["patient"]) != patient or float(item["sampling_rate_hz"]) != 250.0:
            raise RuntimeError("private patient identity/sampling-rate mismatch")
        return {name: item[name].copy() for name in item.files}


def fit_scaler(cache: Path, patients: list[str]) -> RobustScaler:
    values, masks = [], []
    for patient in patients:
        item = load_raw_patient(cache, patient)
        values.append(item["spectra"])
        masks.append(item["window_mask"])
    return fit_robust_scaler(values, masks, visible_bins(250.0))


def prepare_patient(cache: Path, patient: str, scaler: RobustScaler,
                    mode: str = "full", fold: int = 0) -> dict:
    item = load_raw_patient(cache, patient)
    matrices, channel_indices = [], []
    fallback = 0
    for record in range(len(item["channel_present"])):
        present = item["channel_present"][record].astype(bool)
        local = np.flatnonzero(present)
        if not len(local):
            raise RuntimeError("empty synchronized record")
        center_valid = item["center_mask"][record].astype(bool)
        centers = item["centers"][record, center_valid]
        spectra = scaler.transform(item["spectra"][record][:, center_valid, :])
        windows = item["window_mask"][record][:, center_valid]
        states, state_valid = aggregate_states(spectra, centers, windows, scaler.frequency_valid)
        states, state_valid = states[local], state_valid[local]
        reference_permutation = None
        if mode == "reference_shuffle":
            reference_permutation = np.random.default_rng(
                stable_seed(SEED, "reference_shuffle", fold, patient, record)).permutation(len(local))
        elif mode == "temporal_permutation":
            permuted, permuted_valid = np.zeros_like(states), np.zeros_like(state_valid)
            for local_index, canonical_index in enumerate(local):
                permutation = np.random.default_rng(stable_seed(
                    SEED, "temporal_permutation", fold, patient, record, int(canonical_index))).permutation(N_STATES)
                permuted[local_index] = states[local_index, permutation]
                permuted_valid[local_index] = state_valid[local_index, permutation]
            states, state_valid = permuted, permuted_valid
        elif mode != "full":
            raise ValueError(mode)
        current, current_fallback = scm_matrices(states, state_valid,
                                                 reference_permutation=reference_permutation)
        fallback += current_fallback
        matrices.append(current)
        channel_indices.append(local.astype(np.int64))
    all_indices = np.concatenate(channel_indices)
    n_channels = len(item["labels"])
    if set(all_indices.tolist()) != set(range(n_channels)):
        raise RuntimeError("canonical patient channel lacks synchronized raw input")
    return {"matrices": torch.from_numpy(np.concatenate(matrices)),
            "channel_index": torch.from_numpy(all_indices),
            "labels": torch.from_numpy(item["labels"].astype(np.float32)),
            "n_channels": n_channels, "fallback_count": fallback}


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int8); scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) != 2:
        raise RuntimeError("historical VLOO patient lacks both classes")
    order = np.argsort(-scores, kind="stable")
    rank = int(np.flatnonzero(labels[order] == 1)[0])
    return {"auroc": float(roc_auc_score(labels, scores)),
            "ap": float(average_precision_score(labels, scores)),
            "mrr": float(1.0 / (rank + 1)), "top1": float(labels[order[0]])}


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = np.asarray(scores) >= float(threshold)
    return {"macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
            "ez_f1": float(f1_score(labels, prediction, pos_label=1, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, prediction))}


def patient_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    return {**ranking_metrics(labels, scores), **threshold_metrics(labels, scores, threshold)}


def predict(model: SCMEZ, patients: dict[str, dict]) -> dict[str, dict]:
    model.eval(); result = {}
    with torch.inference_mode():
        for patient, item in patients.items():
            scores = torch.sigmoid(model(item["matrices"], item["channel_index"], item["n_channels"])).numpy()
            result[patient] = {"labels": item["labels"].numpy().astype(np.int8).tolist(),
                               "scores": scores.astype(float).tolist()}
    return result


def run_fold(args) -> None:
    started = time.perf_counter()
    folds = read_manifest(args.manifest)
    split = folds[args.fold]
    work = args.runtime / "folds" / f"fold{args.fold}"
    status_path = work / "FOLD_STATUS.json"
    protocol_sha = digest(args.protocol)
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "COMPLETE" and status.get("protocol_sha256") == protocol_sha:
            print(json.dumps({"status": "REUSED", "fold": args.fold}), flush=True); return
        raise RuntimeError("non-complete fold status requires inspection")
    work.mkdir(parents=True, exist_ok=True)
    scaler = fit_scaler(args.cache, split["fit"])
    np.savez(work / "FIT_SCALER_PRIVATE.npz", median=scaler.median, iqr=scaler.iqr,
             frequency_valid=scaler.frequency_valid)
    fit = {patient: prepare_patient(args.cache, patient, scaler, fold=args.fold) for patient in split["fit"]}
    validation = {patient: prepare_patient(args.cache, patient, scaler, fold=args.fold)
                  for patient in split["validation"]}
    set_seed(SEED + args.fold)
    model = SCMEZ()
    if parameter_count() >= 15000:
        raise RuntimeError("SCM-EZ parameter budget failed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    predictions, metric_rows = [], []
    epoch_seconds = []
    for epoch in range(1, 41):
        epoch_started = time.perf_counter(); model.train(); optimizer.zero_grad(set_to_none=True)
        losses = []
        for patient in split["fit"]:
            item = fit[patient]
            logits = model(item["matrices"], item["channel_index"], item["n_channels"])
            weights = torch.where(item["labels"] > 0.5, 2.0, 1.0)
            patient_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, item["labels"], weight=weights, reduction="mean")
            (patient_loss / len(fit)).backward(); losses.append(float(patient_loss.detach()))
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)); optimizer.step()
        snapshot = predict(model, validation); predictions.append(snapshot)
        metrics = [patient_metrics(np.asarray(value["labels"]), np.asarray(value["scores"]))
                   for value in snapshot.values()]
        row = {"fold": args.fold, "epoch": epoch, "train_patient_equal_loss": float(np.mean(losses)),
               "gradient_norm_before_clip": gradient_norm,
               **{f"validation_patient_equal_{name}": float(np.mean([metric[name] for metric in metrics]))
                  for name in METRICS}}
        metric_rows.append(row); epoch_seconds.append(time.perf_counter() - epoch_started)
        checkpoint = {"protocol_sha256": protocol_sha, "fold": args.fold, "epoch": epoch,
                      "model": model.state_dict()}
        torch.save(checkpoint, work / f"checkpoint_epoch_{epoch:02d}.pt")
        atomic_json(work / f"validation_epoch_{epoch:02d}_private.json",
                    {"protocol_sha256": protocol_sha, "fold": args.fold, "epoch": epoch,
                     "private": snapshot, "test_accessed": False})
        print(json.dumps(row), flush=True)
    write_csv(work / "VALIDATION_HISTORY.csv", metric_rows)
    status = {"status": "COMPLETE", "fold": args.fold, "protocol_sha256": protocol_sha,
              "fit_patients": len(fit), "validation_patients": len(validation),
              "epochs": 40, "patience": 7,
              "patience_application": "target-excluded offline selection; full grid retained to prevent target leakage",
              "mean_seconds_per_epoch": float(np.mean(epoch_seconds)),
              "total_seconds": time.perf_counter() - started,
              "reference_fallback_counter": int(sum(item["fallback_count"] for item in [*fit.values(), *validation.values()])),
              "outer_test_accessed": False}
    atomic_json(status_path, status); print(json.dumps(status, indent=2), flush=True)


def load_grid(work: Path, fold: int, protocol_sha: str) -> tuple[list[dict], list[str]]:
    grid = []
    for epoch in range(1, 41):
        path = work / f"validation_epoch_{epoch:02d}_private.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_sha256") != protocol_sha or int(payload["epoch"]) != epoch or \
                len(payload["private"]) != 13:
            raise RuntimeError("private validation grid mismatch")
        grid.append(payload["private"])
    patients = sorted(grid[0])
    if any(sorted(snapshot) != patients for snapshot in grid):
        raise RuntimeError("validation membership changed across epochs")
    return grid, patients


def choose_excluding_with_patience(grid: list[dict], patients: list[str], target: str) -> tuple[int, int]:
    others = [patient for patient in patients if patient != target]
    best_index, best_key, stale, stop_index = 0, None, 0, len(grid) - 1
    for epoch, snapshot in enumerate(grid):
        values = [patient_metrics(np.asarray(snapshot[patient]["labels"]),
                                  np.asarray(snapshot[patient]["scores"]), 0.5) for patient in others]
        key = tuple(round(float(np.mean([value[name] for value in values])), 12)
                    for name in ("auroc", "ap", "mrr", "macro_f1"))
        if best_key is None or key > best_key:
            best_key, best_index, stale = key, epoch, 0
        else:
            stale += 1
        if stale >= 7:
            stop_index = epoch; break
    return best_index, stop_index


def choose_threshold(snapshot: dict, patients: list[str], target: str) -> float:
    others = [patient for patient in patients if patient != target]
    candidates = []
    for index, threshold in enumerate(THRESHOLDS):
        values = [threshold_metrics(np.asarray(snapshot[patient]["labels"]),
                                    np.asarray(snapshot[patient]["scores"]), threshold) for patient in others]
        key = tuple(round(float(np.mean([value[name] for value in values])), 12)
                    for name in ("macro_f1", "ez_f1", "balanced_accuracy"))
        key += (-round(abs(float(threshold) - 0.5), 6), -index)
        candidates.append((key, float(threshold)))
    return max(candidates)[1]


def fixed_query(n_channels: int, fold: int, patient: str, repetition: int) -> np.ndarray:
    permutation = np.random.default_rng(stable_seed(SEED, fold, patient, repetition, "split")).permutation(n_channels)
    return permutation[n_channels // 2:]


def query_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = scores >= threshold
    result = {"macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
              "ez_f1": float(f1_score(labels, prediction, pos_label=1, zero_division=0)),
              "balanced_accuracy": (float(balanced_accuracy_score(labels, prediction))
                                    if len(np.unique(labels)) == 2 else float("nan"))}
    if len(np.unique(labels)) < 2:
        result.update({key: float("nan") for key in ("auroc", "ap", "mrr", "top1")})
    else:
        result.update(ranking_metrics(labels, scores))
    return result


def aggregate(rows: list[dict]) -> dict[str, float]:
    return {name: float(np.nanmean([row[name] for row in rows])) for name in METRICS}


def read_a1_folds(path: Path) -> dict[int, dict[str, float]]:
    result = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            result[int(row["fold"])] = {"auroc": float(row["patient_ez_auroc"]),
                                        "ap": float(row["patient_ez_auprc"])}
    if set(result) != set(range(1, 6)):
        raise RuntimeError("historical A1 fold reference changed")
    return result


def finalize(args) -> None:
    protocol_sha = digest(args.protocol); a1 = read_a1_folds(args.a1_folds)
    private_selection, private_queries, fold_rows = [], [], []
    patient_public = []
    for fold in range(1, 6):
        work = args.runtime / "folds" / f"fold{fold}"
        status = json.loads((work / "FOLD_STATUS.json").read_text(encoding="utf-8"))
        if status.get("status") != "COMPLETE" or status.get("protocol_sha256") != protocol_sha:
            raise RuntimeError("five completed protocol-matched folds required")
        grid, patients = load_grid(work, fold, protocol_sha)
        fold_cells, fold_patient_metrics = [], []
        for patient in patients:
            epoch, stop_epoch = choose_excluding_with_patience(grid, patients, patient)
            threshold = choose_threshold(grid[epoch], patients, patient)
            labels = np.asarray(grid[epoch][patient]["labels"], dtype=np.int8)
            scores = np.asarray(grid[epoch][patient]["scores"], dtype=np.float64)
            private_selection.append({"fold": fold, "patient_private": patient,
                                      "selected_epoch": epoch + 1, "target_excluded_stop_epoch": stop_epoch + 1,
                                      "selected_threshold": threshold})
            patient_cells = []
            for repetition in range(20):
                query = fixed_query(len(labels), fold, patient, repetition)
                row = {"fold": fold, "patient_private": patient, "repetition": repetition,
                       **query_metrics(labels[query], scores[query], threshold)}
                fold_cells.append(row); patient_cells.append(row); private_queries.append(row)
            fold_patient_metrics.append(aggregate(patient_cells))
        metric = aggregate(fold_cells)
        fold_rows.append({"fold": fold, "validation_patients": len(patients), "query_cells": len(fold_cells),
                          **metric, "historical_A1_auroc": a1[fold]["auroc"],
                          "historical_A1_ap": a1[fold]["ap"],
                          "auroc_delta_vs_A1": metric["auroc"] - a1[fold]["auroc"]})
        for name in METRICS:
            values = np.asarray([row[name] for row in fold_patient_metrics], dtype=np.float64)
            patient_public.append({"fold": fold, "metric": name,
                                   "aggregation": "13_target_summary_no_patient_rows",
                                   "n_targets": len(values), "mean": float(np.nanmean(values)),
                                   "std": float(np.nanstd(values)), "min": float(np.nanmin(values)),
                                   "max": float(np.nanmax(values))})
    if len(private_queries) != 1300 or len(private_selection) != 65:
        raise RuntimeError("historical 65-target x 20-query structure changed")
    overall = aggregate(private_queries)
    positive_folds = int(sum(row["auroc_delta_vs_A1"] > 0 for row in fold_rows))
    checks = {"auroc_ge_0_751382": overall["auroc"] >= 0.751382,
              "ap_ge_0_576743": overall["ap"] >= 0.576743,
              "positive_folds_ge_3": positive_folds >= 3,
              "mrr_ge_0_730": overall["mrr"] >= 0.730,
              "macro_f1_ge_0_606": overall["macro_f1"] >= 0.606}
    passed = all(checks.values())
    terminal = ("SCM_ICTAL_STRONG_PASS" if overall["auroc"] >= .77 and overall["ap"] >= .585
                else "SCM_ICTAL_PASS") if passed else "STOP_SCM_ICTAL_GATE_FAILED"
    gate = {"status": terminal, "pass": passed, "checks": checks,
            "positive_folds": positive_folds, "SCM_EZ": overall, "historical_A1": HISTORICAL,
            "omni_authorized": passed, "outer_test_accessed": False, "test_used_for_tuning": False}
    private = args.runtime / "private_final"; private.mkdir(parents=True, exist_ok=True)
    write_csv(private / "VLOO_SELECTION_PRIVATE.csv", private_selection)
    write_csv(private / "VLOO_QUERY_METRICS_PRIVATE.csv", private_queries)
    write_csv(EXPERIMENT / "ICTAL_VALIDATION_METRICS.csv", [*fold_rows,
              {"fold": "overall", "validation_patients": 65, "query_cells": 1300, **overall}])
    write_csv(EXPERIMENT / "ICTAL_FOLD_METRICS.csv", fold_rows)
    write_csv(EXPERIMENT / "ICTAL_PATIENT_METRICS.csv", patient_public)
    atomic_json(EXPERIMENT / "ICTAL_GATE.json", gate)
    print(json.dumps(gate, indent=2), flush=True)


def load_scaler(path: Path) -> RobustScaler:
    with np.load(path, allow_pickle=False) as item:
        return RobustScaler(item["median"], item["iqr"], item["frequency_valid"].astype(bool))


def diagnose_fold(args) -> None:
    protocol_sha = digest(args.protocol)
    selection = []
    with (args.runtime / "private_final" / "VLOO_SELECTION_PRIVATE.csv").open(newline="", encoding="utf-8") as stream:
        selection = [row for row in csv.DictReader(stream) if int(row["fold"]) == args.fold]
    if len(selection) != 13:
        raise RuntimeError("private target selection absent")
    work = args.runtime / "folds" / f"fold{args.fold}"
    scaler = load_scaler(work / "FIT_SCALER_PRIVATE.npz")
    cells = {mode: [] for mode in ("full", "patient_reference_shuffle", "temporal_state_permutation")}
    maximum_difference = 0.0
    for row in selection:
        patient = row["patient_private"]; epoch = int(row["selected_epoch"])
        threshold = float(row["selected_threshold"])
        checkpoint = torch.load(work / f"checkpoint_epoch_{epoch:02d}.pt", map_location="cpu", weights_only=False)
        if checkpoint["protocol_sha256"] != protocol_sha or checkpoint["fold"] != args.fold:
            raise RuntimeError("diagnostic checkpoint mismatch")
        model = SCMEZ(); model.load_state_dict(checkpoint["model"], strict=True); model.eval()
        variants = {"full": prepare_patient(args.cache, patient, scaler, "full", args.fold),
                    "patient_reference_shuffle": prepare_patient(args.cache, patient, scaler,
                                                                   "reference_shuffle", args.fold),
                    "temporal_state_permutation": prepare_patient(args.cache, patient, scaler,
                                                                    "temporal_permutation", args.fold)}
        scores = {}
        with torch.inference_mode():
            for mode, item in variants.items():
                scores[mode] = torch.sigmoid(model(item["matrices"], item["channel_index"],
                                                   item["n_channels"])).numpy()
        frozen = json.loads((work / f"validation_epoch_{epoch:02d}_private.json").read_text(encoding="utf-8"))
        reference = np.asarray(frozen["private"][patient]["scores"], dtype=np.float64)
        maximum_difference = max(maximum_difference, float(np.max(np.abs(reference - scores["full"]))))
        labels = variants["full"]["labels"].numpy().astype(np.int8)
        for repetition in range(20):
            query = fixed_query(len(labels), args.fold, patient, repetition)
            for mode in cells:
                cells[mode].append({"fold": args.fold, "patient_private": patient,
                                    "repetition": repetition,
                                    **query_metrics(labels[query], scores[mode][query], threshold)})
    if maximum_difference >= 1e-6:
        raise RuntimeError(f"frozen diagnostic replay mismatch: {maximum_difference}")
    atomic_json(work / "DIAGNOSTIC_AUDIT.json", {"status": "PASS", "fold": args.fold,
                "max_abs_full_prediction_difference": maximum_difference,
                "fixed_checkpoint_and_threshold": True, "retrained": False, "test_accessed": False})
    # Historical fixed-query cells may contain a single class.  Their undefined
    # ranking fields are private NaN values and are aggregated with nanmean.
    atomic_json(work / "DIAGNOSTIC_CELLS_PRIVATE.json", cells, allow_nan=True)
    print(json.dumps({"status": "DIAGNOSTIC_FOLD_COMPLETE", "fold": args.fold,
                      "max_abs_difference": maximum_difference}), flush=True)


def finalize_diagnostics(args) -> None:
    modes = ("full", "patient_reference_shuffle", "temporal_state_permutation")
    combined = {mode: [] for mode in modes}; audits = []
    for fold in range(1, 6):
        work = args.runtime / "folds" / f"fold{fold}"
        audits.append(json.loads((work / "DIAGNOSTIC_AUDIT.json").read_text(encoding="utf-8")))
        cells = json.loads((work / "DIAGNOSTIC_CELLS_PRIVATE.json").read_text(encoding="utf-8"))
        for mode in modes: combined[mode].extend(cells[mode])
    if any(audit.get("status") != "PASS" for audit in audits):
        raise RuntimeError("diagnostic fold audit failed")
    full = aggregate(combined["full"]); rows = []
    for mode in modes:
        metric = aggregate(combined[mode])
        rows.append({"intervention": mode, **metric,
                     "delta_auroc_vs_full": metric["auroc"] - full["auroc"],
                     "delta_ap_vs_full": metric["ap"] - full["ap"],
                     "delta_mrr_vs_full": metric["mrr"] - full["mrr"],
                     "fixed_checkpoint_and_threshold": True, "retrained": False})
    gate = json.loads((EXPERIMENT / "ICTAL_GATE.json").read_text(encoding="utf-8"))
    if abs(full["auroc"] - gate["SCM_EZ"]["auroc"]) >= 1e-8:
        raise RuntimeError("diagnostic full replay differs from frozen development result")
    write_csv(EXPERIMENT / "ICTAL_MATRIX_DIAGNOSTICS.csv", rows)
    atomic_json(EXPERIMENT / "MATRIX_CONSTRUCTION_AUDIT.json", {
        "status": "PASS", "input_shape": [34, 6, 6],
        "components": ["16_self_frequency_channels", "16_patient_frequency_channels",
                       "1_self_validity_mask", "1_patient_validity_mask"],
        "full_replay_max_abs_difference_by_fold": [a["max_abs_full_prediction_difference"] for a in audits],
        "interventions_retrained": False, "outer_test_accessed": False})
    print(json.dumps({"status": "DIAGNOSTICS_COMPLETE", "rows": rows}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run-fold", "finalize", "diagnose-fold", "finalize-diagnostics"))
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--a1-folds", type=Path)
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if args.manifest and digest(args.manifest) != json.loads(args.protocol.read_text(encoding="utf-8"))["ictal_fold_manifest_sha256"]:
        raise RuntimeError("frozen manifest hash mismatch")
    if args.command == "run-fold": run_fold(args)
    elif args.command == "finalize": finalize(args)
    elif args.command == "diagnose-fold": diagnose_fold(args)
    else: finalize_diagnostics(args)


if __name__ == "__main__":
    main()
