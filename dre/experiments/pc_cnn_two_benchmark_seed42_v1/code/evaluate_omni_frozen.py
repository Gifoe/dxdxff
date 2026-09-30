"""One frozen official Omni test evaluation, no model/threshold updates.

The primary metric unit is pooled (EDF,channel), with 60-s clip probabilities
averaged inside that unit. Frozen validation-selected numeric thresholds are
applied without any test re-optimization. Patient/channel predictions remain
in private runtime; public files contain only aggregate metrics and CIs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, roc_auc_score)

from frozen_omni_test_bank import FrozenOmniTestBank
from official_spectrum import OfficialSpectrum, load_official_module
from pc_cnn import PCCNN
from raw_metrics import binary_metrics
from train_pccnn import evaluate as evaluate_pc
from train_rawcnn import digest, evaluate as evaluate_raw, save_json


CENTERS = ("HUP", "Open-iEEG", "SourceSink", "Zurich")


def table(path: Path, rows):
    names = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def flatten(private, patients):
    y = np.concatenate([private[patient]["labels"] for patient in patients]).astype(np.int8)
    score = np.concatenate([private[patient]["scores"] for patient in patients]).astype(float)
    group = np.concatenate([np.full(len(private[patient]["labels"]), index, dtype=np.int16)
                            for index, patient in enumerate(patients)])
    return y, score, group


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
    return binary_metrics([labels[name] for name in names],
                          [np.mean(grouped[name]) for name in names])


def scored(private, patients, threshold):
    y, score, _ = flatten(private, patients)
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    rank = [patient_ranking(private[patient]) for patient in patients]
    values = {key: [row[key] for row in rank if row[key] is not None]
              for key in ("auroc", "ap", "mrr", "top1")}
    return {"edf_channel_units": len(y), "patients": len(patients),
            "positive_units": int(y.sum()),
            "auroc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
            "pooled_ap": float(average_precision_score(y, score)) if y.sum() else None,
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "pathological_f1": float(f1_score(y, pred, pos_label=1, zero_division=0))
            if y.sum() else None,
            "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "accuracy": float((tp + tn) / len(y)),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "patient_equal_auroc": float(np.mean(values["auroc"])) if values["auroc"] else None,
            "patient_ap": float(np.mean(values["ap"])) if values["ap"] else None,
            "mrr": float(np.mean(values["mrr"])) if values["mrr"] else None,
            "top1": float(np.mean(values["top1"])) if values["top1"] else None}


def bootstrap_paired(raw_private, pc_private, patients, raw_threshold, pc_threshold):
    ry, rs, group = flatten(raw_private, patients)
    py, ps, other_group = flatten(pc_private, patients)
    if not np.array_equal(ry, py) or not np.array_equal(group, other_group):
        raise RuntimeError("Official EDF-channel paired alignment differs")
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(patients), size=(10000, len(patients)))
    deltas = {key: np.empty(len(draws)) for key in
              ("auroc", "ap", "macro_f1", "pathological_f1", "sensitivity",
               "specificity", "mrr", "top1")}
    rankings = {"RawCNN": [patient_ranking(raw_private[p]) for p in patients],
                "PC-CNN": [patient_ranking(pc_private[p]) for p in patients]}
    ranking_deltas = {
        metric: np.asarray([
            left[metric] - right[metric] if left[metric] is not None and
            right[metric] is not None else np.nan
            for left, right in zip(rankings["PC-CNN"], rankings["RawCNN"])], dtype=float)
        for metric in ("mrr", "top1")}
    for index, sampled in enumerate(draws):
        counts = np.bincount(sampled, minlength=len(patients))
        weights = counts[group]
        active = weights > 0
        if len(np.unique(ry[active])) < 2:
            deltas["auroc"][index] = np.nan
            deltas["ap"][index] = np.nan
        else:
            deltas["auroc"][index] = (roc_auc_score(py, ps, sample_weight=weights) -
                                       roc_auc_score(ry, rs, sample_weight=weights))
            deltas["ap"][index] = (average_precision_score(py, ps, sample_weight=weights) -
                                    average_precision_score(ry, rs, sample_weight=weights))
        confusion = {}
        for name, score, threshold in (("RawCNN", rs, raw_threshold),
                                       ("PC-CNN", ps, pc_threshold)):
            predict = score >= threshold
            tn, fp, fn, tp = confusion_matrix(ry, predict, labels=[0, 1],
                                               sample_weight=weights).ravel()
            p_f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            n_f1 = 2 * tn / max(2 * tn + fp + fn, 1)
            confusion[name] = {"macro_f1": (p_f1 + n_f1) / 2,
                               "pathological_f1": p_f1,
                               "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
                               "specificity": tn / (tn + fp) if tn + fp else np.nan}
        for metric in confusion["RawCNN"]:
            deltas[metric][index] = confusion["PC-CNN"][metric] - confusion["RawCNN"][metric]
        for metric in ("mrr", "top1"):
            deltas[metric][index] = np.nanmean(ranking_deltas[metric][sampled])
    return [{"candidate": "PC-CNN", "reference": "RawCNN", "metric": key,
             "delta_mean_bootstrap": float(np.nanmean(value)),
             "ci_low": float(np.nanquantile(value, 0.025)),
             "ci_high": float(np.nanquantile(value, 0.975)),
             "draws": 10000, "cluster_unit": "patient"}
            for key, value in deltas.items()]


def prediction_digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def cached_patient_predictions(name, model, preprocessor, bank, patients,
                               normalizer, device, runtime, freeze_sha,
                               checkpoint_sha, protocol_sha, official_cnn_sha):
    """Resume frozen inference only at completed patients after native crashes."""
    cache = runtime / "omni" / "FROZEN_TEST_PATIENT_CACHE_PRIVATE"
    cache.mkdir(parents=True, exist_ok=True)
    result = {}
    variants = {"PC-CNN": (True, True),
                "PC_no_physiology": (False, True),
                "PC_no_context": (True, False),
                "PC_all_disabled": (False, False)}
    for ordinal, patient in enumerate(patients):
        path = cache / f"{name}_{ordinal:03d}.json"
        binding = {"freeze_sha256": freeze_sha,
                   "checkpoint_sha256": checkpoint_sha,
                   "protocol_sha256": protocol_sha,
                   "official_cnn_sha256": official_cnn_sha,
                   "model": name, "ordinal": ordinal, "patient": patient,
                   "cohort_patients": len(patients)}
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if any(saved.get(key) != value for key, value in binding.items()) or \
                    saved.get("prediction_sha256") != prediction_digest(
                        saved.get("prediction")):
                raise RuntimeError("Frozen patient inference cache provenance mismatch")
            prediction = saved["prediction"]
            reused = True
        else:
            if name == "RawCNN":
                _, private = evaluate_raw(model, preprocessor, bank, [patient],
                                          "omni", 0, device)
            else:
                physiology, context = variants[name]
                _, private = evaluate_pc(model, preprocessor, bank, [patient],
                                         "omni", 0, normalizer, device,
                                         physiology=physiology, context=context)
            prediction = private[patient]
            save_json(path, {**binding, "prediction": prediction,
                             "prediction_sha256": prediction_digest(prediction)})
            reused = False
        result[patient] = prediction
        print(json.dumps({"status": "FROZEN_TEST_PATIENT_COMPLETE", "model": name,
                          "ordinal": ordinal + 1, "of": len(patients),
                          "reused": reused}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--official-cnn", type=Path, required=True)
    p.add_argument("--omni-test-cache", type=Path, required=True)
    p.add_argument("--omni-official-split", type=Path, required=True)
    p.add_argument("--cohort-audit", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze["status"] != "FROZEN_BEFORE_OFFICIAL_TEST" or \
            freeze["protocol_sha256"] != digest(args.protocol) or \
            not freeze["model_frozen_before_final_test"]:
        raise RuntimeError("Omni official TEST requires frozen selection")
    selected = freeze["benchmark_checkpoints"]["omni/fold1"]
    for model in selected.values():
        if digest(Path(model["private_path"])) != model["sha256"]:
            raise RuntimeError("Omni selected checkpoint changed after freeze")
    bank = FrozenOmniTestBank(args.omni_test_cache, args.omni_official_split,
                              args.cohort_audit, args.freeze, args.protocol)
    patients = bank.patients()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = load_official_module(args.official_cnn)
    preprocessor = OfficialSpectrum(module, channel_chunk=4)
    raw = module.NeuralCNN(in_channels=1, outputs=1).to(device)
    raw_state = torch.load(selected["RawCNN"]["private_path"],
                           map_location=device, weights_only=False)
    raw.load_state_dict(raw_state["raw_state"])
    freeze_sha, protocol_sha = digest(args.freeze), digest(args.protocol)
    official_cnn_sha = digest(args.official_cnn)
    raw_private = cached_patient_predictions(
        "RawCNN", raw, preprocessor, bank, patients, None, device,
        args.runtime, freeze_sha, selected["RawCNN"]["sha256"],
        protocol_sha, official_cnn_sha)
    normalizer_path = args.runtime / "omni" / "fold1" / "descriptor_normalization.json"
    if digest(normalizer_path) != selected["PC-CNN"]["train_fit_descriptor_norm_sha256"]:
        raise RuntimeError("Omni descriptor normalization changed after freeze")
    normalizer = json.loads(normalizer_path.read_text(encoding="utf-8"))
    pc = PCCNN(module.NeuralCNN(in_channels=1, outputs=1)).to(device)
    pc_state = torch.load(selected["PC-CNN"]["private_path"],
                          map_location=device, weights_only=False)
    pc.load_state_dict(pc_state["model_state"])
    private = {"RawCNN": raw_private}
    for name, physiology, context in (("PC-CNN", True, True),
                                       ("PC_no_physiology", False, True),
                                       ("PC_no_context", True, False),
                                       ("PC_all_disabled", False, False)):
        private[name] = cached_patient_predictions(
            name, pc, preprocessor, bank, patients, normalizer, device,
            args.runtime, freeze_sha, selected["PC-CNN"]["sha256"],
            protocol_sha, official_cnn_sha)
    args.output.mkdir(parents=True, exist_ok=True)
    thresholds = {"RawCNN": selected["RawCNN"]["validation_threshold"]["threshold"],
                  "PC-CNN": selected["PC-CNN"]["validation_threshold"]["threshold"]}
    threshold_rows = [{"model": name, **value["validation_threshold"]}
                      for name, value in selected.items()]
    table(args.output / "OMNI_THRESHOLD_SELECTION.csv", threshold_rows)
    totals = []
    for name, rows in private.items():
        tau = thresholds["RawCNN"] if name == "RawCNN" else thresholds["PC-CNN"]
        totals.append({"benchmark": "Omni", "model": name,
                       "threshold": tau, **scored(rows, patients, tau)})
    table(args.output / "OMNI_RAWCNN_METRICS.csv", [totals[0]])
    table(args.output / "OMNI_PCCNN_METRICS.csv", [totals[1]])
    table(args.output / "OMNI_INTERVENTION.csv", totals)
    center_rows = []
    for center in CENTERS:
        subset = [patient for patient in patients if bank.center_by_patient[patient] == center]
        if not subset:
            center_rows.append({"center": center, "model": "not_estimable",
                                "reason": "no_official_test_patients"})
            continue
        for name in ("RawCNN", "PC-CNN"):
            metrics = scored(private[name], subset, thresholds[name])
            has_positive = any(sum(private[name][patient]["labels"])
                               for patient in subset)
            if not has_positive:
                for key in ("auroc", "pooled_ap", "pathological_f1", "sensitivity"):
                    metrics[key] = "not_estimable"
            center_rows.append({"center": center, "model": name,
                                "reason": "" if has_positive else "no_positive",
                                **metrics})
    table(args.output / "OMNI_CENTER_METRICS.csv", center_rows)
    table(args.output / "OMNI_BOOTSTRAP.csv",
          bootstrap_paired(private["RawCNN"], private["PC-CNN"], patients,
                           thresholds["RawCNN"], thresholds["PC-CNN"]))
    save_json(args.runtime / "omni" / "FROZEN_TEST_SCORES_PRIVATE.json",
              {"freeze_sha256": digest(args.freeze), "patient_scores": private})
    save_json(args.output / "OMNI_OFFICIAL_TEST_STATUS.json",
              {"status": "ONE_FROZEN_OFFICIAL_TEST_PASS_COMPLETE",
               "patients": len(patients), "edf_channels": totals[0]["edf_channel_units"],
               "test_used_for_tuning": False, "historical_test_previously_viewed": True})
    print(json.dumps({"status": "ONE_FROZEN_OFFICIAL_TEST_PASS_COMPLETE",
                      "patients": len(patients), "test_used_for_tuning": False}), flush=True)


if __name__ == "__main__":
    main()
