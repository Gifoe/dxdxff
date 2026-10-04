#!/usr/bin/env python3
"""The single OOF evaluation pass for the frozen five-fold patient protocol.

This program refuses to start until every fold/variant completed the locked 30
training epochs.  It writes patient/channel predictions only to the private
runtime, and writes non-identifying aggregate tables separately for the public
experiment directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)

from modeling import (A1ContextResidual, FixedSegmentSpectrum, TimeConvMorphology,
                      load_official_module, sha256)
from train_fold import (EPOCHS, RATE, SEGMENTS_PER_RECORD, SEGMENT_SAMPLES,
                        RecordBank, patient_forward)


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
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def cache_name(patient: str) -> str:
    return hashlib.sha256(patient.encode("utf-8")).hexdigest()[:24] + ".npz"


def atomic_prediction_cache(path: Path, *, protocol_sha: str, freeze_sha: str,
                            checkpoint_sha: str, channels: list[str], labels: np.ndarray,
                            scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, protocol_sha256=np.asarray(protocol_sha), freeze_sha256=np.asarray(freeze_sha),
                            checkpoint_sha256=np.asarray(checkpoint_sha),
                            channels=np.asarray(channels, dtype=str), labels=labels, scores=scores)
    os.replace(temporary, path)


def load_prediction_cache(path: Path, *, protocol_sha: str, freeze_sha: str,
                          checkpoint_sha: str) -> tuple[list[str], np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        if str(data["protocol_sha256"].item()) != protocol_sha or str(data["freeze_sha256"].item()) != freeze_sha or \
                str(data["checkpoint_sha256"].item()) != checkpoint_sha:
            raise RuntimeError("Private OOF patient cache provenance mismatch")
        return (data["channels"].astype(str).tolist(), data["labels"].astype(np.int8),
                data["scores"].astype(np.float64))


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    if len(np.unique(labels)) != 2:
        raise RuntimeError("Both-class patient condition violated during evaluation")
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order]
    first_positive = int(np.flatnonzero(ordered == 1)[0])
    discounts = 1.0 / np.log2(np.arange(2, len(labels) + 2, dtype=float))
    dcg = float((ordered * discounts).sum())
    ideal = np.sort(labels)[::-1]
    idcg = float((ideal * discounts).sum())
    predicted = (scores >= 0.5).astype(np.int8)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "mrr": float(1.0 / (first_positive + 1)),
        "top1": float(ordered[0] == 1), "ndcg": float(dcg / idcg) if idcg else float("nan"),
        "macro_f1_0_5": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "pathological_f1_0_5": float(f1_score(labels, predicted, pos_label=1, zero_division=0)),
        "normal_f1_0_5": float(f1_score(labels, predicted, pos_label=0, zero_division=0)),
        "balanced_accuracy_0_5": float(balanced_accuracy_score(labels, predicted)),
    }


def aggregate(rows: list[dict], key: str) -> dict:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return {f"{key}_{name}": float(value) for name, value in {
        "mean": values.mean(), "sd": values.std(ddof=0), "median": np.median(values),
        "q25": np.quantile(values, .25), "q75": np.quantile(values, .75)}.items()}


def load_model(checkpoint: Path, variant: str, source, device: torch.device) -> tuple[TimeConvMorphology, A1ContextResidual | None]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if int(state["epoch"]) != EPOCHS or state["variant"] != variant:
        raise RuntimeError("Unfrozen or wrong-variant checkpoint")
    raw = TimeConvMorphology(source).to(device)
    raw.load_state_dict(state["raw_state"]); raw.eval()
    plugin = None
    if variant == "plugin":
        plugin = A1ContextResidual().to(device)
        plugin.load_state_dict(state["plugin_state"]); plugin.eval()
    return raw, plugin


def score_patient(raw, plugin, spectrum, paths: list[Path], device: torch.device) -> tuple[list[str], np.ndarray, np.ndarray]:
    sum_logits: dict[str, float] = {}
    counts: dict[str, int] = {}
    label_by_channel: dict[str, int] = {}
    with torch.inference_mode():
        for segment in range(SEGMENTS_PER_RECORD):
            logits, labels, channels = patient_forward(raw, plugin, spectrum, paths, segment, device, False)
            for channel, logit, label in zip(channels, logits.detach().cpu().tolist(), labels.detach().cpu().tolist()):
                sum_logits[channel] = sum_logits.get(channel, 0.0) + float(logit)
                counts[channel] = counts.get(channel, 0) + 1
                if label >= 0:
                    previous = label_by_channel.setdefault(channel, int(label))
                    if previous != int(label):
                        raise RuntimeError("STOP_LABEL_CONFLICT during OOF evaluation")
    channels = sorted(sum_logits)
    if not all(counts[channel] == SEGMENTS_PER_RECORD for channel in channels):
        raise RuntimeError("Incomplete fixed 60-s segment aggregation")
    labels = np.asarray([label_by_channel.get(channel, -1) for channel in channels], dtype=np.int8)
    scores = 1.0 / (1.0 + np.exp(-np.asarray([sum_logits[channel] / counts[channel] for channel in channels])))
    return channels, labels, scores


def bootstrap(deltas: np.ndarray, seed: int = 42, draws: int = 10000) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(deltas), size=(draws, len(deltas)))
    values = deltas[index].mean(axis=1)
    return values, {"mean": float(values.mean()), "ci_low": float(np.quantile(values, .025)),
                    "ci_high": float(np.quantile(values, .975)), "draws": draws, "seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--official-cnn", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--patient-cache", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol_sha = sha256(args.protocol)
    freeze_sha = os.environ.get("OMNI_OOF_FREEZE_SHA")
    if not freeze_sha:
        raise RuntimeError("OOF evaluator requires a pre-test freeze hash")
    if sha256(args.private_manifest) != protocol["patient_manifest_private_sha256"]:
        raise RuntimeError("Private manifest differs from protocol lock")
    manifest = pd.read_csv(args.private_manifest)
    if manifest.patient_name.duplicated().any() or not (manifest.n_pathological_channels.gt(0) & manifest.n_normal_channels.gt(0)).all():
        raise RuntimeError("STOP_PROTOCOL_INVALID manifest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = load_official_module(args.official_cnn)
    spectrum = FixedSegmentSpectrum(source)
    private_root = args.runtime / "oof_private"
    private_root.mkdir(parents=True, exist_ok=True)

    all_patient_metrics: list[dict] = []
    all_predictions: list[dict] = []
    fold_metrics: list[dict] = []
    for fold in range(1, 6):
        held = manifest.loc[manifest.fold == fold].sort_values("patient_name")
        test_patients = held.patient_name.astype(str).tolist()
        bank = RecordBank(args.records, set(test_patients))
        for variant in ("baseline", "plugin"):
            checkpoint = args.runtime / "checkpoints" / f"fold_{fold}" / variant / "last.pt"
            checkpoint_sha = sha256(checkpoint)
            raw, plugin = load_model(checkpoint, variant, source, device)
            current: list[dict] = []
            for row in held.itertuples(index=False):
                patient = str(row.patient_name)
                cache = args.patient_cache / f"fold_{fold}" / variant / cache_name(patient)
                restored = load_prediction_cache(cache, protocol_sha=protocol_sha, freeze_sha=freeze_sha,
                                                  checkpoint_sha=checkpoint_sha)
                if restored is None:
                    channels, labels, scores = score_patient(raw, plugin, spectrum, bank.by_patient[patient], device)
                    atomic_prediction_cache(cache, protocol_sha=protocol_sha, freeze_sha=freeze_sha,
                                            checkpoint_sha=checkpoint_sha, channels=channels, labels=labels, scores=scores)
                    reused = False
                else:
                    channels, labels, scores = restored; reused = True
                valid = labels >= 0
                if not bool((labels[valid] == 1).any() and (labels[valid] == 0).any()):
                    raise RuntimeError("Both-class guarantee failed at evaluation")
                metrics = ranking_metrics(labels[valid], scores[valid])
                patient_row = {"patient_name": patient, "center": str(row.center), "fold": fold,
                               "variant": variant, "n_labeled_channels": int(valid.sum()), **metrics}
                current.append(patient_row); all_patient_metrics.append(patient_row)
                for channel, label, score in zip(channels, labels.tolist(), scores.tolist()):
                    all_predictions.append({"patient_name": patient, "center": str(row.center), "fold": fold,
                                            "variant": variant, "channel_name": channel, "label": label,
                                            "pathological_score": score})
                print(json.dumps({"stage": "OOF_PATIENT", "fold": fold, "variant": variant,
                                  "patient_cache_reused": reused, "channels": len(channels)}), flush=True)
            fold_metrics.append({"fold": fold, "variant": variant, "n_test_patients": len(current),
                                 **{key: float(np.mean([value[key] for value in current])) for key in
                                    ("auroc", "ap", "mrr", "top1", "ndcg", "macro_f1_0_5")}})
            del raw, plugin
            if device.type == "cuda": torch.cuda.empty_cache()

    patient_df = pd.DataFrame(all_patient_metrics)
    prediction_df = pd.DataFrame(all_predictions)
    # Exact pairing assertions occur before any aggregate inference.
    pivot = patient_df.pivot(index="patient_name", columns="variant", values=["auroc", "ap"])
    if len(pivot) != len(manifest) or pivot.isna().any().any():
        raise RuntimeError("Incomplete paired OOF patient results")
    delta = pd.DataFrame({"patient_name": pivot.index,
                          "baseline_patient_AUROC": pivot[("auroc", "baseline")].to_numpy(),
                          "plugin_patient_AUROC": pivot[("auroc", "plugin")].to_numpy(),
                          "delta_AUROC": (pivot[("auroc", "plugin")] - pivot[("auroc", "baseline")]).to_numpy(),
                          "baseline_patient_AP": pivot[("ap", "baseline")].to_numpy(),
                          "plugin_patient_AP": pivot[("ap", "plugin")].to_numpy(),
                          "delta_AP": (pivot[("ap", "plugin")] - pivot[("ap", "baseline")]).to_numpy()})
    patient_center = manifest.set_index("patient_name").center.astype(str)
    delta["center"] = delta.patient_name.map(patient_center)
    observed_auroc = float(delta.delta_AUROC.mean())
    observed_ap = float(delta.delta_AP.mean())
    auroc_draws, auroc_boot = bootstrap(delta.delta_AUROC.to_numpy())
    ap_draws, ap_boot = bootstrap(delta.delta_AP.to_numpy())
    paired_bootstrap = pd.DataFrame({"draw": np.arange(1, len(auroc_draws) + 1),
                                     "delta_patient_equal_AUROC": auroc_draws,
                                     "delta_patient_equal_AP": ap_draws})

    summary_rows = []
    for variant in ("baseline", "plugin"):
        subset = patient_df.loc[patient_df.variant == variant]
        row = {"variant": variant, "n_patients": len(subset)}
        for key in ("auroc", "ap", "mrr", "top1", "ndcg", "macro_f1_0_5", "pathological_f1_0_5", "normal_f1_0_5", "balanced_accuracy_0_5"):
            row.update(aggregate(subset.to_dict("records"), key))
        summary_rows.append(row)
    secondary_rows = []
    for variant in ("baseline", "plugin"):
        subset = prediction_df.loc[(prediction_df.variant == variant) & (prediction_df.label >= 0)]
        secondary_rows.append({"variant": variant, "unit": "patient_channel", "n_units": len(subset),
                               "pooled_AUROC": float(roc_auc_score(subset.label, subset.pathological_score)),
                               "pooled_AP": float(average_precision_score(subset.label, subset.pathological_score))})
    center_rows = []
    for (center, variant), subset in patient_df.groupby(["center", "variant"], sort=True):
        center_rows.append({"center": center, "variant": variant, "n_patients": len(subset),
                            "patient_equal_AUROC": float(subset.auroc.mean()), "patient_equal_AP": float(subset.ap.mean()),
                            "MRR": float(subset.mrr.mean()), "Top1": float(subset.top1.mean()), "NDCG": float(subset.ndcg.mean())})

    # Private, identifying artifacts: intentionally not copied into git.
    write_rows(private_root / "OOF_PATIENT_CHANNEL_PREDICTIONS.csv", all_predictions)
    write_rows(private_root / "OOF_PATIENT_METRICS.csv", all_patient_metrics)
    write_rows(private_root / "PAIRED_PATIENT_DELTAS.csv", delta.to_dict("records"))
    write_rows(private_root / "PAIRED_BOOTSTRAP.csv", paired_bootstrap.to_dict("records"))
    # Public-safe compact aggregates.
    args.public_output.mkdir(parents=True, exist_ok=True)
    write_rows(args.public_output / "FOLD_METRICS.csv", fold_metrics)
    write_rows(args.public_output / "PAIRED_BOOTSTRAP.csv", [
        {"metric": "AUROC", "observed_mean": observed_auroc, "bootstrap_mean": auroc_boot["mean"], **{key: value for key, value in auroc_boot.items() if key != "mean"}},
        {"metric": "AP", "observed_mean": observed_ap, "bootstrap_mean": ap_boot["mean"], **{key: value for key, value in ap_boot.items() if key != "mean"}},
    ])
    write_rows(args.public_output / "CENTERWISE_OOF_METRICS.csv", center_rows)
    write_rows(args.public_output / "SECONDARY_POOLED_METRICS.csv", secondary_rows)
    write_rows(args.public_output / "OOF_SUMMARY.csv", summary_rows)
    atomic_json(args.public_output / "OOF_EVALUATION_STATUS.json", {
        "status": "ONE_FROZEN_OOF_PASS_COMPLETE", "protocol_sha256": protocol_sha,
        "patients": int(len(manifest)), "folds": 5, "temporal_segments_per_record": SEGMENTS_PER_RECORD,
        "private_prediction_artifacts_retained_server_side": True,
        "test_fold_labels_used_for_optimization": False,
        "paired_auroc": {"observed_mean": observed_auroc, "bootstrap_mean": auroc_boot["mean"], **{key: value for key, value in auroc_boot.items() if key != "mean"},
                         "fraction_improved": float((delta.delta_AUROC > 0).mean()),
                         "fraction_unchanged": float((delta.delta_AUROC == 0).mean()),
                         "fraction_worsened": float((delta.delta_AUROC < 0).mean())},
        "paired_ap": {"observed_mean": observed_ap, "bootstrap_mean": ap_boot["mean"], **{key: value for key, value in ap_boot.items() if key != "mean"}},
    })


if __name__ == "__main__":
    main()
