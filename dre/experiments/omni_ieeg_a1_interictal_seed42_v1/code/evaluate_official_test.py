"""One frozen A1-Omni test pass, followed by fixed descriptive statistics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, ndcg_score, roc_auc_score)

from train_a1_omni import (Bank, check_source, make_model, sha256, to_device)


def patient_metrics(frame: pd.DataFrame) -> dict:
    y = frame["soz"].to_numpy(dtype=np.int8)
    score = frame["score_ez"].to_numpy(dtype=np.float64)
    order = np.argsort(-score, kind="stable")
    ranks = np.flatnonzero(y[order] == 1)
    ap = float(average_precision_score(y, score)) if y.sum() else 0.0
    auc = float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan")
    return {
        "patient": str(frame["patient"].iat[0]), "dataset": str(frame["dataset"].iat[0]),
        "channels": len(y), "soz_channels": int(y.sum()), "ap": ap, "auroc": auc,
        "mrr": 1.0 / (1 + int(ranks[0])) if len(ranks) else 0.0,
        "top1_soz": int(y[order[0]]) if len(y) else 0,
        "ndcg": float(ndcg_score(y[None, :], score[None, :])) if y.sum() else 0.0,
    }


def aggregate(frame: pd.DataFrame, patient: pd.DataFrame) -> dict:
    y = frame["soz"].to_numpy(dtype=np.int8)
    score = frame["score_ez"].to_numpy(dtype=np.float64)
    pred = (score >= 0.5).astype(np.int8)
    return {
        "patients": int(patient.shape[0]), "channels": int(len(y)),
        "soz_channels": int(y.sum()), "soz_prevalence": float(y.mean()),
        "pooled_ez_ap": float(average_precision_score(y, score)),
        "pooled_auroc": float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan"),
        "patient_equal_mean_ap": float(patient["ap"].mean()),
        "patient_median_ap": float(patient["ap"].median()),
        "patient_ap_q1": float(patient["ap"].quantile(0.25)),
        "patient_ap_q3": float(patient["ap"].quantile(0.75)),
        "patient_equal_mrr": float(patient["mrr"].mean()),
        "patient_equal_top1": float(patient["top1_soz"].mean()),
        "patient_equal_ndcg": float(patient["ndcg"].mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
        "ez_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "predicted_soz_fraction": float(pred.mean()),
        "patients_without_soz": int((patient["soz_channels"] == 0).sum()),
    }


def bootstrap(frame: pd.DataFrame, patient: pd.DataFrame, draws: int = 10000) -> pd.DataFrame:
    """Patient-cluster bootstrap with fixed score order for fast weighted AP/AUC."""
    names = patient["patient"].tolist()
    ids = {name: idx for idx, name in enumerate(names)}
    group = frame["patient"].map(ids).to_numpy(dtype=np.int32)
    y = frame["soz"].to_numpy(dtype=np.int8)
    score = frame["score_ez"].to_numpy(dtype=np.float64)
    pred = (score >= 0.5).astype(np.int8)
    order = np.argsort(-score, kind="stable")
    ys = y[order]
    groups = group[order]
    score_sorted = score[order]
    group_start = np.r_[0, np.flatnonzero(score_sorted[1:] != score_sorted[:-1]) + 1]
    rng = np.random.default_rng(42)
    rows = []
    ap_p = patient["ap"].to_numpy()
    mrr_p = patient["mrr"].to_numpy()
    top_p = patient["top1_soz"].to_numpy()
    for draw in range(draws):
        counts = np.bincount(rng.integers(0, len(names), size=len(names)), minlength=len(names))
        w = counts[group]
        ws = counts[groups]
        positives = float(np.dot(w, y))
        negatives = float(w.sum() - positives)
        positive_at_score = np.add.reduceat(ws * ys, group_start)
        total_at_score = np.add.reduceat(ws, group_start)
        tp_by_score = np.cumsum(positive_at_score)
        total_by_score = np.cumsum(total_at_score)
        precision = np.divide(tp_by_score, total_by_score,
                              out=np.zeros_like(tp_by_score, dtype=float), where=total_by_score > 0)
        ap = float(np.dot(positive_at_score, precision) / positives) if positives else float("nan")
        fp_by_score = total_by_score - tp_by_score
        auc = float(np.trapezoid(np.r_[0, tp_by_score / positives],
                                 np.r_[0, fp_by_score / negatives])) if positives and negatives else float("nan")
        tp = float(np.dot(w, (y == 1) & (pred == 1)))
        fp = float(np.dot(w, (y == 0) & (pred == 1)))
        fn = float(np.dot(w, (y == 1) & (pred == 0)))
        tn = float(np.dot(w, (y == 0) & (pred == 0)))
        f1_ez = 2 * tp / max(2 * tp + fp + fn, 1)
        f1_nez = 2 * tn / max(2 * tn + fp + fn, 1)
        rows.append({"draw": draw, "pooled_ez_ap": ap, "pooled_auroc": auc,
                     "patient_equal_ap": float(np.dot(counts, ap_p) / len(names)),
                     "mrr": float(np.dot(counts, mrr_p) / len(names)),
                     "top1": float(np.dot(counts, top_p) / len(names)),
                     "macro_f1": 0.5 * (f1_ez + f1_nez)})
        if (draw + 1) % 1000 == 0:
            print(f"bootstrap {draw+1}/{draws}", flush=True)
    result = pd.DataFrame(rows)
    ci = result.drop(columns="draw").quantile([0.025, 0.975]).T.reset_index()
    ci.columns = ["metric", "ci_lower", "ci_upper"]
    return ci


def inference(args):
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if not freeze.get("model_frozen_before_official_test") or freeze["protocol_sha256"] != sha256(args.protocol):
        raise RuntimeError("Missing or changed pre-test freeze")
    checkpoint = args.runtime / "final/last.pt"
    normalizer_path = args.runtime / "final/normalizer.npz"
    if sha256(checkpoint) != freeze["final_checkpoint_sha256"] or sha256(normalizer_path) != freeze["final_normalizer_sha256"]:
        raise RuntimeError("Frozen model/normalizer changed")
    output_file = args.output / "CHANNEL_PREDICTIONS.csv"
    if output_file.exists():
        raise RuntimeError("Official test predictions already exist; do not rerun model inference")
    collate, Model = check_source()
    bank = Bank(args.cohort, args.features, None, official_split="test")
    with np.load(normalizer_path) as norm:
        mean, std = norm["mean"], norm["std"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(Model, collate, bank, bank.patients[0], mean, std, device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    cohort = pd.read_csv(args.cohort)
    test = cohort.loc[cohort["official_split"] == "test"]
    dataset_of = test.groupby("patient")["dataset"].first().to_dict()
    private_predictions = args.runtime / "official_test_predictions"
    private_predictions.mkdir(parents=True, exist_ok=True)
    started = private_predictions / "TEST_ACCESS_STARTED.json"
    if started.exists():
        previous = json.loads(started.read_text(encoding="utf-8"))
        if previous["final_checkpoint_sha256"] != freeze["final_checkpoint_sha256"]:
            raise RuntimeError("Cannot resume test with a different frozen checkpoint")
    else:
        started.write_text(json.dumps({"final_checkpoint_sha256": freeze["final_checkpoint_sha256"],
                                       "test_used_for_tuning": False}, indent=2) + "\n", encoding="utf-8")
    files = []
    with torch.inference_mode():
        for number, patient in enumerate(bank.patients, start=1):
            if any(part in patient for part in ("/", "\\", "..")):
                raise RuntimeError("Unsafe patient identifier")
            patient_file = private_predictions / f"{patient}.csv"
            files.append(patient_file)
            if patient_file.exists():
                cached = pd.read_csv(patient_file)
                if len(cached) != len(bank.canonical[patient]) or set(cached["channel"]) != set(bank.canonical[patient]):
                    raise RuntimeError(f"Partial test prediction mismatch for {patient}")
                print(f"official_test patient={number}/{len(bank.patients)} reused", flush=True)
                continue
            example = bank.example(patient, mean, std, epoch=None)
            batch = to_device(collate([example]), device)
            logits = model(batch)["logits"][0].detach().cpu().numpy()
            scores = 1.0 - torch.sigmoid(torch.from_numpy(logits)).numpy()
            names = bank.canonical[patient]
            labels = [bank.labels[patient][name] for name in names]
            rows = [{"patient": patient, "dataset": dataset_of[patient],
                     "channel": name, "soz": label, "score_ez": float(score)}
                    for name, label, score in zip(names, labels, scores)]
            partial = patient_file.with_suffix(".csv.tmp")
            pd.DataFrame(rows).to_csv(partial, index=False)
            partial.replace(patient_file)
            print(f"official_test patient={number}/{len(bank.patients)} channels={len(names)} records={len(example['b0_features'])}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    tmp = output_file.with_suffix(".csv.tmp")
    pd.concat([pd.read_csv(path) for path in files], ignore_index=True).to_csv(tmp, index=False)
    tmp.replace(output_file)
    (args.output / "TEST_ACCESS_AUDIT.json").write_text(
        json.dumps({"official_test_accessed": True, "test_used_for_tuning": False,
                    "model_frozen_before_test": True, "patients": len(bank.patients),
                    "final_checkpoint_sha256": freeze["final_checkpoint_sha256"],
                    "predictions_sha256": sha256(output_file)}, indent=2) + "\n", encoding="utf-8")


def summarize(args):
    frame = pd.read_csv(args.output / "CHANNEL_PREDICTIONS.csv")
    patient = pd.DataFrame([patient_metrics(group) for _, group in frame.groupby("patient", sort=True)])
    patient.to_csv(args.output / "PATIENT_LEVEL_METRICS.csv", index=False)
    primary = aggregate(frame, patient)
    (args.output / "PRIMARY_METRICS.json").write_text(json.dumps(primary, indent=2) + "\n", encoding="utf-8")
    groups = []
    for dataset, subset in frame.groupby("dataset", sort=True):
        scores = aggregate(subset, patient.loc[patient["dataset"] == dataset])
        groups.append({"dataset": dataset, **scores})
    pd.DataFrame(groups).to_csv(args.output / "DATASET_STRATIFIED_METRICS.csv", index=False)
    ci = bootstrap(frame, patient, 10000)
    ci.to_csv(args.output / "PATIENT_CLUSTER_BOOTSTRAP.csv", index=False)
    print(json.dumps({"status": "COMPLETE", "primary": primary}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["inference", "summary"], required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "inference":
        inference(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
