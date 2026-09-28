"""Single frozen A1-Omni v2 test inference and prespecified summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, f1_score, ndcg_score,
                             roc_auc_score, roc_curve)

from train_v2 import V2Bank, atomic_json, check_source, make_model, sha256, to_device


def verify_freeze(args):
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    threshold = json.loads(args.threshold.read_text(encoding="utf-8"))
    if not freeze.get("model_frozen_before_official_test") or \
            not freeze.get("threshold_frozen_before_official_test") or \
            freeze.get("test_used_for_tuning") or \
            freeze["protocol_sha256"] != sha256(args.protocol) or \
            freeze["threshold_json_sha256"] != sha256(args.threshold) or \
            freeze["frozen_threshold"] != threshold["threshold"]:
        raise RuntimeError("Frozen v2 protocol/checkpoint/threshold mismatch")
    checkpoint = args.runtime / "final/last.pt"
    normalizer = args.runtime / "final/normalizer.npz"
    if sha256(checkpoint) != freeze["final_checkpoint_sha256"] or \
            sha256(normalizer) != freeze["final_normalizer_sha256"]:
        raise RuntimeError("Frozen A1 model or normalizer changed")
    return freeze, threshold


def inference(args):
    freeze, _ = verify_freeze(args)
    output_file = args.output / "CHANNEL_PREDICTIONS.csv"
    if output_file.exists():
        raise RuntimeError("Official v2 test predictions already exist; do not rerun inference")
    private = args.runtime / "official_test_predictions"
    private.mkdir(parents=True, exist_ok=True)
    started = private / "TEST_ACCESS_STARTED.json"
    if started.exists():
        old = json.loads(started.read_text(encoding="utf-8"))
        if old["final_checkpoint_sha256"] != freeze["final_checkpoint_sha256"] or \
                old["threshold_json_sha256"] != freeze["threshold_json_sha256"]:
            raise RuntimeError("Cannot resume official test with different model/threshold")
    else:
        atomic_json(started, {"final_checkpoint_sha256": freeze["final_checkpoint_sha256"],
                              "threshold_json_sha256": freeze["threshold_json_sha256"],
                              "test_used_for_tuning": False})
    collate, Model = check_source()
    bank = V2Bank(args.cohort, args.features, None, official_split="test",
                  protocol=args.protocol)
    if len(bank.patients) != 96:
        raise RuntimeError("Official-labeled test patient coverage changed")
    with np.load(args.runtime / "final/normalizer.npz") as norm:
        mean, std = norm["mean"], norm["std"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(Model, collate, bank, bank.patients[0], mean, std, device)
    state = torch.load(args.runtime / "final/last.pt", map_location=device,
                       weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    files = []
    with torch.inference_mode():
        for number, patient in enumerate(bank.patients, start=1):
            if any(piece in patient for piece in ("/", "\\", "..")):
                raise RuntimeError("Unsafe patient identifier")
            patient_file = private / f"{patient}.csv"
            files.append(patient_file)
            expected = sum(len(record["names"]) for record in bank.by_patient[patient])
            if patient_file.exists():
                prior = pd.read_csv(patient_file)
                if len(prior) != expected or prior.duplicated(["edf", "channel"]).any():
                    raise RuntimeError(f"Partial test prediction mismatch: {patient}")
                print(f"v2 official_test patient={number}/{len(bank.patients)} reused", flush=True)
                continue
            batch = to_device(collate([bank.example(patient, mean, std, epoch=None)]), device)
            logits = model(batch)["logits"][0].detach().cpu().numpy()
            scores = 1.0 - torch.sigmoid(torch.from_numpy(logits)).numpy()
            score_of = dict(zip(bank.canonical[patient], scores.tolist()))
            rows = []
            for record in bank.by_patient[patient]:
                for name in record["names"]:
                    rows.append({"patient": patient, "dataset": bank.patient_dataset[patient],
                                 "edf": record["edf"], "channel": name,
                                 "pathology": bank.edf_labels[record["edf"]][name],
                                 "score_pathology": float(score_of[name])})
            if len(rows) != expected:
                raise RuntimeError(f"Test EDF-channel expansion mismatch: {patient}")
            temp = patient_file.with_suffix(".csv.tmp")
            pd.DataFrame(rows).to_csv(temp, index=False)
            temp.replace(patient_file)
            print(f"v2 official_test patient={number}/{len(bank.patients)} edf_channels={expected}",
                  flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    if len(frame) != 8104 or frame.duplicated(["edf", "channel"]).any():
        raise RuntimeError("Official test EDF-channel denominator changed")
    temp = output_file.with_suffix(".csv.tmp")
    frame.to_csv(temp, index=False)
    temp.replace(output_file)
    atomic_json(args.output / "TEST_ACCESS_AUDIT.json", {
        "official_test_accessed": True, "model_frozen_before_test": True,
        "threshold_frozen_before_test": True, "test_used_for_tuning": False,
        "patients": 96, "edf_channel_records": len(frame),
        "final_checkpoint_sha256": freeze["final_checkpoint_sha256"],
        "threshold_json_sha256": freeze["threshold_json_sha256"],
        "predictions_sha256": sha256(output_file)})


def classification_metrics(frame: pd.DataFrame, threshold: float) -> dict:
    y = frame["pathology"].to_numpy(dtype=np.int8)
    score = frame["score_pathology"].to_numpy(dtype=np.float64)
    pred = (score >= threshold).astype(np.int8)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    positives = int(y.sum())
    negatives = len(y) - positives
    return {"edf_channel_records": int(len(y)), "pathological_records": positives,
            "pathological_prevalence": float(positives / len(y)),
            "threshold": float(threshold),
            "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
            "pathological_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
            "sensitivity": float(tp / positives) if positives else float("nan"),
            "specificity": float(tn / negatives) if negatives else float("nan"),
            "balanced_accuracy": float(0.5 * (tp / positives + tn / negatives))
            if positives and negatives else float("nan"),
            "pooled_ap": float(average_precision_score(y, score)) if positives else float("nan"),
            "pooled_auroc": float(roc_auc_score(y, score)) if positives and negatives else float("nan"),
            "predicted_pathological_fraction": float(pred.mean())}


def unique_patient_channels(frame: pd.DataFrame) -> pd.DataFrame:
    consistency = frame.groupby(["patient", "channel"])[["pathology", "score_pathology"]].nunique()
    if (consistency > 1).any().any():
        raise RuntimeError("Repeated EDF patient-channel label/score differs")
    return frame.drop_duplicates(["patient", "channel"]).copy()


def patient_metrics(frame: pd.DataFrame) -> dict:
    y = frame["pathology"].to_numpy(dtype=np.int8)
    score = frame["score_pathology"].to_numpy(dtype=np.float64)
    positive = int(y.sum())
    result = {"patient": str(frame["patient"].iat[0]),
              "dataset": str(frame["dataset"].iat[0]),
              "channels": len(y), "pathological_channels": positive,
              "estimable_for_ranking": bool(positive),
              "non_estimable_reason": "" if positive else "zero_pathological_channels"}
    if not positive:
        result.update({"ap": float("nan"), "mrr": float("nan"),
                       "top1": float("nan"), "ndcg": float("nan")})
        return result
    order = np.argsort(-score, kind="stable")
    ranks = np.flatnonzero(y[order] == 1)
    result.update({"ap": float(average_precision_score(y, score)),
                   "mrr": 1.0 / (1 + int(ranks[0])),
                   "top1": int(y[order[0]]),
                   "ndcg": (1.0 if len(y) == 1 else
                            float(ndcg_score(y[None, :], score[None, :])))})
    return result


def bootstrap(frame: pd.DataFrame, patient: pd.DataFrame, threshold: float,
              draws: int = 10000) -> pd.DataFrame:
    """Patient-cluster bootstrap; ties grouped to match sklearn AP/AUROC."""
    names = patient["patient"].tolist()
    patient_index = {name: idx for idx, name in enumerate(names)}
    groups = frame["patient"].map(patient_index).to_numpy(dtype=np.int32)
    y = frame["pathology"].to_numpy(dtype=np.int8)
    score = frame["score_pathology"].to_numpy(dtype=np.float64)
    pred = (score >= threshold).astype(np.int8)
    order = np.argsort(-score, kind="stable")
    sorted_y = y[order]
    sorted_group = groups[order]
    sorted_score = score[order]
    group_start = np.r_[0, np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1]
    estimable = patient.loc[patient["estimable_for_ranking"]]
    rank_ids = [patient_index[name] for name in estimable["patient"]]
    rng_class = np.random.default_rng(42)
    rng_rank = np.random.default_rng(42)
    rows = []
    for draw in range(draws):
        count = np.bincount(rng_class.integers(0, len(names), size=len(names)), minlength=len(names))
        w = count[groups]
        ws = count[sorted_group]
        positive = float(np.dot(w, y))
        negative = float(w.sum() - positive)
        score_positive = np.add.reduceat(ws * sorted_y, group_start)
        score_total = np.add.reduceat(ws, group_start)
        cum_positive = np.cumsum(score_positive)
        cum_total = np.cumsum(score_total)
        precision = np.divide(cum_positive, cum_total, out=np.zeros_like(cum_positive, dtype=float),
                              where=cum_total > 0)
        ap = float(np.dot(score_positive, precision) / positive) if positive else float("nan")
        cum_negative = cum_total - cum_positive
        auc = float(np.trapezoid(np.r_[0, cum_positive / positive],
                                 np.r_[0, cum_negative / negative])) if positive and negative else float("nan")
        tp = float(np.dot(w, (y == 1) & (pred == 1)))
        tn = float(np.dot(w, (y == 0) & (pred == 0)))
        fp = float(np.dot(w, (y == 0) & (pred == 1)))
        fn = float(np.dot(w, (y == 1) & (pred == 0)))
        macro = 0.5 * (2 * tp / max(2 * tp + fp + fn, 1) +
                       2 * tn / max(2 * tn + fp + fn, 1))
        rank_draw = rng_rank.integers(0, len(rank_ids), size=len(rank_ids))
        sampled = estimable.iloc[rank_draw]
        rows.append({"draw": draw, "pooled_ap": ap, "pooled_auroc": auc,
                     "macro_f1": macro,
                     "sensitivity": tp / positive if positive else float("nan"),
                     "specificity": tn / negative if negative else float("nan"),
                     "patient_equal_ap": float(sampled["ap"].mean()),
                     "mrr": float(sampled["mrr"].mean()),
                     "top1": float(sampled["top1"].mean())})
        if (draw + 1) % 1000 == 0:
            print(f"v2 bootstrap {draw+1}/{draws}", flush=True)
    ci = pd.DataFrame(rows).drop(columns="draw").quantile([0.025, 0.975]).T.reset_index()
    ci.columns = ["metric", "ci_lower", "ci_upper"]
    return ci


def summarize(args):
    freeze, frozen_threshold = verify_freeze(args)
    frame = pd.read_csv(args.output / "CHANNEL_PREDICTIONS.csv")
    if len(frame) != 8104 or frame["patient"].nunique() != 96 or \
            frame.duplicated(["edf", "channel"]).any():
        raise RuntimeError("Official v2 test prediction coverage mismatch")
    unique = unique_patient_channels(frame)
    patient = pd.DataFrame([patient_metrics(group)
                            for _, group in unique.groupby("patient", sort=True)])
    patient.to_csv(args.output / "PATIENT_LEVEL_METRICS.csv", index=False)
    estimable = patient.loc[patient["estimable_for_ranking"]]
    frozen = classification_metrics(frame, frozen_threshold["threshold"])
    f05 = classification_metrics(frame, 0.5)
    fpr, tpr, thresholds = roc_curve(frame["pathology"], frame["score_pathology"])
    youden = classification_metrics(frame, float(thresholds[np.argmax(tpr - fpr)]))
    primary = {"classification_unit": "official EDF-channel pair",
               "patients": len(patient), "edfs": int(frame["edf"].nunique()),
               "unique_patient_channels": len(unique),
               "estimable_ranking_patients": len(estimable),
               "non_estimable_ranking_patients": len(patient) - len(estimable),
               "non_estimable_reason": "zero_pathological_channels",
               "patient_equal_ap": float(estimable["ap"].mean()),
               "patient_median_ap": float(estimable["ap"].median()),
               "patient_ap_q1": float(estimable["ap"].quantile(0.25)),
               "patient_ap_q3": float(estimable["ap"].quantile(0.75)),
               "patient_equal_mrr": float(estimable["mrr"].mean()),
               "patient_equal_top1": float(estimable["top1"].mean()),
               "patient_equal_ndcg": float(estimable["ndcg"].mean()),
               **frozen}
    atomic_json(args.output / "PRIMARY_METRICS.json", primary)
    ablation = pd.DataFrame([{"setting": "fixed_0.5", **f05},
                             {"setting": "validation_frozen", **frozen},
                             {"setting": "posthoc_test_youden_reproduction_only", **youden}])
    ablation.to_csv(args.output / "THRESHOLD_ABLATION.csv", index=False)
    strata = []
    for dataset, subset in frame.groupby("dataset", sort=True):
        group_patients = patient.loc[patient["dataset"] == dataset]
        group_estimable = group_patients.loc[group_patients["estimable_for_ranking"]]
        classification = classification_metrics(subset, frozen_threshold["threshold"])
        binary_evaluable = 0 < classification["pathological_records"] < len(subset)
        if not binary_evaluable:
            for metric in ("macro_f1", "pathological_f1", "sensitivity",
                           "balanced_accuracy", "pooled_ap", "pooled_auroc"):
                classification[metric] = float("nan")
        strata.append({"dataset": dataset, "patients": len(group_patients),
                       "binary_evaluable": binary_evaluable,
                       "non_evaluable_reason": "" if binary_evaluable else "single_class_official_labels",
                       "estimable_ranking_patients": len(group_estimable),
                       "patient_equal_ap": float(group_estimable["ap"].mean())
                       if len(group_estimable) else float("nan"),
                       "mrr": float(group_estimable["mrr"].mean())
                       if len(group_estimable) else float("nan"),
                       "top1": float(group_estimable["top1"].mean())
                       if len(group_estimable) else float("nan"),
                       **classification})
    pd.DataFrame(strata).to_csv(args.output / "DATASET_STRATIFIED_METRICS.csv", index=False)
    ci = bootstrap(frame, patient, frozen_threshold["threshold"], draws=10000)
    ci.to_csv(args.output / "PATIENT_CLUSTER_BOOTSTRAP.csv", index=False)
    print(json.dumps({"status": "COMPLETE", "primary": primary,
                      "freeze_checkpoint_sha256": freeze["final_checkpoint_sha256"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["inference", "summary"], required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--threshold", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "inference":
        inference(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
