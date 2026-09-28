"""Frozen N1 one-time official test inference and patient-paired summaries."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_tf_omni import (V2TFBank, make_model, wrap_collate, check_source,
                           to_device, sha256, atomic_json)

V2_CODE = Path(__file__).resolve().parents[2] / "omni_ieeg_a1_interictal_v2_seed42/code"
if not V2_CODE.is_dir():
    V2_CODE = Path("E:/DRE-nips/new-pipeline/7-11/omni_a1_v2/code")
sys.path.insert(0, str(V2_CODE))
from evaluate_v2 import (classification_metrics, unique_patient_channels,
                         patient_metrics, bootstrap)  # noqa: E402

LOCK = "ace2017e01d0d21e3dda8ddd4d42551366704430b8a10f3666f955150150630d"


def verify(a):
    freeze = json.loads(a.freeze.read_text(encoding="utf-8"))
    marker = json.loads((a.runtime / "official_test_access_started.json").read_text(encoding="utf-8"))
    if freeze["protocol_sha256"] != LOCK or sha256(a.protocol) != LOCK:
        raise RuntimeError("A1-TF protocol changed after freeze")
    if marker["checkpoint_sha256"] != freeze["final_checkpoint_sha256"] or not marker["training_resume_forbidden"]:
        raise RuntimeError("Official test access marker does not match frozen model")
    if sha256(a.runtime / "final/last.pt") != freeze["final_checkpoint_sha256"]:
        raise RuntimeError("N1 checkpoint changed after test access")
    if sha256(a.threshold) != freeze["threshold_json_sha256"]:
        raise RuntimeError("N1 frozen threshold changed")
    return freeze


def inference(a):
    freeze = verify(a)
    private = a.runtime / "official_test_predictions"
    private.mkdir(parents=True, exist_ok=True)
    collate_old, Model = check_source()
    bank = V2TFBank(a.cohort, a.features, a.tf_features, None,
                    a.v2_protocol, a.protocol, official_split="test")
    if len(bank.patients) != 96:
        raise RuntimeError("Official test patient coverage changed")
    norm = a.runtime / "final/normalizer.npz"
    with np.load(norm) as values:
        mean, std, tf_mean, tf_std = [values[k] for k in ("mean", "std", "tf_mean", "tf_std")]
    bank.set_tf_normalizer(tf_mean, tf_std)
    collate = wrap_collate(collate_old)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(Model, collate, bank, bank.patients[0], mean, std, device)
    state = torch.load(a.runtime / "final/last.pt", map_location=device, weights_only=False)
    if state["protocol_sha256"] != LOCK or state["epoch"] != freeze["selected_inner_epoch"]:
        raise RuntimeError("Frozen N1 epoch/protocol differs")
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    files = []
    with torch.inference_mode():
        for number, patient in enumerate(bank.patients, 1):
            if any(piece in patient for piece in ("/", "\\", "..")):
                raise RuntimeError("Unsafe patient identifier")
            path = private / f"{patient}.csv"
            files.append(path)
            expected = sum(len(record["names"]) for record in bank.by_patient[patient])
            if path.is_file():
                old = pd.read_csv(path)
                if len(old) != expected or old.duplicated(["edf", "channel"]).any() or \
                        not {"score_pathology", "score_alpha0"} <= set(old):
                    raise RuntimeError(f"Partial N1 prediction mismatch: {patient}")
                print(f"N1 official_test patient={number}/96 reused", flush=True)
                continue
            batch = to_device(collate([bank.example(patient, mean, std, epoch=None)]), device)
            original_alpha = model.alpha.detach().clone()
            full = model(batch)["logits"][0].detach().cpu().numpy()
            model.alpha.zero_()
            zero = model(batch)["logits"][0].detach().cpu().numpy()
            model.alpha.copy_(original_alpha)
            scores = (1.0 - torch.sigmoid(torch.from_numpy(full))).numpy()
            no_tf = (1.0 - torch.sigmoid(torch.from_numpy(zero))).numpy()
            index = {name: idx for idx, name in enumerate(bank.canonical[patient])}
            rows = []
            for record in bank.by_patient[patient]:
                for name in record["names"]:
                    idx = index[name]
                    rows.append({"patient": patient, "dataset": bank.patient_dataset[patient],
                                 "edf": record["edf"], "channel": name,
                                 "pathology": bank.edf_labels[record["edf"]][name],
                                 "score_pathology": float(scores[idx]),
                                 "score_alpha0": float(no_tf[idx])})
            if len(rows) != expected:
                raise RuntimeError(f"N1 patient denominator mismatch: {patient}")
            temporary = path.with_suffix(".csv.tmp")
            pd.DataFrame(rows).to_csv(temporary, index=False)
            os.replace(temporary, path)
            print(f"N1 official_test patient={number}/96 edf_channels={expected}", flush=True)
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    if len(frame) != 8104 or frame["patient"].nunique() != 96 or frame.duplicated(["edf", "channel"]).any():
        raise RuntimeError("Official test EDF-channel coverage changed")
    all_file = private / "N1_CHANNEL_PREDICTIONS_PRIVATE.csv"
    temporary = all_file.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, all_file)
    atomic_json(a.output / "INTERICTAL_TEST_ACCESS_AUDIT.json", {
        "official_test_accessed": True, "model_frozen_before_test": True,
        "threshold_frozen_before_test": True, "test_used_for_tuning": False,
        "patients": 96, "edf_channel_records": 8104,
        "final_checkpoint_sha256": freeze["final_checkpoint_sha256"],
        "predictions_sha256": sha256(all_file),
        "prediction_rows_private_not_published": True,
    })


def weighted_ap_auc(sorted_y, sorted_groups, start, patient_weights):
    sorted_weight = patient_weights[sorted_groups]
    positive_by_tie = np.add.reduceat(sorted_weight * sorted_y, start)
    total_by_tie = np.add.reduceat(sorted_weight, start)
    positives = positive_by_tie.sum()
    negatives = total_by_tie.sum() - positives
    cumulative_positive = np.cumsum(positive_by_tie)
    cumulative_total = np.cumsum(total_by_tie)
    precision = np.divide(cumulative_positive, cumulative_total,
                          out=np.zeros_like(cumulative_total, dtype=float), where=cumulative_total > 0)
    ap = float(np.dot(positive_by_tie, precision) / positives) if positives else float("nan")
    auc = (float(np.trapezoid(np.r_[0, cumulative_positive / positives],
                             np.r_[0, (cumulative_total - cumulative_positive) / negatives]))
           if positives and negatives else float("nan"))
    return ap, auc


def paired_bootstrap(n1, n0, patients_n1, patients_n0, threshold1, threshold0, draws=10000):
    keys = ["patient", "edf", "channel"]
    merged = n1.merge(n0[keys + ["score_pathology", "pathology"]], on=keys,
                      suffixes=("_n1", "_n0"), validate="one_to_one")
    if len(merged) != 8104 or not (merged["pathology_n1"] == merged["pathology_n0"]).all():
        raise RuntimeError("N0/N1 official test labels or EDF-channel rows differ")
    names = sorted(n1["patient"].unique())
    by_name = {name: idx for idx, name in enumerate(names)}
    group = merged["patient"].map(by_name).to_numpy(dtype=np.int32)
    y = merged["pathology_n1"].to_numpy(dtype=np.int8)
    scores1 = merged["score_pathology_n1"].to_numpy(dtype=np.float64)
    scores0 = merged["score_pathology_n0"].to_numpy(dtype=np.float64)
    patient1 = patients_n1.set_index("patient").loc[names]
    patient0 = patients_n0.set_index("patient").loc[names]
    estimable = patient1["estimable_for_ranking"].to_numpy(dtype=bool)
    if not np.array_equal(estimable, patient0["estimable_for_ranking"].to_numpy(dtype=bool)):
        raise RuntimeError("N0/N1 ranking eligibility differs")
    rng = np.random.default_rng(42)
    ranked = []
    for score in (scores1, scores0):
        order = np.argsort(-score, kind="stable")
        sorted_score = score[order]
        ranked.append((y[order], group[order],
                       np.r_[0, np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1]))
    pred1 = scores1 >= threshold1
    pred0 = scores0 >= threshold0
    confusion = []
    for pred in (pred1, pred0):
        confusion.append(np.stack([
            np.bincount(group[(y == truth) & (pred == decision)], minlength=len(names))
            for truth, decision in ((1, 1), (0, 0), (0, 1), (1, 0))]))
    rows = []
    for draw in range(draws):
        weights = np.bincount(rng.integers(0, len(names), size=len(names)), minlength=len(names))
        ap1, auc1 = weighted_ap_auc(*ranked[0], weights)
        ap0, auc0 = weighted_ap_auc(*ranked[1], weights)
        f1 = []
        for table in confusion:
            tp, tn, fp, fn = table @ weights
            f1.append(0.5 * (2 * tp / max(2 * tp + fp + fn, 1) +
                             2 * tn / max(2 * tn + fp + fn, 1)))
        row = {"delta_pooled_ap": ap1 - ap0, "delta_auroc": auc1 - auc0,
               "delta_macro_f1": f1[0] - f1[1]}
        for metric in ("ap", "mrr", "top1", "ndcg"):
            w = weights[estimable]
            denominator = w.sum()
            a = patient1.loc[estimable, metric].to_numpy(dtype=float)
            b = patient0.loc[estimable, metric].to_numpy(dtype=float)
            row["delta_patient_equal_" + metric] = float(np.dot(w, a - b) / denominator)
        rows.append(row)
        if (draw + 1) % 1000 == 0:
            print(f"N1 paired bootstrap {draw+1}/{draws}", flush=True)
    frame = pd.DataFrame(rows)
    return frame.quantile([0.025, 0.975]).T.reset_index().rename(columns={
        "index": "metric", 0.025: "ci_lower", 0.975: "ci_upper"})


def summarize(a):
    freeze = verify(a)
    frame = pd.read_csv(a.runtime / "official_test_predictions/N1_CHANNEL_PREDICTIONS_PRIVATE.csv")
    if len(frame) != 8104:
        raise RuntimeError("N1 official prediction coverage incomplete")
    threshold = float(json.loads(a.threshold.read_text(encoding="utf-8"))["threshold"])
    unique = unique_patient_channels(frame)
    patient = pd.DataFrame([patient_metrics(group) for _, group in unique.groupby("patient", sort=True)])
    patient.to_csv(a.output / "INTERICTAL_PATIENT_LEVEL_METRICS.csv", index=False)
    estimable = patient.loc[patient["estimable_for_ranking"]]
    classification = classification_metrics(frame, threshold)
    primary = {"patients": 96, "edf_channel_records": 8104,
               "estimable_ranking_patients": int(len(estimable)),
               "patient_equal_ap": float(estimable["ap"].mean()),
               "mrr": float(estimable["mrr"].mean()),
               "top1": float(estimable["top1"].mean()),
               "ndcg": float(estimable["ndcg"].mean()),
               "trained_alpha": float(torch.load(a.runtime / "final/last.pt", map_location="cpu",
                                                 weights_only=False)["model"]["alpha"]),
               **classification}
    atomic_json(a.output / "INTERICTAL_PRIMARY_METRICS.json", primary)
    half = classification_metrics(frame, 0.5)
    pd.DataFrame([{"setting": "frozen_validation_threshold", **classification},
                  {"setting": "fixed_0.5", **half}]).to_csv(
                      a.output / "INTERICTAL_THRESHOLD_ABLATION.csv", index=False)
    no_tf = frame.rename(columns={"score_pathology": "score_full",
                                  "score_alpha0": "score_pathology"})
    no_tf_unique = unique_patient_channels(no_tf)
    no_tf_patient = pd.DataFrame([patient_metrics(group)
                                  for _, group in no_tf_unique.groupby("patient", sort=True)])
    no_tf_estimable = no_tf_patient.loc[no_tf_patient["estimable_for_ranking"]]
    off = {"alpha": 0.0, "threshold": threshold,
           "patient_equal_ap": float(no_tf_estimable["ap"].mean()),
           "mrr": float(no_tf_estimable["mrr"].mean()),
           "top1": float(no_tf_estimable["top1"].mean()),
           **classification_metrics(no_tf, threshold)}
    pd.DataFrame([{"setting": "full", **primary}, {"setting": "alpha_zero", **off}]).to_csv(
        a.output / "INTERICTAL_TF_INTERVENTION.csv", index=False)
    strata = []
    for dataset, subset in frame.groupby("dataset", sort=True):
        group_patients = patient.loc[patient["dataset"] == dataset]
        group_estimable = group_patients.loc[group_patients["estimable_for_ranking"]]
        metrics = classification_metrics(subset, threshold)
        if not 0 < metrics["pathological_records"] < len(subset):
            for key in ("macro_f1", "pathological_f1", "balanced_accuracy",
                        "pooled_ap", "pooled_auroc"):
                metrics[key] = float("nan")
        strata.append({"dataset": dataset, "patients": len(group_patients),
                       "patient_equal_ap": float(group_estimable["ap"].mean())
                       if len(group_estimable) else float("nan"),
                       "mrr": float(group_estimable["mrr"].mean()) if len(group_estimable) else float("nan"),
                       "top1": float(group_estimable["top1"].mean()) if len(group_estimable) else float("nan"),
                       **metrics})
    pd.DataFrame(strata).to_csv(a.output / "INTERICTAL_DATASET_STRATIFIED_METRICS.csv", index=False)
    n0 = pd.read_csv(a.n0_predictions)
    n0_patient = pd.DataFrame([patient_metrics(group)
                               for _, group in unique_patient_channels(n0).groupby("patient", sort=True)])
    n0_threshold = float(json.loads(a.n0_threshold.read_text(encoding="utf-8"))["threshold"])
    primary_ci = bootstrap(frame, patient, threshold)
    primary_ci["comparison"] = "N1_estimate"
    paired = paired_bootstrap(frame, n0, patient, n0_patient, threshold, n0_threshold)
    paired["comparison"] = "N1_minus_N0"
    pd.concat([primary_ci, paired], ignore_index=True).to_csv(
        a.output / "INTERICTAL_PATIENT_BOOTSTRAP.csv", index=False)
    comparison = {"N0_macro_f1": classification_metrics(n0, n0_threshold)["macro_f1"],
                  "N0_auroc": classification_metrics(n0, n0_threshold)["pooled_auroc"],
                  "N0_patient_equal_ap": float(n0_patient.loc[n0_patient["estimable_for_ranking"], "ap"].mean()),
                  "N1_macro_f1": primary["macro_f1"], "N1_auroc": primary["pooled_auroc"],
                  "N1_patient_equal_ap": primary["patient_equal_ap"],
                  "N1_frozen_threshold": threshold,
                  "checkpoint_sha256": freeze["final_checkpoint_sha256"],
                  "test_used_for_tuning": False}
    atomic_json(a.output / "INTERICTAL_COMPARISON.json", comparison)
    print(json.dumps(comparison, sort_keys=True), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("inference", "summary"), required=True)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--tf-features", type=Path, required=True)
    p.add_argument("--v2-protocol", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--threshold", type=Path, required=True)
    p.add_argument("--n0-predictions", type=Path)
    p.add_argument("--n0-threshold", type=Path)
    a = p.parse_args()
    if a.phase == "inference":
        inference(a)
    else:
        if a.n0_predictions is None or a.n0_threshold is None:
            raise RuntimeError("Paired N0 official scores and frozen threshold required")
        summarize(a)


if __name__ == "__main__":
    main()
