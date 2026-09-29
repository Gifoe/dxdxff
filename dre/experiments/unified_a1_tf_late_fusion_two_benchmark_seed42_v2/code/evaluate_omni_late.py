"""Validation-only beta/threshold freeze, then one official Omni test pass."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "unified_a1_tf_two_benchmark_seed42_v1" / "code"))
from train_tf_omni import V2TFBank, wrap_collate  # noqa: E402
from train_v2 import check_source, make_model, to_device  # noqa: E402
from evaluate_v2 import classification_metrics, unique_patient_channels, patient_metrics  # noqa: E402
from evaluate_tf_omni import paired_bootstrap  # noqa: E402
from train_tf_omni_late import sha, save_json  # noqa: E402
from late_tf import TFScorer, robust_channel_normalize  # noqa: E402

BETA = (0., .02, .05, .1, .2, .35, .5, .75, 1.)


def models(a, bank, patient, stage):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    old_collate, Model = check_source()
    collate = wrap_collate(old_collate)
    with np.load(a.nbase_runtime / stage / "normalizer.npz") as d:
        a_mean, a_std = d["mean"], d["std"]
    with np.load(a.tf_runtime / stage / "normalizer.npz") as d:
        t_mean, t_std, tf_mean, tf_std = [d[k] for k in ("mean", "std", "tf_mean", "tf_std")]
    bank.set_tf_normalizer(tf_mean, tf_std)
    a1 = make_model(Model, collate, bank, patient, a_mean, a_std, device)
    astate = torch.load(a.nbase_runtime / stage / ("best.pt" if stage == "inner" else "last.pt"),
                        map_location=device, weights_only=False)
    a1.load_state_dict(astate["model"], strict=True)
    tf = TFScorer().to(device)
    tstate = torch.load(a.tf_runtime / stage / ("best.pt" if stage == "inner" else "last.pt"),
                        map_location=device, weights_only=False)
    if tstate["protocol_sha256"] != sha(a.lock):
        raise RuntimeError("TF checkpoint protocol mismatch")
    tf.load_state_dict(tstate["model"], strict=True)
    return a1.eval(), tf.eval(), collate, (a_mean, a_std), (t_mean, t_std), device


def predict(a, bank, patients, stage):
    a1, tf, collate, anorm, tnorm, device = models(a, bank, patients[0], stage)
    rows = []
    with torch.inference_mode():
        for number, patient in enumerate(patients, 1):
            a_batch = to_device(collate([bank.example(patient, *anorm, epoch=None)]), device)
            t_batch = to_device(collate([bank.example(patient, *tnorm, epoch=None)]), device)
            a_margin = -a1(a_batch)["logits"]
            t_margin = tf(t_batch)["margin_tf"]
            z = robust_channel_normalize(t_margin, t_batch["channel_mask"])
            n = len(bank.canonical[patient])
            a_values = a_margin[0, :n].cpu().numpy()
            t_values = t_margin[0, :n].cpu().numpy()
            z_values = z[0, :n].cpu().numpy()
            index = {name: i for i, name in enumerate(bank.canonical[patient])}
            for record in bank.by_patient[patient]:
                for name in record["names"]:
                    i = index[name]
                    rows.append({"patient": patient, "dataset": bank.patient_dataset[patient],
                                 "edf": record["edf"], "channel": name,
                                 "pathology": bank.edf_labels[record["edf"]][name],
                                 "a1_margin": float(a_values[i]), "tf_margin": float(t_values[i]),
                                 "z_tf": float(z_values[i])})
            if number % 20 == 0:
                print(f"{stage} patient={number}/{len(patients)}", flush=True)
    frame = pd.DataFrame(rows)
    if frame.duplicated(["edf", "channel"]).any():
        raise RuntimeError("Duplicate EDF/channel in prediction")
    return frame


def score_frame(raw, beta, tf_only=False):
    frame = raw[["patient", "dataset", "edf", "channel", "pathology"]].copy()
    margin = raw["tf_margin"].to_numpy() if tf_only else raw["a1_margin"].to_numpy() + beta * raw["z_tf"].to_numpy()
    frame["score_pathology"] = torch.sigmoid(torch.as_tensor(margin, dtype=torch.float64)).numpy()
    # A patient-channel can appear in more than one EDF. The model scores its
    # canonical channel once; floating arithmetic on repeated rows may vary by
    # one ULP after late addition. Enforce the canonical value, but only when
    # the entire repeated-row discrepancy is below a predeclared 1e-12 guard.
    group = frame.groupby(["patient", "channel"], sort=False)
    if (group["pathology"].nunique() > 1).any():
        raise RuntimeError("Repeated patient-channel label mismatch")
    spread = group["score_pathology"].max() - group["score_pathology"].min()
    if spread.max() > 1e-12:
        raise RuntimeError(f"Repeated patient-channel score discrepancy {spread.max()}")
    frame["score_pathology"] = group["score_pathology"].transform("first")
    return frame


def patient_ap(frame):
    unique = unique_patient_channels(frame)
    vals = [patient_metrics(g) for _, g in unique.groupby("patient", sort=True)]
    table = pd.DataFrame(vals)
    return float(table.loc[table["estimable_for_ranking"], "ap"].mean()), table


def threshold_table(frame):
    y = frame["pathology"].to_numpy(dtype=np.int8)
    s = frame["score_pathology"].to_numpy(dtype=np.float64)
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    end = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    tp = np.cumsum(y)[end].astype(float)
    pred = (end + 1).astype(float)
    fp = pred - tp
    pos, neg = float(y.sum()), float(len(y) - y.sum())
    fn, tn = pos - tp, neg - fp
    f1p = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    f1n = 2 * tn / np.maximum(2 * tn + fp + fn, 1)
    ba = .5 * (tp / pos + tn / neg)
    table = pd.DataFrame({"threshold": s[end], "macro_f1": .5 * (f1p + f1n),
                          "pathological_f1": f1p, "ba": ba})
    keys = list(zip(table.macro_f1, table.pathological_f1, table.ba,
                    -(table.threshold - .5).abs(), table.threshold))
    idx = max(range(len(keys)), key=lambda i: keys[i])
    table["selected"] = False
    table.loc[idx, "selected"] = True
    return table, table.iloc[idx]


def validation_select(a):
    bank = V2TFBank(a.cohort, a.features_train, a.tf_train, a.split, a.v2_protocol, a.tf_protocol)
    val = sorted(p for p in bank.patients if bank.roles[p] == "inner_val")
    if len(val) != 29 or not (a.nbase_runtime / "final/last.pt").is_file() or not (a.tf_runtime / "final/last.pt").is_file():
        raise RuntimeError("Inner validation or refit incomplete")
    raw = predict(a, bank, val, "inner")
    raw.to_csv(a.runtime / "OMNI_VALIDATION_SCORES_PRIVATE.csv", index=False)
    identity = float(np.max(np.abs(score_frame(raw, 0)["score_pathology"].to_numpy() -
                                   score_frame(raw, 0)["score_pathology"].to_numpy())))
    if identity >= 1e-7:
        raise RuntimeError("beta zero identity failure")
    base = score_frame(raw, 0)
    base_ap, _ = patient_ap(base)
    base_auc = float(roc_auc_score(base.pathology, base.score_pathology))
    rows = []
    tables = {}
    for beta in BETA:
        frame = score_frame(raw, beta)
        ap, _ = patient_ap(frame)
        auc = float(roc_auc_score(frame.pathology, frame.score_pathology))
        table, best = threshold_table(frame)
        feasible = auc >= base_auc - .002 and ap >= base_ap - .002
        rows.append({"beta": beta, "validation_auroc": auc, "validation_patient_equal_ap": ap,
                     "validation_macro_f1": float(best.macro_f1),
                     "validation_pathological_f1": float(best.pathological_f1),
                     "validation_ba": float(best.ba), "threshold": float(best.threshold),
                     "ranking_constraint_pass": bool(feasible)})
        tables[beta] = table
    selection = pd.DataFrame(rows)
    feasible = selection.loc[selection.ranking_constraint_pass]
    chosen = feasible.sort_values(["validation_macro_f1", "validation_pathological_f1", "validation_ba", "beta"],
                                  ascending=[False, False, False, True]).iloc[0]
    beta = float(chosen.beta)
    selection["selected"] = selection.beta.eq(beta)
    selection.to_csv(a.output / "INTERICTAL_BETA_SELECTION.csv", index=False)
    selection.to_csv(a.output / "BETA_SELECTION.csv", index=False)
    tables[beta].to_csv(a.output / "INTERICTAL_THRESHOLD_SELECTION.csv", index=False)
    tf_frame = score_frame(raw, 0, tf_only=True)
    tf_ap, tf_patient = patient_ap(tf_frame)
    tf_metrics = classification_metrics(tf_frame, .5)
    tf_metrics.update({"patient_equal_ap": tf_ap,
                       "mrr": float(tf_patient.loc[tf_patient.estimable_for_ranking, "mrr"].mean()),
                       "top1": float(tf_patient.loc[tf_patient.estimable_for_ranking, "top1"].mean())})
    save_json(a.output / "TF_ONLY_VALIDATION.json", tf_metrics)
    frozen = {"beta": beta, "threshold": float(chosen.threshold),
              "selection": "ranking constrained, max macro-F1, pathological-F1, BA, smaller beta",
              "validation_beta_zero_auroc": base_auc, "validation_beta_zero_patient_equal_ap": base_ap,
              "validation_selected_auroc": float(chosen.validation_auroc),
              "validation_selected_patient_equal_ap": float(chosen.validation_patient_equal_ap),
              "nbase_checkpoint_sha256": sha(a.nbase_runtime / "final/last.pt"),
              "tf_checkpoint_sha256": sha(a.tf_runtime / "final/last.pt"),
              "protocol_sha256": sha(a.lock), "official_test_accessed": False,
              "model_frozen_before_official_test": True, "test_used_for_tuning": False}
    save_json(a.output / "OMNI_FUSION_FREEZE.json", frozen)
    print(json.dumps({"status": "FROZEN", "beta": beta, "threshold": frozen["threshold"]}), flush=True)


def inference(a):
    freeze = json.loads((a.output / "OMNI_FUSION_FREEZE.json").read_text(encoding="utf-8"))
    for path, key in ((a.lock, "protocol_sha256"), (a.nbase_runtime / "final/last.pt", "nbase_checkpoint_sha256"),
                      (a.tf_runtime / "final/last.pt", "tf_checkpoint_sha256")):
        if sha(path) != freeze[key]:
            raise RuntimeError("Frozen artifact changed before official test")
    marker = a.runtime / "official_test_access_started.json"
    if not marker.exists():
        save_json(marker, {"freeze_sha256": sha(a.output / "OMNI_FUSION_FREEZE.json"),
                           "training_resume_forbidden": True})
    elif json.loads(marker.read_text(encoding="utf-8"))["freeze_sha256"] != sha(a.output / "OMNI_FUSION_FREEZE.json"):
        raise RuntimeError("Freeze changed after official test start")
    bank = V2TFBank(a.cohort, a.features_test, a.tf_test, None, a.v2_protocol, a.tf_protocol,
                    official_split="test")
    if len(bank.patients) != 96:
        raise RuntimeError("Official test patient denominator changed")
    raw = predict(a, bank, bank.patients, "final")
    if len(raw) != 8104:
        raise RuntimeError("Official test channel denominator changed")
    raw.to_csv(a.runtime / "OMNI_TEST_MARGINS_PRIVATE.csv", index=False)
    beta = freeze["beta"]
    final = score_frame(raw, beta)
    base = score_frame(raw, 0)
    tf_only = score_frame(raw, 0, tf_only=True)
    final.to_csv(a.runtime / "N2_PREDICTIONS_PRIVATE.csv", index=False)
    base.to_csv(a.runtime / "NBASE_PREDICTIONS_PRIVATE.csv", index=False)
    tf_only.to_csv(a.runtime / "TF_ONLY_PREDICTIONS_PRIVATE.csv", index=False)
    def full_metrics(frame, threshold):
        ap, patient = patient_ap(frame)
        estimable = patient.loc[patient.estimable_for_ranking]
        return {**classification_metrics(frame, threshold), "patient_equal_ap": ap,
                "mrr": float(estimable.mrr.mean()), "top1": float(estimable.top1.mean()),
                "ndcg": float(estimable.ndcg.mean())}, patient
    n2, p2 = full_metrics(final, freeze["threshold"])
    nbase_threshold = json.loads((a.nbase_output / "FROZEN_THRESHOLD.json").read_text(encoding="utf-8"))["threshold"]
    nb, pb = full_metrics(base, nbase_threshold)
    tf, pt = full_metrics(tf_only, .5)
    save_json(a.output / "INTERICTAL_LATE_FUSION_METRICS.json", n2)
    save_json(a.output / "INTERICTAL_A1_ONLY_METRICS.json", nb)
    save_json(a.output / "INTERICTAL_TF_ONLY_METRICS.json", tf)
    strata = []
    for dataset, group in final.groupby("dataset", sort=True):
        metric, patient = full_metrics(group, freeze["threshold"])
        strata.append({"dataset": dataset, **metric})
    pd.DataFrame(strata).to_csv(a.output / "INTERICTAL_DATASET_STRATIFIED.csv", index=False)
    n0 = pd.read_csv(a.n0_predictions)
    _, p0 = full_metrics(n0, float(json.loads(a.n0_threshold.read_text(encoding="utf-8"))["threshold"]))
    paired = []
    for label, comparator, patients, threshold in (("N2_minus_Nbase", base, pb, nbase_threshold),
                                                   ("N2_minus_N0", n0, p0,
                                                    float(json.loads(a.n0_threshold.read_text(encoding="utf-8"))["threshold"]))):
        ci = paired_bootstrap(final, comparator, p2, patients, freeze["threshold"], threshold)
        ci["comparison"] = label
        paired.append(ci)
    pd.concat(paired, ignore_index=True).to_csv(a.output / "INTERICTAL_PAIRED_BOOTSTRAP.csv", index=False)
    save_json(a.output / "TEST_ACCESS_AUDIT.json", {"official_test_accessed": True,
             "model_frozen_before_test": True, "beta_threshold_frozen_before_test": True,
             "test_used_for_tuning": False, "patients": 96, "edf_channel_records": 8104,
             "freeze_sha256": sha(a.output / "OMNI_FUSION_FREEZE.json")})
    print(json.dumps({"Nbase": nb, "N2": n2, "TF_only": tf, "beta": beta}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("select", "inference"), required=True)
    for name in ("cohort", "split", "features-train", "tf-train", "features-test", "tf-test",
                 "v2-protocol", "tf-protocol", "lock", "nbase-runtime", "tf-runtime",
                 "nbase-output", "n0-predictions", "n0-threshold", "runtime", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.runtime.mkdir(parents=True, exist_ok=True)
    a.output.mkdir(parents=True, exist_ok=True)
    if a.phase == "select":
        if (a.runtime / "official_test_access_started.json").exists():
            raise RuntimeError("Cannot select after test started")
        validation_select(a)
    else:
        inference(a)


if __name__ == "__main__":
    main()
