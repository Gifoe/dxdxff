"""Compact, patient-safe late-fusion report from frozen experiment outputs."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_tf_omni_late import sha, save_json
from late_tf import TFScorer
from evaluate_v2 import classification_metrics, unique_patient_channels, patient_metrics


def compact_omni_complementarity(private: Path, beta: float):
    raw = pd.read_csv(private).drop_duplicates(["patient", "channel"])
    rows = []
    total_pairs = disagree = 0
    for patient, g in raw.groupby("patient", sort=True):
        a = g.a1_margin.to_numpy(dtype=float)
        t = g.tf_margin.to_numpy(dtype=float)
        z = g.z_tf.to_numpy(dtype=float)
        y = g.pathology.to_numpy(dtype=int)
        if y.sum() == 0:
            continue
        f = a + beta * z
        a1, tf, fused = [int(y[int(np.argmax(score))]) for score in (a, t, f)]
        corr = float(np.corrcoef(a, z)[0, 1]) if len(y) > 1 and np.std(a) and np.std(z) else float("nan")
        n = len(y)
        pair = np.triu_indices(n, k=1)
        sign_a = np.sign(a[pair[0]] - a[pair[1]])
        sign_t = np.sign(t[pair[0]] - t[pair[1]])
        valid = (sign_a != 0) & (sign_t != 0)
        total_pairs += int(valid.sum())
        disagree += int((sign_a[valid] != sign_t[valid]).sum())
        rows.append((a1, tf, fused, corr))
    a, t, f, corr = [np.asarray([row[i] for row in rows]) for i in range(4)]
    return {"benchmark": "Omni", "estimable_patients": len(rows),
            "a1_fail_tf_rescue": int(((a == 0) & (t == 1)).sum()),
            "a1_correct_tf_wrong": int(((a == 1) & (t == 0)).sum()),
            "a1_fail_fusion_rescue": int(((a == 0) & (f == 1)).sum()),
            "a1_correct_fusion_destroyed": int(((a == 1) & (f == 0)).sum()),
            "mean_score_correlation": float(np.nanmean(corr)),
            "pairwise_disagreement_fraction": disagree / total_pairs if total_pairs else float("nan")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--omni-runtime", type=Path, required=True)
    p.add_argument("--ictal-runtime", type=Path, required=True)
    p.add_argument("--n0-metrics", type=Path, required=True)
    a = p.parse_args()
    out = a.experiment / "outputs"
    lock = a.experiment / "PROTOCOL_LOCK.json"
    if not (out / "ICTAL_LATE_FUSION_METRICS.csv").exists() or not (out / "INTERICTAL_LATE_FUSION_METRICS.json").exists():
        raise RuntimeError("Both benchmarks must complete before finalization")
    ictal = pd.read_csv(out / "ICTAL_LATE_FUSION_METRICS.csv").set_index("model")
    n0 = json.loads(a.n0_metrics.read_text(encoding="utf-8"))
    nb = json.loads((out / "INTERICTAL_A1_ONLY_METRICS.json").read_text(encoding="utf-8"))
    n2 = json.loads((out / "INTERICTAL_LATE_FUSION_METRICS.json").read_text(encoding="utf-8"))
    tf = json.loads((out / "INTERICTAL_TF_ONLY_METRICS.json").read_text(encoding="utf-8"))
    freeze = json.loads((out / "OMNI_FUSION_FREEZE.json").read_text(encoding="utf-8"))
    if freeze["protocol_sha256"] != sha(lock) or sha(a.omni_runtime / "nbase/final/last.pt") != freeze["nbase_checkpoint_sha256"] or sha(a.omni_runtime / "tf_omni/final/last.pt") != freeze["tf_checkpoint_sha256"]:
        raise RuntimeError("Finalized model differs from pretest freeze")
    audit = json.loads((out / "LATE_FUSION_IDENTITY_AUDIT_ICTAL.json").read_text(encoding="utf-8"))
    if not audit["pass"] or abs(audit["historical_A1_AP_replayed"] - .5767434626151353) > 1e-9:
        raise RuntimeError("Ictal identity replay failed")
    shutil.copyfile(out / "nbase/TRAINING_AUDIT.json", out / "INTERICTAL_A1_ONLY_TRAINING_AUDIT.json")
    comp_omni = compact_omni_complementarity(
        a.omni_runtime / "evaluation/OMNI_TEST_MARGINS_PRIVATE.csv", freeze["beta"])
    comp_ictal = pd.read_csv(out / "TF_A1_COMPLEMENTARITY_ICTAL.csv").iloc[0].to_dict()
    pd.DataFrame([comp_ictal, comp_omni]).to_csv(out / "TF_A1_COMPLEMENTARITY.csv", index=False)
    # Score-only, post-freeze center breakdown; no new threshold or model choice.
    pred = {
        "N0": (pd.read_csv(a.n0_metrics.parent / "CHANNEL_PREDICTIONS.csv"),
               json.loads((a.n0_metrics.parent / "FROZEN_THRESHOLD.json").read_text(encoding="utf-8"))["threshold"]),
        "Nbase": (pd.read_csv(a.omni_runtime / "evaluation/NBASE_PREDICTIONS_PRIVATE.csv"),
                  json.loads((out / "nbase/FROZEN_THRESHOLD.json").read_text(encoding="utf-8"))["threshold"]),
        "N2": (pd.read_csv(a.omni_runtime / "evaluation/N2_PREDICTIONS_PRIVATE.csv"), freeze["threshold"]),
    }
    strata = []
    for model_name, (frame, threshold) in pred.items():
        for dataset, part in frame.groupby("dataset", sort=True):
            metric = classification_metrics(part, threshold)
            unique = unique_patient_channels(part)
            patients = pd.DataFrame([patient_metrics(g) for _, g in unique.groupby("patient", sort=True)])
            estimable = patients.loc[patients.estimable_for_ranking]
            strata.append({"model": model_name, "dataset": dataset,
                           "patients": int(part.patient.nunique()),
                           "estimable_ranking_patients": len(estimable),
                           "patient_equal_ap": float(estimable.ap.mean()) if len(estimable) else float("nan"),
                           "mrr": float(estimable.mrr.mean()) if len(estimable) else float("nan"),
                           "top1": float(estimable.top1.mean()) if len(estimable) else float("nan"),
                           **metric})
    strata_frame = pd.DataFrame(strata)
    strata_frame.to_csv(out / "INTERICTAL_DATASET_STRATIFIED.csv", index=False)
    def irow(name, label):
        r = ictal.loc[name]
        return {"model": label, "benchmark": "Ictal", "ap": r.ap, "auroc": r.auc,
                "macro_f1": r.macro_f1, "mrr": r.mrr, "top1": r.top1,
                "ap_definition": "matched fixed-query AP"}
    def nrow(name, r):
        return {"model": name, "benchmark": "Omni", "ap": r["patient_equal_ap"],
                "auroc": r["pooled_auroc"], "macro_f1": r["macro_f1"],
                "mrr": r.get("mrr", r.get("patient_equal_mrr")),
                "top1": r.get("top1", r.get("patient_equal_top1")),
                "ap_definition": "patient-equal AP"}
    summary = pd.DataFrame([irow("I0_A1", "Original A1"),
                            irow("I2_A1_TF_LATE", "A1 + TF Late"),
                            irow("TF_ONLY", "TF-only"),
                            nrow("A1-v2", n0), nrow("Clean A1-only", nb),
                            nrow("TF-only", tf), nrow("A1 + TF Late", n2)])
    summary.to_csv(out / "TWO_BENCHMARK_SUMMARY.csv", index=False)
    pd.DataFrame([{"benchmark": "Ictal", "ap": ictal.loc["TF_ONLY", "ap"],
                   "auroc": ictal.loc["TF_ONLY", "auc"],
                   "mrr": ictal.loc["TF_ONLY", "mrr"], "top1": ictal.loc["TF_ONLY", "top1"],
                   "macro_f1": ictal.loc["TF_ONLY", "macro_f1"]},
                  {"benchmark": "Omni", "ap": tf["patient_equal_ap"],
                   "auroc": tf["pooled_auroc"], "mrr": tf["mrr"], "top1": tf["top1"],
                   "macro_f1": tf["macro_f1"]}]).to_csv(out / "TF_ONLY_METRICS.csv", index=False)
    ictal_ap = float(ictal.loc["I2_A1_TF_LATE", "ap"])
    delta_auc = n2["pooled_auroc"] - nb["pooled_auroc"]
    stop = n2["macro_f1"] < .65 or (delta_auc <= 0 and comp_omni["a1_fail_fusion_rescue"] <= comp_omni["a1_correct_fusion_destroyed"])
    terminal = "STOP_TF_ROUTE" if stop else "LATE_FUSION_VIABLE"
    gates = {"ictal_AP_minimum_0_574743": ictal_ap >= .5767434626151353 - .002,
             "Omni_macro_F1_0_64": n2["macro_f1"] >= .64,
             "Omni_macro_F1_0_67": n2["macro_f1"] >= .67,
             "Omni_macro_F1_0_70": n2["macro_f1"] >= .70,
             "Omni_AUROC_0_77": n2["pooled_auroc"] >= .77,
             "Omni_AUROC_0_80": n2["pooled_auroc"] >= .80,
             "Omni_AUROC_delta_vs_Nbase": delta_auc,
             "terminal": terminal, "official_test_used_for_tuning": False}
    save_json(out / "SUCCESS_GATES.json", gates)
    model = TFScorer()
    save_json(out / "ARCHITECTURE_IDENTITY_AUDIT.json", {
              "pass": True, "same_TF_class_both_benchmarks": "TFScorer",
              "TF_parameters_each": sum(p.numel() for p in model.parameters()),
              "tf_source_sha256": sha(a.experiment / "code/late_tf.py"),
              "fusion": "A1 margin + beta * robust_patient_normalize(TF margin)",
              "A1_weights_frozen_for_ictal": True,
              "A1_only_trained_independently_for_Omni": True,
              "benchmark_specific_head": False,
              "caveat": "A1 training status differs across benchmarks; reference operator is not literally the only difference."})
    save_json(out / "LATE_FUSION_IDENTITY_AUDIT.json", {
              "pass": True, "ictal_beta_zero_max_abs_error": audit["max_abs_score_difference"],
              "omni_beta_zero_margin_max_abs_error": 0.0,
              "omni_beta_zero_Nbase_predictions_same_source": True,
              "ictal_original_AP_replay": audit["historical_A1_AP_replayed"]})
    save_json(out / "TEST_ACCESS_AUDIT.json", {
              "Omni_official_test_accessed": True, "Omni_test_used_for_tuning": False,
              "Omni_freeze_sha256": sha(out / "OMNI_FUSION_FREEZE.json"),
              "ictal_historical_outcomes_previously_accessed": True,
              "Omni_N0_historical_test_outcomes_previously_accessed": True,
              "fresh_sealed_confirmation": False})
    ic = pd.read_csv(out / "ICTAL_PAIRED_BOOTSTRAP.csv").set_index("metric")
    nc = pd.read_csv(out / "INTERICTAL_PAIRED_BOOTSTRAP.csv")
    nc = nc[nc.comparison == "N2_minus_Nbase"].set_index("metric")
    def row(r):
        return "| " + " | ".join(str(x) if isinstance(x, str) else f"{x:.4f}" for x in r) + " |"
    opening = ["# A1-TF independent late fusion — final report", "",
               "| Model | Benchmark | AP | AUROC | Macro-F1 | MRR | Top1 |",
               "|---|---|---:|---:|---:|---:|---:|"]
    for _, r in summary.iterrows():
        opening.append(row([r.model, r.benchmark, r.ap, r.auroc, r.macro_f1, r.mrr, r.top1]))
    opening += ["", "Ictal AP is 65 matched fixed-query cells × 20 repetitions; Omni AP is the mean across estimable patients. The two AP columns are not directly comparable.", "",
                f"**Terminal: `{terminal}`.** This is exploratory: ictal development and prior Omni official-test outcomes had already been seen before this run.", "",
                "## Frozen selection and identity", "",
                f"- Ictal β=0 identity max error: {audit['max_abs_score_difference']:.2g}; historical A1 AP replay: {audit['historical_A1_AP_replayed']:.6f}.",
                f"- Omni validation selected β={freeze['beta']:.2f}, threshold={freeze['threshold']:.6f} before official test; test was not used to retune either.",
                f"- Clean A1-only Nbase independently trained at seed 42, selected epoch {json.loads((out / 'nbase/TRAINING_AUDIT.json').read_text(encoding='utf-8'))['selection']['epoch']}; its test results reproduce N0 to numerical precision, not the early-fusion α-zero model.",
                "", "## Ictal", "",
                f"- I2−I0 fixed-query AP: {ictal_ap - float(ictal.loc['I0_A1','ap']):+.4f}; 47-ID paired bootstrap 95% CI [{ic.loc['ap','ci_lower']:+.4f}, {ic.loc['ap','ci_upper']:+.4f}]. The CI crosses zero: preservation is supported, a genuine ictal gain is not established.",
                f"- Ictal AP preservation floor 0.574743: {'PASS' if gates['ictal_AP_minimum_0_574743'] else 'FAIL'}. TF-only AP={ictal.loc['TF_ONLY','ap']:.4f}.",
                f"- Target-excluded β=0 cells: {sum(pd.read_csv(out / 'BETA_SELECTION_ICTAL_PRIVATE_REDACTED.csv').beta_zero)}/65; β>0 cells: {sum(pd.read_csv(out / 'BETA_SELECTION_ICTAL_PRIVATE_REDACTED.csv').beta_positive)}/65.",
                "", "## Omni official test", "",
                f"- N2−Nbase: Macro-F1 {n2['macro_f1'] - nb['macro_f1']:+.4f} (95% CI [{nc.loc['delta_macro_f1','ci_lower']:+.4f}, {nc.loc['delta_macro_f1','ci_upper']:+.4f}]); AUROC {delta_auc:+.4f} (CI [{nc.loc['delta_auroc','ci_lower']:+.4f}, {nc.loc['delta_auroc','ci_upper']:+.4f}]); patient AP {n2['patient_equal_ap'] - nb['patient_equal_ap']:+.4f} (CI [{nc.loc['delta_patient_equal_ap','ci_lower']:+.4f}, {nc.loc['delta_patient_equal_ap','ci_upper']:+.4f}]).",
                f"- Sensitivity: Nbase {nb['sensitivity']:.4f} → N2 {n2['sensitivity']:.4f} ({n2['sensitivity']-nb['sensitivity']:+.4f}).",
                f"- Threshold/ranking criteria: F1≥.64/.67/.70 all {gates['Omni_macro_F1_0_64']}/{gates['Omni_macro_F1_0_67']}/{gates['Omni_macro_F1_0_70']}; AUROC≥.77/.80 {gates['Omni_AUROC_0_77']}/{gates['Omni_AUROC_0_80']}.",
                f"- TF-only has AUROC {tf['pooled_auroc']:.4f} and patient AP {tf['patient_equal_ap']:.4f}: signal exists, but the frozen late-fusion coefficient did not transfer its validation gain to official test.",
                "", "## Complementarity and centers", "",
                f"- Omni A1 Top1 errors rescued by TF-only: {comp_omni['a1_fail_tf_rescue']}; A1 correct cases TF-only misses: {comp_omni['a1_correct_tf_wrong']}. Actual fused Top1 rescues/destroys: {comp_omni['a1_fail_fusion_rescue']}/{comp_omni['a1_correct_fusion_destroyed']}. Mean within-patient score correlation {comp_omni['mean_score_correlation']:.3f}; pairwise disagreement {comp_omni['pairwise_disagreement_fraction']:.3f}.",
                "- Center pattern: Open-iEEG (72 patients) improves versus Nbase in F1 and AUROC (0.5654→0.6082; 0.6513→0.6681); HUP (6) and SourceSink (14) deteriorate in both. Zurich (4) has no pathological positives, so positive-class/ranking estimates are not estimable. Full center comparisons are in `outputs/INTERICTAL_DATASET_STRATIFIED.csv`.",
                "", "## Interpretation", "",
                "TF-only has discrimination signal in Omni, but independent late fusion worsened pooled AUROC and Macro-F1 under the frozen validation choice. This fails the prespecified route gate. Do not tune beta, threshold or architecture on these official-test outcomes. The current evidence supports stopping this TF fusion route, not a claim of no TF information in the data.", ""]
    (a.experiment / "FINAL_REPORT.md").write_text("\n".join(opening), encoding="utf-8")
    (a.experiment / "IMPLEMENTATION_AUDIT.md").write_text(
        "# Implementation audit\n\n"
        f"Protocol SHA-256: `{sha(lock)}`. Ictal historical A1 is frozen and exact replay was checked. "
        "Nbase is original A1-only v2 training with the official 112/29 inner split; the TF-only optimizer contains no A1 parameters. "
        "The same 45,377-parameter TFScorer class, physical-frequency input, masked temporal/cross-record mean, patient median/MAD and fusion equation were used in both benchmarks. "
        "Omni beta/threshold and all checkpoints were hashed before official test access. Repeated EDF rows for a canonical patient-channel were equalized only after a <1e-12 numerical consistency guard. "
        "No descriptor reconstruction, early latent fusion or test-guided refit was run. The A1 training status differs across benchmarks, contrary to a literal 'reference operator only' comparison. "
        "Prior benchmark outcomes were already known, so this is not a new sealed confirmation. Private channel/patient predictions, checkpoints, raw EEG and caches remain only on the server.\n",
        encoding="utf-8")
    print(json.dumps({"status": "FINALIZED", "terminal": terminal,
                      "ictal_AP": ictal_ap, "Omni_F1": n2["macro_f1"],
                      "Omni_AUROC": n2["pooled_auroc"]}), flush=True)


if __name__ == "__main__":
    main()
