"""Read-only numerical audit and compact report after the single v2 test run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_v2 import classification_metrics, patient_metrics, unique_patient_channels
from train_v2 import atomic_json, sha256


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value, digits=4):
    if value is None or not np.isfinite(float(value)):
        return "not estimable"
    return f"{float(value):.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--v1-predictions", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment
    output = root / "outputs"
    protocol = root / "PROTOCOL_LOCK.json"
    if not protocol.is_file():
        protocol = root / "code/PROTOCOL_LOCK.json"
    counts = read_json(root / "audit/OFFICIAL_LABEL_COUNTS.json")
    cross = read_json(root / "CROSS_CHANNEL_REFERENCE_AUDIT.json")
    split_path = root / "TRAIN_VAL_SPLIT.csv"
    if not split_path.is_file():
        split_path = root / "audit/TRAIN_VAL_SPLIT.csv"
    split = pd.read_csv(split_path)
    freeze = read_json(output / "TEST_SCORE_FREEZE_AUDIT.json")
    threshold = read_json(output / "FROZEN_THRESHOLD.json")
    access = read_json(output / "TEST_ACCESS_AUDIT.json")
    training = read_json(output / "TRAINING_AUDIT.json")
    primary = read_json(output / "PRIMARY_METRICS.json")
    d1 = read_json(output / "V1_OFFICIAL_COHORT_RESCORE.json")
    v1 = pd.read_csv(args.v1_predictions).rename(columns={"soz": "pathology",
                                                        "score_ez": "score_pathology"})
    v1_exact = classification_metrics(v1, 0.5)
    v1_patient = pd.DataFrame([patient_metrics(group) for _, group in v1.groupby("patient")])
    v1_estimable_ap = float(v1_patient.loc[v1_patient["estimable_for_ranking"], "ap"].mean())
    frame = pd.read_csv(output / "CHANNEL_PREDICTIONS.csv")
    patients = pd.read_csv(output / "PATIENT_LEVEL_METRICS.csv")
    ablation = pd.read_csv(output / "THRESHOLD_ABLATION.csv").set_index("setting")
    strata = pd.read_csv(output / "DATASET_STRATIFIED_METRICS.csv")
    ci = pd.read_csv(output / "PATIENT_CLUSTER_BOOTSTRAP.csv").set_index("metric")
    selection = pd.read_csv(output / "CHECKPOINT_SELECTION.csv")
    threshold_grid = pd.read_csv(output / "THRESHOLD_SELECTION.csv")
    checks = {
        "frozen_before_test": bool(freeze["model_frozen_before_official_test"] and
                                   freeze["threshold_frozen_before_official_test"]),
        "test_used_for_tuning_false": access["test_used_for_tuning"] is False and
                                      freeze["test_used_for_tuning"] is False,
        "official_test_access_recorded": access["official_test_accessed"] is True,
        "protocol_hash": freeze["protocol_sha256"] == sha256(protocol),
        "threshold_hash": freeze["threshold_json_sha256"] == sha256(output / "FROZEN_THRESHOLD.json"),
        "prediction_hash": access["predictions_sha256"] == sha256(output / "CHANNEL_PREDICTIONS.csv"),
        "v1_frozen_prediction_hash": d1["v1_prediction_sha256"] == sha256(args.v1_predictions),
        "d1_after_v2_freeze": d1["v2_freeze_sha256"] == sha256(output / "TEST_SCORE_FREEZE_AUDIT.json"),
        "cross_channel_reference_tests": cross["pass"] is True,
        "official_patients": frame["patient"].nunique() == 96,
        "official_edfs": frame["edf"].nunique() == 174,
        "official_record_count": len(frame) == 8104,
        "official_positive_count": int(frame["pathology"].sum()) == 807,
        "unique_patient_channels": len(unique_patient_channels(frame)) == 5055,
        "unique_pathological_patient_channels": int(unique_patient_channels(frame)["pathology"].sum()) == 722,
        "binary_official_labels": set(frame["pathology"].unique()) == {0, 1},
        "no_duplicate_edf_channels": not frame.duplicated(["edf", "channel"]).any(),
        "test_label_counts_match_metadata_audit": counts["labeled_channel_records"]["test"] == len(frame)
                                            and counts["pathological_channel_records"]["test"] == int(frame["pathology"].sum()),
        "train_val_patients": len(split) == 141 and split["patient"].is_unique and
                              (split["role"] == "inner_train").sum() == 112 and
                              (split["role"] == "inner_val").sum() == 29,
        "model_parameters": freeze["model_parameters"] == 27713,
        "one_ap_selected_epoch": selection["selected"].sum() == 1 and
                                int(selection.loc[selection["selected"], "epoch"].iat[0]) == freeze["selected_inner_epoch"],
        "one_validation_threshold": threshold_grid["selected"].sum() == 1 and
                                    float(threshold_grid.loc[threshold_grid["selected"], "threshold"].iat[0]) == threshold["threshold"],
        "patient_metric_rows": len(patients) == 96 and patients["patient"].is_unique,
        "bootstrap_metrics": {"pooled_ap", "pooled_auroc", "macro_f1", "sensitivity",
                              "specificity", "patient_equal_ap", "mrr", "top1"} <= set(ci.index),
        "attribution_overlap_declared": d1["full_cohort_D1_attribution_identifiable"] is False,
        "v1_overlap_contains_all_official_positives":
            d1["official_edf_channel_pair_metrics_at_0_5_on_overlap"]["pathological_records"] == 807,
    }
    recomputed = classification_metrics(frame, threshold["threshold"])
    checks["primary_metric_recompute"] = all(np.isclose(primary[key], recomputed[key], rtol=0,
                                                          atol=1e-10)
                                              for key in ("macro_f1", "pathological_f1",
                                                          "balanced_accuracy", "pooled_ap", "pooled_auroc"))
    checks = {key: bool(value) for key, value in checks.items()}
    validation = {"pass": all(checks.values()), "checks": checks,
                  "protocol_sha256": sha256(protocol),
                  "official_test_predictions_sha256": sha256(output / "CHANNEL_PREDICTIONS.csv")}
    atomic_json(output / "VALIDATION.json", validation)
    if not validation["pass"]:
        raise RuntimeError("V2 final validation failed: " +
                           ", ".join(key for key, ok in checks.items() if not ok))
    label_audit = {
        "official_evaluation_rule": read_json(protocol)["cohort"]["official_label_order"],
        "official_test_labeled_edf_channel_records": len(frame),
        "official_test_pathological_records": int(frame["pathology"].sum()),
        "official_test_normal_records": int((frame["pathology"] == 0).sum()),
        "official_test_unique_patient_channels": 5055,
        "non_estimable_ranking_patients": int((~patients["estimable_for_ranking"]).sum()),
        "estimable_ranking_patients": int(patients["estimable_for_ranking"].sum()),
        "unlabeled_good_channels_used_only_for_label_free_reference": True,
        "official_test_used_for_training_or_threshold": False,
        "source_of_labels": "frozen Omni channels.tsv and final_split.csv",
        "normal_precedes_soz_conflicts": counts["normal_precedes_soz_conflict_records"],
    }
    atomic_json(output / "LABEL_USAGE_AUDIT.json", label_audit)
    frozen_f1 = primary["macro_f1"]
    f05 = float(ablation.loc["fixed_0.5", "macro_f1"])
    y_opt = float(ablation.loc["posthoc_test_youden_reproduction_only", "macro_f1"])
    d1_original = d1["intersection_v1_soz_edf_channel_pair_metrics_at_0_5"]
    d1_official = d1["official_edf_channel_pair_metrics_at_0_5_on_overlap"]
    binary = strata.loc[strata["binary_evaluable"]]
    best = binary.sort_values("macro_f1", ascending=False).iloc[0]
    worst = binary.sort_values("macro_f1").iloc[0]
    ci_ap = ci.loc["patient_equal_ap"]
    success = "STRONG" if frozen_f1 >= 0.635 and primary["pooled_auroc"] >= 0.73 else (
        "MINIMUM_WITHOUT_AUROC_REGRESSION" if frozen_f1 > 0.620 and
        primary["pooled_auroc"] >= v1_exact["pooled_auroc"] else "NOT_MET")
    rows = []
    for item in strata.itertuples(index=False):
        rows.append(f"| {item.dataset} | {item.patients} | {item.edf_channel_records} | "
                    f"{fmt(item.pathological_prevalence)} | {fmt(item.pooled_ap)} | "
                    f"{fmt(item.pooled_auroc)} | {fmt(item.macro_f1)} | "
                    f"{fmt(item.patient_equal_ap)} | {fmt(item.mrr)} | {fmt(item.top1)} |")
    report = f"""# Omni-iEEG A1 interictal v2 seed 42 — final report

Validation: **PASS**. Single official test run after model and threshold freeze.
Scientific criterion: **{success}**. This is one seed, not a stability estimate.

## Required answers

1. **Official labels:** `good=1`; first `outcome=1 and resection=0` → normal, else `soz=1` → pathological, else excluded. The official code's ordered branches supersede the prompt's verbal summary. There are 185 train and 66 test EDF-channel SOZ/normal overlaps classified normal.
2. **Eligible cohort:** train 141 patients / 296 EDFs / 13,350 labeled EDF-channel records; test 96 patients / 174 EDFs / 8,104 records (5,055 unique patient-channels).
3. **Pathology prevalence:** train 1,355/13,350 = 10.15%; test 807/8,104 = {100*primary['pathological_prevalence']:.2f}%.
4. **Frozen v1 re-score:** On the *observable overlap only* ({d1['intersection_edf_channel_records']:,}/8,104 official test EDF-channel records), v1 original-SOZ Macro-F1/AP/AUROC at 0.5 = {fmt(d1_original['macro_f1'])}/{fmt(d1_original['pooled_ap'])}/{fmt(d1_original['pooled_auroc'])}; with official labels on that same overlap = {fmt(d1_official['macro_f1'])}/{fmt(d1_official['pooled_ap'])}/{fmt(d1_official['pooled_auroc'])}. This overlap contains all 807 officially pathological records but omits {8104-d1['intersection_edf_channel_records']:,} normal records, so its prevalence is {100*d1_official['pathological_prevalence']:.2f}% rather than the full cohort's {100*primary['pathological_prevalence']:.2f}%. Its elevated F1/AP must not be mistaken for a full-cohort v1 result.
5. **Cross-channel views:** synthetic persistent-abnormality/permutation tests and a real 88-reference-channel EDF pilot passed; all 36D features finite. Labels do not choose reference channels.
6. **AP-selected inner epoch:** {freeze['selected_inner_epoch']} of 30.
7. **Validation-frozen threshold:** {fmt(threshold['threshold'], 6)}.
8. **Validation Macro-F1:** {fmt(threshold['validation_macro_f1'])} on official-labeled validation EDF-channel pairs.
9. **Official test Macro-F1 at 0.5:** {fmt(f05)}.
10. **Official test Macro-F1 at frozen threshold:** {fmt(frozen_f1)}.
11. **Pathological F1 at frozen threshold:** {fmt(primary['pathological_f1'])}.
12. **Balanced accuracy:** {fmt(primary['balanced_accuracy'])}.
13. **Pooled AP:** {fmt(primary['pooled_ap'])}.
14. **Pooled AUROC:** {fmt(primary['pooled_auroc'])}.
15. **Ranking eligibility:** {primary['estimable_ranking_patients']} estimable / {primary['patients']} total test patients; {primary['non_estimable_ranking_patients']} have zero official pathological channels and are excluded only from ranking denominators.
16. **Patient-equal AP:** {fmt(primary['patient_equal_ap'])}.
17. **Patient-equal AP 95% patient bootstrap CI:** [{fmt(ci_ap['ci_lower'])}, {fmt(ci_ap['ci_upper'])}].
18. **MRR:** {fmt(primary['patient_equal_mrr'])}.
19. **Top1:** {fmt(primary['patient_equal_top1'])}.
20. **NDCG:** {fmt(primary['patient_equal_ndcg'])}.
21. **Label-only attribution:** same frozen v1 predictions and same observable overlap yield Macro-F1 difference {d1['paired_label_only_delta_macro_f1_on_overlap']:+.4f}; this is *not* the full official-cohort effect because v1 lacks {8104-d1['intersection_edf_channel_records']:,} officially normal EDF-channel scores.
22. **Threshold contribution within v2:** frozen-validation threshold minus 0.5 on the same official test scores = {frozen_f1-f05:+.4f} Macro-F1. Test-derived Youden yields {fmt(y_opt)} but is posthoc reproduction-only.
23. **Reference+retraining contribution:** not separately identifiable. The residual across v1→v2 also includes changed score coverage, official labels, patient set and final refit; no new architecture/large retraining ablation was allowed.
24. **Macro-F1 > 0.620:** {'YES' if frozen_f1 > 0.620 else 'NO'}.
25. **Macro-F1 ≥ 0.635 and AUROC ≥ 0.73:** {'YES' if frozen_f1 >= 0.635 and primary['pooled_auroc'] >= 0.73 else 'NO'}.
26. **Published context:** Omni TimeConv-CNN Table 5 reports Macro-F1 0.6469 and AUC 0.8061 under a **test-derived Youden threshold**. Our {fmt(frozen_f1)} uses a train-validation-frozen threshold; {fmt(y_opt)} is the posthoc Youden reproduction. The models and sample construction differ, so this is context, not a head-to-head controlled comparison. [Omni-iEEG ICLR 2026 paper](https://openreview.net/pdf?id=rv9lQpY5cG).
27. **Dataset strata:** highest evaluable frozen-threshold Macro-F1 is {best['dataset']} ({fmt(best['macro_f1'])}); lowest is {worst['dataset']} ({fmt(worst['macro_f1'])}). Zurich's normal-only stratum has no binary AUROC/AP or ranking estimate; it remains in pooled official classification.
28. **Test leakage:** none detected. AP chose epoch, validation Macro-F1 chose threshold, then full-train refit and all hashes were frozen before test-feature access. Test-derived Youden was never used for selection.
29. **Final benchmark use:** {'YES, as a one-seed frozen-protocol result, with the above comparability caveats' if success != 'NOT_MET' else 'NO as a positive final model; retain as a valid frozen negative/weak result'}; do not tune on this official test.

## Dataset breakdown

| Dataset | Patients | EDF-channel records | Positive prevalence | Pooled AP | AUROC | Macro-F1 | Patient-equal AP | MRR | Top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

## Provenance and caveats

- Omni code: `57c22a75a59b5c3a98006806ad42000f6a3fa5b6`; dataset: `73b9c5180a57828ab2a83c040e7e9d112e77b2cc`.
- Protocol SHA-256: `{sha256(protocol)}`; final checkpoint SHA-256: `{freeze['final_checkpoint_sha256']}`; frozen threshold file SHA-256: `{freeze['threshold_json_sha256']}`.
- The exact frozen v1 prediction CSV used in D1 has corrected estimable-patient AP {fmt(v1_estimable_ap)}, pooled AP {fmt(v1_exact['pooled_ap'])}, AUROC {fmt(v1_exact['pooled_auroc'])}, Macro-F1 at 0.5 {fmt(v1_exact['macro_f1'])}. The prompt cited approximate 0.3313/0.2361/0.7086/0.6042; the non-AP values do not exactly match this frozen CSV and are not substituted for verified file-derived metrics. In either case v1 uses a different label cohort, so direct subtraction from v2 does not estimate an architecture effect.
- `CHANNEL_PREDICTIONS.csv` contains only pseudonymous official EDF-channel scores, not waveforms or cache tensors. No runtime, checkpoint, or raw EEG is committed.
"""
    (root / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    audit = f"""# Implementation audit

Validation: **PASS** (`outputs/VALIDATION.json`).

- Frozen source split SHA-256: `{read_json(protocol)['official_split_sha256']}`. Train/test patient assignments unchanged.
- Actual model parameter count: {freeze['model_parameters']}; exact historical A1 source hash checks ran in training/inference.
- One 30-epoch inner AP-selection run; best epoch {freeze['selected_inner_epoch']}. Validation Macro-F1 used only after checkpoint selection to choose threshold {fmt(threshold['threshold'], 6)}.
- Full 141-patient train refit used selected epoch count. Frozen model/normalizer/threshold/protocol hashes were checked before official test inference.
- Final official cohort: 96 patients, 174 EDFs, 8,104 labeled EDF-channel records; 807 pathological, 7,297 normal. Unknown labels excluded from loss/evaluation but all good signal-present channels contribute to label-free cross-channel reference.
- Test selection/tuning: **NO**. A test-derived Youden threshold is flagged posthoc and excluded from the primary claim.
- D1 compares only v1-score-observable official channels; full label-only effect is unidentifiable. D2 shares the identical v2 scores at two thresholds.
- Source/cache and checkpoint tensors remain private. Committed outputs are compact audits and pseudonymous score rows only.
"""
    (root / "IMPLEMENTATION_AUDIT.md").write_text(audit, encoding="utf-8")
    print(json.dumps({"status": "VALIDATED", "macro_f1": frozen_f1,
                      "patient_equal_ap": primary["patient_equal_ap"],
                      "auroc": primary["pooled_auroc"], "success": success}))


if __name__ == "__main__":
    main()
