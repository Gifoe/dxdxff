# Omni-iEEG A1 interictal v2 seed 42 — final report

Validation: **PASS**. Single official test run after model and threshold freeze.
Scientific criterion: **NOT_MET**. This is one seed, not a stability estimate.

## Required answers

1. **Official labels:** `good=1`; first `outcome=1 and resection=0` → normal, else `soz=1` → pathological, else excluded. The official code's ordered branches supersede the prompt's verbal summary. There are 185 train and 66 test EDF-channel SOZ/normal overlaps classified normal.
2. **Eligible cohort:** train 141 patients / 296 EDFs / 13,350 labeled EDF-channel records; test 96 patients / 174 EDFs / 8,104 records (5,055 unique patient-channels).
3. **Pathology prevalence:** train 1,355/13,350 = 10.15%; test 807/8,104 = 9.96%.
4. **Frozen v1 re-score:** On the *observable overlap only* (5,493/8,104 official test EDF-channel records), v1 original-SOZ Macro-F1/AP/AUROC at 0.5 = 0.6408/0.3959/0.7198; with official labels on that same overlap = 0.6428/0.3769/0.7265. This overlap contains all 807 officially pathological records but omits 2,611 normal records, so its prevalence is 14.69% rather than the full cohort's 9.96%. Its elevated F1/AP must not be mistaken for a full-cohort v1 result.
5. **Cross-channel views:** synthetic persistent-abnormality/permutation tests and a real 88-reference-channel EDF pilot passed; all 36D features finite. Labels do not choose reference channels.
6. **AP-selected inner epoch:** 9 of 30.
7. **Validation-frozen threshold:** 0.515586.
8. **Validation Macro-F1:** 0.6272 on official-labeled validation EDF-channel pairs.
9. **Official test Macro-F1 at 0.5:** 0.5901.
10. **Official test Macro-F1 at frozen threshold:** 0.5937.
11. **Pathological F1 at frozen threshold:** 0.2718.
12. **Balanced accuracy:** 0.5974.
13. **Pooled AP:** 0.2327.
14. **Pooled AUROC:** 0.6887.
15. **Ranking eligibility:** 88 estimable / 96 total test patients; 8 have zero official pathological channels and are excluded only from ranking denominators.
16. **Patient-equal AP:** 0.5505.
17. **Patient-equal AP 95% patient bootstrap CI:** [0.4663, 0.6363].
18. **MRR:** 0.6302.
19. **Top1:** 0.5682.
20. **NDCG:** 0.7002.
21. **Label-only attribution:** same frozen v1 predictions and same observable overlap yield Macro-F1 difference +0.0020; this is *not* the full official-cohort effect because v1 lacks 2,611 officially normal EDF-channel scores.
22. **Threshold contribution within v2:** frozen-validation threshold minus 0.5 on the same official test scores = +0.0036 Macro-F1. Test-derived Youden yields 0.5428 but is posthoc reproduction-only.
23. **Reference+retraining contribution:** not separately identifiable. The residual across v1→v2 also includes changed score coverage, official labels, patient set and final refit; no new architecture/large retraining ablation was allowed.
24. **Macro-F1 > 0.620:** NO.
25. **Macro-F1 ≥ 0.635 and AUROC ≥ 0.73:** NO.
26. **Published context:** Omni TimeConv-CNN Table 5 reports Macro-F1 0.6469 and AUC 0.8061 under a **test-derived Youden threshold**. Our 0.5937 uses a train-validation-frozen threshold; 0.5428 is the posthoc Youden reproduction. The models and sample construction differ, so this is context, not a head-to-head controlled comparison. [Omni-iEEG ICLR 2026 paper](https://openreview.net/pdf?id=rv9lQpY5cG).
27. **Dataset strata:** highest evaluable frozen-threshold Macro-F1 is sourcesink (0.6395); lowest is openieeg (0.5654). Zurich's normal-only stratum has no binary AUROC/AP or ranking estimate; it remains in pooled official classification.
28. **Test leakage:** none detected. AP chose epoch, validation Macro-F1 chose threshold, then full-train refit and all hashes were frozen before test-feature access. Test-derived Youden was never used for selection.
29. **Final benchmark use:** NO as a positive final model; retain as a valid frozen negative/weak result; do not tune on this official test.

## Dataset breakdown

| Dataset | Patients | EDF-channel records | Positive prevalence | Pooled AP | AUROC | Macro-F1 | Patient-equal AP | MRR | Top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hup | 6 | 952 | 0.0819 | 0.2225 | 0.7249 | 0.6090 | 0.4759 | 0.5828 | 0.5000 |
| openieeg | 72 | 4108 | 0.1400 | 0.2392 | 0.6513 | 0.5654 | 0.4968 | 0.5722 | 0.5000 |
| sourcesink | 14 | 433 | 0.3557 | 0.6547 | 0.7283 | 0.6395 | 0.8430 | 0.9325 | 0.9286 |
| zurich | 4 | 2611 | 0.0000 | not estimable | not estimable | not estimable | not estimable | not estimable | not estimable |

## Provenance and caveats

- Omni code: `57c22a75a59b5c3a98006806ad42000f6a3fa5b6`; dataset: `73b9c5180a57828ab2a83c040e7e9d112e77b2cc`.
- Protocol SHA-256: `68aa2ffe40cff05abf32a1d8eac92111bf4092b1c12a00552218ecd806f2f082`; final checkpoint SHA-256: `273b918a87f01cf4bd9740cb211052054f5f8035a21a8c59ce1013fba001546c`; frozen threshold file SHA-256: `21d170bf6fa697e21cec9d885566605b8abfb565ef500dc5c8f85fc740ebe94b`.
- The exact frozen v1 prediction CSV used in D1 has corrected estimable-patient AP 0.3313, pooled AP 0.2279, AUROC 0.7096, Macro-F1 at 0.5 0.6030. The prompt cited approximate 0.3313/0.2361/0.7086/0.6042; the non-AP values do not exactly match this frozen CSV and are not substituted for verified file-derived metrics. In either case v1 uses a different label cohort, so direct subtraction from v2 does not estimate an architecture effect.
- `CHANNEL_PREDICTIONS.csv` contains only pseudonymous official EDF-channel scores, not waveforms or cache tensors. No runtime, checkpoint, or raw EEG is committed.
