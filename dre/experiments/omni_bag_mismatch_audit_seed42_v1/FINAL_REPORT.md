# Omni Train-Inference Bag Mismatch Audit

| Setting | Mean segments/channel | AUROC | AP | MRR | Top1 |
|---|---:|---:|---:|---:|---:|
| TRAIN original 5-clip (labeled) | 4.950 | 0.957155 | 0.829912 | 0.949253 | 0.928000 |
| TRAIN full-record (labeled) | 10.865 | 0.958678 | 0.833831 | 0.949104 | 0.928000 |
| TEST full-record (labeled; frozen) | 11.220 | 0.798767 | 0.330298 | 0.831977 | 0.772727 |

The prompt's `240,074 / 8,104 = 29.6` estimate mixes all selected TEST segment rows with only labeled channel units. The consistent denominators are: 240,074 rows / 16,543 all units = 14.512, or 90,930 labeled rows / 8,104 labeled units = 11.220. The tables use labeled units for comparability with TRAIN metrics.

## Replay gates

TEST replay: **PASS**, AUROC 0.7987673466, absolute error 0.

TRAIN-5 replay: **PASS**, AUROC 0.9571552723, absolute error 0.

Pipeline identity: **PASS_WITH_CUDA_NUMERICAL_TOLERANCE**. HDF5 waveform replay max absolute error is 0.0; current CUDA forward differs from the historical logits by at most 0.002536, bounding the sigmoid probability discrepancy by 0.000634.

## Required findings

1. **TRAIN-5 exact replay:** yes, 0.9571552723.
2. **TRAIN segment counts:** min 2, median 5.0, max 5, mean 4.950; P(N=5)=0.979925. Stored historical windows overlap in 91.892% of EDFs.
3. **TEST segment counts:** labeled min 1, median 5.0, max 119, mean 11.220. All-unit mean is 14.512.
4. **TRAIN-FULL bag size:** mean 10.865, median 5.0.
5. **TRAIN-FULL versus TEST bags:** close for the labeled comparison (both medians 5; KS 0.060, Wasserstein 1.018).
6. **TRAIN-FULL AUROC:** 0.958678.
7. **Delta AUROC:** +0.001523 versus historical TRAIN-5.
8. **Other metrics:** AP +0.003918, MRR -0.000148, Top1 +0.000000, NDCG +0.002180.
9. **Random-5 variance:** AUROC mean 0.954317, SD 0.001199, 95% empirical interval [0.952083, 0.956924].
10. **Rare-event misses:** pathological tau=0.7 capture is 81.993% with FULL and 77.179% on random-5, a +4.814% difference. This is measurable but below the locked +10% criterion.
11. **Performance versus K:** it rises sharply from K=1 to K=5/10, then changes little; the detailed means are below.
12. **Saturation:** K=10 under the locked ±0.005 AUROC/AP rule.
13. **Prediction correlation:** Pearson r=0.968817; channel-level shifts exist even though pooled AUROC barely changes.
14. **Center concentration:** yes. HUP changes by -0.020081 AUROC, Open-iEEG by +0.012394, and SourceSink by -0.028971; `Other` is single-class and its AUROC is not estimable. Opposing center effects cancel in the pooled result.
15. **Duration confounding:** duration-label r=0.144494, but duration-delta r=-0.007425; duration relates to cohort composition but does not linearly explain the score change.
16. **Gap A explanatory fraction:** the TRAIN-to-TEST AUROC gap changes from 0.158388 to 0.159911; FULL closes -0.96% of the gap (negative means it slightly worsens it).
17. **Remaining Gap B:** 0.159911 AUROC.
18. **Final forced A/B/C decision:** **BAG_MISMATCH_NOT_PRIMARY**.

## Random-5 Monte Carlo

| Metric | Mean | SD | p2.5 | p50 | p97.5 |
|---|---:|---:|---:|---:|---:|
| AUROC | 0.954317 | 0.001199 | 0.952083 | 0.954376 | 0.956924 |
| AP | 0.827052 | 0.002504 | 0.821952 | 0.827171 | 0.831945 |
| Patient-equal AP | 0.867482 | 0.003431 | 0.860602 | 0.867448 | 0.872669 |
| MRR | 0.943939 | 0.004916 | 0.934646 | 0.944310 | 0.951931 |
| Top1 | 0.923440 | 0.007859 | 0.912000 | 0.920000 | 0.936000 |

## Performance versus bag size

| K | AUROC mean | AP mean | MRR mean | Top1 mean |
|---:|---:|---:|---:|---:|
| 1 | 0.928887 | 0.775466 | 0.917686 | 0.888000 |
| 2 | 0.943697 | 0.804004 | 0.930434 | 0.905600 |
| 3 | 0.949340 | 0.815911 | 0.936172 | 0.912320 |
| 5 | 0.954183 | 0.826593 | 0.945869 | 0.926240 |
| 10 | 0.956654 | 0.830630 | 0.949199 | 0.929280 |
| 20 | 0.957748 | 0.832271 | 0.950054 | 0.930560 |
| ALL | 0.958678 | 0.833831 | 0.949104 | 0.928000 |

## Prediction change

- Gap A (TRAIN-5 to TRAIN-FULL): delta AUROC +0.001523; AP +0.003918; MRR -0.000148; Top1 +0.000000.
- Gap B (TRAIN-FULL to TEST): AUROC difference +0.159911.
- Prediction shift: mean absolute delta 0.019126, median 0.000255, p90 0.056778, Pearson r 0.968817.

## H2 evidence summaries

- normal: mean score delta -0.000007, max-segment delta +0.023321, Q90 delta +0.002033.
- pathological: mean score delta -0.005584, max-segment delta +0.041476, Q90 delta +0.014138.

## Decision

**BAG_MISMATCH_NOT_PRIMARY**

> The large train–test degradation cannot be primarily attributed to five-segment sampling; the dominant problem is likely cross-cohort/domain generalization of the encoder.

The strict Case-1 supporting correlation check (`r>0.98`) is not met (`r=0.968817`). The forced A decision follows because the prespecified material AUROC/capture criteria for B and C are also not met; it should not be misreported as evidence that every channel prediction is stable.

This audit changes no model, label, split, threshold, or TEST prediction. Macro-F1 uses the locked 0.5 threshold and is diagnostic only.
