# Patient-relative output coordinate: seed-42 validation-only analysis

## A. Original stopped experiment

The original experiment stopped with `OUTPUT_COORDINATE_RANK_INVARIANCE_FAILED`. That result is preserved in `RANK_INVARIANCE_AUDIT.json`; it has not been erased or rewritten.

## B. Engineering amendment

The original audit compared source float32 sigmoid probabilities against transformed float64 probabilities. Distinct logits sometimes collapsed to tied float32 probabilities. The separately committed amendment changes only the rank audit to canonical float64 EZ logit scores; hard-decision scores, selection, and gates are unchanged.

## C. Revised rank audit

All results use frozen A1 checkpoints and validation patients only. No new model was trained and no current outer-test result was read.
Source A1 VLOO reproduction: PASS; mean 0.6259962097; max fold error 0.00e+00.
Canonical logit-space ranking invariance: PASS across 1950 patient-epochs per transform. Float32-probability tie cases (diagnostic only): 5.

## D. Scientific C1/C2 VLOO results

The rank-metric columns below retain the original hard-score probability semantics for source comparability. They are evaluated at each coordinate's separately VLOO-selected epochs, so their aggregate differences reflect checkpoint selection and occasional finite-precision ties; they are not evidence that a fixed-checkpoint transform improves mathematical ordering.

| Coordinate | VLOO Macro-F1 | EZ-F1 | BA | EZ-AUPRC | EZ-AUROC | EZ-MRR | Top-1 EZ | Pred. EZ fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0_RAW | 0.625996 | 0.414294 | 0.673415 | 0.557200 | 0.743069 | 0.742103 | 0.676923 | 0.223438 |
| C1_CENTERED | 0.586782 | 0.362154 | 0.653972 | 0.548710 | 0.739396 | 0.744157 | 0.676923 | 0.256517 |
| C2_ROBUSTZ | 0.597249 | 0.364980 | 0.656515 | 0.545124 | 0.737888 | 0.754450 | 0.692308 | 0.226324 |

| Fold | C0 Macro-F1 | C1 Macro-F1 | C2 Macro-F1 | C1−C0 | C2−C0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.655000 | 0.641326 | 0.644840 | -0.013674 | -0.010160 |
| 2 | 0.641096 | 0.568017 | 0.560277 | -0.073079 | -0.080819 |
| 3 | 0.610119 | 0.585686 | 0.579733 | -0.024432 | -0.030386 |
| 4 | 0.632979 | 0.641133 | 0.609539 | +0.008154 | -0.023440 |
| 5 | 0.590788 | 0.497748 | 0.591859 | -0.093040 | +0.001071 |

Mean deltas: C1−C0 -0.039214; C2−C0 -0.028747; C2−C1 +0.010467. Positive folds: C1 1/5, C2 1/5.

Preferred by the locked C1-vs-C2 rule: C2_ROBUSTZ; retained vs C0: False; gain -0.028747; positive folds 1/5.
Candidate APPARENT_FULLVAL mean/worst-fold Macro-F1: 0.631782/0.591859.

## E. Threshold stability and F. Label-using oracle diagnostic

| Coordinate | VLOO threshold std | IQR | entropy (bits) | unique | oracle std | oracle IQR | oracle Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0_RAW | 0.063832 | 0.100000 | 2.244611 | 6 | 0.141701 | 0.200000 | 0.707179 |
| C1_CENTERED | 0.053149 | 0.000000 | 1.599901 | 5 | 0.131325 | 0.250000 | 0.693396 |
| C2_ROBUSTZ | 0.038951 | 0.100000 | 1.566596 | 3 | 0.133625 | 0.250000 | 0.694628 |

Oracle thresholds are **LABEL_USING_DIAGNOSTIC_ONLY** at each patient's VLOO-selected epoch; they were not used to choose an epoch, coordinate, global threshold, or gate outcome. A finite 19-point grid can change each patient's oracle Macro-F1 after a monotone score transform, because it samples different cut points in raw-logit space.

## G. Development gate

Strong-gate checks:
- source_C0_reproduced: PASS
- normalized_candidate_retained: FAIL
- candidate_vloo_macro_f1_ge_0_640: FAIL
- gain_vs_C0_ge_0_010: FAIL
- positive_folds_ge_4: FAIL
- mean_ez_f1_nondecreasing: FAIL
- apparent_fullval_mean_ge_0_665: FAIL
- apparent_fullval_worst_fold_ge_0_620: FAIL
- ranking_invariant: PASS
- no_score_pathology: PASS

**Terminal: `OUTPUT_COORDINATE_DEVELOPMENT_GATE_FAILED`.**

Stop at validation. Do not read current outer test or tune another transformation after this result.

Threshold distribution, full-validation selection, raw score-statistic aggregates, and oracle diagnostic aggregates are in `development/`. Patient-level data and logits remain private.
