# A1/A2 objective-only seed-42 development

Exploratory fixed-80 development. No current outer-test result was read or evaluated.
A0 was retrained for 30 epochs under this VLOO selection protocol; historical R0 validation/test numbers are not the matched control.

| Variant | VLOO Macro-F1 | EZ-F1 | NEZ-F1 | BA | EZ-AUPRC | EZ-AUROC | EZ-MRR | Top-1 EZ | Pred. EZ fraction | Positive folds vs A0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | 0.615099 | 0.396443 | 0.833756 | 0.664829 | 0.545074 | 0.737959 | 0.709324 | 0.615385 | 0.225066 | n/a |
| A1 | 0.625996 | 0.414294 | 0.837699 | 0.673415 | 0.557200 | 0.743069 | 0.742103 | 0.676923 | 0.223438 | 3/5 |
| A2 | 0.617376 | 0.398794 | 0.835959 | 0.665744 | 0.558925 | 0.741345 | 0.749471 | 0.692308 | 0.221232 | 2/5 |

| Fold | A0 Macro-F1 | A1 Macro-F1 | A2 Macro-F1 | A1−A0 | A2−A1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.655791 | 0.655000 | 0.637586 | -0.000792 | -0.017414 |
| 2 | 0.630956 | 0.641096 | 0.652194 | +0.010140 | +0.011098 |
| 3 | 0.567179 | 0.610119 | 0.586949 | +0.042939 | -0.023170 |
| 4 | 0.628252 | 0.632979 | 0.617880 | +0.004727 | -0.015100 |
| 5 | 0.593319 | 0.590788 | 0.592273 | -0.002531 | +0.001485 |

A1−A0 mean VLOO Macro-F1: +0.010897 (3/5 positive folds).
A1−A0 EZ-F1: +0.017851; EZ-AUPRC: +0.012125.
A2−A1 mean VLOO Macro-F1: -0.008620; positive in 2/5 folds.
A2 replacement minimum +0.005 met: False.
Locked candidate: A1.
Candidate−A0 mean VLOO Macro-F1: +0.010897; positive in 3/5 folds.
Candidate APPARENT_FULLVAL mean/worst-fold Macro-F1: 0.654667/0.607337.

Locked readiness checks:
- mean_vloo_macro_f1_gain_ge_0_010: PASS
- positive_folds_ge_4: FAIL
- mean_ez_f1_nondecreasing: PASS
- ez_auprc_delta_ge_minus_0_005: PASS
- apparent_fullval_mean_macro_f1_ge_0_665: FAIL
- apparent_fullval_worst_fold_macro_f1_ge_0_620: FAIL
- no_pathology: PASS

**Terminal: `A1_A2_DEVELOPMENT_GATE_FAILED`.**

Stop at development. No seed52/62 or current outer test is authorized. The >0.650 test target is not evaluated in this experiment.

Per-fold selected epochs, VLOO threshold distributions, full-validation epoch/threshold choices, and loss diagnostics are in the aggregate CSV files under `development/`.
Patient identifiers, channel records, checkpoints and private cache paths are not published.
