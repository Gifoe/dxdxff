# PaReSet-EZ full, original B0 protocol, seed 42

## Result

The five historical folds completed on the original Windows server. The 80-patient, patient-equal Macro-F1 was **0.616170** for PaReSet-EZ full versus **0.644589** for historical CDEL seed 42, a difference of **-2.842 percentage points**. Patient-equal balanced accuracy was **0.669786** versus **0.688029** (-1.824 pp). The new full model was lower on Macro-F1 in all five folds.

| Fold | Patients | Selected epoch | Full Macro-F1 | CDEL Macro-F1 | Delta (pp) | Full BA | CDEL BA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 36 | 0.571183 | 0.596366 | -2.518 | 0.647954 | 0.677239 |
| 2 | 16 | 32 | 0.615814 | 0.673036 | -5.722 | 0.660024 | 0.723129 |
| 3 | 17 | 26 | 0.640891 | 0.660092 | -1.920 | 0.720167 | 0.698480 |
| 4 | 15 | 26 | 0.590215 | 0.613462 | -2.325 | 0.606191 | 0.618806 |
| 5 | 16 | 8 | 0.659577 | 0.677073 | -1.750 | 0.707469 | 0.717512 |

## Protocol and checks

- Full PaReSet-EZ architecture and original objective were unchanged. Only this model was trained; no new control run was launched.
- Used the historical fixed five-fold fit/validation/test partition, historical B0 four-view evidence implementation (nine source descriptors times abs/delta/zdelta/ratio), fit-only normalization, validation-only checkpoint/threshold selection, and historical CDEL patient-level evaluator.
- The adapter's source input was numerically identical to the historical feature path over three real patients and 991,908 compared values (maximum absolute difference 0). This test is a sample-level parity check, not an exhaustive all-patient proof.
- Training used seed 42, AdamW learning rate 1e-4, weight decay 1e-3, 45-epoch maximum, minimum 18 epochs before early stopping, patience 6, batch size four patients, and the unchanged full-model loss. Historical PRQ's coarse 0.05 threshold grid selected the epoch; historical CDEL's fine 0.005 grid selected the final validation threshold. CDEL itself is an ensemble with model-specific training and potentially different input-feature settings. Patient membership and final evaluation semantics match, but this is not an identical-input or identical-optimization head-to-head comparison.
- Five completion records exist; the held-out patients across folds total 80 unique IDs; stderr is empty. Protocol lock SHA-256: `61997ec1f2ba16628adfe957c9f1996eb1cbb22e945c028ccfa906bd0393d8b9`.
- This is exploratory: these historical outer outcomes were previously viewed in the broader project. Do not present it as a fresh sealed confirmation. One seed is insufficient to assess seed stability.

Server output: `D:\nips-temp\pareset_ez_v1\full_original_b0_seed42_v1`.

Historical CDEL source: `D:\nips-temp\task1_aaai_completion_training\final_pooled\seed_42\cdel\metrics\by_fold.csv` and `overall.csv`.
