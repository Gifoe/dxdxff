# Pooling and capacity ablations: fixed no-Late DREChannNet, seed42

Actual original80-patient five-fold Outer,25cells:10strict reuses and15fresh fits. Variance, Head Interaction and Late Gate remain absent. All25choices froze before newOuter; all25scores sealed before reporting. Patient-equal metrics, not pooled channels or the best fold. This is exploratory repeated-test following historical Outer viewing, not independent confirmation. No result changes the fixed main.

## Operations fixed before fitting

- P1:0-20s valid-window population Mean/SD followed by equal-seizure Mean/SD,96D. This changes both temporal partition and the within-seizure statistic; it is not a pure isolation of segmentation.
- P2:masked learned time-aware weighting separately in0-8s/8-20s, then original equal-seizure Mean/SD96D. Shared26coefficients on24latent,t/20,(t/20)^2,zero initialization; no labels/physiology in scorer.10403parameters versus Full10377:capacity difference is explicit.
- A2:noEarlyFiLM, ordinary head widened64to73;10337parameters(-40,-0.39% versus Full10377). Original noEarly has9329. Shared64head neurons retain original untrained weights;9extra effective neurons use an independent fixed initializer. Direct physiology and finalPR remain; no physiological CNN modulation. This controls approximate parameter count, not exact functional capacity or compute.
- Same complete-patient FP32/TF32off/chunk4096 updates,seed42 and original optimizer/loss/validation-only selection; no new data/features/extraction/search/performance gate. All groups complete Outer.

| Group | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac. bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL | 0.653465 | 0.765286 | 0.593488 | 0.464629 | 0.842301 | 0.772424 | 0.846368 | 0.800000 | -0.034239 |
| WO_EARLY_FILM | 0.644071 | 0.766014 | 0.568578 | 0.448547 | 0.839594 | 0.764299 | 0.779291 | 0.675000 | -0.035998 |
| P1_TEMPORAL_STATS | 0.650744 | 0.773571 | 0.583444 | 0.472280 | 0.829208 | 0.754780 | 0.816476 | 0.750000 | 0.007175 |
| P2_TIME_ATTENTION | 0.653678 | 0.768380 | 0.595115 | 0.464480 | 0.842876 | 0.773263 | 0.834055 | 0.775000 | -0.035619 |
| A2_CAPACITY_MATCHED | 0.650836 | 0.774698 | 0.577704 | 0.459176 | 0.842495 | 0.768851 | 0.803202 | 0.700000 | -0.035785 |

EZ-fraction bias is predicted minus true EZ fraction; closeness to zero is desirable. [Folds](FOLD_RESULTS.csv), [fold sample SD](FOLD_SD.csv), [sources](SOURCE_RESULTS.csv), [pooled separately](POOLED_RESULTS.csv), [coverage](RAW_COVERAGE.csv).

## Paired comparisons

10000seed42patient draws;5absolute+4Full-minus+1A2-minus-unmatched-noEarly yield90nominal intervals. Ten additional Bonferroni-percentile intervals cover five paired Macro-F1/AUROC contrasts,tail0.05/20. No seedSD or training-seed robustness is implied.

| Contrast | Delta Macro-F1 | Delta AUROC | Folds both positive | Family lower bounds: Macro-F1,AUROC |
| --- | ---: | ---: | ---: | --- |
| FULL_MINUS_WO_EARLY_FILM | +0.009394 | -0.000728 | 2/5 | -0.010411, -0.030370 |
| FULL_MINUS_P1_TEMPORAL_STATS | +0.002721 | -0.008284 | 1/5 | -0.016453, -0.036379 |
| FULL_MINUS_P2_TIME_ATTENTION | -0.000213 | -0.003094 | 2/5 | -0.006224, -0.011903 |
| FULL_MINUS_A2_CAPACITY_MATCHED | +0.002629 | -0.009412 | 2/5 | -0.014899, -0.037343 |
| A2_MINUS_UNMATCHED_NO_EARLY | +0.006765 | +0.008684 | 3/5 | -0.003594, -0.001296 |

[All90 nominal intervals](INTERVALS.csv) · [10family intervals](PRIMARY_FAMILY_INTERVALS.csv) · [fold directions](FOLD_DIRECTIONS.csv)

A2-minus-unmatched-noEarly isolates this particular ordinary-head capacity increase. Full-minus-A2 compares physiological modulation with approximately equal parameter count; it does not rule out every ordinary CNN/head architecture. P2 differences also include26attention parameters, so a win cannot be attributed exclusively to temporal selectivity. P1/P2 do not change the12active physiological summary coordinates or their preprocessing.

Independent verification: 400patient rows,3600metric values,90nominal/10family intervals; maximum interval discrepancy8.88e-16.
Only code/hashes/anonymous aggregates are public; signals,identities,individual scores,weights,preprocessors,caches and runtime logs remain on the trusted server.
