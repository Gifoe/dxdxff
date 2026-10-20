# No-Early/no-Late DREChannNet: complete17-group selected-seed ablations

The user fixed seeds42,44,62 and promoted the existing six-feature no-Early/no-Late graph AFTER viewing historical Outer outcomes. This is outcome-informed seed-subset sensitivity on exploratory repeated-test patients, not random-seed robustness or independent confirmation. The unselected historical43/52main results are retained below; they are not hidden or merged into the requested primary estimate.

All255model choices froze before new Outer; all255scores sealed before metrics. Exactly50strict reuses and205fresh fits. Each group/seed retains80patients,7635channels,256seizures and1471965validRaw250windows (physiology-only consumes no Raw). Variance and Head Interaction remain absent. Same patient roles, loss, optimizer, FP32/TF32off/chunk4096/full-patient updates, original validation-only epoch/threshold rules, no performance gate or post-Outer tuning.

## Patient-equal means across the requested three seeds

| Group | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac. bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL | 0.643771 | 0.766660 | 0.570795 | 0.451637 | 0.835906 | 0.760550 | 0.787698 | 0.695833 | -0.026136 |
| EARLY_FILM_RESTORED | 0.641733 | 0.765902 | 0.592297 | 0.450877 | 0.832589 | 0.758554 | 0.842269 | 0.791667 | -0.020472 |
| WO_FEATURE_PR | 0.621511 | 0.736473 | 0.543781 | 0.407149 | 0.835872 | 0.759370 | 0.773404 | 0.687500 | -0.051387 |
| WO_RAW_PR | 0.627177 | 0.763588 | 0.561702 | 0.438996 | 0.815359 | 0.744547 | 0.758845 | 0.662500 | 0.002629 |
| WO_BOTH_PR | 0.492631 | 0.749737 | 0.554090 | 0.370815 | 0.614448 | 0.618952 | 0.766148 | 0.670833 | 0.192474 |
| WO_PRE_REFERENCE | 0.616957 | 0.731702 | 0.535751 | 0.410349 | 0.823565 | 0.744283 | 0.770916 | 0.679167 | -0.024319 |
| RAW_ONLY | 0.614705 | 0.728789 | 0.535852 | 0.392629 | 0.836781 | 0.756971 | 0.762341 | 0.679167 | -0.065077 |
| PHYSIOLOGY_ONLY | 0.624862 | 0.749580 | 0.542016 | 0.424626 | 0.825098 | 0.750816 | 0.744907 | 0.645833 | -0.018529 |
| DROP_BETA | 0.642765 | 0.766926 | 0.570446 | 0.449590 | 0.835940 | 0.760613 | 0.788130 | 0.687500 | -0.024522 |
| DROP_LOW_GAMMA | 0.647015 | 0.769884 | 0.576576 | 0.456121 | 0.837910 | 0.763774 | 0.796900 | 0.704167 | -0.028850 |
| DROP_PARTIAL_HIGH_GAMMA | 0.643546 | 0.766981 | 0.567885 | 0.450601 | 0.836492 | 0.761798 | 0.790520 | 0.691667 | -0.026735 |
| DROP_RMS | 0.639698 | 0.760393 | 0.558835 | 0.442262 | 0.837134 | 0.761607 | 0.781581 | 0.687500 | -0.033793 |
| DROP_LINE_LENGTH | 0.647034 | 0.765649 | 0.573932 | 0.457006 | 0.837061 | 0.762393 | 0.790701 | 0.691667 | -0.024521 |
| DROP_SPECTRAL_ENTROPY | 0.642594 | 0.761528 | 0.562885 | 0.448509 | 0.836680 | 0.761682 | 0.806374 | 0.716667 | -0.026856 |
| P1_TEMPORAL_STATS | 0.642490 | 0.766303 | 0.564955 | 0.449103 | 0.835877 | 0.759624 | 0.784692 | 0.679167 | -0.029727 |
| P2_TIME_ATTENTION | 0.644494 | 0.767045 | 0.572194 | 0.452035 | 0.836954 | 0.761993 | 0.790476 | 0.700000 | -0.028929 |
| CAPACITY_MATCHED | 0.643904 | 0.770690 | 0.572816 | 0.450481 | 0.837327 | 0.762405 | 0.789081 | 0.687500 | -0.029609 |

[Each seed](SEED_RESULTS.csv), [seed sample SD](SEED_SD.csv), [all255folds](FOLD_RESULTS.csv), [fold sample SD](FOLD_SD.csv), [sources](SOURCE_RESULTS.csv), [pooled separately](POOLED_RESULTS.csv), [coverage](RAW_COVERAGE.csv). These average endpoint values, not probabilities, best seeds or best folds. EZ-fraction bias is predicted minus observed EZ fraction. Sources below5patients have suppressed metrics.

## Comparisons

10000seed42draws of the same80unique patients AFTER averaging their three seed metrics.17absolute+16Full-minus+1PRfactorial interaction yield306nominal intervals;32additional Bonferroni-percentile intervals cover paired Macro-F1/AUROC,tail0.05/64. Intervals are conditional on this outcome-informed selected seed set.

| Contrast | Delta Macro-F1 | Delta AUROC | Seeds both positive | Fold-seed cells both positive |
| --- | ---: | ---: | ---: | ---: |
| FULL_MINUS_EARLY_FILM_RESTORED | +0.002038 | +0.000758 | 1/3 | 5/15 |
| FULL_MINUS_WO_FEATURE_PR | +0.022261 | +0.030188 | 3/3 | 15/15 |
| FULL_MINUS_WO_RAW_PR | +0.016594 | +0.003073 | 1/3 | 8/15 |
| FULL_MINUS_WO_BOTH_PR | +0.151140 | +0.016924 | 3/3 | 11/15 |
| FULL_MINUS_WO_PRE_REFERENCE | +0.026815 | +0.034958 | 3/3 | 12/15 |
| FULL_MINUS_RAW_ONLY | +0.029066 | +0.037871 | 3/3 | 15/15 |
| FULL_MINUS_PHYSIOLOGY_ONLY | +0.018909 | +0.017080 | 3/3 | 11/15 |
| FULL_MINUS_DROP_BETA | +0.001006 | -0.000266 | 1/3 | 7/15 |
| FULL_MINUS_DROP_LOW_GAMMA | -0.003244 | -0.003224 | 0/3 | 3/15 |
| FULL_MINUS_DROP_PARTIAL_HIGH_GAMMA | +0.000225 | -0.000321 | 2/3 | 5/15 |
| FULL_MINUS_DROP_RMS | +0.004073 | +0.006267 | 2/3 | 9/15 |
| FULL_MINUS_DROP_LINE_LENGTH | -0.003262 | +0.001011 | 0/3 | 4/15 |
| FULL_MINUS_DROP_SPECTRAL_ENTROPY | +0.001177 | +0.005132 | 2/3 | 9/15 |
| FULL_MINUS_P1_TEMPORAL_STATS | +0.001282 | +0.000357 | 1/3 | 6/15 |
| FULL_MINUS_P2_TIME_ATTENTION | -0.000723 | -0.000385 | 1/3 | 4/15 |
| FULL_MINUS_CAPACITY_MATCHED | -0.000133 | -0.004030 | 0/3 | 2/15 |

[All nominal intervals](INTERVALS.csv), [family intervals](PRIMARY_FAMILY_INTERVALS.csv), [all fold-seed directions](FOLD_SEED_DIRECTIONS.csv). Negative directions are retained. Restored-Early and other structural controls are not deletions: negative Full-minus-restored means the restored model did better.

## What the fixed controls isolate

- WO_BOTH_PR removes both direct physiological patientZ and finalRaw96DpatientZ. Ordinary waveform normalization and own pre-reference remain. WindowPR has no operative path in the no-guidance main and is only an alias, not another fit/hypothesis.
- Six single deletions mask two direct summary coordinates BEFORE patient-relative statistics. With no guidance modules, no hidden window-conditioning route remains. A beneficial deletion is reported as such; these tests do not establish a globally optimal subset.
- Raw-only/physiology-only keep original untrained head slices, scales and64hidden width, not compensation by widening.
- EARLY_FILM_RESTORED restores the exact old no-Late model:10377parameters versus main9329. CAPACITY_MATCHED uses ordinary noEarly head73:10337parameters,40fewer than restoredEarly(-0.39%). It controls approximate parameter count, not all function classes or compute.
- P1 changes both partitioning and within-seizure statistic: whole0-20meanSD then equal-seizure meanSD96D. It is not a pure segmentation-only test. P2 retains stages but adds26time-aware attention coefficients:9355parameters; comparisons include this capacity increment. Both pooling arms have no FiLM/Gate. Unsupported scorer inputs are masked BEFORE matmul to prevent NaN-gradient contamination; no valid-window removal.
- No meaningful FiLM mismatch or Early/Late factorial interaction exists after both modules are absent. Their deletion aliases are documented without duplicate patient rows or hypotheses.

## Previously viewed main seeds: complete disclosure

| Existing seed | Included in requested subset | Macro-F1 | AUROC | AP | EZ-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 42 | True | 0.644071 | 0.766014 | 0.568578 | 0.448547 |
| 43 | False | 0.633324 | 0.756958 | 0.554850 | 0.435442 |
| 44 | True | 0.641568 | 0.760241 | 0.563333 | 0.448053 |
| 52 | False | 0.633450 | 0.762518 | 0.561388 | 0.428970 |
| 62 | True | 0.645675 | 0.773726 | 0.580476 | 0.458311 |

Selected3main Macro-F10.643771; descriptive all5historical-main average0.639618. The historical five-seed figure is not a newly run five-seed ablation. No new arm/seed is selected from this table.

Independent PASS:4080patient-seed rows,36720metric values,306nominal/32family intervals; maximum discrepancy7.77e-16. Source/init/reuse/preprocessing/validation proofs and all completion records accompany the tables. Signals,identities,individual scores,weights,preprocessors,caches and runtime logs remain private.
