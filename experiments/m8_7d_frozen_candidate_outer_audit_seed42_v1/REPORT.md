# Frozen 7D single-head candidates: complete five-fold Outer results

## Scope and endpoint

Per the user's revised instruction, all ten already-trained candidates from the five recent studies were evaluated without a new Development eligibility screen, together with the original C2 reference. This is a separate evaluation protocol; none of the earlier failed Development improvement gates was rewritten or represented as passing.

The primary endpoint is **equal-patient mean binary Macro-F1 across all 80 original Outer patients**. Every arm covers 7,635 channels, all original five folds and the original seven physiological measurements. Each new arm retains 256 seizures and all 1,471,965 valid raw250 windows. There was **no new training, epoch selection, feature selection, threshold tuning, patient exclusion or window subsampling**. Checkpoints, epochs, thresholds, FIT-only preprocessors and sources were frozen before new Outer inference. Full-cohort metric reporting followed the complete prediction seal.

All models use one scalar classification readout. G0/G1 use the previously trained C2 body plus the previously selected adjustment; the other candidates reuse their own already-trained end-to-end checkpoints. Five C2 fold-score artifacts were reused exactly, not counted as new inference. The 50 new candidate fold cells were inferred once with atomic, hash-bound private patient caches; no failures or restarts occurred in this Outer audit.

## All eleven arms, fixed protocol order

All values below are **patient averages**, not pooled-channel or unweighted fold averages. AP denotes average precision; EZ-F1 is the positive-class F1. Macro-F1 = (EZ-F1 + NEZ-F1)/2 for each patient. ACC is accuracy. EZ-fraction bias is predicted minus true EZ fraction (signed).

| Arm | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac. bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C2 reference | 0.639658 | 0.769704 | 0.582443 | 0.448853 | 0.830463 | 0.758853 | 0.803898 | 0.712500 | -0.026113 |
| R0 mean-only capacity control | 0.642014 | 0.767949 | 0.585839 | 0.456562 | 0.827466 | 0.756033 | 0.819055 | 0.737500 | -0.013702 |
| R1 Raw temporal dispersion | 0.634627 | 0.768879 | 0.582190 | 0.451125 | 0.818129 | 0.745582 | 0.817224 | 0.737500 | +0.000884 |
| F0 early FiLM after GELU | 0.647977 | 0.754932 | 0.587757 | 0.453985 | 0.841968 | 0.770914 | 0.831379 | 0.775000 | -0.041987 |
| F1 early FiLM before GELU | 0.645873 | 0.755353 | 0.584401 | 0.456207 | 0.835540 | 0.763618 | 0.830355 | 0.775000 | -0.022591 |
| G0 global hidden affine | 0.638981 | 0.769721 | 0.582357 | 0.449010 | 0.828952 | 0.757270 | 0.803898 | 0.712500 | -0.022136 |
| G1 patient-context hidden affine | 0.640252 | 0.769938 | 0.582852 | 0.450407 | 0.830097 | 0.758610 | 0.795565 | 0.700000 | -0.027000 |
| P0 pointwise capacity control | 0.637059 | 0.768179 | 0.575322 | 0.447233 | 0.826884 | 0.754020 | 0.789057 | 0.687500 | -0.019783 |
| P1 channel relation | 0.639469 | 0.764839 | 0.570221 | 0.456946 | 0.821992 | 0.750273 | 0.798065 | 0.700000 | -0.000380 |
| T0 globally normalized objective | 0.635366 | 0.761377 | 0.576252 | 0.437363 | 0.833370 | 0.758012 | 0.820825 | 0.750000 | -0.028474 |
| T1 patient-balanced objective | 0.637017 | 0.764273 | 0.570657 | 0.442724 | 0.831309 | 0.755975 | 0.795537 | 0.700000 | -0.021928 |

Unrounded source: [MAIN_RESULTS.csv](results/MAIN_RESULTS.csv). R0, F0, F1 and G1 exceed 0.64 as observed point estimates. The protocol selects **no arm**; identifying the largest observed number is descriptive, not a prospective model-selection rule.

## Paired uncertainty

10,000 seed42 patient-bootstrap draws preserve the identical 80-patient pairing across arms. These nominal 95% intervals describe this repeatedly evaluated cohort; they do not repair selection bias from the broader research history.

| Candidate minus C2 | Macro-F1 difference | Nominal 95% paired CI |
| --- | ---: | --- |
| R0 | +0.002356 | [-0.004065, 0.009471] |
| R1 | -0.005031 | [-0.012836, 0.003216] |
| F0 | +0.008319 | [-0.008144, 0.025137] |
| F1 | +0.006215 | [-0.010174, 0.023490] |
| G0 | -0.000677 | [-0.002399, 0.001140] |
| G1 | +0.000594 | [-0.002353, 0.003481] |
| P0 | -0.002599 | [-0.006833, 0.001158] |
| P1 | -0.000189 | [-0.007206, 0.006675] |
| T0 | -0.004292 | [-0.014643, 0.008263] |
| T1 | -0.002641 | [-0.013558, 0.010316] |

All ten candidate-versus-C2 intervals include zero. For F0, the additional 99.5% percentile interval for the ten-comparison Macro-F1 family is **[-0.014250, 0.032465]**. These Bonferroni-level bootstrap intervals are approximate exploratory multiplicity summaries, not exact familywise guarantees or independent confirmation.

The within-study differences are also fully reported in [INTERVALS.csv](results/INTERVALS.csv). R1 minus R0 is -0.007387, nominal CI [-0.014734, -0.000228]; this suggests dispersion did not help relative to the matched mean-only capacity control, but that within-family interval is not multiplicity-adjusted. The F1-minus-F0 interval includes zero; the experiment does not establish that the after-GELU insertion is intrinsically superior to before-GELU insertion.

## What the largest observed Macro-F1 does and does not mean

F0 has Macro-F1 **0.6479765335907922**, exceeding the requested strict observed point target of 0.64. Its difference from C2 is **0.00831854**, or 0.832 percentage points. This is not an EZ-F1 of 0.648, not a pooled F1, and not a rounded value just over the cutoff.

Compared with C2, F0 changes EZ-F1 by +0.005132, NEZ-F1 by +0.011505, ACC by +0.012061, MRR by +0.027481 and Top-1 by +0.062500. AUROC changes by **-0.014771**, while AP changes by +0.005314. Therefore the observed result is not an across-metric improvement. Its more negative EZ-fraction bias (-0.041987 versus -0.026113) also indicates greater average underprediction of the EZ fraction. In pooled counts, false positives decrease from 879 to 738, but true positives also decrease from 761 to 736. These tradeoffs must not be hidden behind the Macro-F1 maximum.

F0 uses the same seven inputs to condition the early Raw representation (seven inputs to a 32-unit conditioner, producing 12 gains and 12 shifts after the first GELU), while retaining the original late physiological gate and one readout. It has 11,426 parameters versus 10,378 for C2. This establishes the actual tested configuration, not that the extra capacity or a specific physiological mechanism has been isolated as the sole cause of its observed gain.

## Fold and source stability

| Fold | Patients | C2 Macro-F1 | F0 Macro-F1 | Difference |
| --- | ---: | ---: | ---: | ---: |
| 1 | 16 | 0.604546 | 0.599251 | -0.005294 |
| 2 | 16 | 0.645517 | 0.668781 | +0.023264 |
| 3 | 17 | 0.641141 | 0.654744 | +0.013603 |
| 4 | 15 | 0.612202 | 0.637668 | +0.025466 |
| 5 | 16 | 0.693075 | 0.678372 | -0.014704 |

F0 improves Macro-F1 in three of five folds, not all five. Fold sample SD is 0.031155 for F0 versus 0.034914 for C2; five folds alone are insufficient to establish a robust stability advantage. Fold sizes differ, so the primary total is patient-weighted, not the simple average of five fold means.

| Source | Patients | C2 Macro-F1 | F0 Macro-F1 | Difference |
| --- | ---: | ---: | ---: | ---: |
| HUP | 36 | 0.666225 | 0.686079 | +0.019853 |
| LZU | 21 | 0.570828 | 0.575699 | +0.004872 |
| Multicenter aggregate | 15 | 0.666519 | 0.664572 | -0.001948 |
| Pediatric | 8 | 0.650420 | 0.635129 | -0.015290 |

The gain is not uniform across sources. Detailed aggregate AUROC/AP/ranking results for every arm and source are in [SOURCE_RESULTS.csv](results/SOURCE_RESULTS.csv); source strata with fewer than five patients would be suppressed, without dropping those patients from the total. None occurs in this cohort.

## Integrity and limits

- All 55 existing checkpoints/thresholds and source closures were frozen before new Outer inference; adapter replay on one existing validation patient per cell had maximum logit error **0**. The previous independent full-validation verifications remained PASS. No new Development metrics were calculated for screening.
- All 50 new fold cells completed and every arm retained all 80 patients and the complete Raw census. Five C2 cells were reused byte-identically. The original seven-feature preprocessing and one-readout architecture constraint were verified.
- Independent verification rebuilt 880 patient rows and **7,920** metric values from sealed predictions, checked **234** nominal intervals and **10** additional family intervals, and checked pooled/fold/source/coverage summaries. Maximum nominal interval discrepancy was 6.661338147750939e-16.
- This is **exploratory repeated-test, multi-candidate evidence**, not a fresh independent test set. F0 is the largest observed Macro-F1 among the fixed eleven reported arms, not a prospectively selected winner. No model, threshold or selection rule was changed after these outcomes.
- The point target was observed, but statistical superiority and broad generalization were not established. Additional claims require genuinely independent evidence rather than further tuning on these Outer outcomes.

Only source, hashes and anonymous aggregate tables are published. Waveforms, weights, scalers, patient/channel identities, patient predictions and runtime logs remain on the trusted server.
