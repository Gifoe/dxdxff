# EZ-conditioned reference analysis and composition sensitivity

Real computation on all 80 patients and 7,635 channels. Twelve unchanged physiological coordinates; 100 fixed rankings/partitions per patient. These are exploratory supplements after viewing the parent results. No classifier was fitted.

## Two primary endpoints

| Endpoint | Mean [nominal 95% CI] | Two-primary adjusted 97.5% CI |
| --- | --- | --- |
| H: NEZ gain minus EZ gain | 0.0172548 [-0.00339, 0.0392144] | [-0.00614822, 0.0423736] |
| D: Full 25% minus Full 100% CRPS | -0.00670058 [-0.0191565, 0.000437646] | [-0.0225247, 0.000690296] |

Positive H means NEZ reference-fitting gains exceed EZ gains. Positive D means the smaller reference set incurs greater forecast error. Neither establishes localization performance or a clinical reliability threshold.

## E1: original frozen channel halves

| Method | EZ CRPS | NEZ CRPS | EZ gain | NEZ gain | H [nominal 95% CI] |
| --- | --- | --- | --- | --- | --- |
| GLOBAL | 0.499017 | 0.444759 | 0 | 0 | 0 (identity) |
| SOURCE | 0.504355 | 0.446324 | -0.00533751 | -0.00156516 | 0.00377235 [-0.00282984, 0.0105044] |
| CENTER | 0.35328 | 0.308211 | 0.145737 | 0.136548 | -0.0091895 [-0.0267387, 0.00825573] |
| SCALE | 0.483553 | 0.429408 | 0.0154644 | 0.0153504 | -0.00011403 [-0.00718073, 0.00750037] |
| FULL | 0.235389 | 0.163876 | 0.263628 | 0.280883 | 0.0172548 [-0.00339, 0.0392144] |

Each class is averaged within each patient before equal-patient aggregation. Class-weighted reconstruction reproduces the parent all-channel CRPS. E1 uses the exact original reference moments, not the new leave-one-channel-out references.

## E2: nested leave-one-channel-out references

| Other-channel reference fraction | Global CRPS | Source CRPS | Center CRPS | Scale CRPS | Full CRPS [95% CI] | Full EZ-NEZ absolute gap [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| 25% | 0.450934 | 0.452609 | 0.31622 | 0.437478 | 0.170377 [0.131259, 0.220568] | 1.87545 [0.224922, 4.08448] |
| 50% | 0.450934 | 0.452609 | 0.313353 | 0.436005 | 0.174052 [0.131699, 0.22771] | 1.04537 [0.0523543, 2.27547] |
| 75% | 0.450934 | 0.452609 | 0.311816 | 0.434502 | 0.175382 [0.13199, 0.230272] | 0.787713 [0.0184329, 1.68312] |
| 100% | 0.450934 | 0.452609 | 0.310925 | 0.434066 | 0.177078 [0.132641, 0.23345] | 0.612643 [-0.0188278, 1.28293] |

Fractions refer to OTHER channels; every query is excluded even at 100%. Counts use ceil(fraction*(C-1)). All fractions use the same queries and nested sets. Full-reference moments are calculated once and reused across repetitions. Global/Source invariance is an engineering check, not empirical evidence.

## Reference EZ composition

| Fraction | CRPS change per +10pp reference EZ [11-family interval] | Identifiable / assessed patients |
| --- | --- | --- |
| 25% | 0.00190484 [-0.00430413, 0.00735483] | 80 / 80 |
| 50% | 0.00309936 [-0.00599913, 0.00934568] | 80 / 80 |
| 75% | 0.00207155 [-0.0124661, 0.010288] | 80 / 80 |
| 100% | Structurally not estimable | No repeated composition variation |

Both predictor and Full CRPS are centered within the same patient/query channel over repeated references before the patient slope is calculated. These slopes are associations. Channel count does not measure anatomical implantation coverage.

## All eleven secondary slots

| Endpoint | Mean [11-family adjusted interval] |
| --- | --- |
| E1_H_SOURCE | 0.00377235 [-0.00546385, 0.014091] |
| E1_H_CENTER | -0.0091895 [-0.034219, 0.016101] |
| E1_H_SCALE | -0.00011403 [-0.0103265, 0.0109165] |
| E2_CRPS_FULL_50_MINUS_100 | -0.00302533 [-0.0105369, 6.15239e-05] |
| E2_CRPS_FULL_75_MINUS_100 | -0.00169612 [-0.00685237, 0.000264022] |
| E2_ABSGAP_25_MINUS_100 | 1.26281 [-0.210773, 3.72009] |
| E2_ABSGAP_50_MINUS_100 | 0.432729 [-0.0706791, 1.42355] |
| E2_ABSGAP_75_MINUS_100 | 0.17507 [-0.0339023, 0.649636] |
| COMPOSITION_SLOPE_25 | 0.00190484 [-0.00430413, 0.00735483] |
| COMPOSITION_SLOPE_50 | 0.00309936 [-0.00599913, 0.00934568] |
| COMPOSITION_SLOPE_75 | 0.00207155 [-0.0124661, 0.010288] |

## Four-source E1 H and E2 curves

| Source | E1 Full H [95% CI] | Full CRPS 25% | 50% | 75% | 100% |
| --- | --- | --- | --- | --- | --- |
| hup | 0.0375589 [0.00874133, 0.0671424] | 0.183251 | 0.19219 | 0.194373 | 0.19702 |
| lzu | -0.00477489 [-0.0345728, 0.0332626] | 0.183544 | 0.184194 | 0.186127 | 0.188021 |
| multicenter | 0.00253791 [-0.0698686, 0.0774129] | 0.178542 | 0.176272 | 0.175526 | 0.175634 |
| pediatric | 0.0113087 [-0.027288, 0.05525] | 0.0625726 | 0.06165 | 0.0614437 | 0.061322 |

All source and coordinate results are descriptive. Complete tables retain every source, all 12 coordinates, all five arms, and both class losses. No best-coordinate or favorable-source replacement is permitted.

## Legacy absolute-atypicality evidence retained

Original EZ-minus-NEZ absolute-relative gap: 0.958464 [-0.2508, 2.56101] with its original six-family correction. The exact legacy CSV is copied byte-for-byte. Its inconclusive result is not replaced by new H.

## Conditional permutations and limitations

All 1,000 within-patient count-preserving label permutations are retained for each of the four nonidentity arms and each scope. Ties use the frozen numerical tolerance. Their plus-one tails are conditional diagnostics, not independent population tests.
All resampling operates on patients after repeat/query aggregation: 100 repetitions do not increase the independent sample size. Intervals condition on the historical overlapping TRAIN banks; they do not capture refitting uncertainty. The old 79 original-FIT plus one original-validation assessment exception remains explicit. No anatomical, biological-causal, information-conservation or clinical-localization conclusion follows from H, D or composition slopes alone.
