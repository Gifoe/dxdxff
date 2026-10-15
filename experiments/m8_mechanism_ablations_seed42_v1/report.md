# M8 single-mechanism ablations and simple late fusion — completed five-fold Outer

## Actual execution

Completed on 2026-10-16 (Asia/Shanghai), seed42. All eight reported arms cover the same original 80 patients, 7,635 canonical channels and 256 seizures. The original masks and 1,471,965 valid raw channel-windows were preserved; no new extraction or window subsampling occurred. A1–A4 are **20 newly trained correction-model cells**, not inference-time switches on trained full M8. A6 uses **five newly validation-selected probability fusions of independent frozen component models**. Three historical reference arms were hash-verified and reused without new inference.

Primary reporting is the mean of 80 patient metrics, not channel-pooled scores or an equal-weight mean of five unequal-sized folds. EZ is the positive class. Thresholds are the original validation-only NEZ-probability thresholds; EZ is predicted when pNEZ is strictly below the frozen threshold.

## Patient-macro classification results

| Model | AUROC | AP | Macro-F1 | EZ-F1 | NEZ-F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Full M8, frozen reference | 0.762397 | 0.554067 | 0.634097 | 0.434087 | 0.834107 | 0.760166 |
| A1 w/o Feature PR, Raw PR retained | 0.752247 | 0.539495 | 0.615195 | 0.428061 | 0.802329 | 0.739213 |
| A2 w/o Raw PR, Feature PR retained | 0.738449 | 0.537326 | 0.623616 | 0.413664 | 0.833569 | 0.761052 |
| A3 Identity Gate | 0.760983 | 0.549366 | 0.633452 | 0.431143 | 0.835761 | 0.761929 |
| A4 w/o Raw pre-ictal latent reference | 0.741058 | 0.544288 | 0.625461 | 0.420872 | 0.830051 | 0.756275 |
| A6 Simple Late Fusion | 0.739324 | 0.533581 | 0.618049 | 0.401593 | 0.834506 | 0.757102 |
| Independent D1 PR, frozen reference | 0.732636 | 0.527175 | 0.616556 | 0.408531 | 0.824582 | 0.748950 |
| Independent M8 Raw-only, frozen reference | 0.725813 | 0.533535 | 0.610596 | 0.390272 | 0.830919 | 0.749193 |

## Ranking and EZ-fraction bias

| Model | MRR | Top-1 | EZ-fraction bias |
|---|---:|---:|---:|
| Full M8 | 0.749310 | 63.75% | -0.029065 |
| A1 w/o Feature PR | 0.744476 | 62.50% | +0.030740 |
| A2 w/o Raw PR | 0.720280 | 60.00% | -0.046897 |
| A3 Identity Gate | 0.743507 | 62.50% | -0.034647 |
| A4 w/o Raw pre-ictal reference | 0.756637 | 66.25% | -0.023854 |
| A6 Simple Late Fusion | 0.745466 | 62.50% | -0.053995 |
| Independent D1 PR | 0.689399 | 57.50% | -0.027451 |
| Independent M8 Raw-only | 0.756390 | 67.50% | -0.059205 |

Top-1 is whether the highest-scoring canonical channel for a patient is EZ; ties use canonical channel name. Bias = predicted EZ fraction minus true EZ fraction, averaged across patients. For example -0.029065 is -2.9065 percentage points, not a relative percent change. Ranking and thresholded F1 measure different properties; the higher Raw-only Top-1 does not make its Macro-F1 better than full M8.

## Paired comparisons with full M8

Difference is **ablation minus full M8**. Intervals are 10,000 paired patient-bootstrap percentile 95% intervals (seed42), fixed models/thresholds/weights, not multiplicity-adjusted.

| Ablation | ΔAUROC [95% CI] | ΔMacro-F1 [95% CI] | ΔEZ-F1 [95% CI] |
|---|---|---|---|
| A1 w/o Feature PR | -0.010150 [-0.035190, 0.014181] | -0.018902 [-0.040362, 0.003463] | -0.006026 [-0.045123, 0.035231] |
| A2 w/o Raw PR | -0.023948 [-0.049923, -0.002268] | -0.010481 [-0.029124, 0.007803] | -0.020423 [-0.053470, 0.012008] |
| A3 Identity Gate | -0.001413 [-0.003333, 0.000668] | -0.000645 [-0.007478, 0.005723] | -0.002944 [-0.015164, 0.008080] |
| A4 w/o Raw pre-ictal reference | -0.021338 [-0.044401, -0.002731] | -0.008636 [-0.024171, 0.006352] | -0.013215 [-0.039130, 0.012034] |
| A6 Simple Late Fusion | -0.023072 [-0.044994, -0.003117] | -0.016048 [-0.032645, -0.000035] | -0.032494 [-0.060606, -0.005438] |

### Answers to the research questions

1. **Is feature PR necessary?** Removing it reduced mean Macro-F1 by 1.8902 percentage points, but the paired interval includes zero. This supports a favorable point estimate for retaining feature PR, not a demonstrated necessity claim. A1 uses a matched no-PR D1 trained on the same FIT patients and its necessary FIT-only preprocessing, while newly training the Raw correction; it is not simply feeding unnormalized features to a PR-trained base.
2. **Does Raw PR add value?** Its removal reduced AUROC by 2.3948 points; the unadjusted paired interval excludes zero. Macro-F1 fell 1.0481 points but its interval includes zero. Evidence is stronger for patient-local ranking discrimination than for a definite thresholded F1 gain.
3. **Does physiological gating matter?** Macro-F1 is 0.633452 without the learned gate versus 0.634097 with it, a difference of only 0.0645 percentage points. This experiment does not establish an important Macro-F1/AUROC gain from gating. AP fell 0.004701 without the gate, CI [-0.010176, -0.000324], a small unadjusted AP benefit that should not be suppressed or inflated. Identity gating removes 1,048 trainable parameters; this is not a parameter-count-matched conditioning control and does not remove physiology from the frozen D1 branch.
4. **Does the temporal reference help?** Removing the Raw latent pre-reference reduced AUROC by 2.1338 points, with an unadjusted interval excluding zero. Macro-F1 fell 0.8636 points but remains uncertain. Top-1 increased by 2.5 points, CI [-6.25, 11.25] points: no claim that the reference improves every ranking endpoint is justified. A4 leaves the seven physiological DELTA definitions and ordinary window normalization intact.
5. **Is residual fusion better than simple ensemble?** Full M8 beat this specific independent probability-fusion baseline by 2.3072 AUROC points, 1.6048 Macro-F1 points and 3.2494 EZ-F1 points. The Macro-F1 interval barely excludes zero before multiplicity adjustment; do not call it decisive. This compares two concrete packages, not a proof that every residual fusion beats every possible ensemble. A6 itself improved Macro-F1 over D1 by only 0.001493, CI [-0.023094, 0.026841], and over Raw-only by 0.007454, CI [-0.010173, 0.025339].

Overall, the clearest conditional AUROC evidence favors retaining Raw PR and the pre-onset Raw reference. The learned gate has little demonstrated Macro-F1 contribution in this seed42 run. No model was changed, reselected or retrained in response to these observations.

## A6 selected weights — validation only

Probability fusion: pNEZ = (1-w) pD1 + w pRaw. Each component is independently trained/frozen; Raw-only is not extracted from full M8's correction branch. Search was fixed to w = 0, 0.05, ..., 1, with the original validation threshold selector. Ties use patient Macro-F1, EZ-F1, then smaller raw weight.

| Fold | Raw-only weight w | D1 weight | Frozen NEZ threshold |
|---|---:|---:|---:|
| 1 | 0.75 | 0.25 | 0.445 |
| 2 | 0.15 | 0.85 | 0.275 |
| 3 | 0.90 | 0.10 | 0.320 |
| 4 | 0.65 | 0.35 | 0.455 |
| 5 | 1.00 | 0.00 | 0.325 |

An independent post-run verifier reconstructed **all 105 validation candidates**, all five selected weights/thresholds, and all five sealed Outer probability mixtures exactly. It did not fit models or run new inference.

## Execution and verification ledger

- Preserved the original M8 FP32 optimizer, loss, per-patient updates, initialization, early stopping and 30-epoch maximum. A1/A2/A4 each have 9,481 trainable correction parameters; A3 has 8,433; the 1,537-parameter D1 base is frozen.
- Actual epoch counts: A1 82, A2 59, A3 71, A4 65; **277 total epochs** across 20 new fits. Recorded epoch times sum to 497.619 s (A1 157.994, A2 105.415, A3 118.884, A4 115.327 s), excluding admissions, imports, input audits, final checkpoint/validation work, Outer, reporting and transfer. These timings are from this run, not historical estimates.
- Pre-training repairs only: first synthetic process failed on duplicate Intel OpenMP loading; reordered NumPy/SciPy/sklearn before torch, never enabled unsafe duplicate-runtime override. The second synthetic process passed CPU equality but failed an overly strict bitwise GPU-backward assertion. Final outcome-free testing measured new-vs-original GPU maximum gradient drift 6.0536e-9 versus original self-replay 4.1910e-9; CPU forward/gradient and GPU forward differences were zero. Prospective FP32 gradient tolerances and self-replay checks were fixed before any cohort training.
- Four real complete-FIT-patient admissions passed exact enabled-feature parity, zero-initialized residual identity, frozen-base integrity and finite/nonzero gradient-path checks. All synthetic smoke states were discarded before production training. No completed training epoch was discarded or replayed due these pre-admission failures.
- All 20 new checkpoints, five fusion choices and historical references were frozen before the new Outer pass. Forty arm-fold score files were sealed before reading full evaluation labels; per-patient atomic private caches remain bound to the same freeze/checkpoint.
- Independent sklearn confusion/F1/AUROC/AP/float64 logloss and direct ranking checks passed for **640 patient rows** across eight arms. Historical D1/full-M8/Raw-only primary metrics reproduced exactly to 1e-12. Base replays were within the registered 1e-6 limit. Protected source, checkpoints, caches and reference artifacts were unchanged.
- Direct bootstrap resampling matched count-matrix computation for 256 draws at 1e-12 absolute tolerance. Public artifact hashes, source hashes, forbidden-identity-column checks and weighted-fold reconstruction also passed locally after transfer.
- Freeze SHA256: `d9bb40652bb24d2422edcf1372dde1f544803f166ad4a6004e4217348f1bf1e4`.
- Score seal SHA256: `dcc6516cb73213db7ec88e23c87d55c7b2aaab837f23a52e0c3490b293516ad3`.
- Exact final terminal: **`M8_MECHANISM_5FOLD_OUTER_COMPLETE`**; `INDEPENDENT_VERIFICATION.json: PASS`; `INDEPENDENT_FUSION_AUDIT.json: PASS`. `RUN_STATUS.json` is the last training snapshot; `FINAL_STATUS.json` is the authoritative final state.

## Artifacts and limits

`results/main_results.csv` has exact patient-macro values; other files separately contain pooled/fold/center metrics, confusion counts, ranking, validity coverage, fold mean/SD, model/epoch selection, the complete fusion search and paired intervals. Source files and anonymous audits are included; EEG, identities, individual scores/labels/features, models/scalers and runtime logs remain only on the trusted server.

**EXPLORATORY_REPEATED_TEST**: these Outer patients have been viewed before. One training seed, conditional patient-bootstrap uncertainty, no multiplicity correction and no inclusion of model-selection/training-seed uncertainty. A1's base must differ because its preprocessing/training differ; A3 changes parameter count; A6 uses the already declared standalone bounded Raw-only readout. These are limitations of the specified comparisons, not reasons to tune against Outer or selectively omit unfavorable endpoints.
