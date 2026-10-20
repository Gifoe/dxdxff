# Relative-guidance F0: original80-patient protocol

This study modifies the exact historical six-feature no-Variance/no-head-interaction F0 architecture. The original CNN, pre-reference, stage/seizure pooling, late Gate and single64-hidden head remain. Full and all PR controls have11425parameters; deleting early FiLM or late Gate gives10377. No trained original weights initialize a fresh fit.

All45models/preprocessors/validation-selected epochs and thresholds froze before new Outer inference; all45score cells sealed before metrics.40fresh fits and5exact original reference reuses. Original80patients,7635channels,256seizures and1471965valid Raw windows, original five patient-held-out roles, seed42, optimizer/loss/budget and threshold rule are unchanged. No extra Development admission performance gate is used.

All results are exploratory repeated-test: the reference architecture and this historical Outer cohort have repeatedly been viewed. This is not independent confirmation. No configuration, epoch or threshold is selected from these Outer results.

## Equal-patient averages

| Configuration | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac. bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original 0.653828 main (exact reuse) | 0.653828 | 0.752255 | 0.585240 | 0.467424 | 0.840232 | 0.770976 | 0.845674 | 0.800000 | -0.022662 |
| Fixed relative-guidance F0 | 0.646339 | 0.765888 | 0.589527 | 0.447095 | 0.845582 | 0.774722 | 0.836034 | 0.775000 | -0.054110 |
| w/o physiological PR (summary and window) | 0.636037 | 0.744249 | 0.575278 | 0.448413 | 0.823662 | 0.748958 | 0.817611 | 0.750000 | -0.002099 |
| w/o Raw PR | 0.630820 | 0.763988 | 0.563932 | 0.431853 | 0.829787 | 0.762890 | 0.777221 | 0.687500 | -0.038453 |
| w/o BOTH PR | 0.516322 | 0.759725 | 0.570728 | 0.371850 | 0.660795 | 0.662387 | 0.803417 | 0.725000 | 0.129884 |
| w/o relative window guidance | 0.652906 | 0.764621 | 0.591357 | 0.463938 | 0.841873 | 0.770852 | 0.848603 | 0.800000 | -0.032339 |
| w/o early FiLM | 0.646647 | 0.770064 | 0.573003 | 0.457289 | 0.836004 | 0.762476 | 0.784167 | 0.687500 | -0.022903 |
| w/o late Gate | 0.653465 | 0.765286 | 0.593488 | 0.464629 | 0.842301 | 0.772424 | 0.846368 | 0.800000 | -0.034239 |
| w/o own pre-ictal reference | 0.618218 | 0.743705 | 0.558772 | 0.410700 | 0.825736 | 0.752566 | 0.788722 | 0.712500 | -0.028346 |

Macro-F1 averages EZ/NEZ F1 within each patient and then equally over80patients. Pooled-channel results are separate; ranking uses original channel-name tie breaking. EZ-fraction bias is predicted minus observed EZ fraction. [Folds](FOLD_RESULTS.csv), [sample SD](FOLD_SD.csv), [sources](SOURCE_RESULTS.csv), [pooled](POOLED_RESULTS.csv), [coverage](RAW_COVERAGE.csv).

## Does PR improve both ranking and classification?

The fixed contrasts are Full minus the stated control. Positive differences favor the candidate. All contrasts are reported, including unfavorable results.10000seed42same-patient bootstrap draws give162nominal95% intervals and14additional Bonferroni-level percentile intervals (seven contrasts times AUROC/Macro-F1, tail0.05/28). These are approximate exploratory intervals conditional on trained models; they do not correct historical architecture adaptation or measure training-seed uncertainty.

| Contrast | Macro-F1 difference | AUROC difference | Folds both positive | Additional-family lows (F1, AUC) | Stable within this seed/protocol |
| --- | ---: | ---: | ---: | --- | --- |
| FULL_MINUS_B0 | -0.007489 | +0.013633 | 2/5 | -0.029443, -0.005792 | False |
| FEATURE_PR | +0.010301 | +0.021639 | 3/5 | -0.018202, -0.002848 | False |
| RAW_PR | +0.015519 | +0.001900 | 3/5 | -0.010791, -0.036523 | False |
| WINDOW_PR | -0.006567 | +0.001268 | 2/5 | -0.025037, -0.012739 | False |
| EARLY_FILM | -0.000308 | -0.004176 | 3/5 | -0.020569, -0.027481 | False |
| LATE_GATE | -0.007126 | +0.000602 | 1/5 | -0.023687, -0.011701 | False |
| PRE_REFERENCE | +0.028120 | +0.022183 | 4/5 | +0.004963, -0.010676 | False |

The predeclared within-protocol evidence rule requires positive means for both metrics, both additional-family lower bounds above zero, and both metric directions positive in at least4of5folds. Fold counts are descriptive correlated estimates, not five independent replications.

## Fixed objective audit

Overall fixed audit: **FAIL**. No result was guaranteed and no winning control is promoted to a new main model.

- all_retained_components_have_stable_two_metric_evidence: False
- macro_f1_within_1pp_of_original: True
- main_auroc_above_original: True
- main_macro_f1_above_0p64: True

## Interpretation and implementation boundaries

- Summary Feature PR uses a FIT-unit-consistent degeneracy rule; relative window guidance blends the original FIT-global DELTA with complete-patient same-seizure/same-valid-time standardized channel contrast. The0.5mixture, epsilon and unsupported-window fallback were frozen before training. No new feature extraction, trainable capacity or Raw dual-view expansion is introduced.
- w/o Feature PR removes BOTH summary patient normalization and relative physiological window guidance, restoring the original FIT-global inputs, not zeros. w/o Raw PR removes only the final Raw96patient normalization. w/o BOTH PR removes both; ordinary waveform normalization and own pre-reference remain. The five newly trained no-PR controls must exactly reproduce the original identical no-PR selected weights, validation, epochs and thresholds.
- w/o window PR isolates the new window context from the direct-summary preprocessing repair. Thus Full minus original B0 is a joint preprocessing/context revision, not solely a window-context effect.
- Early FiLM and late Gate removals reduce trainable capacity; their differences cannot isolate information content from capacity/optimization. No purely predictive association establishes physiology or seizure-propagation causality.
- The PR interaction in INTERVALS.csv is Full minus NoFeaturePR minus NoRawPR plus NoBothPR, on each patient metric. It is a factorial performance interaction, not evidence of biological interaction.
- Invalid/padded windows never enter channel statistics; valid unsupported contexts retain the original global input. No patient, channel, seizure or window is discarded for frequency quality or metric performance.
- This one-seed experiment cannot establish seed robustness or independent external generalization. Source summaries suppress metric cells below five patients while retaining all patients in the main averages.
- Independent checks: 720patient rows, 6480metric values,162nominal and14additional intervals; maximum independent interval discrepancy 8.88e-16.
- Only source, hashes, anonymous counts and aggregate outcomes are public. Signals, identities, individual predictions, weights, preprocessors, caches and runtime logs remain on the trusted server.
