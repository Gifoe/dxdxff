# B2-based single views: different operating tradeoffs, no resolved AUROC winner

| Model | Coordinates | Parameters | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR | Top1-EZ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MAIN_B2: ABS+DELTA+RATIO, six spectral DELTA fields absent | 36 | 3649 | 0.719570 | 0.517331 | 0.642722 | 0.437631 | 0.691219 | 0.584615 |
| ABS_ONLY | 14 | 1537 | 0.689838 | 0.460809 | 0.611523 | 0.397637 | 0.633100 | 0.523077 |
| DELTA_ONLY | 8 | 961 | 0.715868 | 0.494758 | 0.648985 | 0.447061 | 0.687883 | 0.584615 |
| RATIO_ONLY | 14 | 1537 | 0.708113 | 0.506662 | 0.622821 | 0.410130 | 0.668898 | 0.553846 |

Actual completed five-fold **Development**, seed42: equal mean over65 original
validation appearances from47 unique patients. MAIN_B2 reuses its exact frozen
checkpoints/predictions; the three single views are15 new scratch-trained cells,
137 completed epochs. **No new Outer TEST.** Terminal:
`B2_BASED_SINGLEVIEW_DEVELOPMENT_COMPLETE_EXPLORATORY`.

## Was this on the new main model?

Yes. All three view sets come from B2's36 canonical coordinates, not the old
full42 F0 or an88D trainable-mask model. No deleted spectral DELTA field is
restored. DELTA_ONLY has only RMS, variance, line-length/sec and entropy mean/std:
8 coordinates,961 parameters. ABS and RATIO each retain the seven varying
physiological types' mean/std:14 coordinates,1537 parameters.

All retain original Patient Z and original FIT-fitted88D preprocessing, followed
by strict canonical-coordinate gathering. The model reconstructs original88D
normalization with removed positions zero and retained duplicate multiplicities,
not LayerNorm8/14/36. Newly trained networks fold the original seeded **untrained**
initializer, never fine-tune main weights. See `IMPLEMENTATION_AUDIT.md`.

B2 is the **user-designated** main, not a new scientifically proven winner.
Its previous comparison against full42 B1 improved ranking slightly but failed
the registered EZ-F1 stability allowance. That historical gate is not rewritten
by naming B2 the main, nor rescued by this follow-up.

## What differences are actually observed?

1. Main has the highest primary AUROC/AP point estimates. Relative to ABS_ONLY,
   gains are0.029732 AUROC and0.056521 AP. The AP interval is positive; AUROC
   and F1 intervals still include zero. ABS-only has the lowest point estimates
   in the main performance table, but this does not establish every contrast.
2. DELTA_ONLY has the highest single-view AUROC/Macro-F1/EZ-F1 point estimates.
   Main exceeds it by only0.003702 AUROC but by0.022573 AP. DELTA exceeds main
   by0.006263 Macro-F1 and0.009430 EZ-F1. **All four paired intervals include zero.**
   It is not established that full combination is better, or DELTA universally
   superior. Only8 nonspectral-DELTA coordinates retain much of the observed
   ranking point estimate; this is useful descriptive evidence, not noninferiority.
3. RATIO_ONLY has higher AP than DELTA_ONLY (0.506662 versus0.494758), despite
   lower AUROC. DELTA beats RATIO in Macro-F1/EZ-F1 with positive unadjusted
   intervals. Ranking metrics and thresholded localization need not order views
   the same way.
4. All six AUROC difference intervals include zero. There is no statistically
   resolved AUROC ordering in this single seed/fixed reused cohort. The observed
   point differences cannot establish that ABS/RATIO are useless alongside DELTA.

Single-view sufficiency is not incremental conditional utility. Demonstrating
the added value of a view specifically would require matched combination-versus-
combination-minus-that-view training. Those additional arms were **not** run or
claimed here. No two-view rescue or retrospective subset was selected.

## Classification statistics

| Model | Sensitivity | Specificity | NEZ-F1 | Balanced accuracy | Accuracy | Predicted EZ fraction |
|---|---:|---:|---:|---:|---:|---:|
| MAIN_B2 | 0.465446 | 0.869086 | 0.847813 | 0.667266 | 0.777119 | 0.196426 |
| ABS_ONLY | 0.478479 | 0.824227 | 0.825409 | 0.651353 | 0.742592 | 0.242795 |
| DELTA_ONLY | 0.494720 | 0.875019 | 0.850908 | 0.684870 | 0.781633 | 0.189154 |
| RATIO_ONLY | 0.468309 | 0.857613 | 0.835512 | 0.662961 | 0.760857 | 0.195220 |

Each model uses its own original validation-selected threshold, not a test- or
comparison-tuned threshold. Mean patient confusion counts, **not pooled counts**:

| Model | TP | FP | TN | FN |
|---|---:|---:|---:|---:|
| MAIN_B2 | 8.153846 | 11.446154 | 66.538462 | 10.369231 |
| ABS_ONLY | 8.461538 | 15.523077 | 62.461538 | 10.061538 |
| DELTA_ONLY | 8.215385 | 10.553846 | 67.430769 | 10.307692 |
| RATIO_ONLY | 7.461538 | 11.630769 | 66.353846 | 11.061538 |

These means must not be relabeled an80-patient/7635-channel pooled matrix.
Sensitivity/specificity are patient means, not ratios of the displayed means.

## Paired patient-cluster evidence

10,000 common draws, seed42;47 patient-ID clusters carry all their repeated
validation appearances. Fixed selected models and thresholds; no threshold
reoptimization in bootstrap. Unadjusted percentile95% intervals, not corrected
for adaptive historical viewing, multiple contrasts or split/seed uncertainty.

| Comparison | ΔAUROC [95% interval] | ΔAUPRC [95% interval] | ΔMacro-F1 [95% interval] | ΔEZ-F1 [95% interval] |
|---|---|---|---|---|
| Main−ABS | +0.029732 [-0.017658,0.082122] | +0.056521 [0.004414,0.112788] | +0.031199 [-0.000705,0.064263] | +0.039995 [-0.011969,0.093231] |
| Main−DELTA | +0.003702 [-0.042301,0.043624] | +0.022573 [-0.005355,0.049835] | -0.006263 [-0.027039,0.013709] | -0.009430 [-0.043697,0.023531] |
| Main−RATIO | +0.011457 [-0.025481,0.042169] | +0.010668 [-0.020014,0.044820] | +0.019901 [0.001612,0.039736] | +0.027502 [-0.001496,0.057847] |
| ABS−DELTA | -0.026030 [-0.114151,0.050249] | -0.033949 [-0.098651,0.030372] | -0.037462 [-0.078773,0.004073] | -0.049425 [-0.115517,0.017122] |
| ABS−RATIO | -0.018276 [-0.090264,0.042569] | -0.045853 [-0.104904,0.011494] | -0.011298 [-0.045752,0.024300] | -0.012493 [-0.069088,0.044802] |
| DELTA−RATIO | +0.007754 [-0.025832,0.039191] | -0.011904 [-0.044438,0.020481] | +0.026164 [0.003995,0.050920] | +0.036932 [0.002114,0.075926] |

All11 mandatory ranking/classification metrics for every contrast are in
`PAIRED_BOOTSTRAP.csv`. Positive unadjusted bounds are descriptive conditional
evidence, not fresh independent clinical confirmation.

## Fold heterogeneity and weighting

| Fold | Main AUROC | ABS | DELTA | RATIO |
|---|---:|---:|---:|---:|
| 1 | 0.708682 | 0.686041 | 0.699845 | 0.703494 |
| 2 | 0.724360 | 0.668251 | 0.717724 | 0.692329 |
| 3 | 0.710266 | 0.666457 | 0.736658 | 0.732119 |
| 4 | 0.756570 | 0.704436 | 0.777366 | 0.738646 |
| 5 | 0.697972 | 0.724004 | 0.647744 | 0.673979 |

Main beats ABS in4/5, DELTA3/5, RATIO4/5 folds. Main−DELTA ranges from−0.026392
to+0.050227; the near-zero primary mean is not uniform patient equivalence.
Known-center tables are descriptive and do not choose thresholds or features.

The secondary47-ID-equal estimator averages each ID's
repeated appearances before taking the patient mean:

| Model | 47-ID-equal AUROC | AUPRC | Macro-F1 | EZ-F1 |
|---|---:|---:|---:|---:|
| MAIN_B2 | 0.727195 | 0.528731 | 0.653697 | 0.460925 |
| ABS_ONLY | 0.690838 | 0.478186 | 0.620713 | 0.419060 |
| DELTA_ONLY | 0.736837 | 0.517031 | 0.662594 | 0.473472 |
| RATIO_ONLY | 0.722203 | 0.526166 | 0.631212 | 0.431298 |

This weighting reverses the main/DELTA AUROC point ordering. It is a different
estimand, **not** permission to replace the predeclared65-appearance primary
metric by the most favorable aggregation. Patient means are not pooled-channel
AUROC; no primary benchmark definition changes after observing results.

## Completion and limits

All15 cells and137 epochs ran under the unchanged authoritative trainer; total
measured training-invocation seconds11.715, excluding provenance, tests,
inference/statistics and reporting. No main training or additional competitive
arm. Main probabilities replay exactly, actual full-bank folding parity and
deleted-input invariance PASS; all final trained view-isolation tests PASS.
Independent score-only verification checks260 patient/model metric cells and
all66 paired-bootstrap metric/interval calculations. Source and transfer hashes
match, including byte-preserved server CSVs.

The user-selected new main and this experiment followed viewed prior Development
results. The cohort and labels have been extensively reused, and checkpoints/
thresholds are validation selected as in historical A0. These are adaptive
exploratory Development comparisons, not outer TEST or blind confirmation.
Capacity, normalization geometry and parameter-wise optimizer behavior vary
with input deletion; observed differences are not pure causal physiology tests.
In particular, ABS_ONLY is still Patient-Z-normalized, not a no-patient-relative
control. No clinically irrelevant feature claim follows from a weak single-view
model, high correlation or a small coefficient.

Only source, hashes and anonymous aggregate results are public. Raw features,
patient/channel IDs, labels, scores, checkpoints, preprocessing state and logs
remain on the trusted server. No installed skill, raw extraction, new outer,
extra seed, two-view model, calibration rescue or global winning subset. Stop
after this fixed seed42 single-view study; main designation remains the user's
specified B2 and is not automatically reassigned from this table.
