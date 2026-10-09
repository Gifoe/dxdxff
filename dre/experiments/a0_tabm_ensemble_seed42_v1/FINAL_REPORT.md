# A0 + TabM-style K=4 ensemble: completed seed42 development experiment

Terminal: **TABM_NOT_SUPPORTED**. All ten new students completed; A0 was
replayed, not retrained. The fixed continuation gate failed. No outer TEST
prediction, metric evaluation, hyperparameter search or follow-up model was run.

## Matched results

All entries are patient-equal means over the same 65 validation-fold appearances
from 47 unique IDs. They are checkpoint/threshold-selected development results,
not independent outer-test confirmation. Each fold contributes 13 appearances.
The source cohort remains 80 patients, 7,635 unique patient-channels and 88 features.

| Metric | A0, 8,817 params | W1, 10,167 params | W2, 10,132 params |
|---|---:|---:|---:|
| Patient Macro-F1 | 0.638080 | 0.636871 | 0.637767 |
| EZ-F1 | 0.431198 | 0.430768 | 0.430780 |
| NEZ-F1 | 0.844961 | 0.842975 | 0.844754 |
| Balanced accuracy | 0.665436 | 0.666028 | 0.667014 |
| EZ-AUPRC | 0.518235 | 0.524965 | 0.528517 |
| EZ-AUROC | 0.710667 | 0.719558 | 0.720047 |
| EZ-MRR | 0.686550 | 0.731015 | 0.709784 |
| Top1-is-EZ | 0.584615 | 0.646154 | 0.615385 |
| EZ sensitivity | 0.465904 | 0.471154 | 0.467358 |
| NEZ specificity | 0.864968 | 0.860901 | 0.866670 |
| Accuracy | 0.773174 | 0.770932 | 0.773908 |

Exact values: [VALIDATION_SUMMARY.csv](results/VALIDATION_SUMMARY.csv).

### Paired uncertainty

10,000 draws, seed42; resample 47 patient-ID clusters with replacement and retain
all appearances of each sampled ID. Keep checkpoints and numeric thresholds
fixed. The statistic weights the resulting appearance rows equally, matching
the primary development mean. This is not an independent-channel bootstrap,
does not refit models, and does not correct validation-selection optimism.

| Contrast | Macro-F1 delta | Paired 95% percentile interval |
|---|---:|---:|
| W1 − A0 | −0.001208 | [−0.011574, +0.008750] |
| W2 − A0 | −0.000313 | [−0.009571, +0.008879] |
| W2 − W1 | +0.000896 | [−0.009819, +0.011985] |

W2 − A0 EZ-F1 is −0.000418 [−0.015650, +0.014517]. EZ-AUPRC is
+0.010282 [−0.000033, +0.020707]. EZ-AUROC is +0.009380
[+0.002487, +0.016513]: a small secondary ranking signal versus A0, not a
primary F1 success. Against W1, W2 AUROC is only +0.000489
[−0.007477, +0.008586] and AP +0.003552 [−0.004680, +0.012102].
Thus this does not isolate a convincing TabM-specific ranking advantage.
These are unadjusted development intervals across multiple reported metrics.
All 33 contrast/metric intervals: [PAIRED_BOOTSTRAP.csv](results/PAIRED_BOOTSTRAP.csv).

## Folds, centers and selection

| Fold | A0 Macro-F1 | W1 Macro-F1 | W2 Macro-F1 | W2 − A0 | W2 − W1 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.646844 | 0.646897 | 0.650971 | +0.004128 | +0.004075 |
| 2 | 0.657246 | 0.659487 | 0.663103 | +0.005858 | +0.003616 |
| 3 | 0.642299 | 0.642430 | 0.632986 | −0.009313 | −0.009444 |
| 4 | 0.640909 | 0.634712 | 0.643770 | +0.002862 | +0.009058 |
| 5 | 0.603102 | 0.600830 | 0.598004 | −0.005098 | −0.002826 |

W2 wins 3/5 folds against each comparator, not the required 4/5 against A0.
All fold metrics: [VALIDATION_BY_FOLD.csv](results/VALIDATION_BY_FOLD.csv).

| Center | Appearances / unique IDs | A0 Macro-F1 | W1 Macro-F1 | W2 Macro-F1 | W2 − A0 |
|---|---:|---:|---:|---:|---:|
| HUP | 28 / 21 | 0.669631 | 0.668255 | 0.672202 | +0.002571 |
| LZU | 14 / 11 | 0.590312 | 0.601787 | 0.584361 | −0.005951 |
| Multicenter | 11 / 9 | 0.740741 | 0.712562 | 0.720546 | −0.020194 |
| Pediatric | 12 / 6 | 0.526085 | 0.535192 | 0.543845 | +0.017761 |

Pediatric is the worst center for all arms; W2 improves it but remains at
0.543845 Macro-F1. Positive changes are uneven and canceled by other centers;
they are not a general improvement. Center samples are small and not independent
confirmation. [Full center metrics](results/VALIDATION_BY_CENTER.csv),
[worst-center metrics](results/WORST_CENTER_METRICS.csv).

| Fold | A0 selected epoch / NEZ threshold | W1 selected epoch / NEZ threshold | W2 selected epoch / NEZ threshold | W1 / W2 completed epochs |
|---|---:|---:|---:|---:|
| 1 | 3 / 0.465 | 1 / 0.470 | 1 / 0.490 | 7 / 7 |
| 2 | 1 / 0.275 | 1 / 0.220 | 2 / 0.300 | 7 / 8 |
| 3 | 2 / 0.340 | 2 / 0.345 | 3 / 0.320 | 8 / 9 |
| 4 | 2 / 0.355 | 1 / 0.385 | 1 / 0.415 | 7 / 7 |
| 5 | 2 / 0.445 | 2 / 0.435 | 2 / 0.460 | 8 / 8 |

All scratch runs stopped by the original patience rule, not by outcome-based
manual cancellation. Every completed epoch, initial state, best/last checkpoint,
optimizer and RNG state is retained privately. No fold was discarded or retrained.

## Threshold and error correction

At each fold's frozen A0 threshold, Macro-F1 is A0 0.638080,
W1 0.630322, W2 0.625229. W2 − A0 is then −0.012850 and W2 EZ-F1
drops to 0.399237. W2 sensitivity is 0.401830 and specificity 0.891355.
Selected W2 thresholds recover some of this sensitivity/F1 loss but do not
produce an improvement over A0. No threshold was reselected for this diagnostic.
[FIXED_A0_THRESHOLD_DIAGNOSTIC.csv](results/FIXED_A0_THRESHOLD_DIAGNOSTIC.csv).

Across 6,273 validation-channel **appearances**, not unique source channels:

| Model / threshold | A0 errors corrected | A0 correct spoiled | EZ TP gained / lost | EZ FP added / removed |
|---|---:|---:|---:|---:|
| W1 own selected | 126 | 126 | 44 / 31 | 95 / 82 |
| W2 own selected | 153 | 154 | 39 / 42 | 112 / 114 |
| W1 frozen A0 | 122 | 161 | 45 / 34 | 127 / 77 |
| W2 frozen A0 | 200 | 149 | 10 / 97 | 52 / 190 |

The fixed-A0 W2 accuracy benefit therefore largely trades away EZ detection for
fewer NEZ false positives; it is not a localization F1 improvement. Pooled
correction counts are descriptive and do not replace patient-equal metrics.
[ERROR_CORRECTION_AUDIT.csv](results/ERROR_CORRECTION_AUDIT.csv).

W2 changes 9.02% of within-patient pairwise score orders (W1 8.40%). Its mean
within-patient score correlation with A0 is 0.9612, with mean NEZ probability
shift +0.00270 and mean absolute shift 0.04140. Ranking changes are real but
small, and comparable-capacity W1 already captures most of the AUROC gain.
This is a score/operating-point analysis, **not a formal calibration assessment**:
no ECE, reliability curve or calibration guarantee is claimed.
[RANKING_SCORE_AUDIT.csv](results/RANKING_SCORE_AUDIT.csv).

W2 improves Macro-F1 for 36.92% of the 65 appearances and worsens it for 38.46%;
the rest tie. Averaging repeated appearances within each ID first, 34.04% of
47 IDs improve and 44.68% worsen. Both medians are zero. These distributions
are publicly aggregated only; [quantiles and all contrasts](results/PATIENT_IMPROVEMENT_DISTRIBUTION.csv).

## Ensemble diagnostics

Individual members below use the **parent ensemble's selected numeric threshold**.
No member is deployed, pruned or selected from these diagnostic results.

| Member | Macro-F1 | EZ-AUPRC |
|---|---:|---:|
| 0 | 0.633394 | 0.521386 |
| 1 | 0.620898 | 0.516189 |
| 2 | 0.620761 | 0.528429 |
| 3 | 0.623110 | 0.522137 |
| Fixed four-member ensemble | 0.637767 | 0.528517 |

Ensemble Macro-F1 exceeds the best single member at the parent threshold by
0.004373. A separately labeled posthoc member-threshold oracle reduces this gap
to 0.000423. It cannot alter the deployed model. Best-member AP gap is only
0.000088; ensemble AUROC is 0.001809 below the best diagnostic member.
Metric-specific best-member comparisons are descriptive oracles, not formal
benchmarks, and no confidence or deployed-member advantage is inferred from them.

Six mean within-patient pair correlations range 0.9397–0.9540. Mean pairwise
binary disagreement is 5.99%, and mean per-channel member probability variance
is 0.0008102. Members are nonidentical, but highly correlated; diversity does not
translate into the prespecified primary improvement.
[Member metrics](results/MEMBER_DIVERSITY_SUMMARY.csv),
[pair statistics](results/MEMBER_PAIR_DIVERSITY.csv),
[ensemble-versus-best diagnostic](results/ENSEMBLE_VS_BEST_MEMBER_DIAGNOSTIC.csv).

## Runtime, integrity and deviations

W1 total measured optimization+validation time is 3.617 s over 37 completed
epochs; W2 is 4.320 s over 39. These synchronized times exclude checkpoint I/O,
startup, preprocessing, tests and report/audit generation, and are **not end-to-end
task elapsed time**. Peak allocated CUDA memory reaches 22,594,048 bytes for W1
and 26,094,080 for W2, including inputs/model/optimizer but not GPU-reserved memory.
Whole-fold inference times are approximately 0.073–0.109 ms for W1 and
0.120–0.197 ms for W2. W2 evaluates four member predictions/channel versus one
for A0/W1. These are microbenchmarks with 10 warmups/20 repeats, not production
throughput. Historical A0 training time/memory are unavailable, left blank.
[EFFICIENCY_AUDIT.csv](results/EFFICIENCY_AUDIT.csv).

A0 preprocessing arrays and statistics matched bitwise, including feature order
and FIT membership. Frozen A0 replay drift was at most 1.673e-7 (limit 2e-7);
all requested reproduced metrics matched within 1e-12. SHA-bound checkpoints,
inputs, initial states, selected thresholds and saved predictions passed final
integrity checks. Six pretraining synthetic groups and both three-patient-update
real FIT-only smoke tests passed, including exact upstream component parity,
loss/inference rules, memberwise gradients and deterministic resume.

The first hidden launcher exited without training artifacts; its cause is not
established. A subsequent foreground run completed all ten tasks without a
training interruption. Aggregation then twice encountered native access violations;
the diagnostic traceback located repeated pandas column insertion in member-table
assembly. Replacing only that metadata assembly with one concatenation allowed
aggregation to complete. The additional synthetic table-regression test passed.
Original code/logs remain private. No training model, source input, loss, threshold,
metric formula, bootstrap draw or scientific protocol changed for this repair.
All-cohort clinical fields were materialized before sealing original FIT/VAL
arrays; no outer TEST predictions or performance were accessed.

## Prespecified gate and the twelve answers

Gate failed: W2 − A0 Macro-F1 is below +0.015; W2 − W1 below +0.005;
EZ-F1 declines; W2 wins only 3/5 rather than 4/5 against A0. AP nondecline,
3/5 wins against W1 and implementation/isolation checks pass. The +0.030
target is not reached. [VALIDATION_GATE.json](audit/VALIDATION_GATE.json).

1. **W2 beats A0?** No on the primary metric: −0.000313, interval crosses zero.
2. **W2 beats W1?** Only +0.000896 point estimate, not a supported improvement.
3. **Beyond ordinary capacity?** No established gain; W1 has similar AUROC/AP
   changes and better MRR/Top1 point estimates. Neither arm improves A0 F1.
4. **Consistent across folds?** No: W2 loses folds3 and5 to both comparators.
5. **EZ-F1 improves?** No, −0.000418 versus A0.
6. **EZ-AUPRC improves?** Point estimate +0.010282; interval narrowly crosses zero.
7. **Persists at A0 threshold?** No; W2 Macro-F1 becomes 0.625229.
8. **Members diverse?** Nonidentical but highly correlated, disagreement 5.99%.
9. **Ranking or score shift?** A small secondary AUROC gain exists versus A0,
   not convincingly versus W1; threshold transfer degrades F1. Not pure score
   shifting, but not a robust localization improvement either.
10. **Center concentration?** Pediatric/HUP improve; LZU/Multicenter worsen.
11. **+0.03–0.04 target met?** No; the primary point estimate is negative.
12. **Continue or stop?** Stop this approach under the fixed gate. No outer TEST
    or new model is authorized, and none was initiated.

**TABM_NOT_SUPPORTED** describes this A0-specific one-hidden-layer, K4, seed42
development experiment. It is not a claim that full published TabM fails in
general. Source and non-identifiable aggregates alone are committed; patient/
channel predictions, features, checkpoints, RNG/optimizer states and runtime
logs remain on the trusted server. The existing branch is reused, as requested.
