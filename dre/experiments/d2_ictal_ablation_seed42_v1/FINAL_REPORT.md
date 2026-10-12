# D2 ictal complete ablation study, seed42

Original 80-patient cohort, 7635 canonical channels (1743 EZ / 5892 NEZ), exact
original five outer folds. All metrics below are **equal-patient outer TEST
means**; every patient appears once per configuration. Full D2 and w/o Patient Z
reuse verified frozen scores. The five other configurations were independently
retrained from the prescribed original initialization. No new configuration,
seed search, threshold rule, feature extraction or outcome-dependent iteration.

## Table 1. Component ablation

| Model | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full D2 | 0.730445 | 0.533483 | 0.620403 | 0.410909 | 0.737216 |
| w/o Patient Z | 0.660217 | 0.480013 | 0.493978 | 0.259845 | 0.685571 |
| w/o Source | 0.739768 | 0.537887 | 0.624919 | 0.419405 | 0.728620 |

## Table 2. Physiological feature ablation

| Model | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full D2 | 0.730445 | 0.533483 | 0.620403 | 0.410909 | 0.737216 |
| ABS Only | 0.701075 | 0.482466 | 0.602575 | 0.391919 | 0.653983 |
| w/o DELTA | 0.710437 | 0.524947 | 0.602012 | 0.391946 | 0.716668 |
| w/o ZDELTA | 0.733994 | 0.526393 | 0.618839 | 0.409059 | 0.697692 |
| w/o RATIO | 0.726352 | 0.527691 | 0.604838 | 0.390655 | 0.718524 |

DELTA here means deviation from the same-channel pre-ictal reference, **not
delta-frequency band power**. All feature ablations retain the88D network,
LayerNorm, source mechanism and9221 parameters. Effective retained column
counts are18/62/62/70 respectively, with masks after original preprocessing.

## Remaining metrics

| Model | NEZ-F1 | Balanced accuracy | Sensitivity | Specificity | Top1 EZ | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full D2 | 0.829898 | 0.652228 | 0.435823 | 0.868634 | 0.650000 | 0.759548 |
| w/o Patient Z | 0.728112 | 0.569316 | 0.363673 | 0.774958 | 0.600000 | 0.705398 |
| w/o Source | 0.830434 | 0.666405 | 0.471033 | 0.861776 | 0.637500 | 0.758773 |
| ABS Only | 0.813231 | 0.634556 | 0.435365 | 0.833747 | 0.537500 | 0.736568 |
| w/o DELTA | 0.812079 | 0.638145 | 0.436082 | 0.840208 | 0.625000 | 0.736766 |
| w/o ZDELTA | 0.828618 | 0.653030 | 0.446865 | 0.859195 | 0.600000 | 0.760591 |
| w/o RATIO | 0.819021 | 0.643556 | 0.432582 | 0.854530 | 0.612500 | 0.744879 |

Exact full-precision estimates, parameter/feature counts and reuse/training
status are in `results/ABLATION_SUMMARY.csv`. The source-free model has8817
parameters and no V/u/source bias; historical bias-only R0 was not used.

## Paired patient-cluster effects

**Direction: Full D2 minus ablated model**. Positive means Full D2 is better.
Each cell is point effect [percentile95 CI], based on10000 paired resamples of
the same80 unique patients, seed42, with all validation-selected thresholds
fixed. Intervals are **unadjusted descriptive exploratory intervals**, not
familywise-adjusted confirmatory tests. They quantify patient sampling, not
training-seed variability.

| Ablated configuration | ΔAUROC [95% CI] | ΔAUPRC [95% CI] | ΔMacro-F1 [95% CI] | ΔEZ-F1 [95% CI] |
| --- | --- | --- | --- | --- |
| w/o Patient Z | +0.070228 [0.030976, 0.110716] | +0.053470 [0.014419, 0.094825] | +0.126425 [0.089640, 0.162872] | +0.151064 [0.089214, 0.211649] |
| w/o Source | −0.009323 [−0.028279, 0.004816] | −0.004404 [−0.016121, 0.007986] | −0.004516 [−0.017641, 0.008180] | −0.008496 [−0.027337, 0.010500] |
| ABS Only | +0.029370 [0.002152, 0.056167] | +0.051017 [0.016469, 0.086554] | +0.017828 [−0.006472, 0.041385] | +0.018989 [−0.019362, 0.056419] |
| w/o DELTA | +0.020008 [−0.000255, 0.040336] | +0.008536 [−0.011625, 0.028813] | +0.018391 [0.002765, 0.034254] | +0.018963 [−0.005771, 0.044671] |
| w/o ZDELTA | −0.003549 [−0.019283, 0.009397] | +0.007090 [−0.009387, 0.023433] | +0.001565 [−0.013878, 0.018311] | +0.001850 [−0.024069, 0.030724] |
| w/o RATIO | +0.004093 [−0.012969, 0.019431] | +0.005792 [−0.010404, 0.021449] | +0.015565 [0.002256, 0.029291] | +0.020253 [−0.001130, 0.042958] |

`results/PAIRED_BOOTSTRAP.csv` supplies all66 metric effects, intervals and
patient direction counts, including secondary ranking/confusion metrics.
These effects use private paired scores, not reconstructed aggregate averages.

## Fold and center consistency

Counts below indicate strata with a **positive Full-minus-ablated point
difference**, without claiming stratum-level statistical significance.

| Ablation | Macro-F1: folds /5 | Macro-F1: centers /4 | AUROC: folds /5 | AUROC: centers /4 |
| --- | ---: | ---: | ---: | ---: |
| w/o Patient Z | 5 | 4 | 5 | 3 |
| w/o Source | 2 | 1 | 1 | 1 |
| ABS Only | 3 | 3 | 3 | 4 |
| w/o DELTA | 4 | 3 | 4 | 3 |
| w/o ZDELTA | 2 | 2 | 2 | 2 |
| w/o RATIO | 5 | 3 | 3 | 3 |

Source categories are HUP36 patients, LZU21, Multicenter15, Pediatric8. All35
fold rows and28 center rows, with all11 metrics, are in the corresponding CSVs.
Patient-count-weighted fold and center aggregates exactly reproduce each
overall equal-patient estimate within1e-12; unweighted fold means are not used.

Notable heterogeneity:

- Patient Z helps Macro-F1 in every fold/center, but Pediatric AUROC is lower
  with Patient Z by0.036155. Do not claim all centers' ranking improves.
- Removing Source improves Macro-F1 in folds3/4/5 and in LZU, Multicenter and
  Pediatric; removing Source improves AUROC in four folds and three centers.
  Pediatric AUROC improves0.057985 on removal. These are descriptive subgroup
  findings, especially with only8 Pediatric patients.
- Full versus ABS Only improves AUROC at every center, but Pediatric Macro-F1
  is worse by0.047435 and folds3/4 also favor ABS Only for Macro-F1.
- RATIO removal lowers Macro-F1 in all five folds, but improves Pediatric
  Macro-F1 by0.007554. DELTA removal favors the ablation in fold4; ZDELTA effects
  are mixed across both folds and centers.

## Answers to the eight research questions

1. **Patient Z:** removal reduces AUROC0.070228, AUPRC0.053470, Macro-F1
   0.126425 and EZ-F10.151064. All four paired intervals exclude zero in favor
   of Full D2. This is the strongest observed component effect in this study.
2. **Source conditioning:** no supported positive increment on the four
   requested primary metrics. Source-free point estimates are slightly better
   on all four, but all four intervals cross zero. This is neither evidence
   that the source mechanism significantly helps nor proof it significantly
   hurts. Do not label the source-free comparison as R0.
3. **Pre-ictal-reference views beyond ABS:** support is clearest for ranking:
   Full-minus-ABS AUROC+0.029370 and AUPRC+0.051017 have positive unadjusted
   intervals. Macro-F1 and EZ-F1 point gains have intervals spanning zero.
   Reference features cannot be said to conclusively improve every metric.
4. **Largest individual view effect:** DELTA has the largest observed
   Macro-F1(+0.018391) and AUROC(+0.020008) deletion effect among DELTA/ZDELTA/
   RATIO. RATIO has the largest EZ-F1 point effect(+0.020253). DELTA and RATIO
   have positive unadjusted Macro-F1 intervals; no individual-view AUROC/AP/
   EZ-F1 effect has an interval wholly above zero. These correlated deletion
   effects do not uniquely identify causal biological information.
5. **Consistency:** no blanket across-fold/across-center benefit for the full
   structure. Patient Z's Macro-F1 effect is consistent; feature and source
   effects are heterogeneous as shown above.
6. **Unexpected improvements:** w/o Source increases AUROC/AP/Macro-F1/EZ-F1
   point estimates. w/o ZDELTA increases AUROC0.003549, and accuracy increases
   from0.759548 to0.760591. Neither observation licenses further TEST-driven
   architecture selection, and neither proves a significant improvement.
7. **Retaining the full structure:** these results support retaining Patient Z
   and the combined reference views for ranking; they do **not** establish
   necessity of source conditioning or a detectable ZDELTA increment. A claim
   that every original D2 component is validated would overstate the evidence.
   No model is redesigned or retrained again after these findings.
8. **Duplicates:**16 auxiliary dimensions exactly duplicate matching DELTA/
   ZDELTA primary dimensions, numerically verified over all7635 rows and by
   original synthetic generation. Both copies are masked together. The
   separate16 constant-zero delta/theta dimensions are recorded, not removed
   from Full D2. ABS Only has no accessible relative-view copy.

## Integrity and engineering record

- Source/feature audit completed before training. All88 names map to one view;
  group counts18/26/26/18; generation source hashes match historical originals.
- **56 pre-training checks passed**, including original D2 FIT/VAL replay,
  9221/8817 parameter identities, original split exclusions, source-free source
  invariance, shared initialization parity, actual masks, columnwise retained
  transform equivalence, FIT-only preprocessing, label orientation, valid
  gradients and exact CUDA model/Adam/history/RNG resume for three constructors.
- **25 new scientific cells, 250 completed epochs** with original stopping;
  no scientific-training retry or error. The train/freeze/evaluation pipeline
  took31.44 seconds, excluding implementation, audits, independent verification
  and transfer. No unrelated GPU job was altered.
- All35 checkpoint/preprocessor/mask/threshold selections frozen before new
  outer inference; all35 prediction files sealed before new full-cohort label
  materialization. Ten already completed cells reused without changing scores.
- Independent verification checked **560 model-patient cells ×11 metrics**,
  all35 checkpoint/bank/prediction bindings and all66 paired effects. Maximum
  absolute metric discrepancy **4.440892098500626e-16**. Frozen references
  reproduce their previous aggregates within1e-12.
- Fold/center aggregates independently checked to reproduce each overall
  metric under patient weighting. Private historical data, checkpoints and
  prediction hashes remain unchanged.

The original scientific runtime remains PyTorch2.11.0+cu128, NumPy2.4.3,
sklearn1.8.0. Independent verification encountered native Windows access
violations at sklearn's NumPy-backed cached-label unique operation. No
scientific rerun or result change followed. The **identical verification source**
was executed successfully in the already installed analysis runtime with
PyTorch2.11.0+cu128, NumPy2.3.5, sklearn1.7.2, after a hand-calculated tied-score
classification/ranking fixture passed. The alternative runtime is used only
for reading sealed predictions and independently checking arithmetic; it
does not train or run any model. Failure logs stay private.

## Interpretation boundaries and delivery

This is **single-seed exploratory repeated-outer evaluation**, not external
or blind validation. Patient bootstrap does not capture training-seed or
split uncertainty, and intervals are not adjusted for the multiple ablations.
All seven configurations were prescribed before execution; no additional arm,
threshold or feature search was performed. Acquiring-source stratification in
patient-held-out folds does not demonstrate unseen-center generalization.

Public delivery contains code, exact feature mappings, aggregate training
histories, seven-arm metrics, fold/center results, paired bootstrap and audit
reports only. No patient identifiers, raw EEG, scientific feature archive,
checkpoint, private prediction or runtime log is uploaded. **No installed
skill was used.**
