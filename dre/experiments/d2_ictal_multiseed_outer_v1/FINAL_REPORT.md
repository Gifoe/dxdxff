# Original ictal D2: five-seed outer results

## Primary results

The original **D2 with `patient_z` retained**, evaluated on the same fixed
five-fold outer cohort, gives the following equal-patient results. Each seed has
80 unique TEST patients and 7635 patient-channel units. The reported SD is the
**sample SD across seed-level estimates (ddof=1)**, not fold SD or a confidence
interval.

| Metric | Five-seed mean | Seed SD | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: |
| AUROC | 0.730057 | 0.002452 | 0.727171 | 0.733887 |
| AP / AUPRC | 0.536421 | 0.006108 | 0.526999 | 0.541313 |
| Macro-F1 | 0.615377 | 0.005363 | 0.608709 | 0.620403 |
| EZ-F1 | 0.408162 | 0.012329 | 0.391187 | 0.423585 |
| NEZ-F1 | 0.822593 | 0.008001 | 0.812550 | 0.829898 |
| Balanced accuracy | 0.652443 | 0.003748 | 0.647402 | 0.656108 |
| MRR | 0.752013 | 0.014675 | 0.737216 | 0.770589 |
| Top1 EZ | 0.672500 | 0.025617 | 0.650000 | 0.700000 |
| Sensitivity | 0.451245 | 0.022386 | 0.424839 | 0.483012 |
| Specificity | 0.853642 | 0.016854 | 0.829204 | 0.869965 |
| Accuracy | 0.752595 | 0.006661 | 0.744801 | 0.759548 |

The predeclared seeds were **0, 1, 2, 3, 42**. Original seed42 checkpoints and
already frozen outer predictions were reused unchanged; the other four seeds
were fully trained from scratch on all five folds. No seed was discarded or
selected on TEST performance. There are **20 new training cells plus five
reused seed42 cells**, not 25 newly trained cells.

## Individual seeds

| Seed | AUROC | AP | Macro-F1 | EZ-F1 | MRR | Top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.729470 | 0.526999 | 0.608709 | 0.401494 | 0.764053 | 0.700000 |
| 1 | 0.733887 | 0.540231 | 0.610521 | 0.391187 | 0.770589 | 0.700000 |
| 2 | 0.727171 | 0.540079 | 0.618068 | 0.423585 | 0.740408 | 0.650000 |
| 3 | 0.729314 | 0.541313 | 0.619186 | 0.413635 | 0.747797 | 0.662500 |
| 42, reused | 0.730445 | 0.533483 | 0.620403 | 0.410909 | 0.737216 | 0.650000 |

All 11 individual-seed metrics are in `results/OUTER_BY_SEED.csv`; all 25
seed/fold and 20 seed/center aggregate rows are provided separately.

## What changed, and what did not

Only the training seed changed. Keep the original fold rule
`fold_seed = seed + 1009 * fold`, and keep the model's original private CPU
generator rule `42 + 7000 + fold_seed` for its rank4 factor initialization.
Consequently initialization, dropout and shuffled patient update order vary,
while all data memberships and hyperparameters are fixed.

- Exact historical `SourceA0`: 88-dimensional features, LayerNorm, 96-unit GELU
  MLP, dropout 0.15, one NEZ-logit classifier, rank4 source correction;
  **9221 parameters**. No model approximation or additional configuration.
- Original physiological ABS/DELTA/ZDELTA/RATIO feature views and `patient_z`
  retained. Original FIT-only imputer/global scaler and original FIT/validation
  transformed banks reused bitwise. No new raw EEG extraction.
- Original CUDA/PyTorch runtime; original trainer imported without modification.
  AdamW LR0.001, weight decay0.0001, FIT-only NEZ-positive BCE class weight,
  one optimizer update per FIT patient, source penalty0.001, gradient clip1.
  Maximum30 epochs, minimum6, patience6, unchanged early stopping.
- Exact original fixed five-fold FIT/validation/TEST split; TEST patients by
  fold: 16, 16, 17, 15, 16. Do not average these unequal-fold means to construct
  the headline metric.
- Each seed/fold chooses its own threshold on validation using the original
  0.005 grid and original tie rules. Checkpoint selection remains validation
  Macro-F1, then EZ-F1, then earlier epoch. No shared TEST-selected threshold.
- EZ=0, NEZ=1; output is sigmoid(NEZ logit). EZ prediction iff score is strictly
  below that fold's frozen threshold. Ranking uses ascending NEZ probability
  with the original stable canonical channel-string tie order.

## Freeze and verification

1. Pin and verify original code, feature archive, split, banks, seed42 model
   files, seed42 outer freeze and prediction seal hashes.
2. Independently reconstruct original FIT/validation preprocessing and require
   bitwise parity with all five stored banks. Establish five distinct initial
   model hashes per fold, including exact original seed42 initialization parity.
3. Complete all **20 new training/validation selections**. Freeze all 25
   checkpoint/bank/threshold bindings before any new outer inference.
4. Infer the 20 new outer cells using the same CPU evaluation backend as the
   original seed42 outer experiment; reuse the five sealed seed42 cells. Seal
   all 25 prediction artifacts before loading full-cohort labels for metrics.
5. Independently verify 400 seed-patient cells × 11 metrics with sklearn and
   independent stable-ranking code. Maximum absolute metric discrepancy:
   **3.3306690738754696e-16**. Verify all 25 checkpoint bindings and independently
   recompute means, sample SD, extrema and bootstrap intervals. No inference
   rerun, prediction modification or threshold reoptimization in verification.
6. Require seed42 outer metrics to reproduce the previous frozen result within
   1e-12 and all historical dependency hashes to remain unchanged.

Two focused synthetic code-contract tests passed. The four new seeds completed
**259 training epochs**, with original early stopping and no error/retry. The
server's new-training/freeze/evaluation pipeline took **34.64 seconds** after
the preparation gate; this excludes code preparation, independent verification
and artifact transfer. No unrelated server process was interrupted.

## Patient sampling uncertainty, distinct from seed SD

Average each patient's metrics over the five seeds, then resample the **80
unique patients** with replacement, 10,000 draws, seed42. Frozen thresholds
remain fixed. The percentile95 intervals for these seed-averaged metrics are:

| Metric | Patient-cluster 95% CI |
| --- | --- |
| AUROC | [0.687542, 0.771264] |
| AP | [0.482202, 0.589545] |
| Macro-F1 | [0.590302, 0.641888] |
| EZ-F1 | [0.364887, 0.451829] |

Intervals for all 11 metrics are in `results/MULTISEED_SUMMARY.csv`. These
intervals quantify sampling uncertainty of seed-averaged patient metrics;
seed SD quantifies variability across training seeds on a fixed split. Neither
is a CI over new split assignments. The 400 seed-patient observations are
**not 400 independent patients**.

## Interpretation

The original seed42 AUROC0.730445 is close to the five-seed mean0.730057.
Ranking is relatively stable under the tested training seeds (AUROC SD0.002452);
changing the seed does not reveal a substantially better discrimination result.
Seed42 has the highest Macro-F1 among these five, but the correct summary is
**0.615377 ± 0.005363**, not that maximum. EZ-F1 remains approximately0.408,
with mean sensitivity0.451; localization accuracy has not been solved by
repeating seeds.

This is **exploratory repeated-outer evaluation** on the previously viewed
cohort, not blind confirmation or unseen-center generalization. All seeds and
rules were fixed before these new runs, with no test-driven selection. The
historical D2 development continuation gate is not retroactively declared
passed. This experiment does not create a multi-seed no-PR arm or establish a
paired multi-seed ablation effect; the earlier no-`patient_z` experiment remains
a separate seed42 comparison. No automatic extra seeds or model tuning follow.

## Artifact policy

This directory contains code and compact aggregate audits, seed/fold/center
tables, validation training histories and summary statistics. Raw EEG, feature
archives, patient/channel records, private predictions, checkpoints and runtime
logs remain on the original server. They are not uploaded to GitHub.
