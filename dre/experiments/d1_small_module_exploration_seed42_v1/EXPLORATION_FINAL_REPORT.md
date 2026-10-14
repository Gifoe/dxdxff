# Frozen D1 small-module exploration — completed

This is the completed Development report. The subsequent user-authorized fixed Outer is documented separately in [OUTER_REPORT.md](OUTER_REPORT.md): Macro-F1 gain0.017541, below the requested0.02, paired95% CI crosses zero. No-Outer statements below describe this original Development phase, not the later addition. Development numbers and the selected model are unchanged.

**Selected recipe: M8_MACRO_AWARE. Primary Development target met: Macro-F1 +absolute0.02224836 (2.224836 percentage points).**

| Metric | Frozen D1 | M8 | Absolute difference |
|---|---:|---:|---:|
| Patient AUROC |0.725978|0.756633|+0.030655|
| Patient AP |0.503601|0.542226|+0.038625|
| Patient Macro-F1 |0.644546|0.666795|+0.022248|
| Patient EZ-F1 |0.443537|0.477249|+0.033713|
| NEZ-F1 |0.845556|0.856340|+0.010784|
| Balanced accuracy |0.686159|0.710921|+0.024762|
| EZ sensitivity |0.506245|0.544976|+0.038731|
| NEZ specificity |0.866074|0.876867|+0.010793|
| MRR |0.658502|0.699708|+0.041206|
| Top1-EZ |0.523077|0.569231|+0.046154|
| Accuracy |0.775170|0.789846|+0.014677|
| Predicted EZ fraction |0.194371|0.196177|+0.001806|

Primary estimand remains the exact original65 validation patient-fold appearances/47 unique IDs, not pooled-channel scores or an independent80-patient Outer. Four of five folds improve Macro-F1; fold2 declines0.002643. All eight tested candidates are retained in `phase3/results/ALL_EXPLORATION_CANDIDATES.csv`, including weak/failing variants. These are adaptive, repeatedly used, validation-selected Development results. No new Outer was accessed. This establishes the requested exploratory point gain, **not** independently validated clinical generalization.

## What to add to D1

Keep the original1,537-parameter D1, its R14/PatientZ/FIT imputer/scaler and checkpoints frozen. Add a9,481-parameter zero-output-initialized Raw-DELTA correction:

1. Existing independent-window Raw CNN (2,160 parameters), exact250Hz/500-sample genuine input and historical per-window normalization.
2. Same-channel genuine pre-reference latent subtraction; normalized7D physiology drives an identity-initialized multiplicative gate (1,048 parameters).
3. Genuine early/late means, available-seizure mean/populationSD and patient-relative96D normalization; no GRU, attention, graph or larger backbone.
4.96→64→1 correction head (6,273 parameters), **last projection weight/bias zero**. Final NEZ logit = frozenD1 + tanh(correction). No slow zero scalar gate and no competing auxiliary standalone Raw classifier.
5. Train with fixed **0.75 original FIT-weighted BCE(final) +0.25 patient SoftMacroF1(final)**. FIT-only class weights and one update/FITpatient remain unchanged; checkpoint/threshold selection still uses the original validation policy.

Total11,018 parameters:1,537 frozen +9,481 trainable. At initialization and when correction is disabled, final logits exactly equal D1. The physiological scale is FIT-only median-of-patient-median absolute DELTA, floor1e-30/clip8; the previous1e-6 native-unit suppression is not reused. Original D1 features/scalers are not changed. No new Raw/EDF acquisition or physiological feature re-extraction.

## Why this choice, and what is not established

| Candidate | New trainable parameters | Macro-F1 | Gain vs D1 |
|---|---:|---:|---:|
| M1 corrected-scale DGM |9482|0.651390|+0.006843|
| M2 deterministic stage DELTA |722|0.645883|+0.001337|
| M3 nonlinear DELTA set encoder |1354|0.646847|+0.002300|
| M4 R14 direct correction |257|0.644774|+0.000227|
| M5 stage DELTA direct correction |721|0.653227|+0.008681|
| M6 morphology direct correction |9481|0.661629|+0.017083|
| M7 additional2-parameter margin gate |9483|0.661878|+0.017332|
| **M8 morphology +Macro-aware objective** |9481|**0.666795**|**+0.022248**|

The257-parameter head on existing R14 adds almost nothing; deterministic/learned physiology aggregation alone adds less than the Raw correction. More gating is not justified by the current primary result: M7 adds only0.000249 Macro-F1 over M6 and worsens EZ-F10.007119. M8 meets the primary target without another learned module on top of M6. M6/M8 architecture and all five initial states are bit-identical; their declared training objective is the only experimental change. M8−M6 Macro-F1 is+0.005166 with95% paired interval[-0.004782,+0.014359], and AP drops0.004074 relative to M6. **The SoftMacroF1 increment alone is not statistically established or uniformly superior across metrics.** Selection favors M8 because the user specified Macro-F1, not because every candidate contrast favors it.

M6 versus M1 changes residual parameterization/cap and auxiliary objective together; it is not proof which individual change caused the gain. The window-physiology guidance mechanism still lacks a freshly trained matched no-physiology/misalignment matrix in this exploration; dependence under frozen interventions is not causal proof. No universal method novelty or better-than-most-existing-solutions claim is made.

## Paired patient bootstrap

10,000 seed42 draws resample47 unique IDs while retaining repeated appearances, frozen selected thresholds/checkpoints. M8−D1 intervals:

| Metric | Difference | Selection-conditioned95% interval |
|---|---:|---|
| Macro-F1 |+0.022248|[+0.003509,+0.040966]|
| AUROC |+0.030655|[+0.003539,+0.062108]|
| AP |+0.038625|[+0.015870,+0.062295]|
| EZ-F1 |+0.033713|[+0.002591,+0.066136]|

These intervals do not remove adaptive eight-candidate/epoch/threshold selection or extensive prior validation reuse. MRR/Top1 intervals include zero. Do not call this blind confirmation, a guarantee of two points on unseen patients or a new Outer result.

## Delivery and actual completion verification

Forty candidate-fold cells/503 saved epochs actually completed across three separately locked phases. Original D1 checkpoint/bank hashes are unchanged. All BEST/LAST/COMPLETE hashes, source/protocol hashes and public output hashes independently checked. Real FIT gradient admissions and server tests passed before each phase; smoke states discarded. The reusable `tools/best_recipe.py` factory loaded the selected M8 checkpoints strictly and freshly replayed all five Development folds: **logit drift0, decisions exact, aggregate metrics and declared loss exact**. See `final_audit/GOAL_COMPLETION_AUDIT.json`, `BEST_RECIPE_REPLAY.json`, and `MATCHED_MECHANISM_BOOTSTRAP.csv`.

For reproducible execution, use `phase3/README.md`; for a new independent exploration, copy to a newly named protocol/runtime and lock changes prospectively, retaining old outputs. `tools/best_recipe.py` exposes `build(frozen_d1)` and `training_loss(final_nez_logit,nez_labels,fit_pos_weight)`. Original and new clinical data, identities, masks, scalers, weights, per-channel predictions and logs stay exclusively on trusted serverD. GitHub contains only code/tests/hashes/anonymous aggregates/reports. No installed skill, unrelated-GPU interruption or new Outer evaluation.
