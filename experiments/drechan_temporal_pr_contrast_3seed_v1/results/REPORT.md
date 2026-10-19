# Temporal patient-relative contrast: three-seed controlled experiment

All results are exploratory repeated-test on the historically viewed cohort. New within-FIT assessment roles prevent new epoch/threshold selection from using their own assessment labels; they do not erase historical architecture selection or create an independent cohort.

## Fixed design

B0 is the exact six-feature, no-Variance/no-explicit-head-interaction main model. B1/B2/B3 add the same 1776-parameter residual48-to24-to24 module; all share exact initial B0 outputs. Early FiLM, late Gate, losses, input scaling, original patient-held-out folds and budgets remain unchanged. Seeds42/43/44 are all reported.

B1 uses dimension-wise leave-one-channel-out median/MAD at the same valid seizure/time. B2 uses all current TRAIN windows at the same time, recomputed with current CNN weights before every optimizer update. B3 shifts only the reference to other post times with exactly the same full channel-support vector within the same seizure; singleton groups remain unchanged. All reference backgrounds are stop-gradient; recipient/fusion/pooling and final96D patient Z remain differentiable.

Internal roles are new TRAIN/CAL/ASSESS partitions entirely inside each original FIT. TRAIN alone determines preprocessing, scale and B2 reference; CAL alone selects epochs/thresholds. All60 selected cells freeze and replay before assessment; all60 score cells seal before endpoints. No original Development or ownOuter is used in this first stage.

## Predeclared continuation decision

Performance margin gate: **True**. Current-TRAIN global-reference comparison gate: **False**. Continue final stage: **False**.
The1pp rule applies simultaneously to AUROC, AP, Macro-F1 and EZ-F1 mean differences B1-B0, with at least2/3seeds individually within all4margins. This is an engineering point gate, not formal statistical noninferiority. B1 must also exceed B2 on both ranking metrics on average and in at least2/3seeds, without more than1pp mean Macro/EZ loss. B3 is a diagnostic, not a selector. No gate was relaxed after results.

The fixed internal gates failed. No new final fits or Outer evaluation were authorized by this protocol. B1 is not adopted as a replacement. A failure here does not prove all forms of patient-relative learning ineffective.

## INTERNAL equal-patient endpoints

| Arm | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac.bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current six-feature single head | 0.619955 | 0.752344 | 0.572237 | 0.408406 | 0.831505 | 0.752759 | 0.820934 | 0.729798 | -0.107229 |
| Temporal patient-relative contrast | 0.623988 | 0.749647 | 0.574308 | 0.420153 | 0.827823 | 0.748592 | 0.836805 | 0.752525 | -0.095092 |
| Current TRAIN-only global reference | 0.629194 | 0.753739 | 0.575941 | 0.425909 | 0.832478 | 0.755652 | 0.840969 | 0.762626 | -0.095002 |
| Wrong-time within-patient reference | 0.627099 | 0.755880 | 0.580478 | 0.421099 | 0.833099 | 0.755616 | 0.842737 | 0.760101 | -0.105441 |

Primary points average patient-level metrics across all3seeds, not predictions or best seeds. Repeated internal assessment-fold appearances are averaged within each unique patient before equal-patient aggregation.

| Contrast | Endpoint | Difference | Nominal95% interval | Additional12-comparison interval |
| --- | --- | ---: | --- | --- |
| B1_minus_B0 | macro_f1 | +0.004032 | [-0.009849, 0.017418] | [-0.016576, 0.023823] |
| B1_minus_B0 | auroc | -0.002697 | [-0.018781, 0.013520] | [-0.026271, 0.020462] |
| B1_minus_B0 | auprc | +0.002071 | [-0.021733, 0.027087] | [-0.034974, 0.040307] |
| B1_minus_B0 | f1 | +0.011747 | [-0.014186, 0.036855] | [-0.026658, 0.049365] |
| B1_minus_B2 | macro_f1 | -0.005206 | [-0.017126, 0.007887] | [-0.022478, 0.013919] |
| B1_minus_B2 | auroc | -0.004092 | [-0.016996, 0.011539] | [-0.022344, 0.019780] |
| B1_minus_B2 | auprc | -0.001633 | [-0.020673, 0.022637] | [-0.028328, 0.037696] |
| B1_minus_B2 | f1 | -0.005757 | [-0.027963, 0.019066] | [-0.037923, 0.031219] |
| B1_minus_B3 | macro_f1 | -0.003111 | [-0.014922, 0.008259] | [-0.020782, 0.013091] |
| B1_minus_B3 | auroc | -0.006233 | [-0.022286, 0.010157] | [-0.029951, 0.017446] |
| B1_minus_B3 | auprc | -0.006169 | [-0.028370, 0.019361] | [-0.038186, 0.031547] |
| B1_minus_B3 | f1 | -0.000946 | [-0.023308, 0.020913] | [-0.033896, 0.030815] |

Independently verified 480 patient-fold-seed rows and 4320 endpoint values,63nominal and12additional interval values. Unique patients: 33.
See [internal seed results](internal/SEED_RESULTS.csv), [seed SD](internal/SEED_SD.csv), [folds](internal/FOLD_RESULTS.csv), [fold SD](internal/FOLD_SD.csv), [sources](internal/SOURCE_RESULTS.csv), [coverage](internal/COVERAGE.csv), [pooled endpoints](internal/POOLED_RESULTS.csv).

## Interpretation boundaries

- Patient AUROC/AP can change because contextual comparison occurs before nonlinear representation learning. Patient-wise monotone normalization of final logits alone cannot change within-patient rankings; that is not the method tested here.
- B1 versus B0 measures incremental information plus additional capacity/optimization. B1 versus B2 matches added capacity/input width/scale recipe/compute updates; it contrasts local context with a global reference, not every possible non-PR architecture.
- Correct-time evidence requires B1 versus B3 support and adequate changed-reference coverage. A positive point difference or mostly unchanged singleton control cannot establish timing specificity.
- Both original summary-PR operations remain enabled in every arm. This experiment isolates additional temporal PR; it is not a claim that all PR is necessary or that deleting PR should break the network.
- Historical PatientZ AUROC+0.028655 and[0.010792,0.049047] were factorial mean/min/max over8 frozen-CNN probe backgrounds, NOT a patient-paired95% confidence interval. Past weak relation/affine/ranking results remain negative evidence, not proof that this design must fail or succeed.
-10000seed42unique-patient paired bootstrap draws use seed-averaged metrics; conditional intervals do not cover training randomness, overlapping FIT histories or prior researcher selection. Three seeds and source summaries do not establish external stability. Additional intervals use approximate Bonferroni percentile tails0.05/24; they are not exact simultaneous inference.
- No physiological propagation or clinical mechanism is inferred from latent comparisons. Partial high gamma is80-125Hz, entropy10-125Hz; unavailable low-frequency sources are not invented.
- No post-assessment/Outer architecture, threshold, cohort, gate, seed or loss tuning. No best-Outer arm is selected. No guarantee of improvement or global optimality.
- Signals, identifiers, per-patient/channel predictions, weights, preprocessors, raw caches, global-reference tensors and runtime logs stay on the trusted server. Public outputs contain only source, hashes and anonymous aggregates.
