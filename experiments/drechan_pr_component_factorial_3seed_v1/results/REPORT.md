# Six-feature DREChannNet PR component factorial: actual internal results

Exploratory repeated-test. These are the33-unique-patient, three-seed within-original-FIT ASSESS results, NOT a new80-patient Outer. The original main model Outer Macro-F1 remains0.653828; no direct performance difference against that point is estimated here.

## Execution and controls

Three seeds42/43/44 and all original five fold paths;15 exact internal B0 cells verified/reused and45 new full-patient fits. All four arms have11425parameters, the same original budget/loss/initialization and a64-hidden single head. EarlyFiLM, lateGate, six features, pre-reference and complete window/seizure pooling are unchanged.

Feature PR off omits only patient Z, refitting the original imputation/scaling recipe on TRAIN only. Raw PR off omits only final96D patient Z. The two normalizations are independently varied; raw and physiological branch inputs are not artificially zeroed. Numeric-scale changes are part of this original deletion, so these contrasts alone do not isolate physiological information from scale/optimization.

All60 selections replayed/frozen before new assessment scores; all60 scores sealed before metrics. Prior B0 scores reused only after exact index/checkpoint/preprocessor/threshold checks. No new Outer, candidate threshold search or test-driven model edits.

| Arm | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac.bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL | 0.619955 | 0.752344 | 0.572237 | 0.408406 | 0.831505 | 0.752759 | 0.820934 | 0.729798 | -0.107229 |
| WO_FEATURE_PR | 0.601894 | 0.737354 | 0.560529 | 0.373776 | 0.830013 | 0.747472 | 0.814444 | 0.717172 | -0.120184 |
| WO_RAW_PR | 0.599953 | 0.746658 | 0.546728 | 0.420476 | 0.779429 | 0.726282 | 0.798352 | 0.696970 | -0.030975 |
| WO_BOTH_PR | 0.488635 | 0.733040 | 0.557519 | 0.242726 | 0.734543 | 0.690526 | 0.804002 | 0.719697 | -0.069073 |

## Separate PR evidence

| Contrast | Seeds with AUROC and Macro-F1 both positive | Stable positive points | Both nominal lower bounds positive | Both4-comparison lower bounds positive |
| --- | ---: | --- | --- | --- |
| FEATURE_PR_WITH_RAW | 3 | True | True | False |
| RAW_PR_WITH_FEATURE | 2 | True | False | False |

See all108 nominal intervals, four additional primary-family intervals and the difference-of-differences interaction in INTERVALS.csv. Zero-crossing intervals do not establish absence or superiority. In8 factorial contrasts, numerical-scale/calibration effects and nonlinear training can interact; an interaction is not proof of a physiological causal mechanism.

Primary estimates average repeated fold metrics within unique patients, then metrics across three seeds, then patients equally. Pooled appearances are separate. Source summaries with fewer than5patients are suppressed. Bootstrap draws10000unique patients, retaining all paths and seeds paired; intervals are conditional on historical selection and do not cover full training randomness. Reusing the historical internal assessment does not create independent confirmation.

## Active objective remains open

This diagnostic cannot prove that a new model has five-fold Macro-F1>0.64, improves AUROC, or has useful contributions from all retained modules. It identifies separate PR effects for subsequent predeclared optimization. No control is weakened or unfavorable result hidden to manufacture a PR advantage. The previous temporal-contrast failed-gate result remains unchanged.

Raw signals, identities, individual predictions, checkpoints, preprocessors, caches and runtime logs remain on the trusted server. Only source, hashes and anonymous aggregates are exported.
