# F0 without Variance: complete ablations and paired classical classifiers

**Completed and independently verified on 2026-10-19.** All30groups completed the original five-fold Outer:70new neural fits,70new classical models and10exact reference reuses,150model/fold cells,2400patient rows and21600checked metric values.603nominal intervals and37additional Macro-F1 family intervals passed independent verification.

The new main model is the exact frozen prior DROP_VARIANCE selected by the user after historical Outer viewing. Full patient-average Macro-F1 is **0.650225**, AUROC **0.755579**, AP **0.589825**, EZ-F1 **0.471029**, NEZ-F1 **0.829421**, ACC **0.761246**, MRR **0.829984**, Top-1 **0.775000**, and EZ-fraction bias **−0.013162**. This main reference is exact reuse, not a second independent confirmation.

Joint removal of physiological and Raw PR gives Macro-F1 **0.516423**, not the old single-Raw-PR ablation. All seven classical classifiers improve Macro-F1 with PR by0.075447–0.164026; all seven paired Macro-F1 intervals remain positive under the predefined approximate37-comparison adjustment. Other metrics need not all improve.

## Delivered results

- [All30groups, all9metrics, paired intervals and figures](results/REPORT.md)
- [中文结果与解释边界](FINDINGS.md)
- [Unrounded primary table](results/MAIN_RESULTS.csv), [pooled](results/POOLED_RESULTS.csv), [folds](results/FOLD_RESULTS.csv), [fold SD](results/FOLD_SD.csv), [sources](results/SOURCE_RESULTS.csv)
- [603nominal intervals](results/INTERVALS.csv), [37Macro-F1 family intervals](results/MACRO_FAMILY_INTERVALS.csv), [classifier PR intervals](results/PR_PAIRED_INTERVALS.csv)
- [Execution/recovery and compatibility audit](RUN_AUDIT.md), [actual visible figure QA](FIGURE_QA.md)

## Fixed design

Six measurements: beta, low gamma, partial high gamma80–125Hz, RMS, line length and spectral entropy10–125Hz. Neural model keeps original7/14 width with Variance permanently masked in window conditioners and direct summary. Classical models actually receive the remaining12summary coordinates.

The sixteen neural groups comprise Full6D, noEarly, noLate, noBothGuidance, **noBothPR**, noRawPreReference, noHeadInteraction, support-matched earlyFiLM mismatch, Raw-only, physiology-only, and six leave-one-feature-out models. NoBothPR simultaneously disables summary patient Z and final Raw96 patient Z, not waveform normalization.

Seven original classifiers (LR, linear SVM, RBF SVM, RF, XGBoost, MLP, AdaBoost) each train without/with patient PR. Parent legal-validation-selected hyperparameters are frozen per model/fold and shared across PR arms; there is no new search. Classifiers have FIT-only preprocessing, patient/class weighting, and patient-OOF FIT-only SVM calibration; own legal internal validation chooses numeric thresholds.

Target:70new neural fits +10verified reference reuses +70new classical models =150selected model/fold cells,30groups,2400patient rows and21600nine-metric values. Report all30groups regardless of internal performance. All150selections freeze before any newOuter; all150scorecells seal before report. Same80patients/7635channels/256seizures/1471965validRawwindows/seed42/originalfivefolds, no extraction or Raw copying.

Nine endpoints: patient Macro-F1, AUROC, AP, EZ-F1, NEZ-F1, ACC, MRR, Top-1, EZ-fraction bias. Separate pooled, five-fold/sampleSD, sources and coverage;10000seed42paired bootstrap,603nominal interval rows and37additional approximate Macro-F1 family intervals.

Scientific sources/protocol were admitted and pushed before formal fits. Atomic complete-epoch model/optimizer/RNG checkpoints support diagnosed engineering resumes, not blind retries. All new server artifacts live on D; private signals, identities, predictions, weights and logs never leave the server. Only code/hashes and anonymous aggregates are published.

## Actual execution and reproducibility

The14scientific source/protocol files were admitted and pushed before formal fits at **9c84578a83cbae402f28e308de05602931111019** and remain byte unchanged. Source data and all original cohort roles stayed fixed. The70new neural fits completed923unique complete epochs.

Use **run_verified.ps1** as the actual production launcher, not the preserved initial run_all.ps1 snapshot. It invokes the immutable training/evaluation code with two explicit, hash-audited execution adapters: RF probability replay tolerates only machine-level parallel-summation roundoff while all201threshold decisions, selected thresholds and patient rankings remain exact; the Outer adapter resolves one mistaken historical seal basename to its already hash-locked original file. Neither adapter changes training, model weights, preprocessing, prediction math or thresholds. The original failed entries/logs remain preserved.

A native CUDA-library access violation was diagnosed and resumed from the exact complete epoch8checkpoint, preserving43completed fresh cells. It is not described as a failure-free run or a proven root-cause repair. All10MLPfits reached the unchanged400iteration budget with ConvergenceWarning; this is disclosed rather than silently increasing the budget.

Run `python experiments/m8_6d_no_variance_ablation_baselines_seed42_v1/verify_public.py` for public aggregate, report, source/adaptor and manifest checks. After pushing, `verify_git_release.py` checks remote-tip equality and exact committed Git blob hashes. Private-data execution additionally requires the original trusted server data and pinned parent experiments; no public raw dataset is included.

Everything remains **exploratory repeated-test evidence**. This fixed matrix does not choose a new post-Outer winner or prove all six features/modules indispensable. The experimental design preserves patient pairing and distinct aggregation units; the figure workflow binds every plotted value to verified source hashes.

