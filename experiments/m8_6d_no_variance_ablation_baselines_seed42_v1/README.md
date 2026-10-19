# F0 without Variance: complete ablations and paired classical classifiers

Execution protocol, not asserted new results. New main model is the exact frozen prior DROP_VARIANCE selected by the user after historical Outer viewing. All results will therefore remain exploratory repeated-test, not independent confirmation.

Six measurements: beta, low gamma, partial high gamma80–125Hz, RMS, line length and spectral entropy10–125Hz. Neural model keeps original7/14 width with Variance permanently masked in window conditioners and direct summary. Classical models actually receive the remaining12summary coordinates.

The sixteen neural groups comprise Full6D, noEarly, noLate, noBothGuidance, **noBothPR**, noRawPreReference, noHeadInteraction, support-matched earlyFiLM mismatch, Raw-only, physiology-only, and six leave-one-feature-out models. NoBothPR simultaneously disables summary patient Z and final Raw96 patient Z, not waveform normalization.

Seven original classifiers (LR, linear SVM, RBF SVM, RF, XGBoost, MLP, AdaBoost) each train without/with patient PR. Parent legal-validation-selected hyperparameters are frozen per model/fold and shared across PR arms; there is no new search. Classifiers have FIT-only preprocessing, patient/class weighting, and patient-OOF FIT-only SVM calibration; own legal internal validation chooses numeric thresholds.

Target:70new neural fits +10verified reference reuses +70new classical models =150selected model/fold cells,30groups,2400patient rows and21600nine-metric values. Report all30groups regardless of internal performance. All150selections freeze before any newOuter; all150scorecells seal before report. Same80patients/7635channels/256seizures/1471965validRawwindows/seed42/originalfivefolds, no extraction or Raw copying.

Nine endpoints: patient Macro-F1, AUROC, AP, EZ-F1, NEZ-F1, ACC, MRR, Top-1, EZ-fraction bias. Separate pooled, five-fold/sampleSD, sources and coverage;10000seed42paired bootstrap,603nominal interval rows and37additional approximate Macro-F1 family intervals.

Scientific sources/protocol will be admitted then pushed before formal fits. Atomic complete-epoch model/optimizer/RNG checkpoints support diagnosed engineering resumes, not blind retries. All new server artifacts live on D; private signals, identities, predictions, weights and logs never leave the server. Publish only code/hashes and anonymous aggregates.

