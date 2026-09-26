# Frozen A1 output-coordinate study: validation-only stop

**Terminal: `OUTPUT_COORDINATE_RANK_INVARIANCE_FAILED`.** No new model was trained. No current outer-test prediction, label, metric, or result was read.

The source A1 was reproduced exactly: five-fold VLOO patient Macro-F1 was `0.6259962097`, and all five fold values matched the published values with zero observed error. All 150 frozen A1 validation checkpoints were replayed on the 13 canonical validation patients per fold.

For C1 and C2, raw-logit versus transformed-logit Spearman was at least `0.9999999999999998` over 1,950 patient×epoch cases. Nevertheless, the pre-registered audit requires the actual ranking metrics to agree within `1e-8`. Both transformations exceeded that tolerance: maximum absolute EZ-AUPRC difference `0.0138888889` and EZ-AUROC difference `0.0004901961`; EZ-MRR and Top1-is-EZ were unchanged. Three cases per transformation had a ranking-metric difference. In the frozen A1 scores, five cases had distinct logits that became tied after float32 probability conversion. This is consistent with the metric discrepancy, but it does not satisfy the locked audit.

Consequently, the study stopped before interpreting C1/C2 VLOO comparisons. The protocol's questions about offset or scale improvement, 4/5 positive folds, EZ-F1, threshold stability, label-using oracle dispersion, the `0.640` VLOO gate, and the `0.665`/`0.620` apparent-full-validation gate remain **unanswered** under this version. There is no selected normalized coordinate, no frozen output-coordinate manifest, and no justification for outer testing from this experiment.

This failure is narrow: it concerns the strict equality of ranking metrics computed from finite-precision probabilities. It does not show that C1/C2 improve or harm hard decisions, nor that patient-relative modeling is ineffective. Revising the metric definition or tolerance would require a separately declared protocol amendment before any continuation.
