# Implementation audit

- Source: original frozen A1 epoch checkpoints, 5 folds × 30 epochs. No optimizer or training loop is invoked.
- Data: canonical 13 validation patients per fold, using valid channels only. Test loaders are not iterated.
- Stage 0 recomputes A1 raw-logit probabilities and the original per-epoch metric grids, then checks all five published VLOO fold values and their mean.
- Frozen A1 is configured with `positive_label=nez`; Stage 0 checks `sigmoid(logit)` against A1's NEZ score. A 5e-7 floating-point tolerance accommodates CPU/GPU sigmoid rounding; the complete stored validation metric grid must also match within 1e-8.
- Stage 1 applies exactly C0/C1/C2 from the protocol lock. C1/C2 use label-free per-patient logit statistics; neither learns a parameter.
- Ranking is audited for every patient, epoch, and fold before interpreting transformed VLOO estimates. Patient-oracle thresholds use labels for diagnosis only, never selection or gating.
- Public outputs contain only fold and cross-fold aggregates. Private patient-level values remain outside Git.
- A passed development gate freezes the selected coordinate/checkpoint/thresholds; it does not run outer evaluation.
- Source replay passed exactly: all five fold VLOO Macro-F1 values and their mean matched the published A1 values with zero observed error. The source probability-versus-logit tolerance was 5e-7; the per-epoch source metric grids matched within 1e-8.
- The strict rank-metric audit failed. Logit Spearman remained approximately 1, but 3/1950 cases changed ranking metrics beyond 1e-8. The original float32 EZ probabilities had ties between distinct logits in 5/1950 cases; this is consistent with quantization affecting AUROC/AUPRC, but the failed audit is not waived.
- As specified in the protocol lock, C1/C2 candidate selection, full-validation gate interpretation, and outer testing were stopped. No patient-level values are uploaded.
