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

## Frozen V2 rank-audit amendment

- The separate amendment was committed and pushed before V2 analysis. The original protocol lock SHA-256 and all scientific rules remain unchanged.
- Source A1 was reconfirmed from all 150 frozen checkpoints with exact five-fold VLOO reproduction.
- V2 used `-float64(NEZ_logit)` for C0 and the corresponding centered/robust-z EZ scores for C1/C2. It audited every valid-channel pair, stable-sort ordering, Spearman, and the four canonical ranking metrics over 1,950 patient×epoch cases per transform. No sigmoid was used in this audit.
- V2 passed with zero pairwise violations and zero canonical metric deltas. The five float32 source-probability tie cases are diagnostic only.
- Only after V2 passed did the unchanged VLOO hard-decision pipeline run. The development gate failed; no outer test was accessed and no output-coordinate manifest was frozen.
