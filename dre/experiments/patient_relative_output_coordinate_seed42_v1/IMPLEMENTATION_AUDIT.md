# Implementation audit

- Source: original frozen A1 epoch checkpoints, 5 folds × 30 epochs. No optimizer or training loop is invoked.
- Data: canonical 13 validation patients per fold, using valid channels only. Test loaders are not iterated.
- Stage 0 recomputes A1 raw-logit probabilities and the original per-epoch metric grids, then checks all five published VLOO fold values and their mean.
- Frozen A1 emits EZ-positive logits; Stage 0 negates these to obtain the specified NEZ-positive logits and checks `sigmoid(-EZ_logit)` against A1's NEZ score.
- Stage 1 applies exactly C0/C1/C2 from the protocol lock. C1/C2 use label-free per-patient logit statistics; neither learns a parameter.
- Ranking is audited for every patient, epoch, and fold before interpreting transformed VLOO estimates. Patient-oracle thresholds use labels for diagnosis only, never selection or gating.
- Public outputs contain only fold and cross-fold aggregates. Private patient-level values remain outside Git.
- A passed development gate freezes the selected coordinate/checkpoint/thresholds; it does not run outer evaluation.
