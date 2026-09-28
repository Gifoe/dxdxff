# Implementation audit

- Same `A1TFModel` class and parameter topology for I1/N1; synthetic alpha-zero replay gave zero logit error/mismatch; Ictal real FIT batch replay passed.
- Original A1 36-D descriptors and patient-relative downstream are imported from historical source; TF is a 32-bin physical-frequency residual.
- Ictal exact 80-person/5-fold source and 47-ID/65-cell/20-query comparison replayed the historical I0 AP to 1e-9. Stage2 gate used the other 12 validation patients per target, and 135 score-only epoch files were frozen before target evaluation.
- Omni exact official split: 141 train, 96 test patients, 8,104 EDF-channel test records. Checkpoint and threshold SHA were frozen before N1 test extraction.
- Ictal per-epoch **weights** were not retained, only per-epoch score snapshots; its alpha-zero intervention is limited to final stage endpoints, not selected I1.
- Historical I0/N0 outcomes were already seen before A1-TF design. This is exploratory and does not license a blinded generalization claim.
