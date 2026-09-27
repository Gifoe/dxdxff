# Implementation audit

- New branch starts from the exact A1 source tip. Source replay independently checked all 150 validation checkpoints and the five locked Macro-F1 values.
- All new variants used the same frozen five FIT/validation folds, 30 epochs, source optimizer and per-epoch shuffle RNG. No early stopping, no S-RANK selector and no outer-test loader.
- B00 and A0 reuse exact A1 checkpoint grids. B10 is patient class-balanced BCE; B01/B11 differ only in the fixed hard-negative loss. CAP0 uses the prelocked Audit-A gate to choose the BCE base.
- Common CAP parameters were restored from the same A1 fold initialization. Only the feature MLP was replaced; downstream dimension remains 32.
- S-F1 selection excluded each validation patient from checkpoint and threshold choice. Patient-level records, selected checkpoints, optimizer state and logs remain private on the server.
- Label-using imbalance, prevalence quartile, margin and top-K diagnostics were not used for training configuration or selection. FP, if triggered, was diagnostic only.
- Existing source construction indexes cohort metadata, but this audit never builds an outer-test loader or reads outer labels, predictions or performance.
