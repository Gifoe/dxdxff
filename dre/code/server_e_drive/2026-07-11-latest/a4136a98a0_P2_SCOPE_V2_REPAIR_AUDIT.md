# P2-SCOPE-v2 Repair Audit

## Findings

1. `predicted_k_topk` was not recognised by the shared summary code, so formal masks could be replaced by a threshold during final aggregation.
2. The trainer contained a second decoder and both decoder implementations used an incorrect Beta-Binomial combination term.
3. The old validation ledger sorted globally instead of sampling within strata, and did not prove equivalence to runtime outer folds.
4. Fold state, DataLoader randomness, learning-rate schedule, checkpoint persistence, and resume hashes were incomplete.
5. Cardinality compressed channel embeddings into two scalar values; boundary-pair capping depended on channel order; patient/center/fold exports were incomplete.

## Files Changed

- `neuroez_c/p2_scope_v2_decoder.py`
- `neuroez_c/p2_scope_v2.py`
- `neuroez_c/p2_scope_v2_loss.py`
- `neuroez_c/p2_scope_v2_trainer.py`
- `neuroez_c/cane_path_cp_trainer.py`
- `exp_ez_hybrid.py`
- `run_neuroez_c.py`
- `scripts/build_p2_scope_v2_validation_ledger.py`
- `scripts/run_p2_scope_v2_fixed_validation.ps1`
- `scripts/audit_p2_scope_v2_fold.py`
- `tests/test_p2_scope_v2.py`

## Files Intentionally Not Changed

- P2-Q10 backbone structure and losses
- P2-ATC, P2-RTC, P23, N8F, and existing experiment outputs
- Any raw-cache or EDF extraction path

## Formal Prediction Data Flow

`scope_nez_logit -> scope_ez_score = -scope_nez_logit -> cardinality alpha/beta -> shared Beta-Binomial mode -> predicted K Top-K masks -> shared summary metrics`.

The formal decoder accepts scores, valid-channel mask, alpha, and beta only. It cannot receive labels or true K. Formal records carry `decision_rule=predicted_k_topk`, `formal_prediction_source=predicted_k_topk`, `true_count_used_for_prediction=false`, and `classification_threshold=NaN`.

## True-K Diagnostic Data Flow

`labels_ez + scope_ez_score -> true-K Top-K diagnostic metric`.

This diagnostic is calculated separately on validation/test reporting only. It never writes the formal masks, never determines the checkpoint's primary key, and never enters the decoder.

## Formal-Mask Verification

`scripts/audit_p2_scope_v2_fold.py` recomputes fold metrics from saved channel-level `predicted_ez` and `predicted_nez` fields, checks predicted EZ count equals `scope_predicted_k`, requires NaN threshold and `not_used` threshold source, then compares the recomputation with the fold summary at absolute tolerance `1e-10`.
