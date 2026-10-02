# Engineering repair ledger

| Item | Scope | Verification | Scientific effect |
|---|---|---|---|
| P16 availability | Existing cache retained R4 but not P16; deterministic inference from already materialized 60-s feature NPZs added P16 to a private cache. | Frozen-forward identity check was zero within the extractor's configured tolerance. | No checkpoint, EDF input, or optimization change. |
| Windows job persistence | Extraction was moved from a directly spawned SSH child process to scheduled tasks because the remote child was reclaimed when the SSH parent ended. | Atomic private cache markers and expected 296 TRAIN / 237 TEST files. | Execution mechanism only. |
| Extraction throughput | Batch 24 replaced batch 16 after a same-input R4/P16/logit maximum discrepancy of `3.815e-6`. | Per-sample frozen-forward identity still passed. | Numerical extraction implementation only; no score rule changed. |
| Leakage scanner | Identifier scanning was corrected so `unlabeled` is not falsely classified as a label input. | Local smoke test passed. | Audit enforcement only. |
| Baseline gate | TRAIN replay did not match the binding reference. | See `BASELINE_REPLAY_AUDIT.json` and `AGGREGATION_PARITY_AUDIT.json`. | Hard stop; no geometry or adaptation result was produced. |
