# Implementation audit

- Fresh 150-checkpoint A1 replay and selected-epoch R4 prehook/classifier equality were verified before calibration. FIT/validation roles, source locks, selected epoch and VLOO threshold are checked per cell.
- The prior reference's private raw R4 payloads are reused, not its aggregate outcome numbers. FIT scaler is fit only to FIT channel rows for each selected checkpoint.
- Candidate/query split is deterministic, label-blind, fixed across policies and budgets in each repetition. All 20 query-score sets are frozen before any query labels or target oracle are accessed. Only acquired candidate labels enter deployable calibrators.
- Exact B=0 A1 ranking and decision checks pass. Bias-only uses matched support. Oracle-balanced support and full-pool are explicitly nondeployable controls.
- Patient-ID cluster resampling (10,000, seed 42) includes every fold/repetition of sampled IDs. Per-cell files remain private; output is aggregate-only.
- A memory-bound one-cell subprocess driver changes only process lifetime, not samples, optimizer, acquisition or statistical rules.
- Legacy monolithic A1 loader materializes all 80 patient labels while constructing the FIT+validation data object; no outer predictions or metrics are produced, but literal no-outer-label-materialization is not satisfied.
