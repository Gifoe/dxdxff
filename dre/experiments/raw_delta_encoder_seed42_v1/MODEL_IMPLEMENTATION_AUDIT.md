# Model implementation

D1 is the 1,537-parameter exact LN88-folded compact MLP applied to label-blind Raw-derived R14. E1 is the identical compact architecture on historical E14 with matching initialization and independent FIT-only preprocessing.

D2/D3 retain the same selected frozen D1 checkpoint per fold. A shared Conv1d(1,12,9,stride4) → GELU → Conv1d(12,24,7,stride4) → GELU → AdaptiveAvgPool1 produces 24D window embeddings. The CNN alone uses the exact historical RawTiny per-window mean/population-SD floor1e-6 and clip[-8,8]; physical R14 is never computed on normalized waves.

Subtract the mean genuine pre-reference embedding within seizure/channel, then process genuine [0,20) windows with a single forward GRU(input25, hidden24). The extra coordinate is actual relative time/20. No pre-reference window enters a post stage. The fixed summary is early [0,8) / late [8,20) stage means, each aggregated over genuinely available seizures with mean and population SD: stage × mean/SD × 24 = 96. Patient Z precedes Linear(96,64) → GELU → Linear(64,1).

Final NEZ logit is frozen D1 logit + 0.5*tanh(gamma)*tanh(raw logit), gamma initially zero. The auxiliary weighted BCE coefficient is fixed 0.25 so CNN/GRU/head receive nonzero gradients at gamma zero. D1 always remains eval/frozen, even when the enclosing model is put into training mode.

D2/D3 total parameters are 13,643: 12,106 trainable and 1,537 frozen. Identical initial full states, parameter budgets, D1 checkpoint and optimization rules are asserted. D3 only changes the deterministic stage-local order of complete window embeddings, equivalent to moving complete wave windows through an independent CNN; chronological destination time coordinates remain fixed. SHA seed42 identities are private, and eligible permutations deterministically avoid identity.

Thirteen deterministic CPU tests passed locally and on the server, including algebra, parameter counts, actual gradient flow, equal initialization, initial D1 identity, base freezing, invalid-padding exclusion and shuffle preservation. Full-data source sample parity and all-fold inference identity are additional runtime audits, not synthetic test substitutes.
