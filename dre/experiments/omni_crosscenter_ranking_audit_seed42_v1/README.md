# Omni Cross-Center Ranking Decomposition Audit

This is a pure post-hoc audit of frozen Omni TimeConv-CNN and the already
frozen R4 Source Fisher score. It does not train, calibrate, pool, adapt, or
modify a neural model. The only lawful TRAIN representation source is the
validated full-record artifact from `omni_bag_mismatch_audit_seed42_v1`; the
historical five-clip TRAIN feature NPZs are explicitly forbidden.

The audit decomposes pooled AUROC into positive-center × negative-center
ranking cells, then evaluates three predeclared zero-label center transforms:
median/IQR UCS, empirical percentiles, and fixed-clipped CDF Gaussianization.
The TEST result is exploratory because this test set was previously viewed.
Only compact aggregate outputs are versioned; caches, predictions, source EEG,
checkpoints, runtime logs, and patient identities are not.

## Result

The full-record gates and frozen replays passed: CNN AUROC `0.7987677712` and
Source Fisher R4 AUROC `0.7994219077`. Fisher gained `+0.007779674` in the
pair-weighted within-center term but lost `-0.007125537` in the cross-center
term. The two effects almost cancel.

This is not a deployable score-alignment route. All zero-label controls failed
the preregistered TRAIN pseudo-target gate: every Source Fisher alignment had a
negative OOF AUROC delta in all five folds. Their exploratory TEST effects were
also negative (Fisher UCS `-0.035735`, CPA `-0.034774`, Gaussianization
`-0.034774` versus Fisher). The terminal is
`CROSS_CENTER_MISALIGNMENT_NOT_ZERO_SHOT_ACTIONABLE`.

The literal Gaussian CDF clip would introduce tail ties and violate the
required within-center ranking identity. The committed implementation therefore
uses the fixed clip endpoints with adjacent floating-point values to preserve
the original rank order. This is deterministic, label-blind, and does not
change the clip threshold; the audit verifies zero within-center AUC change.
