# Omni class-conditional covariance audit (seed 42)

This is a frozen-CNN post-hoc audit, not a trained model. It exposes R4 (32-D,
before the nonlinear head) and P16 (16-D, immediately before the frozen final
linear layer), then compares source Fisher geometry, diagnostic target Fisher,
and predeclared zero-label target-covariance analytic scores **only after its
binding baseline replay gate passes**.

TRAIN representations are recovered only from the exact full-record artifact
produced by `omni_bag_mismatch_audit_seed42_v1`, then regenerated from the
native signal cache at those frozen starts. Historical five-clip TRAIN feature
NPZs are explicitly excluded. Before any covariance, Fisher, or UTC analysis,
the pipeline hard-gates the regenerated cache on 296 EDFs, 316,364 total
segments, 145,052 labelled segments, 13,350 labelled EDF-channel units, and
frozen-CNN AUROC 0.9586782931. It does not optimize parameters, modify a
checkpoint, or upload private EDF/channel/patient records. `run_audit.py`
enforces the no-label interface `estimate_unlabeled_moments(X, patient_ids)`
for UTC/CORAL/UTC-Mahalanobis construction. Public result files contain
aggregates only.

The final analysis is necessarily exploratory/repeated-test: historical Omni
test outcomes had been viewed before this audit. Test labels are permitted only
for marked diagnostic geometry and evaluation, never for construction of UTC,
diagonal UTC, CORAL, or UTC-Mahalanobis.

## Status

The full-record recovery and regenerated-representation gates both passed:

- `296` EDFs; `316,364` total segments; `145,052` labelled segments; and
  `13,350` labelled EDF-channel units;
- frozen-CNN TRAIN AUROC `0.95867829307722` (target `0.9586782931`);
- maximum regenerated-vs-frozen segment probability discrepancy
  `2.98e-08`.

The one frozen post-hoc audit completed with terminal
`COVARIANCE_SHIFT_NOT_ZERO_SHOT_ACTIONABLE`: full-covariance UTC R4 AUROC was
`0.786162`, versus `0.798768` for the frozen CNN (Δ `-0.012605`; patient
cluster-bootstrap 95% CI `[-0.050473, 0.031241]`). It is therefore not a
usable zero-shot improvement. The target-label Fisher result is retained only
as a diagnostic. See [`outputs/FINAL_REPORT.md`](outputs/FINAL_REPORT.md).

`outputs_legacy_invalid_fiveclip/` contains the superseded 5-clip terminal
artifact for provenance only; it is not a valid result for this protocol.
