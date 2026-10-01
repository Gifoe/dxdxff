# Omni class-conditional covariance audit (seed 42)

This is a frozen-CNN post-hoc audit, not a trained model. It exposes R4 (32-D,
before the nonlinear head) and P16 (16-D, immediately before the frozen final
linear layer), then compares source Fisher geometry, diagnostic target Fisher,
and predeclared zero-label target-covariance analytic scores **only after its
binding baseline replay gate passes**.

The server-only extraction reads already materialized 60-s feature NPZs. It
does not read EDF files, optimize parameters, modify a checkpoint, or upload
private EDF/channel/patient records. `run_audit.py` enforces the no-label
interface `estimate_unlabeled_moments(X, patient_ids)` for UTC/CORAL/
UTC-Mahalanobis construction. Public result files contain aggregates only.

The final analysis is necessarily exploratory/repeated-test: historical Omni
test outcomes had been viewed before this audit. Test labels are permitted only
for marked diagnostic geometry and evaluation, never for construction of UTC,
diagonal UTC, CORAL, or UTC-Mahalanobis.

## Recorded terminal

The frozen official TEST replay passed, but the required TRAIN-FULL reference
did not: the official `(EDF, channel)` score replay was `0.9571564413`, while
the binding reference was `0.9586782931`. The hard stop was therefore applied;
no covariance, Fisher, UTC, CORAL, Mahalanobis, bootstrap, or center result
was produced. See [`outputs/FINAL_REPORT.md`](outputs/FINAL_REPORT.md) and
[`outputs/AGGREGATION_PARITY_AUDIT.json`](outputs/AGGREGATION_PARITY_AUDIT.json).
