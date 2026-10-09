# A0 source-conditioned scorer and patient posterior decoder

Seed42, original Task1 cohort and frozen five folds. Exactly D0–D3; no outer
evaluation. Full rules are frozen in PROTOCOL_LOCK.json. New training consists
of five joint rank4 source-conditioned A0 models and 20 legal inner teachers.
The 20 pre-existing hard-label A0 teachers are deterministically replayed.

Run `code/execute.ps1` on the original trusted server. It isolates steps/folds,
retains separate private logs, and stops on any failed integrity or test gate.
Private model/scaler/prediction files are never repository inputs or uploads.
Resume preserves old logs, uses new attempt names and epoch checkpoints; it is
not permission for new models, thresholds or outcomes-driven changes.

The posterior common-variance Gaussian family, nu10 shrinkage, Beta20 prior,
deterministic MAP solver and all-k ratio-of-expected-counts Macro-F1 approximation
are locked before outcomes. A fixed-prior diagnostic is not a selectable arm.
See FINAL_REPORT.md and audit/results artifacts for completion and findings.

## Completed outcome

Terminal: `POSTERIOR_IDENTIFIABILITY_FAILED`. D0 replay passed; five formal D2
models and all 20 D2 teachers completed. An independent replay of all 40 A0/D2
teacher query outputs passed with zero logit drift. Full matched development
Macro-F1 is 0.638080 for D0 and 0.647926 for D2; the paired patient-cluster gain
is +0.009846, 95% CI [-0.004723, 0.026640], with only 3/5 improving folds.
The source-only continuation gate fails. No outer TEST was accessed.

D1 and D3 fail the frozen global FIT density ordering in folds 3–5. Their full
validation metrics are `not_estimable`; valid partial-fold diagnostics are not
full-arm results. Near-zero patient IQR creates extreme normalized tails that
dominate Gaussian moments. No clipping, exclusion, relabeling or rescue fit was
used. This run does not support the combined posterior mechanism.

The orchestration intentionally stops at that scientific gate. To reproduce
the blocked terminal report from the retained private artifacts, run the
reporting-only scripts on the trusted server, in this order:

```powershell
& $Python tests/verify_oof_and_normalization.py --runtime $Runtime --source $Source
& $Python tests/complete_nonidentifiable.py --runtime $Runtime --protocol $Protocol
```

The variables use the paths in `code/execute.ps1`; invoke from this experiment
directory. These scripts verify completed scorer/teacher artifacts and report
the failed densities; they neither train nor apply invalid posteriors. They
were added after the blockage and are explicitly not new model variants.
`results/RUN_STATUS.json` binds the public aggregates by SHA-256. Private banks,
checkpoint tensors, individual outputs and logs remain on the server.
