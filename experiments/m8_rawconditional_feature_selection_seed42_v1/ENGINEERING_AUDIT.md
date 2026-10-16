# Execution and numerical audit

## Actual scientific work

Twenty strictly inner-cross-fitted Raw encoders completed286 full-protocol
epochs. The original raw data, all80-patient/7635-channel/256-seizure contracts,
valid-window masks and patient-relative computation are unchanged. Every
calibration logit was replayed against the saved model. Feature-choice and
assessment labels were absent from neural training and checkpoint selection.
The20 role/representation admissions are in `results/ENCODER_ADMISSION.csv`
and `results/PATIENT_ROLE_ADMISSION.csv`.

The first closed-form screen executed9344 heads but failed its actual-data
numeric gate. Its outputs and logs remain private and preserved, not silently
replaced. The already declared full sklearn reference then recomputed9344
heads from the same representations. There was no Raw encoder retraining.
An additional25 distinct final-choice head fits fill30 registered control
rows, sharing duplicates; these do not choose or alter any subset. Search
records and control results are in separate tables so actual cost is visible.

## Failed shortcut and unchanged scientific configuration

The synthetic block-ridge and preprocessing tests passed, but the full actual
data admission failed54 of720 comparisons at the original2e-10 absolute
score tolerance. Maximum score discrepancy was0.0006735324859619141;
maximum sigmoid-probability discrepancy was4.14890344302421e-13 and all
calibration thresholds/decisions agreed. Neither the small probability error
nor synthetic success waived the score failure.

The numerical diagnosis found large held-input extrapolation of some legacy
physiological power coordinates under the frozen train-only scaling. It did
not justify declaring a physiological attribute useless. No feature, patient,
window, imputation rule, scaling floor, clip, alpha, target, sample weight,
threshold rule, choice metric or cardinality criterion changed.

`NUMERICAL_ENGINE_AMENDMENT.json` was committed/pushed at62c0bce before the
reference recomputation. `canonical_ridge.py` calls the predeclared full
sklearn Ridge with alpha1, intercept and Cholesky solver. The original seven
scientific files and both parent/follow-up locks remain unchanged. The new
private binding wraps the original input/partition/source identity and the
three numerical-engine source identities.

## Terminal-only metadata recovery

All five reference screens,9344 head evaluations,2805 patient rows, search
tables, chosen feature sets and bootstrap choices were saved before a final
flat-key lookup hit the new nested binding. The exception did not occur
during a fit. `finalize_dense_metadata.py` checks all five private fold
bindings, input/engine identities, completed rows and the original v2 Outer
prediction seal, then restores completion metadata. It performs no head fit
or model inference. Its pending-verification status is an immutable historical
stage record; the later `INTERNAL_REPLAY_VERIFICATION.json` and
`INTERNAL_REPORT_STATUS.json` are authoritative completion records.

`resume_reference_checked.py` is the terminal recovery/reproduction entry.
It refuses absent/partial screens instead of blindly rerunning them, and
does not repeat an existing complete verification/report/control export.
The original wrapper that generated the historical exception is preserved
byte-for-byte, as required by its numerical-engine binding.

## Actual verification and its meaning

- Canonical API admission:720 cases, score error0, unchanged tolerance.
  This is conformance to the chosen reference API, not independent solver or
  clinical replication. The earlier Schur gate stays **FAIL**.
- Full replay:9179 search-head evaluations and165 assessment fits; largest
  choice-utility difference5.551115123125783e-17.
- Independently recomputed patient metrics:2805 rows, including calibrated
  classifications, AUROC/AP and first-hit ranking.
- All patient roles, preprocessing arrays, cardinality decisions, greedy
  paths, best swaps and10 patient-weight re-selections per fold checked.
- Protected original v2 release:135 files unchanged. No original model,
  validation threshold or Outer score was overwritten or rerun.

The actual record does not claim that green computational checks establish
feature minimality, physiological causality or a clinical localization gain.

## Publication and privacy

`export_public_transfer.py` hashes an explicit19-file anonymous allowlist.
`verify_public_release.py` verifies those transferred bytes, the original135
release artifacts, plotted aggregate counts and paired arithmetic. Labels,
patient/channel identifiers, raw/feature/embedding arrays, predictions,
weights, scalers and runtime logs remain on the trusted server. Raw training
uses the existing GPU environment; ridge/statistics use the isolated stable
CPU environment, one BLAS thread and affinity16. No unrelated GPU task,
driver or global environment setting was changed.
