# P2_ATC Implementation Audit

## Frozen Contract

- Cache labels remain `EZ=1`, `NEZ=0`; the training loss derives `labels_nez=1-labels_ez` once.
- Model scores are `score_nez=sigmoid(final_nez_logit)` and `score_ez=1-score_nez`.
- Both cohorts use seed 42, patient-wise outer five-fold splits, four-fold inner OOF threshold selection, and no patient calibration.

## P2 Baseline Path

`P2_TEMPORAL_Q10` uses temporal seizure embeddings, the legacy five-feature q10 tail evidence head, clean-NEZ anchor residual, and one seizure residual term (`0.20 * u_seizure`). The original patient-balanced BCE, soft worst-center, soft Macro-F1, anchor, high-confidence-EZ rank, and residual-L2 losses remain active. Checkpoints retain the original F1 selection and thresholds come only from outer-train inner-OOF predictions.

## ATC Changes

- A0: frozen legacy five-feature q10 tail head and no ATC loss.
- A1: normalized logit-space soft-min with count shrinkage, plus a seven-feature tail head. Raw probability q10 remains a diagnostic field.
- A2: patient-balanced clean-NEZ robust-tail floor, restricted to valid seizure count at least two.
- A3: A2 plus within-patient trusted observed-EZ pairwise robust-tail separation. Trusted-EZ selection reads only detached direct NEZ score and detached anchor NEZ evidence.

No Noisy-OR, EMA, soft labels, raw branch, V3, true-count/cardinality, center-specific rule, patient shift, or patient threshold is enabled.

## Optional Direct-Outer Screening

`run_p2_atc_dual_cohort.ps1 -DirectOuterOnly` is a speed-oriented screening mode. It trains once per outer fold and selects its threshold from an outer-train validation split. Its outputs are isolated below `direct_outer_screening/`; they are not interchangeable with the nested inner-OOF formal protocol and must not be pooled or compared as if they had the same threshold-selection procedure.

## Files

- `neuroez_c/p2_atc_profiles.py`
- `neuroez_c/p23_seizure_tail.py`
- `neuroez_c/cane_path_cp_loss.py`
- `neuroez_c/model.py`
- `neuroez_c/p23_trainer.py`
- `run_neuroez_c.py`
- `scripts/audit_p2_atc_seizure_coverage.py`
- `scripts/run_p2_atc_dual_cohort.ps1`
- `scripts/audit_p2_atc_results.py`
