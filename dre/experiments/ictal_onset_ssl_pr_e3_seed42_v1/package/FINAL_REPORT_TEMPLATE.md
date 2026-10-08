# Ictal Onset SSL + PR: Experiment Report

**Status:** `REAL_DATA_NOT_RUN` until the server outputs replace this template.

## Data and provenance

- Frozen cohort: 80 patients; 256 seizures; 7,635 unique channels (historical provenance only).
- Frozen fit/val/test memberships: **not reaudited against private cache in this conversation**.
- Feature/cache fingerprints: expected hashes in `expected_sha.json`; live server confirmation **pending**.
- Independent EDF/onset proof: **not established**. This must remain separate from historical source-code alignment.

## Variants

| Model | SSL | Patient Relative | Relative Attention | Patient Macro-F1 (outer) | EZ-AUPRC (outer) |
|---|---|---|---|---|---|
| E0 | No | No | No | NOT RUN | NOT RUN |
| E1 | No | Yes | No | NOT RUN | NOT RUN |
| E2 | Yes | No | No | NOT RUN | NOT RUN |
| E3 | Yes | Yes | No | NOT RUN | NOT RUN |
| E4 | Yes | Yes | Yes | GATED/NOT RUN | GATED/NOT RUN |
| E5_SHAM | Shuffled onset pairing | Yes | No | GATED/NOT RUN | GATED/NOT RUN |

## Primary prospective comparisons

1. E1 minus E0: PR contribution conditional on no SSL.
2. E3 minus E1: SSL contribution conditional on PR (primary).
3. E3 minus E2: PR contribution conditional on SSL.
4. E4 minus E3: relative-attention utility after dev gate.
5. E3 minus E5_SHAM: true onset pair structure after dev gate.

## Decision policy

Require validation E3−E1: Macro-F1 >=+0.020, EZ-AUPRC nondecreasing, >=4/5 positive folds. This controls whether E4 and E5_SHAM are trained. Do not select a single best model after inspecting outer test.

## Historical references (not matched new-model results)

- Patient-z MLP: 0.616167 patient Macro-F1; 0.531996 EZ-AUPRC.
- Historical CDEL: ~0.644589 Macro-F1 under a different feature/training setup.
- RawTiny with PR already tried and below A1; new claim must arise from SSL contribution under PR, not the mere presence of Attention.

## Output locations after server execution

`audit.json`, `validation_selections.json`, `validation_gate.json`, `score_selection_lock.json`, `outer_by_fold.csv`, `outer_by_center.csv`, `test_summary.json`, `RUN_STATUS.json`. Keep private individual records and full per-patient scores on the private server only.
