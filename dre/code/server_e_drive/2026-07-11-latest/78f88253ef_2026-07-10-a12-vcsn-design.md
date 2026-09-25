# A12-VCSN Design

## Goal

Implement A12-VCSN as an isolated, fail-closed experiment suite that learns
selective, K-preserving swaps around a frozen V3 OOF prediction ledger. The
suite must produce reproducible audit, model, action, and comparison artifacts
without modifying the existing V3/A9 code paths.

## Confirmed repository contract

The audited V3 anchor source is a five-fold, 90-subject OOF channel prediction
directory. Its channel files contain `fold_idx`, `subject_id`, `center`,
`channel_name`, `true_nez`, `true_ez`, `score_nez_probability`,
`score_ez_probability`, `rank_ez_desc`, `predicted_nez`, and `predicted_ez`.
The existing evaluator measures patient-level macro-F1 with `true_ez=1` and
top-K EZ predictions; it is reused by the A12 adapter rather than redefined.

The production cache adapter expects an existing read-only pickle containing
`run_records`, `patient_index`, and window feature tensors. It discovers and
records all real feature names and channel mappings at runtime. No raw waveform
or EDF extraction is part of A12. Inputs that lack usable window features fail
strict audit; the package remains testable with synthetic fixtures.

## Architecture

`a12_vcsn.schemas` normalizes the supplied V3 ledger into one canonical
patient-channel dataframe with explicit clinical and prediction semantics.
`a12_vcsn.io` reads the frozen ledger and cache, hashes inputs, and emits a
resolved schema. `a12_vcsn.protocol` validates the 90-subject/five-fold anchor,
parity, subject isolation, feature provenance, and OOF-only auxiliary inputs.

Candidate generation, per-channel feature construction, pair construction,
pair labels, utility models, gates, matching, evaluation, and reporting are
separate modules. The orchestrator uses only outer-train subjects for fitting,
inner OOF predictions for calibrator/threshold/gate selection, and reads test
labels only after actions have been frozen.

## Variants and failure policy

V0 is the unchanged frozen anchor and D0 is ground-truth diagnostic-only.
V1--V4 increment CatBoost feature groups, V5 adds an OOF HNC residual only if
such a ledger passes strict joining/provenance checks, V6--V9 add Siamese/TCN
models and optional learned gates, and V10 ensembles calibrated utilities only
when chosen inside each outer fold. A missing optional dependency or valid HNC
input marks only the affected non-strict variant as skipped; strict mode raises
an error. No model is silently substituted or relabelled.

## Public interface and artifacts

`scripts/run_a12_vcsn_suite.py` is the full-suite CLI. It accepts explicit
ledger/cache/output paths, optional allowed-subject/HNC ledgers, variant and
seed selection, resume, strictness, cache rebuild, audit-only, and dry-run
flags. All runs emit resolved configuration, environment/git/input provenance,
audit reports, cache compatibility metadata, per-variant manifests, actions,
final ledgers, patient/fold/center metrics, paired bootstrap confidence
intervals, and an explicitly post-hoc comparison summary.

## Safety and testing

The code exposes clinical labels as audit/evaluation data and rejects them from
candidate or feature registries. Pair rows are weighted to one per subject.
Matching is deterministic, one-to-one, utility-thresholded, and limited to zero
through two swaps. Tests cover semantic mappings, candidate construction,
anchor non-use of labels, masks/topology, pair deltas, calibration/gate/fold
isolation, matching, HNC provenance, oracle-feature exclusion, and an
end-to-end synthetic five-fold suite.

## Non-goal

This implementation cannot establish a real-data score above 0.70 until a
valid all90 cache and the frozen V3 ledger are supplied to the CLI and a full
outer-OOF run completes. The code will report that condition rather than
inventing a result.
