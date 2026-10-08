# Implementation and engineering audit

## Scientific invariants

- Same PRMLP/8,817 parameters, seeded initial state, dropout RNG reset, patient
  order, preprocessing and AdamW settings in all three student arms.
- Original NEZ-positive binary FIT counts define BCE positive weight, including
  for soft targets. One channel-mean update per patient; no pooling patients into
  a channel-count-weighted objective.
- A1 epsilon=.10; A2 lambda is zero through epoch5, .05/.10/.15/.20/.25 at epochs
  6–10, then .25. No target flip/removal; q/u detached.
- Four OOF groups per outer FIT, teacher internal TRAIN/selection/query disjoint,
  no outer VAL/TEST labels passed to teachers. Each FIT channel queried once.
- Teacher preprocessing fit on teacher TRAIN. Student preprocessing fit on outer
  FIT. Within-patient z uses only that patient's complete unlabeled channel context.
- Ten MC passes activate Dropout only and use no_grad. Mean probability,
  predictive/expected entropy, MI and variance are retained privately. Only q/u
  affect the primary target.
- Historical threshold candidates, metric unit and all tie-breaks preserved.
  Vectorized confusion counts accelerate the same 201-point grid; source parity
  is tested, not assumed. No validation/test-derived per-patient threshold.
- Epoch checkpoints bind source/protocol/features/OOF artifacts/initial weights
  and retain optimizer, complete history, stale counter, best model and RNG.
  Changed binding or corrupted completed checkpoint is rejected.
- Development bootstrap resamples patient IDs, preserving every repeated
  patient-fold validation appearance rather than treating it as an independent ID.
- EZ-F1 continuation tolerance is conservatively zero, fixed before outcomes;
  the prompt's undefined 'material' tolerance is not chosen retrospectively.

## Engineering changes before formal training

1. NumPy2 pickle module names require aliases when loading the old feature cache
   in the NumPy1.26 overlay. No bytes/values changed; fresh five-fold gate PASS.
2. The foreground replay process ended natively without a Python error. The
   background replay subsequently completed and exactly matched historical
   metrics; original cached data and checkpoints were never overwritten.
3. Original runtime test attempt1 failed natively in sklearn target validation;
   attempt2 failed during Torch import. Neither admitted formal training because
   no passing test/smoke artifact existed. Logs are retained privately.
4. Windows native exit codes can be negative: cmd `if errorlevel 1` is insufficient.
   Launcher now admits only exact zero. Independent audit-file hard gate also
   prevented accidental training on the failed attempts.
5. Evaluate an isolated Python3.12/Torch2.11 runtime using the already installed
   server environment, leaving shared environments untouched. Admission requires
   new B0 replay, all synthetic tests and exact real FIT-only original-source
   preprocessing/training parity. See `audit/RUNTIME_ENGINEERING_AMENDMENT.json`
   and execution artifacts for actual outcomes. No scientific setting changed.
6. Reaggregation of the large feature cache in Python3.12 also exited natively.
   Instead reuse this task's already freshly reconstructed, hash-verified exact
   88D feature export (not an older or approximate cache). All five frozen B0
   checkpoints and historical metrics replayed exactly in the replacement runtime.
7. Pandas string arrays in the private export have object dtype. Loading them now
   permits pickle only after its exact trusted export SHA gate; no array values,
   identities or preprocessing were changed. Failed pretraining sources/logs kept.
8. The Python3.12 repeated sklearn reference loop also exited natively. Eight
   identical synthetic reference cases were exhaustively computed and tested
   locally (all 201 thresholds, original metric/tie logic). On the server their
   canonical input SHA and exact reference optimum/metrics are checked from the
   frozen synthetic oracle, removing thousands of redundant legacy metric calls.
   Local tests continue to execute the complete reference loop by default. Real
   FIT-only selected-threshold metrics are checked with original `_patient_metrics`.
   Original-source optimization is compared using the independently validated
   equivalent threshold evaluator, with exact parameter hash equality after one
   epoch. This is not a claim that the unstable legacy sklearn loop ran successfully
   on the server. The cause of the native faults is not established.
9. Only the private Python3.12 venv received pytest8.4.2 and pandas2.2.3. Original
   shared environments, source cache and frozen B0 checkpoints were not modified.
10. All 20 teachers/15 students completed and saved their artifacts before a
    native access violation during interpreter shutdown. A new independent
    verifier passed last/best hashes, finite weights, optimizer/RNG/history,
    OOF identities/exclusions/scalers and matched validation predictions for all
    35 cells. Final aggregation and 10,000-draw bootstrap then completed. No
    formal completed epoch was discarded or retrained. Root cause remains
    unestablished. Public launchers were subsequently changed only to canonical
    filenames for handoff; actual versioned server sources/logs remain retained.

## Scope and limitations

Original clinical labels are not independently cleaned/adjudicated. Entropy does
not identify clinical annotation error. A teacher may be confidently wrong.
Validation selects checkpoint/threshold, so development performance is selection-
optimistic. Historical outer outcomes are already viewed. A newly admitted outer
pass remains exploratory, not independent confirmation.

Magnitude matching is limited to OOF target-space diagnostics using a derived
constant epsilon with exactly equal average target movement, without training an
extra arm. Primary A1 is fixed epsilon=.10. This cannot by itself establish a
causal uncertainty-specific effect independent of smoothing strength.
