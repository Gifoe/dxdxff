# A12-VCSN Final Repair Design

## Goal

Complete the existing A12-VCSN implementation against
`CODEX_PROMPT_A12_VCSN_COMPLETE_ONE_SHOT_FINAL.md` without replacing the
current worktree, changing frozen V3 folds, or silently substituting missing
capabilities. The deliverable is code, tests, audits, real-cache fixture smoke
evidence, and one final commit.

## Governing contracts

- Source `fold_idx` is authoritative. No outer fold is regenerated.
- Clinical labels are evaluation-only. Candidate generation, preprocessing,
  fitting, calibration, policy selection, gating, checkpoint selection, and
  ensemble selection cannot inspect outer-test labels.
- All channel joins use `(subject_id, fold_idx when applicable,
  channel_name_norm)` while public tables preserve original and normalized
  names.
- Every external path is supplied at runtime. Example configuration contains
  placeholders only.
- Missing required data fails in strict mode. Non-strict mode skips only the
  dependent variant and records the reason.
- The real feature-cache pickle is read-only. Invalid records are dropped as a
  complete duplicate group or individual invalid record and are fully audited
  with hashed identifiers.

## Architecture

The existing module boundaries remain. `schemas`, `io`, `cache_filtering`,
`protocol`, and `audit` own canonical identities, runtime input adaptation,
cache validation, and fail-closed contracts. `anchor_features`,
`trajectory_store`, `trajectory_features`, `topology_features`,
`candidate_pool`, and `feature_blocks` produce label-free patient-relative
features and candidate source flags. `pair_dataset`, `models`, `calibration`,
`thresholds`, `matching`, `patient_gate`, and `evaluation` own nested training,
cross-fitting, complete action-set simulation, and authoritative metrics.
`provenance`, `reporting`, and `suite` own fingerprinted resume, artifact
manifests, variant isolation, and final summaries.

## Data flow

1. Load and normalize the frozen V3 ledger, optional HNC ledger, and read-only
   cache. Reject duplicate normalized patient-channel keys.
2. Filter cache records, write hashed audit artifacts, and build a masked
   `PatientTrajectoryStore` that preserves `[patient, seizure, time, feature]`.
3. Build anchor coverage, multi-feature patient/seizure-relative trajectory
   summaries, topology features, and explicit HNC semantics.
4. Generate candidate-source unions, apply deterministic caps after union, and
   build evaluator-derived swap pairs.
5. For each outer fold, cross-fit models, calibration, policy selection,
   learned patient gate, CatBoost hyperparameters, checkpoints, and V10 member
   combinations using outer-train subjects only.
6. Freeze the selected policy, infer once on outer-test, apply normalized-key
   action sets, and evaluate with the authoritative patient evaluator.
7. Write provenance, model/training manifests, resume fingerprints, public
   reports, and explicit smoke/partial/full run classifications.

## Training and device behavior

Siamese and TCN models use patient-grouped train/validation splits,
patient-balanced mini-batches, train-only scaling and class weighting, early
stopping with best-checkpoint restore, gradient clipping, deterministic seeds,
and explicit device resolution. TCN input remains `[B,S,T,F]`; only `B*S` is
folded for the temporal encoder before masked cross-seizure pooling. CatBoost
uses grouped inner validation and the registered search grid. V10 selects among
all seven non-empty member combinations using inner predictions and preserves
seed, member, and total uncertainty.

## Error handling

Contract errors include the variant, fold, subjects or differing fingerprint
fields, and the failed invariant. CUDA requested but unavailable fails only
when `strict_device` is true; otherwise the CPU fallback reason is recorded.
Unknown HNC semantics fail strict mode or skip only V5. Anchor/trajectory
coverage failure follows the same strict/non-strict dependency boundary. No
fallback is labeled as a successful implementation of the missing capability.

## Verification

Every new behavior is introduced through a focused failing test, followed by
the smallest implementation and a green focused suite. Completion additionally
requires compileall, the complete pytest suite, unittest discovery,
`git diff --check`, available lint/type checks, the real-cache audit, a
real-structure fixture build, CPU smoke for V2/V3/V4/V6/V7/V8/V9/V10, and the
synthetic V0-D0-V10 matrix. A partial or smoke execution is never reported as a
full five-fold result.
