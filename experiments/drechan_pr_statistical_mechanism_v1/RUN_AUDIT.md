# Execution audit

## Before real-cohort inference

- Base: completed feature-predictiveness study commit
  `78b5c5d26bb8d5d2468324d1f6be23d9f801ac81`.
- Existing A/B reports, definitions, numerical verification and exact preprocessing
  lineage were inspected. No previous classification benchmark is rerun.
- User explicitly required all 80 patients. Before new outcome computation, the
  internal assessment assignment was changed from the historical 33 unique
  patients/40 appearances to 80 unique patients/80 appearances. The old split is
  retained as an immutable audit input, not mislabeled as an 80-patient evaluation.
- Eight synthetic tests cover mathematical identities, affine invariance,
  numerical support, missing values, zero directions, TRAIN/query isolation,
  donor isolation, permutation counts, independent scalar arithmetic and the
  complete/legal metadata-only assignment. These are not cohort findings.
- Pre-execution synthetic debugging corrected a mistaken expectation about a
  tiny but nonconstant feature after global unit scaling, and literal-constant
  floating-point scale detection. Operational global preprocessing explicitly
  preserves historical float64 arithmetic before FP32 bank rounding.
- Source/protocol will be committed, pushed and remote-verified before admission
  or any new ASSESS evaluation. All scientific source hashes are bound at admission
  and enforced at every subsequent stage.

## Trust and resource boundaries

Use only existing frozen feature/label caches, CPU 2 and one BLAS thread. No neural
imports/training, classifier fitting, new feature extraction, or original output
modification. The independently running LOCO experiment is not touched.
Public release contains only anonymous aggregates, schemas, source and hash proofs.
Exact execution logs and individual-data artifacts stay on the trusted server.
