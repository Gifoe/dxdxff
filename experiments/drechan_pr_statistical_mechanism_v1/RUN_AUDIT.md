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

## Admission01: metadata-only feasibility failure

- Source `4f1ab8b7fc854da5d51ceb1754684c64721a53e3` was pushed and remote-verified.
  Its Git archive SHA256 is
  `b6ae138437dc077acc1692b4f8b273daf48788ae533190be866722e7f3ebf331`.
- Real frozen metadata revealed FIT-appearance counts:33patients appear3times,
  11appear2times,33appear4times,2appear1time,and1appears0times. The never-FIT
  patient belongs to pediatric. The all-FIT assignment is mathematically infeasible.
- Failure occurred before representations, direction freezing or ANY ASSESS
  endpoint. Source, input archive and private transcript are preserved unmodified.
- Before outcomes, eligibility was minimally corrected: the sole never-FIT patient
  may be assessed in one original-validation path, never original test. The other79
  remain inside original FIT; all80 stay absent from their own TRAIN/CAL.
- The corrected source will be pushed before a NEW isolated admission/runtime.
  Added a synthetic never-FIT/validation-only eligibility test and a prespecified
  strict79 descriptive sensitivity. Primary remains all80, not a favorable subset.

## Trust and resource boundaries

Use only existing frozen feature/label caches, CPU 2 and one BLAS thread. No neural
imports/training, classifier fitting, new feature extraction, or original output
modification. The independently running LOCO experiment is not touched.
Public release contains only anonymous aggregates, schemas, source and hash proofs.
Exact execution logs and individual-data artifacts stay on the trusted server.
