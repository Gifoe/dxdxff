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
- Initially eight synthetic tests cover mathematical identities, affine invariance,
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

## Admission02 and complete execution

- Corrected source`49db6da4084635bade1cf64e3da10a5ed890e8fb` was committed,
  pushed and verified with `git ls-remote` before real admission02/inference.
  Source archive SHA256:
  `4c295963b2d1778d932df4923758ab8e98f65dc15c8722a66515cea83dfbf612`.
- A new isolated code/runtime attempt preserved admission01 and its source/input
  archive/logs. No outcomes had been computed or discarded at the correction.
- Exact original Feature PR/unit-consistent PR parity and immutable old-bank
  replays pass. Raw inputs, labels, source membership and all original role
  lineage checks pass. Metadata-only assignment digest:
  `878b0a83431b99390e47e46be7069e7aaeec6853965f1c45e57f4e98ed8f1350`.
- All80 are assessed once. TRAIN counts27/27/26/28/28; CAL8 each, unused;
  ASSESS16 each. The sole original-validation exception is in path5.
  Assessed channels1601/1446/1514/1616/1458 sum7,635; EZ393/324/301/402/323
  sum1,743; all original source counts are retained.
- Admission endpoint access is TRAIN-only. Rank-AUC and hence rank-biserial
  invariance error is0; fixed-projection identity error≤2.13×10⁻13; supported
  affine-invariance error≤3.53×10⁻11, with0support crossings. No neural imports.
- All7,070 original/permuted TRAIN directions, donor assignments and preprocessing
  choices were frozen before any new ASSESS endpoint. All179,360 patient-direction
  rows were sealed before aggregate reporting. No performance-based choice was made.
- The independent implementation replays all7,070 directions and717,440 endpoints
  using origin-shifted class moments, direct scalar projected variances and
  bootstrap-count matrices. It passes1,112 nominal and25 family interval checks.
  Maximum endpoint absolute difference1.1428369361965451×10⁻10.
- All100TRAIN label shuffles and matched-H3 diagnostics, both direction estimators,
  all three epsilon rules, all four sources and the strict79 sensitivity are retained.
  No classifier/neural fits, raw encoding, new extraction or feature changes occurred.
- Nine source tests pass locally (Python3.10) and on trusted server (Python3.13.9,
  NumPy2.3.5,SciPy1.16.3,scikit-learn1.7.2). Full private artifacts/logs remain
  trusted-side. Public aggregate export excludes individuals and private paths.
- Frozen trusted-intermediate archive SHA256 (NOT the final public release):
  `792f9caee25efa499ede006b2e0d5d649ff21291ac8228fc76a929bc28f12e89`.
  Integrity/schema checks authenticated the original18payloads and eight source
  hashes. Subsequent publication review identified the complementary80/79 mean
  disclosure risk before any result was committed/pushed publicly. The exact
  strict79 table was blocked from publication, as detailed below.
- Source fingerprint parity is enforced through reporting and release; scientific
  files are not reformatted or changed after inference. Presentation scripts do
  not refit directions, change metrics or alter the authenticated CSV bytes.

## Reproducible commands

Use the environment aliases and entrypoint documented in README; exact private
locations and actual invocation are kept only with trusted runtime logs.

```powershell
python -m unittest discover -s experiments/drechan_pr_statistical_mechanism_v1/tests -v
powershell -NoProfile -File experiments/drechan_pr_statistical_mechanism_v1/run_all.ps1 -Runtime <private-runtime> -Python <trusted-python>
python experiments/drechan_pr_statistical_mechanism_v1/release_check.py --package <privacy-filtered-archive> --sha256 0a356496d52fba9beb8c7643e216b48efdceccf5d18a29e41c11b772c8f349d4 --extract <new-aggregate-directory>
python experiments/drechan_pr_statistical_mechanism_v1/report_public.py
python experiments/drechan_pr_statistical_mechanism_v1/plot_figures.py
python experiments/drechan_pr_statistical_mechanism_v1/verify_public_release.py
```

## Scientific result and interpretation

The80-person primary ΔT is negative but its95% interval includes zero. Direct
angular alignment improves, Fisher alignment worsens, and centering reduces
midpoint drift. The report classifies this as mixed evidence, not proof of increased
transferable SNR or method novelty. All negative/source-specific findings are kept.
The strict79 sensitivity does not replace the primary80.

## Publication privacy correction and safe release

- Publishing precise80 and79 means would disclose the omitted person's endpoint
  by subtraction. The numerical sensitivity remains verified privately, with its
  qualitative negative conclusion retained. No endpoint or hypothesis was changed.
- Publication filter source`3f6b3d459d478eeb10021362399404640b7eb967` was pushed
  and remote-verified before trusted-side filtering. It retained all17 other
  original payloads bit-for-bit and replaced the complementary table with a
  privacy-review/verification JSON. The eight scientific sources are untouched.
- The final privacy-safe archive SHA256 is
  `0a356496d52fba9beb8c7643e216b48efdceccf5d18a29e41c11b772c8f349d4`.
  It contains18payloads plus manifest, with no strict79 numerical table. Native
  pinned transfer verified the whole SHA; local schema/source/privacy checks
  passed BEFORE extraction. The download helper now accepts only this filtered
  archive, not the original intermediate.
- Unpublished intermediate and temporary QA copies were moved to recoverable
  quarantine outside the repository. Trusted originals remain intact. No patient
  identity, individual score, raw wave, model, scaler or runtime log is published.
- The current filter reports the qualitative sensitivity outcome by a generic
  interval rule, not an outcome-dependent export acceptance gate. Its output for
  this completed study has identical payload bytes to the verified safe archive.
- Public release QA verifies all18payload hashes, eight scientific hashes and two
  exact operational reference dependencies, plus schemas and privacy gates.
  Three PDF/SVG/600dpiPNG figure bundles pass embedded-font/editable-text/dimension
  checks. Plot/report regeneration is byte-identical. An actual Git-tree archive
  is used for the final release test, not merely the working directory.
- `REPORT_STATUS.json` is the immutable report-stage snapshot before independent
  verification. Final completion is established by `STATUS.json` together with
  `INDEPENDENT_VERIFICATION.json`; earlier stage snapshots are not rewritten.

## Trust and resource boundaries

Use only existing frozen feature/label caches, CPU 2 and one BLAS thread. No neural
imports/training, classifier fitting, new feature extraction, or original output
modification. The independently running LOCO experiment is not touched.
Public release contains only anonymous aggregates, schemas, source and hash proofs.
Exact execution logs and individual-data artifacts stay on the trusted server.
