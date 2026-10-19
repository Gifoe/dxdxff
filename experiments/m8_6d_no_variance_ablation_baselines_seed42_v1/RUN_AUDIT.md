# Actual execution, engineering recovery and release audit

## Frozen scope and source

All30groups completed:16neural arms and7classifiers×2PR views, each over the original five folds. There are70fresh neural fits,70fresh classical models and10exact reference reuses,150selected cells,2400patient rows and21600independently checked endpoints. The70new neural fits completed923unique complete epochs.

The14scientific source/protocol files were admitted and pushed before formal fits at **9c84578a83cbae402f28e308de05602931111019**. Their exact hashes remain unchanged in SOURCE_PUSH_BEFORE_TRAIN.json, ADMISSION.json and this release. All subsequent repairs are separately published execution/verification/reporting adapters, not hidden edits to the admitted model, preprocessing, optimization, thresholds or prediction math.

The six-feature main model is exactly prior DROP_VARIANCE. Five main and five Raw-only initialization/validation replays had maximum error0; all201threshold decisions were exact. Five changed-path CPU tests and14real complete-FIT gradient admissions passed. The classifier12D views exactly matched corresponding FIT-only bank coordinates in both PR settings over all five folds. All original80patients/7635channels/256seizures and1471965valid Raw windows remain unchanged.

## Pre-training namespace repair

Admission attempt01 failed before formal fitting: importing the parent module had rebound the shared common-module alias, returning the historical namespace. The input equality gate caught the mismatch. The new module now preserves its own module object before parent imports and restores it explicitly. Attempt02 passed the correct five tests and all real-data gates; the old failed log was retained. Source admission/push happened after this repair and before formal fits.

## Native backward failure and exact resume

Formal run03 suffered a native Windows c0000005 access violation during backward for WO_LATE_GATE,fold4. A timestamp-matched Windows Application Error names cublasLt64_12.dll; the underlying root cause is not established. The venv launcher and underlying base-Python child have distinct process IDs, so the audit does not equate wrapper IDs with native event IDs or claim a proven hardware diagnosis.

Before run04, all43completed fresh cells' BEST/validation/completion hashes were verified. The complete epoch8 LAST contained14model tensors,13Adam slots and complete Python/numpy/torch/CUDA RNG state, with best epoch7. A whole-file-verified private backup was retained. LAST SHA256:

**c969dbba9ea1e94a0f60924419dc9f00cade9cf39929d846ad4f745ce71b152c**

The unchanged command resumed at epoch9, skipped completed cells and completed all remaining fits. No complete epoch was discarded. Model/data/loss/hyperparameters/selection/library/scientific sources were not changed. See results/NATIVE_RESUME_AUDIT.json and the published diagnosis helper. GPU memory had ample free capacity after the failure; this is not treated as proof of the exception's cause. Original logs/checkpoints remain private and preserved.

## Verification-only RF roundoff handling

All80neural replay cells passed; the initial classical verifier stopped because it demanded bit-identical probabilities from a two-thread RandomForest reduction. The read-only diagnosis of all70classical models found at most4.440892098500626e-16difference, exclusively in RF. All201threshold decisions and all selected thresholds/decisions remained identical. This was not a training error or evidence of changed weights.

The original verifier was preserved. verify_internal_roundoff.py changes only the RF probability equality check to a strict8×float64-epsilon absolute bound (**1.7763568394002505e-15**); every other classifier remains bit exact. It additionally requires all201threshold decisions, selected thresholds/decisions, and within-patient complete rankings to remain exact. No probability is rounded or rewritten, n_jobs is unchanged, and no model or calibrator is refitted.

Actual final verification maximum: neural **0**, tabular **3.3306690738754696e-16**. All150selected cells passed. Wrapper/source hashes are recorded in results/TABULAR_REPLAY_ROUNDOFF_AUDIT.json and INTERNAL_VERIFICATION.json, and are checked by the public/Git release verifiers. Both the original failed log and the quantitative diagnosis table remain preserved.

## Reference basename adapter

The initial Outer entry failed before producing a new score cell because its mechanical migration requested the new neural-only seal basename in the historical parent runtime, where the historical unified seal has a different name. evaluate_reference_path_adapter.py redirects only that exact requested metadata path to the already pinned original OUTER_SCORE_SEAL_PRIVATE.json with SHA256:

**6dc73306f04ffefa9ffee471470786b3e190d16a304a81b8be5b9b2225a1f033**

It changes no original file, model, input, inference operation, prediction or threshold. The original entry/log remain preserved. Its exact adapter hash and resolution proof are public in results/REFERENCE_PATH_ADAPTER_AUDIT.json. run_verified.ps1 is the actual complete production launcher incorporating both disclosed adapters; run_all.ps1 remains the immutable initial source snapshot, not a misleadingly repaired historical file.

## Freeze, one sealed Outer, and independent verification

- All150model/preprocessor/calibrator/epoch/threshold selections froze before any new Outer inference.
- Global pre-Outer freeze SHA256: **d58f7157a5b02531d8082c53ae5dacc682955fa2ed3126662d78091c0d67606d**. The legacy status string names80neural cells; its models field is150 and binds the separate70-cell tabular freeze.
- Neural model-freeze SHA256: **6c4a60d0586dd63adb649c48b9cf271100aeaecc1850a0c0339431b6be1b7e55**.
- Tabular model-freeze SHA256: **38a21a7e2fb0acddf7550250fddf64782e1d8f5cdadaa2eee6757b7732f67a8b**.
- All150score cells sealed before unified metric reporting. Global score-seal SHA256: **5886048612e9c3d2539d75ed0ad434d8ff104dc5298440e6476670cd00157b05**.
- Independent sklearn/count/ranking checks covered2400patient rows and21600values;603nominal and37additional family intervals were independently recomputed. Maximum interval discrepancy: **8.881784197001252e-16**.
- Each of13new Raw-consuming arms retained1471965valid windows. Physiology-only and all classical models consumed no Raw during inference. Reused main/Raw-only predictions remained unchanged.
- All10MLPfits reached the original400iteration budget with ConvergenceWarning. No post-Outer extension, refit, search or outcome-based omission was made.

Actual production terminal:

```text
SIXD_30_GROUPS_ALL_FIVE_FOLD_OUTER_VERIFIED_COMPLETE
```

Independent private verifier terminal:

```text
F0_SIXD_ALL30_FULL80_INDEPENDENT_VERIFICATION_PASS
```

Public-only aggregate/report/source/adapter/manifest terminal:

```text
SIXD_30_GROUP_PUBLIC_RESULTS_REPORT_AND_MANIFEST_PASS
```

## Public export and visual QA

Only explicit anonymous public outputs were exported. The final v2 package contains47entries,463675bytes, SHA256 **6e0faa17984d22a9686f23072c07f4772866c079229e2cc78463bd26d3283691**. Transfer and all per-file hashes passed. The ZIP is not committed.

The first preserved export was468687bytes, SHA256 **b5cf98b9b6a0d76824e57a531c71ccb39d8108c6f8a07758ec4a640cac7bad81**. Visible QA found overly dense AUROC ticks. A uniform MaxNLocator layout repair and readable classifier labels fixed this; the report also disclosed actual convergence warnings. All **21 numeric CSV hashes were identical between export versions**. No model, prediction, threshold or numeric result was rerun or changed for plotting.

All three final PNGs were actually viewed; editable SVG/live text and vector PDF remain available. See FIGURE_QA.md. The immutable renderer manifest retains its generation-stage status; the separate visible QA ledger completes that step without altering server-export digests.

Two short SSH banner queries failed during the work; the active training connection remained live. They were transport handshakes, not evidence of invalid credentials or a reason to duplicate a job. No askpass script, credential publication or untrusted host-key bypass was used.

Only code/hashes and anonymous aggregates/figures/reports are released. Signals, identities, per-patient/channel predictions, weights, preprocessors/calibrators, mappings, caches and logs remain on the trusted server. No unrelated jobs or data were deleted, stopped or modified. All conclusions remain exploratory repeated-test; no post-Outer winner is selected.
