# PLLR implementation and execution audit

## Frozen scientific settings

See PROTOCOL_LOCK.json. PCA8 (full SVD and FIT projected population standard
deviation), beta=.5, support5/class, median positive conflict tau, class-normalized
weights and epoch1 activation are not selected by outcomes. B1 uses canonical
channel-ID order and SHA-derived patient/class RNG seeds. Only complete
unchanged permutations in nonconstant groups are rejected. Each realized
patient/class multiset is exactly preserved; unchanged channels are audited.

The prompt did not quantify extreme instability or substantial worst-center
deterioration. Before FIT audit/validation, we fixed every-fold median patient
mean bootstrap rank correlation >=.5, every-fold eligibility >=30%, and worst
center nondecline. These are conservative operational gates, not clinical
validity or statistical-significance criteria. Prototype bootstrap uses100
resamples, always excluding the query; its tau is the original FIT tau, not
re-estimated to optimize stability. No prototype or label enters inference.

## Reused, independently gated utilities

The immutable PR-UAS `uas_core.py` is imported only after exact SHA verification.
It supplies the exact network, patient-z/imputation/scaling, threshold and metric
rules, seed/RNG and atomic checkpoint utilities. The new weighted trainer does
not call soft_target, teachers, lambda schedules or smoothing. Both classes keep
their original binary labels and positive BCE weight.

Historical outer checkpoint replay is the sole required outer-data exception.
The already verified private source export contains all80 labels, which the
sealing/replay stage materializes. After that, geometry and formal training read
only per-fold FIT/VAL banks, never outer channels or labels. Do not claim the
original all-cohort export was label-free. No new outer evaluation is authorized.

## Pretraining engineering correction

Initial local synthetic tests caught an exact-gradient mismatch: computing
unreduced BCE and then an external mean changes the floating-point backward
reduction order by tiny rounding. Use the native BCE `weight=` argument with
the original mean reduction instead. It is mathematically the prescribed mean
of weighted channel losses; all-one weights now reproduce original A0 loss,
gradients and complete two-epoch synthetic parameter hashes exactly. This was
fixed before any real formal training, without changing the objective or model.
All6 local synthetic test groups pass. The constant-feature PCA test produces
the expected undefined explained-variance ratio warning; weights remain all1,
finite PCA coordinates and zero stored variance ratios are explicitly checked.

## Runtime and preservation

Use the prior isolated admitted Python3.12/Torch2.11+cu128 environment unchanged.
It previously suffered native faults with no established cause; this task does
not assert those are resolved or modify drivers/shared environments. Each
attempt uses new log names; each completed epoch preserves optimizer, RNG,
best/last states and source/data/weight/protocol binding. No duplicate GPU task,
raw extraction, discarded completed result or outcome-driven repair is permitted.

Execution outcomes and any later engineering corrections are appended here.

## Completed execution

Fresh historical replay passed all five checkpoints (<1.49e-7 max score drift).
All five fresh matched A0 selections reproduced the prior PR-UAS development
protocol (<2.98e-8 probability drift). Every fold shared the exact initial model
hash across A0/B1/B2. Server synthetic tests passed all6 groups; an actual
three-FIT-patient one-epoch original-vs-all-one-weight smoke produced identical
model parameter hashes and zero drift. Weighted B1/B2 gradients were finite.

FIT eligibility and100-resample stability passed all five pre-outcome gates.
All15 matched training runs completed in one formal-training attempt, with
130 completed epochs (A0:40, B1:44, B2:46), epoch histories, best/last checkpoints,
optimizer and RNG states retained privately. Formal training stderr was empty;
no completed epoch or run was discarded. The train log's creation-to-last-write
interval was13.132 seconds; that is not total task time including preprocessing,
feasibility, replay, smoke checks and statistical analysis.

All15 selected artifact bindings passed independent finalizer integrity checks.
Development comprises65 validation appearances/47 unique patient IDs per arm.
The paired10,000-draw seed42 bootstrap clusters repeated appearances by unique
patient ID. Five of six mandatory gates fail; no new outer evaluation follows.

## Statistics-only native interruption and exact retry

The first finalizer suffered a native Pandas access violation in Series.abs /
make_block during decision diagnostics, after writing summary, fold, center,
paired-bootstrap and integrity aggregates. Original logs and those completed
files were preserved in a separate private attempt backup. There was no model
training or inference in the finalizer. The root cause of this native fault is
not established; it is not asserted to be fixed.

One retry of the exact same finalizer command completed with empty stderr.
No code, source protocol, weight, prediction, selected checkpoint, threshold or
bootstrap seed changed. All five already-written artifacts were independently
verified byte-identical to their preserved first-attempt copies:

| Artifact | SHA256 before and after retry |
|---|---|
| VALIDATION_SUMMARY.csv | 77e2b3087bd89179ac21f8b21e18758d8354d046770bb4705d3cc6f80e931540 |
| VALIDATION_BY_FOLD.csv | 8314ccba9f5a9a2cfa8b429b716948d36cf1f624c22f072b0e455b437c9abce0 |
| VALIDATION_BY_CENTER.csv | aa65be1557cad175dd45781ba0f31ed018ba6a32bd3c35d29542e083f2b50ea8 |
| PAIRED_BOOTSTRAP.csv | 47d9f8a70c6818c721d9934fdfe2a7fe5189ba58d5de10a698df9e820e01cb67 |
| ARTIFACT_INTEGRITY_AUDIT.json | dfd2f98e49f84c452b5533f6f7d87edf8867e11ec5fdaad814d15ba6683f3e8d |

The frozen protocol SHA256 remains
`6f32cf49b9d34ff55406d9ca58374edba676603bd64bf1b62f38498a3d141c76`;
the executed finalizer SHA256 remains
`e6ecf4602a63a35f34c25015558b407e2e2aa1ce0c89db7f61f6b8b5782b0599`.
Only compact aggregate outputs and source are published. Raw/private data,
individual predictions, learned artifacts and runtime logs are not uploaded.
