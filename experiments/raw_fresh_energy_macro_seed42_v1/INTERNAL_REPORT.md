# Actual fresh Raw internal factorial and frozen choices

This is the completed **internal** study, not a complete80-patient Outer
result and not an achievement of the >0.65 goal. Final FIT/Development training
was started only after the independent internal gate passed. Its now-completed
results are kept separate in [RUN_REPORT.md](RUN_REPORT.md): final primary
0.624365 and no fixed secondary >0.65.

## Executed and verified

- 42 genuinely random-start Raw neural fits, 1,062 epochs, 68 logical cells
  including fixed references, and 1,157 private patient metric rows.
- All selected states, random initializers, CAL selection and HELD scores were
  independently replayed. Maximum replay error: **0**. All nine patient metrics
  were independently checked against sklearn/ranking calculations.
- Four changed-critical-path tests passed. Actual largest complete training
  patient:164channels/48,380validwindows, allfourmodels with finite full gradients
  and nonzero backbone/head gradients. No admission optimizer update was made.
- No original Development fit or new Outer pass in this internal study.

The complete first-path matrix has **86 assessment appearances / 59 unique
patients per arm**, not 86 independent patients or the full80-patient test.
Repeated appearances are clustered together in10,000-patient-ID bootstrap draws.
Historical/overlapping internal paths remain selection-conditioned.

## Complete matched first-path results

| Configuration | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed Native | 0.601974 | 0.718595 | 0.517026 | 0.370762 | 0.833186 | 0.751410 | 0.738937 | 0.604651 |
| Fixed prior precompand body | 0.604982 | 0.723093 | 0.535747 | 0.382907 | 0.827057 | 0.745545 | 0.746988 | 0.627907 |
| WINDOW_MEAN | 0.600015 | 0.703966 | 0.491574 | 0.368451 | 0.831579 | 0.749116 | 0.722860 | 0.604651 |
| WINDOW_LOGENERGY | 0.599525 | 0.715647 | 0.503814 | 0.381216 | 0.817835 | 0.733184 | 0.721759 | 0.581395 |
| PRE_MEAN | 0.625949 | 0.751279 | 0.555566 | 0.408033 | 0.843866 | 0.765489 | 0.765498 | 0.662791 |
| PRE_LOGENERGY | 0.624857 | 0.751743 | 0.566049 | 0.427910 | 0.821804 | 0.746667 | 0.772248 | 0.697674 |

All four fresh arms have12,529parameters, matched random CNN/head weights and
the same legal data/optimizer/loss/budget. The historical references have a
different training history; matched fresh controls, not Native alone, isolate
normalization/pooling effects. There are no static physiological predictors,
D1 outputs, old embeddings or trained weight initializers in a fresh model.
[MODEL_SPEC.md](MODEL_SPEC.md) states the exact data flow and interpretation
boundary. The energy operator is fixed; the responses it pools are learned.

## Attribution, not a forced positive energy claim

| Patient Macro-F1 contrast | Difference | Nominal paired95% CI |
|---|---:|---|
| PRE_MEAN − WINDOW_MEAN | +0.025934 | [0.004402,0.048477] |
| PRE_LOGENERGY − WINDOW_LOGENERGY | +0.025332 | [-0.001669,0.055832] |
| WINDOW_LOGENERGY − WINDOW_MEAN | -0.000490 | [-0.015243,0.014106] |
| PRE_LOGENERGY − PRE_MEAN | -0.001093 | [-0.016009,0.015194] |
| PRE_MEAN − fixed Native | +0.023976 | [-0.000233,0.049421] |
| PRE_LOGENERGY − fixed Native | +0.022883 | [-0.004851,0.054175] |

The pooling-by-normalization difference-in-differences is -0.000602,
CI[-0.022651,0.022684]. Thus this first-path experiment does **not** establish
energy pooling superiority or a positive pooling/normalization interaction.
The clearer matched signal is the PRE normalization under mean pooling, not
simply a benefit from adding parameters (there are none) or training longer
(all fresh arms share that budget). These are nominal, unadjusted, reused-data
comparisons, not independent confirmation or clinical mechanism proofs.

## Replication and legally frozen procedure

The registered first-path Macro/AUROC gate advanced folds1,3,4,5. Only their
candidate and registered mean controls were run on two additional paths.
The `RACED_PATH_RESULTS.csv` table is a selected subset and is not substituted
for a complete six-arm three-path matrix; unrun cells remain unrun.

| OwnFIT fold | First candidate | Final selected procedure after its legal assessments |
|---|---|---|
| 1 | WINDOW_LOGENERGY | FROZEN_PRE |
| 2 | PRE_MEAN (did not advance) | NATIVE |
| 3 | PRE_LOGENERGY | PRE_LOGENERGY |
| 4 | PRE_LOGENERGY | PRE_LOGENERGY |
| 5 | PRE_LOGENERGY | PRE_LOGENERGY |

The positive first-path WINDOW_LOGENERGY result in fold1 was **not** retained
as the winner after replication. That is why a favorable fold cannot be
promoted to a global result. Each ownFIT chose only from its own legal
assessments; no global winner from overlapping folds was imposed on allOuter.

These five choices were hash-frozen before original Development access. The
final primary is this heterogeneous, legal selected procedure. Allfour fixed
fresh configurations and both fixed references are predeclared secondary
complete fivefold comparisons. A best-Outer secondary arm cannot replace the
primary after outcomes are seen.

## Engineering, provenance and next stage

Allfour initial states match within eachfold and differ from that path's old
trained references. No CAL-selected epoch0 was available: every retained fresh
checkpoint is an actually trained epoch>=1. Model/optimizer/RNG last-complete-
epoch checkpoints and all logs are preserved. No native/training restart was
needed. A local SSH empty-stderr pump timed out while the same server training
PID continued normally; that observation error did not trigger a restart.

The1062 formalepochs averaged2.021689s and totaled2147.033295s, excluding
admission, HELD inference, replay and orchestration. These are actual measured
training times, not an extrapolated whole-project ETA or fair cold/warm
microbenchmark. No unrelated GPU job was modified.

- Internal score seal SHA256: `7ba9c5fb3c1577fb7db42dd2bf4030687cb391757ce7bf3b5bf6c223f0ca461a`
- OwnFIT choice SHA256: `8c1cc33a6ae4644bd4e82acc926e16ff1aed371ed6c6d4502c79a84f7d017cc6`
- Independent terminal: `COMPLETE_VERIFIED_FRESH_ENERGY_INTERNAL` / `PASS`.

Final-stage source and `FINAL_PROTOCOL_LOCK.json` were committed before
original Development access. The unchanged fresh fitter is reused with an
explicit final-role progress adapter: its private generic `HELD` filename then
contains Development-only scores, not independent assessments. The actual
final plan fits20freshmodels, freezes35logicalmodel/thresholdcells, seals all
predictions, and only then opens the complete Outer truth archive. The
historical Outer remains exploratory repeated-test. No final metric is claimed
until that evaluation and independent verification actually finish.

Only code/hashes/anonymous aggregate artifacts are public. Raw, labels,
identities, weights, individual predictions, caches and runtime logs stay on
the trusted server. The original Raw-only reference0.630493 is a full80-patient
Outer value and must not be directly compared with this smaller internal
assessment as if their populations were the same.
