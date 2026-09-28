# Implementation audit

## Invariants verified

- Frozen v1 Teacher targets were numerically recomputed exactly (maximum absolute score error 0); their own scored channel labels were excluded from the relevant cross-fit head. AP eligibility was recomputed from the frozen v1 aggregates rather than hard-coded.
- Each v2 fold used exact A1 architecture, fold-specific A1 epoch-30 state, exact fold-FIT normalization, and patient-disjoint FIT Student-train/meta partitions for v2 updates and selection.
- Teacher and Student KD scores were standardized within each FIT patient; the deterministic pair sampler capped each patient/step at 128 pairs. D3b's reliability weights derived only from source FIT OOF Teacher-minus-A1 AP. Hard-label BCE remained patient-equal.
- The synthetic KD gradient and deterministic-pair unit test passed on the server (`STUDENT_KD_UNIT_PASS`). Thirty FIT selections were frozen, then thirty B=0 target score files and their checkpoints were hash-frozen before the target metric pass.
- Target evaluation replayed the historical A1 AP exactly: `0.5767434626151353`. Every model has 65 cells, 47 unique patient IDs and 1,300 fixed-query records. AP was estimable for 1,289/1,300 records. Paired CIs resampled the 47 unique patient IDs 10,000 times with seed 42.
- The planned endpoint was evaluated once. No KD variant passed the predeclared transfer gate; no outer/external test, post-outcome grid extension, new loss or model redesign was performed.

## Scope and limitations

- Historical v1 per-file hashes are absent; `hash-exact to historical v1 manifest` is **not verifiable**. Exact numerical replay plus a pre-v2 input manifest is the supported claim.
- The source A1 checkpoint had already used FIT-meta patients; the meta split protects against v2 target use but does not produce an independent holdout for source initialization.
- The historical D0 A1 VLOO reference used target-specific choices based on *other* validation patients' labels. The fold-shared D0+ is a matched KD control, not a fully identical no-update replay of that historical D0 procedure.
- FIT-only successive halving is computationally efficient but may discard a candidate whose benefit emerges after epoch 2 or 8.
- The negative finding is limited to the frozen exact-A1 Student, these Teacher targets, loss variants, grid and seed-42 development cohort. It does not prove that privileged information is universally non-distillable.
