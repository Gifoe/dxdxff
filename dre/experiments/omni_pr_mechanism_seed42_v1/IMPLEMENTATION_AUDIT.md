# Implementation and execution contract

## Matched head capacity and optimization

Every correction is the unchanged historical `Head('C1')`: beta[32] and source bias[3], 35 trainable parameters, zero initialization, no U/V, MLP, attention or CNN update. Both new controls invoke the unchanged authoritative historical trainer: AdamW 0.001/0.0001, clipping 1.0, weighted BCE 2:1 normalized within patient, one step per patient, beta L2 0.001, source-count-weighted bias L2 0.001, float32 CPU and deterministic historical patient ordering. Fitted normalization moments are statistics, not additional trainable parameters.

Formal budget is exactly 30 inner runs (three per arm per outer fold), all 50 epochs, and 10 reset full-FIT refits at the inner-selected epoch. FIT-only engineering smoke runs are separate from this formal budget. Existing checkpoints, optimizer history and inner predictions enable exact deterministic resume, not a second optimization search.

## Transformations

`transforms.global_moments` uses two-pass float64 patient-equal global moments, including between-patient variance. Each ABS inner scaler excludes both inner validation and outer held; each refit scaler uses full outer FIT only. The transform uses epsilon=1e-5 and is cast to float32 for the authoritative head.

PR normalization reproduces the historical float32 transform exactly over all observable good channels. SHUFFLE applies one deterministic label-blind derangement to these normalized vectors per patient/fold/partition role. It leaves base logits, labels, channel order and source unchanged. The inference-only shuffle uses the same prelocked held permutation with the old PR beta; it is distinct from the newly trained shuffle head.

## Selection, freeze and metrics

All 50 epochs receive complete inner head-OOF patient metrics. Selection maximizes patient AUROC, then AP, then earlier epoch. The unchanged 201-point threshold selector uses patient Macro-F1, pathological F1, balanced accuracy, distance to 0.5 and smaller threshold. Old C0/PR selections are never recomputed.

All ten new checkpoints, 20 ABS scalers, 620 permutations, thresholds, executable hashes and historical replay hashes are frozen before any new outer score. New outer/diagnostic predictions are atomically private-sealed and immutable. Ranking and classification use unchanged historical metric implementations. FIT-selected and fixed-0.5 classification are separate columns. Bootstrap draws resample the same 124 unique patients jointly across methods, 10,000 draws/seed42 with fixed numeric thresholds; exactly five registered contrasts are reported.

## Mathematical intervention

For fixed beta, variance and patient, subtracting mean changes the corrected logit by only `beta dot (mean/std)`, a patient-constant offset. The scale-only audit compares direct float64 formulas and pairwise ordering, verifies all five ranking metrics, and records any difference from operational historical float32 normalization or probability saturation. Classification changes at old thresholds are reported separately. A PR-vs-ABS training comparison does not causally separate variance scaling, centering-induced fitted beta, optimization conditioning and interactions with the frozen CNN base.

## Integrity and publication

`tests/engineering.py` performs the 36 required pre-evaluation checks, synthetic tests, ten real FIT-only smoke runs and exact resume verification. `tests/completion_verify.py` verifies all training/selection bindings, all population and prediction hashes, historical immutability, bootstrap counts, gates and public aggregate schemas. A hash allowlist permits retrieval of compact audits/results only. Patient/electrode identifiers, private patient rows, clinical labels, embeddings, checkpoints, permutations, optimizer state and runtime logs are never published. Source and executable hashes are locked before the new outcomes; no post-result tuning or rescue is permitted.

The first post-run completion check rejected legal non-applicable CSV metadata: the all-center diagnostic aggregate has an empty center, and SHUFFLE has no global-scaler SHA. The checker was repaired to allow exactly these two explicitly validated cases while continuing to reject every missing/non-finite metric. The pre-repair checker SHA was `9e63fe978ae77d28160feb7a4f0d10c530ae8e379f9105ad9d67ead959f72787`, retained on the server. No training/evaluation/report executable, protocol, input, checkpoint, prediction, threshold or result artifact changed, and neither training nor outer evaluation was rerun.
