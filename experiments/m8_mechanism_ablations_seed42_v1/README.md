# M8 mechanism ablations: A1–A4 and A6, seed42

Four new correction-model ablations plus one validation-selected late fusion, on the exact original five Ictal patient-held-out folds. See `PROTOCOL_LOCK.json` for the prospective contract. These are full retraining ablations, not posthoc inference interventions on trained full M8.

| Arm | Feature Patient Z | Raw96 Patient Z | Physiological gate | Raw pre-onset latent subtraction | Combination |
|---|---|---|---|---|---|
| Full M8, frozen reference | yes | yes | learned | yes | D1 logit + bounded learned correction |
| A1 w/o Feature PR | **no** | yes | learned | yes | same M8 residual structure |
| A2 w/o Raw PR | yes | **no** | learned | yes | same M8 residual structure |
| A3 w/o Physiological Gate | yes | yes | **parameter-free ones** | yes | same M8 residual structure |
| A4 w/o Pre-ictal Reference | yes | yes | learned | **no** | same M8 residual structure |
| A6 Simple Late Fusion | original D1 PR | original standalone Raw-only PR | none in Raw-only | yes in Raw-only | convex probability average of independent models |

A1 reuses the previously trained, hash-bound, matched **CompactDelta1537 DELTA14 no-PR** base and its legal FIT imputer/scaler; the Raw correction is newly trained. It is not a sklearn MLP or the old experiment that removed both PR sites. A2 retains ordinary per-window amplitude normalization; it removes only the cross-channel Z on pooled raw96. A4 removes only Raw latent reference subtraction; the seven physiological DELTA coordinates and their established reference definitions remain unchanged. A3 keeps the physiological D1 branch, removing only the physiological conditioner on raw morphology.

All correction CNN/head initializations match the original M8 fold initializer. A3 constructs the original model before removing gate parameters, preserving initializer RNG consumption. All original masks, genuine windows, channels, available-seizure pooling, class-weighted patient updates, FP32 precision and epoch/early-stop budgets remain fixed. The original frozen D1 base remains in evaluation mode during correction training.

A1, A2 and A4 have 9,481 trainable correction parameters; A3 has 8,433 because its identity gate has no parameters (1,537 frozen D1 parameters are additional in each). Thus A3 measures removal of the learned conditioner as requested, not a parameter-count-matched substitute network. Removing normalization or the reference changes representation scale under the same optimizer budget; the ablations do not receive compensating posthoc learning-rate or threshold rescue.

A6 uses the independently trained M8 Raw-only model already completed in `../m8_raw_only_seed42_v1`, **not** a residual output from a jointly trained full M8. Both component checkpoints are frozen. Each fold searches 21 predefined raw weights (0 to 1 by 0.05) and selects its original validation-only threshold, breaking ties by Macro-F1, EZ-F1, then smaller raw weight. Test probabilities, labels or metrics never select that weight. No new component fit or logit averaging is substituted after seeing outcomes.

## Reproduction and gates

The trusted server contains the pinned historical private artifacts needed by this additive experiment. It is not a standalone data release. Scientific source hashes are frozen before the first training epoch; candidates/epochs/thresholds are frozen before the new Outer pass. Upstream checkpoints, source and all existing results are read-only. Private atomic per-patient inference caches permit engineering recovery with the same freeze only.

1. Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` to 1 in the original server training environment. Import NumPy/SciPy/sklearn before PyTorch to avoid duplicate Intel OpenMP runtimes; never set `KMP_DUPLICATE_LIB_OK`.
2. Run `smoke.py`: synthetic CPU/GPU feature/gradient equivalence, identity-gate delta independence, no-PR channel-subset independence and pre-reference removal checks.
3. Run `runner.py`: hard-gated cohort/source/cache inputs; real whole-FIT-patient forward parity and finite/nonzero gradient admission; 20 correction fits; five validation-only fusion selections; one frozen/sealed five-fold Outer pass; aggregate metrics and independent verification.

The first synthetic process exited on an OpenMP import-order conflict before cohort access. Scientific libraries were moved before torch. A second synthetic run passed CPU checks but showed non-bitwise GPU backward equality; the final outcome-free check measures both new-vs-original and original self-replay drift. CPU forward/gradients and GPU forward are exact; GPU backward uses the explicit fixed FP32 tolerance and original-self-replay comparison in the protocol. No cohort training had occurred during these admission repairs. Prior source copies and logs remain private on the server.

Reported primary values are patient-macro means across the original 80 patients, not channel-pooled scores or equal-weight averages of unequal-sized folds. Separate outputs cover pooled channels, folds, centers, confusion counts, metric validity, ranking (MRR/Top1), NEZ-F1, accuracy and signed EZ-fraction bias. Paired patient-bootstrap intervals use 10,000 draws with seed42 and fixed fitted models/weights/thresholds, with no multiplicity correction or training-seed uncertainty.

All results are **EXPLORATORY_REPEATED_TEST** because this cohort was previously viewed. No installed research/extraction skills are used. Only source, hashes and anonymous aggregate tables belong in the repository; no raw EEG, fitted weights, individual features/predictions/labels, identities, scalers, cache or runtime logs may be published.
