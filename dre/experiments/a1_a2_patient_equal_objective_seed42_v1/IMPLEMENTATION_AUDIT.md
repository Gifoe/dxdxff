# Implementation audit

- Reuses the audited R0 `make_args` and fixed-fold builder from the prior HLV branch, with physics/HLV disabled.
- A0 calls the unchanged core `masked_bce` training loss. A1/A2 replace only the instance loss callback; model construction and forward path are unchanged.
- A1 uses EZ=2/NEZ=1 weighted BCE normalized within each active patient, then averages active patients. It does not call the repository's EZ/NEZ-balanced patient loss.
- A2 uses the same A1 BCE and the same final logits. Its coefficient is 0 through epoch 6, 0.025/0.05/0.075/0.10 at epochs 7/8/9/10, and 0.10 thereafter.
- A fixed fold-and-epoch RNG reset makes the patient shuffle and stochastic model operations matched across all variants. All parameters are copied from one dry-initialized fold state.
- Fit-only normalization and validation-only selection use the fixed manifest; the development runner never builds a test loader, imports prior outer results, or evaluates any test subject in its test role.
- All 30 epoch checkpoints and patient-level validation grids are private. The VLOO selector uses 12 patients to choose an epoch/threshold and scores the excluded thirteenth; full-validation selection is reported separately.

Loss unit tests and active fit/validation class support must be recorded and checked before training. The completed empirical checks and any engineering repairs should be appended below, without changing the scientific lock.

## Completed checks

- Protocol lock SHA-256: `6694da1b351d7130015661fcce1ec91bec643821392e409f1d5728f472b3aba0`. It was pushed before training.
- All six synthetic objective checks passed, including exact batch-one equality with R0 masked BCE, channel duplication/permutation invariance, equal weighting of 50- and 150-channel patients, finite gradient, and fixed beta schedule.
- The fit/validation class-support audit found zero patients missing EZ or NEZ in all five folds. No test dataset was built in that audit.
- A0/A1/A2 have exactly `27,713` parameters per fold. All variants load the identical fold initial state; model construction and inference are unchanged.
- All `15 × 30 = 450` epoch checkpoints and validation grids completed in private runtime. Each validation grid was checked against the audited core patient metric implementation at threshold 0.5. VLOO produced 13 excluded-patient estimates per fold and variant; aggregate outputs contain no IDs.
- A1 and A2 fold-1 training losses are exactly equal through epoch 6, when beta is zero. A2's logged beta reaches 0.025/0.05/0.075/0.10 at epochs 7/8/9/10 as locked.
- A finite-gradient/model-parameter check was added as an engineering assertion after protocol freeze; it did not alter any scientific setting or loss. The development report was expanded to show all requested aggregate metrics. No observed result was used to change the method.
- `TEST_READINESS_GATE.json` is `pass=false`. No current outer-test outcome was read by this development code. Seed52/62 and outer evaluation were not run.
