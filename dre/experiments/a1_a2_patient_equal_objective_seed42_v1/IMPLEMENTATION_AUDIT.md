# Implementation audit

- Reuses the audited R0 `make_args` and fixed-fold builder from the prior HLV branch, with physics/HLV disabled.
- A0 calls the unchanged core `masked_bce` training loss. A1/A2 replace only the instance loss callback; model construction and forward path are unchanged.
- A1 uses EZ=2/NEZ=1 weighted BCE normalized within each active patient, then averages active patients. It does not call the repository's EZ/NEZ-balanced patient loss.
- A2 uses the same A1 BCE and the same final logits. Its coefficient is 0 through epoch 6, 0.025/0.05/0.075/0.10 at epochs 7/8/9/10, and 0.10 thereafter.
- A fixed fold-and-epoch RNG reset makes the patient shuffle and stochastic model operations matched across all variants. All parameters are copied from one dry-initialized fold state.
- Fit-only normalization and validation-only selection use the fixed manifest; the development runner never builds a test loader, imports prior outer results, or evaluates any test subject in its test role.
- All 30 epoch checkpoints and patient-level validation grids are private. The VLOO selector uses 12 patients to choose an epoch/threshold and scores the excluded thirteenth; full-validation selection is reported separately.

Loss unit tests and active fit/validation class support must be recorded and checked before training. The completed empirical checks and any engineering repairs should be appended below, without changing the scientific lock.
