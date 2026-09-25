# P2.1 V3-ASRR Sensitivity80 Report

## Status

Implementation and synthetic validation are complete. No P2.1 fold training result is reported in this document.

Sensitivity80 is a post-hoc cohort. Results from `precomputed_oof_screening` are structural screening only and are not formal nested estimates. Only `nested_fold_safe` runs with passed per-fold V3 isolation audits are eligible for formal reporting.

## Method

P2.1 uses a continuous NEZ-oriented anchor, patient-wise robust median/IQR standardization, clean-NEZ prototype evidence, q10 multi-seizure evidence, deterministic quality-gated ridge-VAR proxy evidence, and a four-way simplex gate. The final correction is bounded to `+/-0.20` and the three legacy P2 residuals are not added again.

Training combines patient-balanced classification, reliability-weighted observed-EZ pairwise ranking, anchor ranking preservation, direct-head auxiliary supervision, clean-NEZ prototype regularization, and optional patient-pooled clean-NEZ center moment alignment. Center identifiers are used only by the training loss and diagnostics, never by model forward or thresholding.

Checkpoint selection uses continuous ranking metrics and does not optimize a threshold. A single fold-global threshold is selected only after checkpoint fixation using validation data or outer-train inner-OOF predictions. Test labels, patient oracle thresholds, and true channel counts are never used for formal predictions.

## Required Interpretation

- Patient oracle, fold-global oracle, and true-count-assisted metrics are diagnostic and not deployable.
- Precomputed global OOF screening is not equivalent to strict nested isolation.
- Ridge-VAR propagation features are proxies and do not establish causality.
- Counterfactual component ablations are inference-only and are not substitutes for retrained R0-R5 ablations.
- The implementation does not guarantee macro-F1 above 0.70.

## Results

Not run. Populate this section only from generated `p21_*` ledgers and preserve each run's `analysis_status`.
