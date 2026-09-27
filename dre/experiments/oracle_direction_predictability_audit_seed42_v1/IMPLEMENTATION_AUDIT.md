# Implementation audit

- Source branch is exact A1 tip `b2871b32873b67d0e6155d2766ef040ee1e9df01`; source lock and input/source hashes checked.
- All 150 A1 validation grids re-evaluated within 1e-6 before audit. Other-12 VLOO selects one source epoch per target. FIT and target representations use the same epoch.
- R3 is captured at the original patient-channel attention input, after source patient-relative normalization. R4 is captured at the original classifier input; classifier replay is exact to 1e-6. No approximate R3 reimplementation.
- FIT-only oracle directions, context PCA, SVD basis, scalers and predictors are rebuilt in each target's selected-epoch coordinates. Target labels are used only for the nondeployable target oracle, expressivity bound and evaluation after prediction.
- The original frozen A1 attention computes each patient's channel representations jointly. All unlabeled channels contribute to R3/R4; no target label is an input to context or direction predictors.
- The inherited monolithic cache initializer materializes labels for all 80 patients before role filtering. Literal no-outer-label-materialization is not met. No outer loader, prediction, metric or outcome-based model choice was made. This is exploratory development, not sealed confirmation.
- Patient-level cell records, direction vectors, teacher models and caches remain in the private runtime. There are 65 fold-by-patient validation cells but only 47 distinct patients. CIs use 10,000 paired patient-ID cluster resamples, preserving all repeated cells; see BOOTSTRAP_UNIT_AMENDMENT.json. FIT patient pairs are repeated across selected epochs and are not independent.
