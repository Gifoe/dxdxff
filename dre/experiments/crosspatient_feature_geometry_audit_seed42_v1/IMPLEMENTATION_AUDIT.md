# Implementation audit

- Source A1 tip and protocol hashes checked at runtime. All 150 checkpoint grids replayed exactly.
- The 28-name feature inventory was committed before richer-feature label evaluation.
- All FIT/validation rows and selected-epoch embeddings remain in private runtime; public files are fold or representation aggregates.
- Source four-view function is reused for all 28 descriptors by temporarily substituting its feature-index selector; RICH9 aggregate replay checks RAW72 numerically.
- Local probes use one fixed stratified cross-fit and evaluate AP/AUROC within heldout folds, weighted by heldout channel count. No cross-model whole-patient MRR or Top1 is computed.
- PCA fits on FIT patient-z channels only. Source and target patients are disjoint in all five folds.
- Random-label control reduced but did not reach prevalence AP; see SANITY_AUDIT.json.
- No outer-test loader or metric is used by these scripts, although the legacy source cache initializer materializes all 80 patient records.
- This is a protocol deviation from the literal no-read-outer-label rule. Do not label the run as strictly outer-sealed.
