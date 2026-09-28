# Implementation audit

Validation: **PASS** (`outputs/VALIDATION.json`).

- Frozen source split SHA-256: `e329bab57037d649a1995074a7e5ee31c7c002f00ea6f0ee54bd48d67be1dc42`. Train/test patient assignments unchanged.
- Actual model parameter count: 27713; exact historical A1 source hash checks ran in training/inference.
- One 30-epoch inner AP-selection run; best epoch 9. Validation Macro-F1 used only after checkpoint selection to choose threshold 0.515586.
- Full 141-patient train refit used selected epoch count. Frozen model/normalizer/threshold/protocol hashes were checked before official test inference.
- Final official cohort: 96 patients, 174 EDFs, 8,104 labeled EDF-channel records; 807 pathological, 7,297 normal. Unknown labels excluded from loss/evaluation but all good signal-present channels contribute to label-free cross-channel reference.
- Test selection/tuning: **NO**. A test-derived Youden threshold is flagged posthoc and excluded from the primary claim.
- D1 compares only v1-score-observable official channels; full label-only effect is unidentifiable. D2 shares the identical v2 scores at two thresholds.
- Source/cache and checkpoint tensors remain private. Committed outputs are compact audits and pseudonymous score rows only.
