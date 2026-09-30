# Leakage audit

- Only the frozen Ictal `fit` and `validation` memberships were opened by a fold trainer. The manifest's `test` role was not loaded for training, selection, or metrics.
- Per-frequency median/IQR scalers were fitted exclusively on each fold's FIT patients.
- Patient references used only synchronized signal-valid channels; labels, outcome, resection, center and patient identity were not inputs.
- Each validation target's patience stop, epoch and threshold were selected from the other 12 validation patients. The target label did not affect its own selection.
- All 40 validation snapshots were retained so physical training termination could not leak the current target into its candidate epoch range.
- The two diagnostics reused frozen checkpoints and thresholds and did not retrain.
- No outer heldout result and no Omni record was accessed. Test-driven iteration did not occur.
