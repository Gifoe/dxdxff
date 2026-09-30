# Engineering ledger

Only engineering repairs were made; none changed data membership, model, loss, hyperparameters, selection, threshold grid or predictions.

1. The source bank used explicit `window_relative_centers_sec`. Three short padded records were aligned by their real onset sample and center; no ordinal-time inference or synthetic window was used.
2. Fold jobs were moved to independent background tasks after a detached OpenSSH child was not persistent. Finished checkpoints/predictions were not discarded.
3. Diagnostic fixed-query cells can be single-class, so undefined private ranking metrics are serialized as `NaN` and aggregated with `nanmean`, matching historical semantics. Full frozen replay matched every score exactly (`max_abs_difference=0`).
4. The runtime-only fallback counter originally counted the universally absent LATE state as a fallback. The counter was corrected to count only nonempty states with fewer than three valid channels. The reference tensor and all predictions were unchanged; the actual low-channel fallback count is zero because each record has at least 40 signal channels.
5. The bootstrap was corrected to preserve the original 1,300-cell weighting when resampling patient clusters. Its point estimate now exactly equals the frozen aggregate.
