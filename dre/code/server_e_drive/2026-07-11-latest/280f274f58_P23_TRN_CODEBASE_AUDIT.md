# P23-TRN Codebase Audit

## P2 baseline inspected

The current P2/CANE-PATH-CP forward is `B0 -> physics dynamics -> masked temporal encoder -> cross-seizure aggregator -> channel classifier`. `direct_nez_logit` is the classifier logit on the patient-channel embedding. It is NEZ-directed: larger means more likely NEZ.

P2 receives `window_centers` in its batch, but the existing `ChannelTemporalEncoder` only consumes the window mask; its masked temporal mean does not use the center times. `seizure_channel_embedding` is `[B,S,C,D]`; the patient aggregator consumes it with `[B,S]` seizure and `[B,S,C]` channel masks.

The P2 anchor generates an NEZ-direction anchor residual. Its multi-seizure residual uses per-seizure NEZ scores and aggregation. The causal residual comes from six cached ridge-VAR proxies. In the P2 score path, all enabled residuals are directly added to `direct_nez_logit`; anchor and seizure bounds are `0.20`, causal is `0.15`.

The causal cache loader can provide validity, window coverage, seizure count, and stability quality fields. P2 checkpoint selection is validation thresholded F1 in the direct-outer-only runner, while the nested P2 runner trains a PATH threshold head. Neither behavior is used by P23.

## P23 changes

P23 keeps the P2 backbone and masked mean pooling. It adds a separate onset-aware temporal delta using real clipped `window_centers`, q10 tail evidence, and from P3 onward replaces direct residual addition with one bounded `direct_nez_logit + delta` fusion. P5 excludes causal evidence. P6 only accepts causal data with all quality fields.

P23 checkpoint selection is validation NEZ/EZ AUPRC harmonic mean. After a checkpoint is selected, a single global threshold is selected only from outer-train inner-OOF predictions. Test labels, true channel counts, center-specific thresholds, V3, raw waveforms, and PATH threshold heads are not input to the P23 prediction path.

P0 is deliberately delegated to the frozen P2 runner, preserving historical P2 code and results. P1-P6 share one P23 model path controlled by the profile object.
