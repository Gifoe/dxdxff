# Next controlled candidate: branch-wise nonlinear PR residual

Draft execution package; not yet admitted, trained or promoted. Current PR-component factorial remains running under its original frozen source.

## Outcome-free motivation

The actual largest-TRAIN-patient admission of the preceding factorial measured12000 raw-summary coordinates. With Raw PR, sum-of-squares was11999.999976; without Raw PR it was0.308121911. Thus the RMS ratio was about197.35. This is an **initialization/input-scale** observation, not a classification result or a general estimate over all patients. Straight PR deletion confounds relative comparison with approximately two orders of magnitude of input-scale change. A performance drop cannot be assigned wholly to contextual information.

The previous temporal-PR study failed its fixed local-versus-global gate. This candidate does not reopen that study, change its gate, borrow a better control, or assert that timing-specific comparisons work. It keeps the original six-feature framework and tests one different, fixed alteration: preserve the old single-head hidden activation and add a small, zero-initialized branch-wise relative residual **after a nonlinear transform**, before the same final classification layer.

## Fixed candidate, no architecture search

Let original raw/physiology first-layer activations be `a` and `b`, and original hidden state `h=GELU(a+b+bias)`. Define `u_raw=GELU(a+bias/2)` and `u_feature=GELU(b+bias/2)`.

`h_new = h + 0.25*tanh(g_raw)*norm_raw(u_raw) + 0.25*tanh(g_feature)*norm_feature(u_feature)`.

Each gain is64-dimensional and initialized to exact zero. No second classifier is added: the hidden width remains64 and the original final scalar Linear is retained. Extra capacity is128parameters, identical in all five new-model variants. Full initialization reproduces the old network function; shared trained weights are never used to initialize formal new fits.

When branch PR is on, keep its original summary patient Z and use coordinate-wise patient Z for its nonlinear residual. When off, remove both patient-Z operations in that branch, keep all its inputs, and use a channel-wise64D LayerNorm without learned affine parameters for the new residual. This control never compares different channels. Feature PR off uses the existing TRAIN-only global imputer/scaler. Raw PR off uses a coordinate-wise global affine preconditioner fitted on **all TRAIN channels under the matched initial encoder**, before any optimizer update. That preconditioner is identical across variants for a given fold/seed and frozen, not described as a current-encoder population reference. All fitting inputs are unlabeled TRAIN waveforms; no CAL/ASSESS/Outer input is used. Finite/constant-coordinate rules are fixed before admission. This reduces gross scale sabotage but is not exact distribution matching during optimization.

Capacity-matched `CHANNEL_CONTEXT` retains original summary PR but substitutes channel-wise LayerNorm for both new residual patient Z operations. It tests whether the additional nonlinear residual helps without its new patient context. This is not the same as removing all existing PR.

Proposed arms: originalB0; newFULL; newWO_FEATURE_PR; newWO_RAW_PR; newWO_BOTH_PR; newCHANNEL_CONTEXT. Three seeds42/43/44, original five within-FIT TRAIN/CAL/ASSESS roles,15 exact original internal baseline reuses and75 fresh fits. Original EarlyFiLM, lateGate, pre-reference, six features, full differentiable pooling, waveform normalization and loss/budget remain unchanged in this first comparison. Subsequent core-module ablations are still required if a candidate qualifies; this package alone cannot establish every retained module's contribution.

## Selection and interpretation

No new Outer is permitted by this draft. Before training, complete executable source/tests, bind the previous completed freeze/seal, lock exact preprocessing/fallback/contrast/gate rules, push GitHub, and admit real complete-patient numerical/gradient paths. All candidate fits must be frozen and scored together. Three seeds, unique-patient metrics, all nine endpoints, paired intervals and source/fold summaries are required. Compare only matched internal estimates, not0.62 internal against0.65 historicalOuter.

Only an internally qualified candidate should enter a separately fixed final five-fold study and core-module ablations. Final success still requires actual80-patient five-fold Macro-F1>0.64, improved AUROC and reproducible, fairly controlled contributions; the current draft or its initial identity does not prove any of these. Historical cohort exposure means exploratory repeated-test throughout. No best-Outer selection, threshold editing, manufactured ablation failure or hidden negative result.
