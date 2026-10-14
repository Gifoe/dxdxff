# D1 small-module exploration — Phase1 Seed42

| Model | New trainable parameters | AUROC | AP | Macro-F1 | EZ-F1 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 0 | 0.725978 | 0.503601 | 0.644546 | 0.443537 | 0.658502 |
| M1_SCALED_DGM | 9482 | 0.729874 | 0.515867 | 0.651390 | 0.457923 | 0.663778 |
| M2_STAGE_DELTA | 722 | 0.726325 | 0.504268 | 0.645883 | 0.447100 | 0.658502 |
| M3_DELTA_SET | 1354 | 0.726550 | 0.505422 | 0.646847 | 0.446867 | 0.674169 |

Fixed target: Macro-F1≥0.664546461935 (D1+absolute0.02). Phase1 quality-gated target met: **False**. Globally chosen candidate: `M1_SCALED_DGM`; this is exploratory candidate selection, never independently confirmed performance.

The scale fix is a new prospectively locked experiment authorized for exploration, not alteration of the completed A2 result. Original1e-6 floor collapsed native V-power dimensions; current1e-30 floor preserves the same FIT-only median-of-patient-median estimator and makes no unit/data/filter changes. No new physiology extraction. All candidates consume exactly the same corrected scale and source windows.

M1 is the old DGM topology retrained with corrected scaling. M2 adds only722 parameters on deterministic early/late DELTA statistics; no waveform CNN. M3 adds1354 parameters on a nonlinear window-set encoder; no waveform CNN, chronology, GRU, attention or graph. Both test whether all-window R14 pooling discards useful stage/distribution information. Their initialization exactly preserves frozen D1 final logits, and auxiliary loss starts the residual gradients. D1 weights/scalers and folds stayed unchanged.

## All targets and uncertainty

| Candidate | ΔMacro-F1 | ΔAUROC | ΔAP | ΔEZ-F1 | Positive MF folds | All quality gates |
|---|---:|---:|---:|---:|---:|---|
| M1_SCALED_DGM | +0.006843 | +0.003897 | +0.012266 | +0.014386 | 3/5 | False |
| M2_STAGE_DELTA | +0.001337 | +0.000347 | +0.000666 | +0.003563 | 3/5 | False |
| M3_DELTA_SET | +0.002300 | +0.000572 | +0.001821 | +0.003330 | 3/5 | False |

All10000 seed42 paired unique-ID draws retain65 appearances/47 IDs, thresholds/checkpoints fixed.95% intervals and secondary98.333% three-candidate primary intervals are descriptive and still condition on validation checkpoint/threshold selection and prior repeated Development usage. They are not independent significance or a correction for years of research reuse.

## Execution/scope

All15 predefined candidate-fold training cells ran; no poor-fold early truncation, extra seed, hyperparameter rescue or new Outer access. Own-fold legal FIT labels are used for training only; validation labels select checkpoint/threshold. Public files have no individual ID/channel/label or score records. All private data/weights/scales/predictions/logs remain server D:. Frozen residual0 is exact D1 logits but evaluated at each candidate threshold.

Sources: [Deep Sets](https://arxiv.org/abs/1703.06114) supplies the general permutation-invariant encoder/pooling idea; [FiLM](https://arxiv.org/abs/1709.07871) supplies general feature conditioning context. Neither paper establishes this clinical task’s incremental performance. This implementation and the measurements above, not external literature, determine the result.
