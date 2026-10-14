# D1 small-module exploration — Phase3 and complete candidate table

| Model | New parameters | AUROC | AP | Macro-F1 | EZ-F1 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 0 | 0.725978 | 0.503601 | 0.644546 | 0.443537 | 0.658502 |
| M1_SCALED_DGM | 9482 | 0.729874 | 0.515867 | 0.651390 | 0.457923 | 0.663778 |
| M2_STAGE_DELTA | 722 | 0.726325 | 0.504268 | 0.645883 | 0.447100 | 0.658502 |
| M3_DELTA_SET | 1354 | 0.726550 | 0.505422 | 0.646847 | 0.446867 | 0.674169 |
| M4_FEATURE_CORRECTION | 257 | 0.726044 | 0.504661 | 0.644774 | 0.443991 | 0.656817 |
| M5_STAGE_CORRECTION | 721 | 0.733558 | 0.527209 | 0.653227 | 0.456208 | 0.720722 |
| M6_MORPH_CORRECTION | 9481 | 0.753007 | 0.546300 | 0.661629 | 0.471723 | 0.717188 |
| M7_MARGIN_GATED | 9483 | 0.751264 | 0.546727 | 0.661878 | 0.464605 | 0.732848 |
| M8_MACRO_AWARE | 9481 | 0.756633 | 0.542226 | 0.666795 | 0.477249 | 0.699708 |

Target Macro-F1≥0.664546461935: point target met `True`, complete quality gate met `True`. Selected common candidate `M8_MACRO_AWARE`: Macro-F10.666795, D1 gain+0.022248. This is adaptive repeated Development, not new Outer or independent clinical confirmation.

Forty candidate-fold training cells were actually completed across three prospectively frozen phases. Phase3 was registered after all six previous models/results; it is not a pre-independent eight-model confirmation. No outcomes were overwritten, poor folds excluded or best architectures picked independently by fold. All new variants use the same frozen D1 checkpoints and preprocessing, same masks/Raw/DELTA/scales, original FIT weights and patient-equal optimizer/schedule/validation selection.

M7 adds exactlytwo zero-initialized scalar parameters to the measured direct-error morphology branch. Its conditional multiplier uses tanh(abs(D1logit)), which is an uncalibrated magnitude, **not** certified uncertainty. Its inference-only margin-identity intervention separates dependence on this added gate from physiology-gate identity. M8 adds no new architecture parameters and uses fixed.75 weightedBCE+.25 patient SoftMacroF1; actual FIT labels/counts participate only in the supervised objective, never model inference. No separate threshold is optimized inside the training loss. Both retain initial/final residual-disabled exact D1 identity.

Selected-checkpoint/threshold bootstrap:10000 seed42 paired unique-ID draws retaining65 appearances/47 IDs.95% intervals are unadjusted and selection-conditioned, not a correction for threshold/epoch/candidate selection or years of prior validation reuse. Report all AUROC/AP/EZF1/sensitivity tradeoffs and fold/center tables; a target point improvement does not prove transport to new patients.

Only source/tests/hashes/anonymous aggregate reports are public. Actual legal-FIT gradient tests preceded new candidate training and their smoke states were discarded. All private weights/scalers, IDs/labels, waveforms/window banks, per-channel predictions and logs stay on serverD. No installed skill, new Raw extraction, additional seed or Outer access.
