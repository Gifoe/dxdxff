# Frozen D1 small-module exploration — Phase2 direct correction

| Candidate | New parameters | AUROC | AP | Macro-F1 | EZ-F1 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 0 | 0.725978 | 0.503601 | 0.644546 | 0.443537 | 0.658502 |
| M1_SCALED_DGM | 9482 | 0.729874 | 0.515867 | 0.651390 | 0.457923 | 0.663778 |
| M2_STAGE_DELTA | 722 | 0.726325 | 0.504268 | 0.645883 | 0.447100 | 0.658502 |
| M3_DELTA_SET | 1354 | 0.726550 | 0.505422 | 0.646847 | 0.446867 | 0.674169 |
| M4_FEATURE_CORRECTION | 257 | 0.726044 | 0.504661 | 0.644774 | 0.443991 | 0.656817 |
| M5_STAGE_CORRECTION | 721 | 0.733558 | 0.527209 | 0.653227 | 0.456208 | 0.720722 |
| M6_MORPH_CORRECTION | 9481 | 0.753007 | 0.546300 | 0.661629 | 0.471723 | 0.717188 |

Target patient Macro-F1≥0.664546461935 (D1+absolute.02). Any point target met: False. Any complete quality gate met: False. Current globally selected candidate `M6_MORPH_CORRECTION`, gain+0.017083. These remain adaptive repeated validation results, not independent confirmation.

Phase2 was prospectively frozen after observing Phase1; prior models/results remain unchanged. It tests a different training mechanism: zero last-output initialization makes step0 exactlyD1, while direct weightedBCE(final) trains corrections to the frozen D1 rather than a competing standalone auxiliary classifier. The fixed residual cap is1.0; no alpha needs to leave zero. This changes both correction parameterization and objective deliberately; it is not an isolated proof of which change caused a gain.

M4 adds257 parameters on original preprocessed R14; M5 adds721 on exact Phase1 stage-DELTA28; M6 adds9481 on the unchanged morphology/gate path. Original D1 weights/preprocessing, original folds, FIT weights, patient-equal updates, optimizer/schedule/validation selection and corrected FIT-only scales remain fixed. M4 adds no new information source: any gain is capacity/error-correction, not physiological discovery. M5/M6 use the existing pinned Raw250-derived physiology; no new extraction/filter/frequency/onset proof. All models and failed gates are reported.

Actual legal-FIT three-update gradient tests precede each new candidate, then the smoke model is discarded and seeded construction reset. Zero last-layer initialization means first hidden/CNN gradients are expected zero on step1, gate output starts after head movement, and the earlier gate layer follows after its output moves. D1 has no gradients, always eval and its state hashes remain exact. Residual0 logits exactly equalD1 but use the candidate-selected threshold in interventions.

10000 paired seed42 unique-ID draws retain65 appearances/47 IDs with thresholds fixed. Intervals condition on checkpoint/threshold/candidate selection and the repeated Development history. No fresh Outer inference, extra seeds, hidden configurations, patient/fold exclusion or per-fold best-model assembly. Raw/labels/IDs/weights/predictions/logs stay server D; only code/hashes/anonymous aggregates are exported.
