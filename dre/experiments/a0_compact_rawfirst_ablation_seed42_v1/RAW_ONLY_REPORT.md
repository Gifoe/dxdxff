# Raw-only versus complete A0-compact — completed first

Completed 2026-10-14. Stage terminal: **RAW_ONLY_EVALUATION_COMPLETE_ABLATIONS_PENDING**. This completed report is published before any F1–F5 feature training; their entire matrix was frozen before new raw outcomes.

## Table A — separate input/model baseline

Primary outer is equal mean over80 unique held-out patients, each once in the original five folds. It is not pooled-channel performance or an equal mean of unequal fold means.

| Input/model | Parameters | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR | Top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| F0 complete engineered A0-compact, historical verified reuse | 8817 | 0.736977 | 0.530653 | 0.624236 | 0.414671 | 0.694267 | 0.587500 |
| R0 new waveform-only RawTiny + A0-form head | 18153 | 0.633626 | 0.408496 | 0.543958 | 0.321269 | 0.588369 | 0.475000 |

F0 is masked88D with62 retained ABS/DELTA/RATIO columns, native input LayerNorm, no Source. R0 uses raw waveforms only, no engineered coordinates or scaler. It is not the historical RawTiny A1 experiment and no historical RawTiny score is inserted here.

## Full outer metrics and matched effects

Direction is **F0 minus R0**. 10,000 common paired80-ID bootstrap draws, seed42, numeric thresholds/checkpoints fixed; unadjusted percentile95% intervals.

| Metric | F0 | R0 | F0−R0 | 95% CI |
|---|---:|---:|---:|---:|
| AUROC | 0.736977 | 0.633626 | 0.103351 | [0.054102, 0.152366] |
| AUPRC | 0.530653 | 0.408496 | 0.122156 | [0.065015, 0.179153] |
| Macro-F1 | 0.624236 | 0.543958 | 0.080277 | [0.043348, 0.116817] |
| EZ-F1 | 0.414671 | 0.321269 | 0.093401 | [0.028778, 0.155591] |
| NEZ-F1 | 0.833800 | 0.766647 | 0.067153 | [0.036213, 0.100178] |
| Balanced accuracy | 0.663990 | 0.574269 | 0.089721 | [0.052226, 0.127113] |
| MRR | 0.694267 | 0.588369 | 0.105898 | [-0.015117, 0.220413] |
| Top1 | 0.587500 | 0.475000 | 0.112500 | [-0.037500, 0.262500] |
| Sensitivity | 0.460046 | 0.368038 | 0.092008 | [0.006113, 0.177530] |
| Specificity | 0.867935 | 0.780500 | 0.087435 | [0.036582, 0.141011] |
| Accuracy | 0.763172 | 0.699405 | 0.063767 | [0.033698, 0.093805] |
| Predicted EZ fraction | 0.203485 | 0.241122 | -0.037637 | [-0.086408, 0.009600] |

The AUROC/AUPRC/Macro-F1/EZ-F1 intervals exclude zero in favor of F0 under this conditional descriptive bootstrap. MRR/Top1 intervals cross zero. These intervals do not remove historical cohort viewing, user-selected F0 identity, multiple-comparison adaptivity, seed uncertainty or bandwidth/architecture differences.

## Confusion matrices — do not mix units

| Model | Mean patient TP | Mean FP | Mean TN | Mean FN | Pooled TP | Pooled FP | Pooled TN | Pooled FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F0 | 9.112500 | 9.987500 | 63.662500 | 12.675000 | 729 | 799 | 5093 | 1014 |
| R0 | 8.037500 | 15.437500 | 58.212500 | 13.750000 | 643 | 1235 | 4657 | 1100 |

R0 has436 additional FP and86 additional FN in the matched pooled counts; both patient-mean sensitivity and specificity are lower. Ranking AUROC/AP is also lower, so this is not merely a threshold-calibration difference. The pooled counts are secondary descriptions; headline F1/AP/AUROC remain patient means.

## Development — legal validation only

65 validation-fold appearances from47 IDs, original equal-appearance estimator. The separate47-ID-equal summary is provided and is not substituted for the primary.

| Model | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR |
|---|---:|---:|---:|---:|---:|
| F0 | 0.716569 | 0.508559 | 0.642067 | 0.433516 | 0.676591 |
| R0 | 0.630896 | 0.389998 | 0.567631 | 0.347431 | 0.580004 |

## Exact training and resources

Each raw fold uses the original shared A0 **initial** classifier weights, not fitted F0 weights. Jointly trained RawTinyEncoder (3,616) + Linear(64→88) (5,720) + original classifier (8,817) = **18,153** parameters. Every500-sample valid window is encoded; mean within each seizure, mean and population std across available seizures; std is zero for a single seizure. All valid canonical channels participate in differentiable patient-Z with denominator SD floor1e-8. No A0 engineered-coordinate imputer/scaler, source head, A1 task head, old loss or old checkpoint is used.

Original unchanged trainer: seed42+1009×fold; patient-equal updates, original FIT-only class-weighted NEZ-positive BCE, AdamW0.001/wd0.0001, clip1, max30/min6/patience6, own validation Macro-F1/EZ-F1/earlier-epoch selection and original0.005 threshold-grid/ties. No search or test-driven iteration.

| Fold | Completed epochs | Selected epoch | Frozen NEZ threshold | Measured trainer/collection seconds |
|---|---:|---:|---:|---:|
| 1 | 10 | 4 | 0.375 | 118.895 |
| 2 | 11 | 5 | 0.395 | 134.018 |
| 3 | 7 | 1 | 0.390 | 76.970 |
| 4 | 11 | 5 | 0.540 | 145.182 |
| 5 | 12 | 6 | 0.375 | 152.849 |

Total51 completed epochs; measured trainer/collection wall time **627.914 seconds (10.47 minutes)**, excluding implementation, admission, waveform banking, tests, outer/verification and transfer. Maximum PyTorch allocated GPU memory 179.08 MiB; reserved 204.00 MiB. These allocator counts exclude CUDA/library/desktop/other-process overhead; observed total-device use around3.2GiB is not attributed wholly to this model. CPU waveform banks occupy approximately2.98GB. RTX5090, PyTorch2.11.0+cu128.

## Integrity and engineering history

- Whole raw and feature-window pickle hashes match historical frozen RawTiny sources. Actual compound/run/channel/sample matching admits all80 patients/256 records/7635 canonical channels,24995 run-channel incidences and1471965 valid channel-windows (15074 record-windows), with zero duplicate physical windows or invalid retained feature windows.
- All raw500-sample windows are whole-array/sample verified against the source waveform using valid-start/valid-samples bounds. No padding, additional filter/resampling, arbitrary easy-window selection or patient exclusion.
- Raw data-only metadata allowlist contains no engineered vectors, clinical labels or source input. Input-purity, missing-seizure/window, differentiable patientZ and full-patient grouping tests PASS. A single legal-FIT label-swap one-step diagnostic changes parameters by0.002000004; it is not an extra competitive arm.
- The first test fixture omitted seeding before model construction and was corrected before any clinical raw training. A summed synthetic coordinate-gradient probe exceeded fixed tolerances; the mean-loss-unit probe (consistent with actual averaged BCE/pooling) passes unchanged3e-5 limits. Both diagnostic results and earlier bindings are retained. No data, architecture, clinical objective, hyperparameter or threshold changed.
- Raw chunk forward drift5.96e-7 and averaged-probe gradient drift1.69e-6 pass the fixed bounds. Exact CPU model/Adam/history resume and CPU/CUDA RNG-state restoration PASS. GPU floating-point gradient self replay is not bitwise deterministic; bitwise clinical GPU trajectory/resume is not claimed. No clinical training restart occurred.
- An initial detached launch did not remain alive and produced no epoch/checkpoint; actual training ran in a held-open foreground SSH session. A later local stderr-pump timeout did not stop the healthy trainer. Server training/outer/verification stderr files are empty and every selected checkpoint/completion hash was independently checked.
- All5 raw checkpoints/thresholds and source/registry bindings freeze before new outer scores. All5 new outer score files seal before the new process reads global clinical-y. Historical baseline labels were used separately only for explicitly authorized Stage I reuse verification. Legacy pickle deserialization materializes other fields, then discards them without label/feature indexing; this is not physically blinded serialization.
- Independent verification PASS:290 patient/model metric cells,32 paired intervals, every primary/fold/center/unique-ID aggregate; maximum original-versus-sklearn metric drift3.33e-16. It reads sealed scores and performs no new inference or threshold reoptimization.

## Interpretation and next stage

This fixed R0 raw-only model is clearly weaker in discrimination and thresholded localization than the selected compact engineered F0 under this repeated cohort and protocol. That is not a universal verdict against raw EEG: the fixed tiny frontend, mean/std pooling, per-window normalization, budget and raw250Hz Nyquist125Hz can limit it. Some original engineered high-gamma features may use higher native-rate information. Raw cache extraction/filter history and all EDF+D physical clocks are not fully independently certified; no low-frequency-ground-truth or onset-propagation claim is made.

All outcomes are exploratory repeated-test evidence, not blind clinical confirmation. The already frozen F1–F5 matrix **will still run**, regardless of this negative raw result. No replacement baseline, extra loss, normalization, grid, ensemble or v2 is introduced. Source code and anonymous aggregates only are published; weights, waveform banks, records, labels, patient/channel predictions, clinical identities and logs remain private.
