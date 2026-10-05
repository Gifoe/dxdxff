# Final report — Omni A1 physiological-feature complementarity audit (seed 42)

## Scope and protocol status

This completed audit reused the fixed 124 both-class-patient manifest and five patient-held-out folds from `omni_patient5fold_a1plugin_seed42_v1`.  Labels remained `1=pathological/SOZ`, `0=normal`.  No TimeConv checkpoint was retrained, no cohort/fold/label was changed, and no held-out label selected a classifier checkpoint, a normalizer, or a fusion coefficient.

The stored TimeConv OOF predictions replayed at patient-equal AUROC **0.7621631**, versus the locked reference 0.7621631 (absolute discrepancy below `1e-10`), so the mandatory baseline gate passed.  The private waveform cache provenance audit found 137 records.  All extraction uses 30 fixed two-second windows per record, followed by arithmetic mean over windows and then EDF occurrences; no label-aware pooling was used.  F9 and F36 feature heads used the locked 30 epochs, AdamW (`lr=1e-4`, `weight_decay=1e-3`), patient-equal BCE, and pathological weight 2.  For every outer fold, population mean/std were fitted only on the other four folds.

The reported OOF metrics are development diagnostics on this fixed 124-patient cohort.  They are not an unseen external test result.

## Feature provenance

The audit dynamically reused the checked historical A1 spectral implementation (`historical_a1_ez_features.py`, SHA-256 `1b046b542b791f2416467582fe95e2dfa991d04ebb535bf188e04b9a1a1439d6`, function `compute_spectral_channel_features`) and the historical interictal comparison implementation (`historical_a1_dataset.py`, SHA-256 `9826059603242eeb686c4e896b42d2c3c5a33d5860830f58b3df8e94487863a2`, function `_interictal_comparison_features`).  It did not reimplement or redesign the descriptors.

F9 is exactly `log_bp_delta, log_bp_theta, log_bp_beta, log_bp_low_gamma, log_bp_high_gamma, rms, variance, line_length_per_sec, spectral_entropy`.  The source uses Welch power spectra with `nperseg=min(samples, round(2*fs))`, 50% overlap, no detrending, trapezoid integration, and `log1p` band powers.  Bands are 1–4, 4–8, 13–30, 30–80, and 80–150 Hz.  RMS uses `+1e-8`; entropy floors/normalizes PSD at `1e-8`; line length is `sum(abs(diff(signal)))/duration_seconds`.

F36 is the exact historical `[ABS, DELTA, ZDELTA, LOGR]` view, with percentile and missing-indicator extensions disabled.  For Omni, the historical interictal fallback uses the 30 windows of the EDF-channel as its robust reference: median and `max(1.4826*MAD, 1e-5)`.  This is an interictal local reference, not a seizure-onset delta and not population normalization.

## Primary OOF results

| Model | Patient-equal AUROC | AP | MRR | Top-1 | NDCG | Macro-F1 at 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen TimeConv | 0.762163 | 0.477255 | 0.607458 | 0.508065 | 0.670700 | 0.545188 |
| F9 | 0.597671 | 0.290750 | 0.402978 | 0.282258 | 0.530103 | 0.465082 |
| F36 | 0.719959 | 0.383934 | 0.549596 | 0.419355 | 0.613840 | 0.540421 |

F36 is materially stronger than F9 as a standalone feature model, but it is still weaker than TimeConv by 0.042204 AUROC.  Neither descriptor-only classifier is a replacement for morphology.

## Pair complementarity

Primary pair measures are patient-equal: compute each patient's pathological-versus-normal pair rates first, then average patients equally.  Micro results are pair-count weighted and are included to show that the conclusion is not a weighting artefact.

| Feature | Weighting | Feature pair accuracy | Rescue on TimeConv-wrong pairs | Destroy on TimeConv-right pairs | Net rescue minus destroy | Oracle pair accuracy |
|---|---|---:|---:|---:|---:|---:|
| F9 | patient-equal | 0.597671 | 0.444541 | 0.365152 | 0.086763 | 0.863701 |
| F9 | micro | 0.626132 | 0.461104 | 0.318386 | 0.142718 | 0.864408 |
| F36 | patient-equal | 0.719959 | 0.526149 | 0.234774 | 0.291528 | 0.885902 |
| F36 | micro | 0.709627 | 0.509858 | 0.223210 | 0.286648 | 0.876676 |

F36 is unambiguously more complementary than F9 by all four rates above: it rescues more TimeConv errors and destroys fewer correct TimeConv pairs.  But the primary rescue rate, 0.52615, does **not** meet the predeclared meaningful threshold of 0.60; it is also below the predeclared weak-complementarity threshold of 0.55.  The positive net rate means the errors are not simply redundant, but it does not prove a trainable plugin will improve generalization.

The 124-patient pair table was generated privately as required.  It is not committed because it contains patient-level results.  It shows valid rescue/destroy denominators for 114 patients; the remaining 10 have no TimeConv-wrong or no TimeConv-right pair denominator for the corresponding conditional rate.  Consequently, claims about a patient distribution are limited to the aggregate and center summaries rather than a public patient listing.

### Center stability

| Center | Feature | Patient-equal rescue | Patient-equal destroy | Net | Feature AUROC |
|---|---|---:|---:|---:|---:|
| HUP (n=10) | F9 | 0.483839 | 0.484802 | 0.035993 | 0.514068 |
| Open-iEEG (n=103) | F9 | 0.443894 | 0.363290 | 0.084915 | 0.602226 |
| SourceSink (n=11) | F9 | 0.417922 | 0.273820 | 0.144102 | 0.631020 |
| HUP (n=10) | F36 | 0.544774 | 0.323138 | 0.225836 | 0.664588 |
| Open-iEEG (n=103) | F36 | 0.530789 | 0.223678 | 0.307016 | 0.731917 |
| SourceSink (n=11) | F36 | 0.471257 | 0.258335 | 0.212922 | 0.658327 |

F36 has positive net complementarity in every reported center, rather than being driven entirely by one center.  Its conditional rescue nevertheless remains below 0.55 in every center and is particularly weak in SourceSink.  The small HUP and SourceSink samples also preclude strong claims of cross-center consistency.  This is mixed evidence, not robust confirmation.

The pooled secondary diagnostic agrees directionally but is not the primary protocol: TimeConv pooled AUROC was 0.715747; F9 and F36 were 0.626296 and 0.687309.  Pooled F36 rescue/destroy were 0.449503/0.218249; F9 were 0.441004/0.300117.

## Fixed fusion diagnostic

No train-side OOF-compatible inner prediction bank existed from which to select beta independently per outer fold.  Therefore no learned or chosen beta is legal.  The three values below were fixed in advance and all are reported; none may be promoted as a selected, deployable fusion result.

| Feature | Fixed beta | Fused patient-equal AUROC | Delta versus TimeConv |
|---|---:|---:|---:|
| F9 | 0.10 | 0.763936 | +0.001773 |
| F9 | 0.25 | 0.758412 | -0.003751 |
| F9 | 0.50 | 0.735468 | -0.026695 |
| F36 | 0.10 | 0.777899 | +0.015736 |
| F36 | 0.25 | 0.783106 | +0.020943 |
| F36 | 0.50 | 0.774178 | +0.012015 |

Thus individual fixed F36 diagnostics exceed +0.01 and +0.02, but none reaches +0.03.  The largest observed fixed-grid difference is +0.020943 at beta 0.25; because the grid was inspected on the same OOF outcomes, it is descriptive only and cannot justify a post-hoc beta selection or a claimed deployable gain.  F9 has no meaningful practical fusion gain.

## Oracle ceiling

The label-aware oracle is explicitly `DIAGNOSTIC_ONLY_NOT_DEPLOYABLE`.  For every pathological-normal pair it counts the pair correct when either TimeConv or the feature model is correct.  It produced pair accuracies of 0.864408 for F9 and 0.876676 for F36 (micro), versus TimeConv's 0.748390, corresponding to descriptive headroom of +0.116018 and +0.128285.  This says the two score sequences make many nonidentical ranking errors.  It does **not** imply that an executable model can capture that headroom: F36's standalone score is weak and the legal beta was not selected independently.

## Required answers and decision

1. **Did TimeConv reproduce approximately 0.7622?** Yes: 0.7621631, exact replay within `1e-10`.
2. **F9 patient-equal AUROC?** 0.5976709.
3. **F36 patient-equal AUROC?** 0.7199595.
4. **F9 rescue?** 0.444541 patient-equal (0.461104 micro).
5. **F36 rescue?** 0.526149 patient-equal (0.509858 micro).
6. **Destroy rates?** F9 0.365152 patient-equal (0.318386 micro); F36 0.234774 (0.223210 micro).
7. **Is F36 materially more complementary than F9?** Yes on observed pair rescue, destruction, net complementarity, and every fixed F36 fusion diagnostic; no, it is not a stronger standalone model than TimeConv.
8. **Consistent across patients?** Aggregate patient-equal net complementarity is positive, but conditional-rate denominators exist for only 114/124 patients and no public patient-level table is released.  It is not defensible to claim uniformly strong patient-level rescue.
9. **Consistent across centers?** Positive F36 net complementarity is present in HUP, Open-iEEG, and SourceSink, but rescue is below 0.55 in all three and two centers are small.  The evidence is heterogeneous and insufficiently strong.
10. **Does simple legal fusion improve TimeConv?** Fixed F36 diagnostics do; F9 does not meaningfully.  No beta was legally selected for deployment.
11. **By how much?** F36 deltas are +0.015736, +0.020943, and +0.012015 for fixed beta 0.10, 0.25, and 0.50.
12. **Any fixed legal diagnostic reach +0.01?** Yes: all three F36 values.
13. **Reach +0.02?** Yes: F36 beta 0.25 is +0.020943.
14. **Reach +0.03?** No.
15. **Oracle ceiling?** F36 0.876676 micro pair accuracy, +0.128285 descriptive headroom over TimeConv pair accuracy; F9 0.864408, +0.116018.  Neither is deployable.
16. **Enough headroom to plausibly justify an embedding plugin?** The oracle suggests potential nonredundancy, but the primary rescue gate (<0.60) and lack of nested train-only fusion selection do not provide sufficient evidence to justify a new training experiment now.
17. **Mostly redundant or genuinely complementary?** F9 is largely noncompetitive.  F36 is weakly/nontrivially complementary in error overlap, but not strongly complementary under the predeclared decision rule.
18. **Train actual TimeConv + A1 feature residual plugin next?** **No.** Stop after this audit.  Do not migrate the feature route on the present evidence.

### Terminal decision

`A1_FEATURES_WEAKLY_COMPLEMENTARY_BUT_PLUGIN_GATE_NOT_MET`.

This wording is deliberately stricter than the original Case B label.  The result lies between the prompt's Case A and B thresholds: F36 rescue is greater than 0.50 and net-positive, but below 0.55, while predeclared fixed beta diagnostics include a +0.020943 result without an independently selected beta.  It fails Case C (`rescue >= 0.60` and legal selected gain >= +0.02) and fails Case D.  Claiming `A1_FEATURE_PLUGIN_WORTH_TRAINING` would be test-driven overinterpretation.

## Why this is not a repeat of prior experiments

The historical PC-CNN result (roughly 0.6895 to 0.7064) jointly changed a weaker RawCNN baseline, 36-D physiology, patient context, and later CNN/classifier layers; its interventions could not exactly restore an independent RawCNN.  It therefore could not isolate physiological evidence.  A1-TF tested A1 plus a learned time-frequency residual, not TimeConv plus explicit A1 descriptors.  The preceding strict A1-context run changed 0.7622 to only 0.7630, indicating that channel/patient context itself was not the source of an actionable gain.  This audit instead held the current TimeConv OOF model fixed and measured descriptor-only error overlap.

## Reproducibility and release

`results/` contains only aggregate results, source hashes, normalizer hashes, and the locked protocol.  Raw waveform cache, feature tensors, checkpoints, per-patient pair records, and predictions remain private.  No external official test was read for model or hyperparameter selection.
