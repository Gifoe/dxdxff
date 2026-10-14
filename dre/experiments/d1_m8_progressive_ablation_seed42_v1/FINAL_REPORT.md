# D1 → M8 progressive ablation — completed seed42

| Model | DELTA | PR | Raw correction | AUROC | AUPRC | Macro-F1 | EZ-F1 |
|---|---|---|---|---|---|---|---|
| A0 Raw ABS | No | No | No | 0.638904 | 0.436295 | 0.470580 | 0.298905 |
| A1 Raw DELTA | Yes | No | No | 0.653323 | 0.459677 | 0.479030 | 0.318277 |
| A2 frozen D1 | Yes | Yes | No | 0.732636 | 0.527175 | 0.616556 | 0.408531 |
| A3 frozen M8 | Yes | Yes | Yes | 0.762397 | 0.554067 | 0.634097 | 0.434087 |

Primary values are actual **Outer TEST patient-equal** estimates, not Development or pooled-channel metrics. **Observed strict ordering A0<A1<A2<A3 holds for AUROC, AUPRC, Macro-F1 and EZ-F1.** Completion terminal: PROGRESSIVE_IMPROVEMENT_OBSERVED; execution status: COMPLETE_VERIFIED. “Observed” refers to point estimates, not statistically confirmed benefit at every step.

## Progressive contributions

| Increment | ΔAUROC | ΔAUPRC | ΔMacro-F1 | ΔEZ-F1 | MF1-improving folds | MF1-improving patients |
|---|---|---|---|---|---|---|
| DELTA_effect | +0.014419 | +0.023382 | +0.008450 | +0.019372 | 4/5 | 36/80 |
| PR_effect | +0.079312 | +0.067498 | +0.137527 | +0.090255 | 5/5 | 61/80 |
| RawCorrection_package_effect | +0.029761 | +0.026892 | +0.017541 | +0.025556 | 5/5 | 44/80 |

The last row is the **complete M8 package**, including the physiology gate and changed optimization objective. It must not be relabeled a pure CNN effect.

## Paired patient bootstrap

10,000 draws, seed42, same80 patient IDs in each arm; per-fold numeric thresholds fixed, no reoptimization. Unadjusted95% percentile intervals:

| Increment | Metric | Difference | Unadjusted paired95% interval |
|---|---|---|---|
| DELTA_effect | ez_auroc | +0.014419 | [-0.027119, +0.059002] |
| DELTA_effect | ez_auprc | +0.023382 | [-0.015729, +0.064919] |
| DELTA_effect | macro_f1 | +0.008450 | [-0.031964, +0.048564] |
| DELTA_effect | ez_f1 | +0.019372 | [-0.019536, +0.059325] |
| PR_effect | ez_auroc | +0.079312 | [+0.033290, +0.127182] |
| PR_effect | ez_auprc | +0.067498 | [+0.023406, +0.113151] |
| PR_effect | macro_f1 | +0.137527 | [+0.096971, +0.178380] |
| PR_effect | ez_f1 | +0.090255 | [+0.033863, +0.144974] |
| RawCorrection_package_effect | ez_auroc | +0.029761 | [+0.006912, +0.056661] |
| RawCorrection_package_effect | ez_auprc | +0.026892 | [+0.003448, +0.051300] |
| RawCorrection_package_effect | macro_f1 | +0.017541 | [-0.001153, +0.036806] |
| RawCorrection_package_effect | ez_f1 | +0.025556 | [-0.006790, +0.059056] |

## Interpretation limits

The strictly increasing **Outer point estimates** in all four primary metrics support an observed progression under this fixed seed/model/training matrix. They do not establish three independently significant or causal physiological improvements. The DELTA-only contrast has all four95% intervals crossing zero; it improves Macro-F1 for only36/80 patients, despite4/5 improved fold averages. Patient Z is the largest observed contribution and improves61/80 patient Macro-F1 values. Its four primary paired intervals exclude zero under this descriptive, unadjusted analysis.

The M8 package improves AUROC/AP with intervals above zero, but its Macro-F1/EZ-F1 intervals include zero. It improves44/80 patient Macro-F1 values and all five fold averages; this is not universal patient benefit. A2→A3 includes morphology correction, physiology gate, different readout/residual formulation and the original macro-aware objective. It is **not a pure CNN-only effect**. A1→A2 identifies input-level Patient Z in the D1 physiology pathway, not the separate Raw-branch Patient Z effect in M8.

Ranking and classification are distinct: PR raises patient specificity0.657272→0.854288 while sensitivity decreases0.495166→0.477193. Pooled FP falls1,945→844 while FN rises849→1,021. Thus its F1 gain is not evidence of improved EZ sensitivity. M8 then reduces both pooled FP844→799 and FN1,021→996. Confusion totals pool channels; reported primary sensitivity/specificity/F1 average patients and must not be recalculated from pooled counts as substitutes.

Development is65 appearances/47 unique IDs, whereas Outer is80 unique patients once each. Development A0→A1 AUROC decreases0.684288→0.681432, so monotonicity is **not** a universal property across evaluation scopes. The very low no-PR Outer Macro-F1 values0.470580/0.479030 versus Development0.555853/0.566046 show poor transfer of their legally selected operating points; no test threshold is changed to fix this. All new numeric thresholds are listed below and retained.

Scientific status: **EXPLORATORY_REPEATED_TEST, single seed42**. Original M8 was selected in adaptive, extensively reused Development research; that architecture selection was not independently nested within these five folds. The cohort was also viewed repeatedly in historical Outer work. Descriptive unadjusted patient-bootstrap intervals do not account for architecture/epoch/threshold selection, test reuse, training-seed variability or multiple testing. This is not blind external/clinical confirmation. No center, patient or window was selected after looking at results.


## All additional patient-equal Outer metrics

| Model | NEZ-F1 | Balanced accuracy | Sensitivity | Specificity | MRR | Top1-EZ | Accuracy | Predicted EZ fraction |
|---|---|---|---|---|---|---|---|---|
| A0 Raw ABS | 0.642254 | 0.567974 | 0.482193 | 0.653755 | 0.558785 | 0.437500 | 0.615806 | 0.375189 |
| A1 Raw DELTA | 0.639783 | 0.576219 | 0.495166 | 0.657272 | 0.654289 | 0.562500 | 0.628425 | 0.373115 |
| A2 frozen D1 | 0.824582 | 0.665741 | 0.477193 | 0.854288 | 0.689399 | 0.575000 | 0.748950 | 0.210810 |
| A3 frozen M8 | 0.834107 | 0.681352 | 0.497392 | 0.865313 | 0.749310 | 0.637500 | 0.760166 | 0.209195 |

## Pooled confusion counts (not pooled primary metrics)

| Model | EZ TP | EZ FP | NEZ TN | EZ FN |
|---|---|---|---|---|
| A0 Raw ABS | 864 | 2002 | 3890 | 879 |
| A1 Raw DELTA | 894 | 1945 | 3947 | 849 |
| A2 frozen D1 | 722 | 844 | 5048 | 1021 |
| A3 frozen M8 | 747 | 799 | 5093 | 996 |

## Five Outer folds

| Fold | Patients | Channels | A0 AUROC/MF1 | A1 AUROC/MF1 | A2 AUROC/MF1 | A3 AUROC/MF1 |
|---|---|---|---|---|---|---|
| 1 | 16 | 1631 | 0.596595/0.436411 | 0.624072/0.496379 | 0.713926/0.600166 | 0.716668/0.616079 |
| 2 | 16 | 1498 | 0.630570/0.401464 | 0.690363/0.407861 | 0.775697/0.579680 | 0.782367/0.586124 |
| 3 | 17 | 1746 | 0.674576/0.509585 | 0.698664/0.516982 | 0.740615/0.634277 | 0.774181/0.655064 |
| 4 | 15 | 1425 | 0.614190/0.492895 | 0.656847/0.500080 | 0.659013/0.593764 | 0.727089/0.601743 |
| 5 | 16 | 1335 | 0.674816/0.511499 | 0.594058/0.472790 | 0.768828/0.672363 | 0.808734/0.708143 |

## Four acquisition centers

| Center | Patients | A0 AUROC/MF1 | A1 AUROC/MF1 | A2 AUROC/MF1 | A3 AUROC/MF1 |
|---|---|---|---|---|---|
| hup | 36 | 0.709077/0.527796 | 0.688589/0.508007 | 0.781675/0.665235 | 0.806963/0.676924 |
| lzu | 21 | 0.579501/0.442539 | 0.531970/0.423789 | 0.639171/0.547782 | 0.657057/0.553756 |
| multicenter | 15 | 0.626019/0.451928 | 0.741765/0.493451 | 0.806417/0.612662 | 0.814129/0.654739 |
| pediatric | 8 | 0.503217/0.321684 | 0.647352/0.466600 | 0.618964/0.585337 | 0.741365/0.613568 |

Full precision AP/EZ-F1/sensitivity/specificity and all other metrics for every fold/center are in the CSVs. No center was excluded; the center summaries are descriptive, especially the small pediatric subset.

## Development (separate estimand)

| Model | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR | Top1-EZ | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|---|
| A0 Raw ABS | 0.684288 | 0.463550 | 0.555853 | 0.323305 | 0.669599 | 0.569231 | 0.412047 | 0.809406 |
| A1 Raw DELTA | 0.681432 | 0.470875 | 0.566046 | 0.332951 | 0.665033 | 0.584615 | 0.387291 | 0.834094 |
| A2 frozen D1 | 0.725978 | 0.503601 | 0.644546 | 0.443537 | 0.658502 | 0.523077 | 0.506245 | 0.866074 |
| A3 frozen M8 | 0.756633 | 0.542226 | 0.666795 | 0.477249 | 0.699708 | 0.569231 | 0.544976 | 0.876867 |

## Exact selections

| Model | Fold | Selected epoch | Frozen NEZ threshold | New epochs run |
|---|---|---|---|---|
| A0 | 1 | 23 | 0.535 | 29 |
| A1 | 1 | 7 | 0.415 | 13 |
| A0 | 2 | 2 | 0.465 | 8 |
| A1 | 2 | 2 | 0.520 | 8 |
| A0 | 3 | 3 | 0.400 | 9 |
| A1 | 3 | 19 | 0.375 | 25 |
| A0 | 4 | 2 | 0.450 | 8 |
| A1 | 4 | 8 | 0.425 | 14 |
| A0 | 5 | 2 | 0.500 | 8 |
| A1 | 5 | 1 | 0.520 | 7 |
| A2 | 1 | 2 | 0.435 | reused |
| A2 | 2 | 19 | 0.260 | reused |
| A2 | 3 | 5 | 0.300 | reused |
| A2 | 4 | 3 | 0.320 | reused |
| A2 | 5 | 5 | 0.400 | reused |
| A3 | 1 | 4 | 0.410 | reused |
| A3 | 2 | 1 | 0.255 | reused |
| A3 | 3 | 10 | 0.295 | reused |
| A3 | 4 | 13 | 0.365 | reused |
| A3 | 5 | 15 | 0.395 | reused |

## Execution and integrity

Ten new scratch training cells actually completed 129 saved epochs. A0/A1 parameter count is1,537, with exact historical88-position duplicate/zero-aware LayerNorm folding, not LayerNorm14. The five paired initial parameter hashes match each other and the initial hashes recorded in the **original trained D1 checkpoints**, verified separately on the server. No trained checkpoint warm start was used. All preprocessing fits use only own-fold FIT patients. Numeric thresholds and checkpoints follow the original validation policy, independently for each new arm.

All80 patients/256 seizures/7,635 canonical channels remain, with1,743EZ/5,892NEZ. Fold counts remain16/16/17/15/16 patients and1,631/1,498/1,746/1,425/1,335 channels. New feature extraction uses the pinned, existing Raw250 waveforms,500-sample windows and exact original physiology formulas without additional filtering/acquisition. All1,471,965 genuine channel-windows and original masks/alignment are retained. Actual new DELTA14 reconstruction matches frozen D1 R14 elementwise: maximum error0. ABS14 uses the same samples/families but no pre-reference subtraction. Missing seizure channels remain masked; single-seizure population SD is zero.

All ten new checkpoint/preprocessor/validation-threshold selections plus original D1/M8 provenance were hash-frozen before new Outer inference. Twenty cells (ten new, ten exact reused original ledgers) were sealed before the new full archive TEST-label read. This safeguard does not mean that other historical folds/researchers never saw these patients' labels. D1/M8 were not retrained, refitted, recalibrated or neurally replayed. Their new combined patient metrics match the original verified private ledgers exactly.

Freeze SHA256: 53afbec76d6e9837b40a4435fbcc65159298ee1851eac33aa3b3a7bc86950510.

Prediction-seal SHA256: 9a8cd7d353092c51e2b628c4f42f9adfd29bc2b054d6b7eb8488a656329ee17d.

Protocol SHA256: 2386f9685b0eee462b9b559a06cfa3d85695159bf9506fb45f2cda25a25c943b.

Independent verification is PASS:320 patient-model metric rows checked by sklearn/confusion calculations, original frozen checkpoint hashes, aggregate tables and all36 bootstrap metric contrasts checked. Bootstrap distributions also match independent patient-multiplicity matrix calculations. Three synthetic contract tests passed before training. No engineering failure, discarded completed epoch, model search or post-Outer rescue was needed in this experiment.


## Delivery

Only source, protocol/hash records, anonymous aggregates and reports are committed. Raw data, IDs, labels, checkpoints, scalers, per-channel/per-patient predictions, extraction intermediates and runtime logs remain on the trusted server. No installed skill was used. Source code and results are in this experiment directory; previous experiments are unchanged. Work stops after this fixed matrix, with no additional model, seed or test-driven correction.
