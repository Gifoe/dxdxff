# Six-feature F0 without head interaction: approved fourteen-group ablation

The main model is the exact previously frozen six-feature WO_HEAD_INTERACTION configuration. Variance and head interaction remain absent in every group. Original80patients,7635channels,256seizures; seed42 and original five patient-held-out folds. All70selected models/preprocessors/thresholds froze before newOuter; all70score cells sealed before unified reporting.55genuine fresh fits and15exact reused cells.

This reference was chosen after historical Outer viewing. All results are exploratory repeated-test, not independent confirmation. No group, threshold or model was chosen or changed from this pass. No Development eligibility filter is used.

## Equal-patient averages

| Configuration | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac. bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Main: six features, no head interaction | 0.653828 | 0.752255 | 0.585240 | 0.467424 | 0.840232 | 0.770976 | 0.845674 | 0.800000 | -0.022662 |
| w/o early FiLM | 0.638450 | 0.766268 | 0.576278 | 0.457544 | 0.819355 | 0.747129 | 0.832323 | 0.762500 | 0.003023 |
| w/o both guidance | 0.635337 | 0.763667 | 0.568775 | 0.453781 | 0.816893 | 0.743496 | 0.813047 | 0.725000 | 0.005559 |
| w/o BOTH PR | 0.516322 | 0.759725 | 0.570728 | 0.371850 | 0.660795 | 0.662387 | 0.803417 | 0.725000 | 0.129884 |
| w/o Raw pre-reference | 0.608670 | 0.741359 | 0.556211 | 0.399424 | 0.817916 | 0.744404 | 0.798735 | 0.725000 | -0.024654 |
| Early FiLM mismatch | 0.637951 | 0.762301 | 0.573701 | 0.449240 | 0.826662 | 0.753649 | 0.805629 | 0.725000 | -0.020528 |
| Raw-only | 0.621288 | 0.727387 | 0.543166 | 0.403061 | 0.839516 | 0.761493 | 0.774164 | 0.700000 | -0.067481 |
| Physiology-only | 0.625022 | 0.745238 | 0.541467 | 0.432788 | 0.817256 | 0.746628 | 0.750419 | 0.650000 | -0.004029 |
| w/o beta | 0.642729 | 0.751716 | 0.583951 | 0.462707 | 0.822752 | 0.752023 | 0.844880 | 0.787500 | -0.001311 |
| w/o low gamma | 0.632982 | 0.753187 | 0.594837 | 0.450875 | 0.815089 | 0.742093 | 0.848512 | 0.800000 | 0.008469 |
| w/o partial high gamma80-125Hz | 0.638606 | 0.752474 | 0.577113 | 0.440225 | 0.836987 | 0.762240 | 0.832408 | 0.775000 | -0.040450 |
| w/o RMS | 0.624215 | 0.752131 | 0.578262 | 0.435301 | 0.813130 | 0.735770 | 0.810425 | 0.750000 | 0.002387 |
| w/o line length | 0.629786 | 0.745476 | 0.571816 | 0.429622 | 0.829951 | 0.756658 | 0.830043 | 0.775000 | -0.041672 |
| w/o entropy10-125Hz | 0.641416 | 0.750527 | 0.586901 | 0.446435 | 0.836397 | 0.762345 | 0.839559 | 0.787500 | -0.028807 |

Macro-F1 is the mean of EZ-F1 and NEZ-F1 for each patient, then averaged equally over80patients. Pooled-channel metrics are separate. EZ-fraction bias is predicted minus observed EZ fraction; zero is preferable. See [folds](FOLD_RESULTS.csv), [sample SD](FOLD_SD.csv), [sources](SOURCE_RESULTS.csv), [pooled](POOLED_RESULTS.csv), [coverage](RAW_COVERAGE.csv).

## Patient-paired differences

Full minus ablation; positive favors the fixed main model.10000seed42same-patient bootstrap draws.243nominal95% intervals cover14absolute and13paired contrasts across9metrics.13additional Macro-F1 intervals use Bonferroni-level percentile tails0.05/(2x13); approximate exploratory intervals, not exact simultaneous coverage or correction for historical researcher selection.

| Ablation | Macro-F1 difference | Nominal95% interval | Additional13-comparison interval |
| --- | ---: | --- | --- |
| w/o early FiLM | +0.015378 | [-0.001511, 0.032717] | [-0.009135, 0.042521] |
| w/o both guidance | +0.018491 | [0.001508, 0.036205] | [-0.006033, 0.044342] |
| w/o BOTH PR | +0.137506 | [0.098996, 0.178225] | [0.083062, 0.198608] |
| w/o Raw pre-reference | +0.045158 | [0.023842, 0.067260] | [0.015097, 0.077543] |
| Early FiLM mismatch | +0.015877 | [-0.000486, 0.032985] | [-0.007845, 0.040435] |
| Raw-only | +0.032540 | [0.013752, 0.051842] | [0.004185, 0.061329] |
| Physiology-only | +0.028806 | [0.004830, 0.052839] | [-0.006101, 0.064140] |
| w/o beta | +0.011099 | [0.000712, 0.022413] | [-0.003575, 0.028097] |
| w/o low gamma | +0.020845 | [0.008061, 0.034166] | [0.001578, 0.040766] |
| w/o partial high gamma80-125Hz | +0.015222 | [0.002425, 0.028256] | [-0.003658, 0.034857] |
| w/o RMS | +0.029612 | [0.014411, 0.045555] | [0.007526, 0.052802] |
| w/o line length | +0.024041 | [0.011733, 0.037474] | [0.006459, 0.044566] |
| w/o entropy10-125Hz | +0.012412 | [-0.001149, 0.025607] | [-0.007547, 0.032996] |

## Interpretation limits

- The user excluded the isolated Late Gate deletion and interaction add-back. No new results for these rows are generated. A full early-by-late factorial interaction is not identifiable and is not reported.
- w/o PR removes BOTH physiological-summary patient Z and final Raw96 patient Z. Per-window waveform normalization and Raw pre-reference remain unchanged.
- Each single-feature deletion masks both summary coordinates and matching early/late window inputs during fresh training. Variance is always masked. Input widths and remaining shared initial tensors stay matched; one64-hidden classifier is retained.
- Mismatch changes early FiLM donor trajectories only, within same patient/seizure/identical full support; direct physiology and late Gate remain correctly aligned. Singleton coverage is retained and reported.
- Raw-only and Physiology-only retain the corresponding original head slices and branch scaling. Branch comparisons may include parameter-capacity/optimization differences, not only information content.
- No single deletion proves global feature optimality; Raw may encode correlated physiological properties. Partial high gamma is80-125Hz, not full80-150Hz. Spectral entropy is10-125Hz.
- Previously completed seven classifiers in both PR settings are separately reused as [historical context](HISTORICAL_CLASSIFIER_REFERENCE.csv); they are unaffected by neural head interaction. No fresh classical fit or classifier comparison is included in this14-group CI family.
- One training seed and descriptive source summaries do not establish seed or external-cohort generalization. Patient bootstrap does not cover model-reference selection or training randomness.
- Independently verified 1120 patient rows, 10080 metric values,243nominal intervals and13additional intervals. Maximum independent interval discrepancy 5.55e-16.
- Only code, hashes and anonymous aggregate reports are public. Signals, patient/channel identities, predictions, weights, preprocessors, mismatch maps and runtime logs remain on the trusted server.
