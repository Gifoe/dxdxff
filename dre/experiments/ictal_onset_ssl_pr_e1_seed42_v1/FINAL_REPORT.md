# E1 vs frozen E3: source-only development validation

Status: E1_VALIDATION_COMPLETE_TEST_NOT_ACCESSED.

| Arm | Macro-F1 | EZ-F1 | EZ-AUPRC | EZ-AUROC | MRR | Top1 |
|---|---:|---:|---:|---:|---:|---:|
| E1 | 0.613295 | 0.387922 | 0.487951 | 0.706686 | 0.709271 | 0.615385 |
| E3 | 0.616545 | 0.397900 | 0.494767 | 0.691482 | 0.666778 | 0.553846 |

Primary values average five patient-equal validation folds (65 patient-fold cells, 47 unique patients). Same 80-patient, 255-seizure, 7,635-channel amended export and original fold roles as E3. Exactly one previously approved padded seizure is excluded; source caches are unchanged.

E1 uses the same 44,401-parameter PR model, 30 supervised epochs, seeds, AdamW settings, patient-equal EZ:NEZ 2:1 BCE and validation checkpoint/threshold rules, but no SSL pretraining or pretrained-state loading. E3 checkpoints and outcomes are not modified or rerun.

Paired comparison resamples 47 patient IDs 10,000 times, seed42, including all of each sampled patient's validation cells. Point estimates retain the 65-cell primary weighting. These are post-selection validation intervals, not independent test confirmation. The separate single-arm bootstrap first averages repeated appearances and uses unique-patient weighting.

E3 trained with NumPy2.2.6 and finalized with isolated1.26.4; E1 uses isolated1.26.4 throughout. PyTorch2.8.0+cu128 and model/data/metric code remain unchanged. Prior NumPy metric, sampling and real validation forward parity passed; E1 original-loop/resume parity is separately audited. This engineering runtime difference must be disclosed, not hidden as an identical binary environment.

Predeclared E3-vs-E1 development gate passed: False. E4/E5 are not authorized or run even if eligible. No outer test or independently confirmed EDF onset replay was performed. Neither these selected validation values nor their paired intervals establish independent clinical generalization or a benchmark Macro-F1>=0.70.

## Does SSL add a measured increment here?

**The predeclared development criterion is not met.** E3-minus-E1 Macro-F1 is only **+0.003251**, versus the required +0.020, and only **3/5** folds are positive rather than the required 4/5. EZ-AUPRC changes by +0.006816. The patient-cluster intervals for all six contrasts cross zero. These observations do not support a stable SSL increment; they also do not establish that E1 is significantly superior to E3.

| E3 minus E1 | Primary mean difference | Patient-cluster bootstrap 95% interval |
|---|---:|---:|
| Macro-F1 | +0.003251 | [-0.018788, +0.026011] |
| EZ-F1 | +0.009978 | [-0.027639, +0.048296] |
| EZ-AUPRC | +0.006816 | [-0.027909, +0.043788] |
| EZ-AUROC | -0.015205 | [-0.048088, +0.016432] |
| MRR | -0.042493 | [-0.102000, +0.019857] |
| Top1 | -0.061538 | [-0.163934, +0.037037] |

Source: [paired_bootstrap_E3_minus_E1.csv](results/paired_bootstrap_E3_minus_E1.csv). No model, epoch-selection rule, threshold rule or prediction was changed after this comparison. E4/E5 are not run.

## Selected E1 operating points

| Fold | Selected epoch | Global validation-selected EZ threshold | Macro-F1 | EZ-AUPRC | EZ-AUROC |
|---|---:|---:|---:|---:|---:|
| 1 | 5 | 0.400 | 0.587208 | 0.455601 | 0.662746 |
| 2 | 3 | 0.475 | 0.618588 | 0.491603 | 0.696058 |
| 3 | 27 | 0.575 | 0.613419 | 0.414129 | 0.687154 |
| 4 | 8 | 0.275 | 0.646473 | 0.568189 | 0.766941 |
| 5 | 11 | 0.375 | 0.600785 | 0.510232 | 0.720532 |

Source: [validation_by_fold.csv](results/validation_by_fold.csv). Each fold has 13 validation patients. Its threshold applies to every channel of every validation patient in that fold, not a patient-specific oracle.

## Center diagnostics

These are descriptive validation-cell means, not center-held-out tests.

| Center | Validation cells | E1 Macro-F1 | E3 Macro-F1 | E1 EZ-AUPRC | E1 EZ-AUROC |
|---|---:|---:|---:|---:|---:|
| HUP | 28 | 0.643100 | 0.670706 | 0.520339 | 0.757203 |
| LZU | 14 | 0.543756 | 0.540329 | 0.485742 | 0.607366 |
| Multicenter | 11 | 0.646406 | 0.605607 | 0.510989 | 0.690572 |
| Pediatric | 12 | 0.594525 | 0.589115 | 0.393835 | 0.719461 |

The small aggregate E3 F1 gain is not consistent across centers: it is positive in HUP and negative in the other three groups. This is descriptive heterogeneity, not a causal center explanation or an independent robustness result.

## Execution and integrity

- All five E1 folds completed **150 supervised epochs, zero SSL epochs** in the first attempt, with empty stderr and no native-crash restart.
- Recorded epoch-body time totals **115.61 seconds** (mean 0.771 seconds/epoch). This excludes model loading, synthetic tests, validation threshold grids, auditing and transfer.
- [E1_PARITY_AUDIT.json](audit/E1_PARITY_AUDIT.json): original supplied E1 loop and interrupted/resumed loop have maximum synthetic CPU parameter difference **0**, with identical selected epoch, threshold and metrics. Same supervised initialization, architecture and parameter count were verified.
- [COMPLETION_AUDIT.json](audit/COMPLETION_AUDIT.json): 65 matched validation cells, 47 unique patients and exact channel/class counts; all E1 checkpoint hashes match; all E3 checkpoint hashes, selection lock and private ledger remain unchanged. Independent summary/bootstrap recalculation differs by at most **3.33e-16**.
- The primary summary weights 65 patient-fold cells; the separate single-arm bootstrap in [validation_patient_bootstrap.csv](results/validation_patient_bootstrap.csv) first collapses repeated patient appearances and weights 47 unique patients equally. Their point estimates and intervals must not be mixed.
- The source-only limitation remains: **0/256** original records have independent clinical EDF-to-cache onset confirmation. The single approved padded record is excluded in both arms; no patient or channel is removed.

Private selected E1 checkpoints: `C:\ictal_onset_ssl_pr_e1_seed42_runtime\validation\fold_1_E1.pt` through `fold_5_E1.pt`. Private optimizer/RNG checkpoints and individual patient records remain on the server. Public delivery contains only code and aggregate artifacts.
