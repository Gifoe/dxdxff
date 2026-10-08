# E3 source-only five-fold validation

Status: VALIDATION_COMPLETE_TEST_NOT_ACCESSED.

## Primary five-fold validation results

Each fold contains 13 validation patients. The primary summary below averages the five patient-equal fold metrics, equivalently the 65 patient-fold cells. Repeated validation appearances are not independent new patients.

| Metric | Five-fold mean |
|---|---:|
| Macro-F1 | 0.616545 |
| EZ-F1 | 0.397900 |
| EZ-AUPRC | 0.494767 |
| EZ-AUROC | 0.691482 |
| MRR | 0.666778 |
| Top1 | 0.553846 |

| Fold | Selected supervised epoch | Validation-selected EZ threshold |
|---|---:|---:|
| 1 | 4 | 0.350 |
| 2 | 3 | 0.450 |
| 3 | 7 | 0.375 |
| 4 | 8 | 0.325 |
| 5 | 11 | 0.500 |

No checkpoint or threshold was selected using outer-test outcomes. Detailed metrics are in [validation_by_fold.csv](results/validation_by_fold.csv).

## Data and interpretation

The user approved excluding the single audited padded seizure. The model uses 255 seizures, all 80 patients and 7,635 channels, with original FIT/VAL/TEST membership. The source caches are untouched. Independent EDF onset confirmation remains 0/256.

Validation contains 65 patient-fold cells and 47 unique patients. Checkpoints use validation EZ-AP (tie F1@0.5, earlier epoch); a single global threshold per fold uses validation Macro-F1. No outer patient was scored in its held-out fold. These post-selection validation metrics are optimistic and are not an 80-patient outer benchmark or an independent test.

The following secondary summary first averages a patient's metrics across their validation appearances, then weights the 47 unique patients equally. The 10,000-draw seed42 bootstrap resamples those patients. It is descriptive and does not correct checkpoint/threshold selection optimism; its intervals do not apply to the primary 65-cell mean above.

| Metric | Mean across unique validation patients | Descriptive patient-bootstrap 95% CI |
|---|---:|---:|
| macro_f1 | 0.618765 | [0.579296, 0.659154] |
| ez_f1 | 0.404793 | [0.339391, 0.468919] |
| ez_ap | 0.505304 | [0.433096, 0.575710] |
| ez_auroc | 0.685416 | [0.628122, 0.739270] |
| mrr | 0.689338 | [0.578629, 0.796334] |
| top1 | 0.585106 | [0.446809, 0.723404] |

All five folds used 25 SSL epochs and 30 supervised epochs, the supplied E3 topology, VICReg, patient-equal EZ:NEZ 2:1 BCE, seeds and learning rates. E0/E1/E2/E4/E5 were not trained. The contribution of SSL versus E1 cannot be inferred. No claim of benchmark Macro-F1 >=0.70 is made.

Numerical resume tests and package tests are reported separately. Private checkpoints, patient records, waveforms and runtime logs must remain on the private server.

## Center diagnostics

These are patient-fold-cell means, not independent center-held-out evaluations.

| Center | Validation cells | Macro-F1 | EZ-AUPRC | EZ-AUROC |
|---|---:|---:|---:|---:|
| HUP | 28 | 0.670706 | 0.550657 | 0.750762 |
| LZU | 14 | 0.540329 | 0.485593 | 0.619759 |
| Multicenter | 11 | 0.605607 | 0.429900 | 0.636653 |
| Pediatric | 12 | 0.589115 | 0.434520 | 0.687097 |

The observed center variation is substantial. These results do not establish improvement over historical A1, do not isolate an SSL effect without E1, and do not support a Macro-F1 >=0.70 claim. A historical development score from another evaluation protocol is not a paired control for this run.

## Execution verification

- The supplied E3 model is unchanged, with 44,401 trainable parameters.
- All 125 SSL and 150 supervised epochs completed. Recorded epoch-body time totals 210.80 seconds (96.28 SSL + 114.52 supervised); this excludes export, loading, final threshold sweeps, tests and native-crash recovery.
- All 13 package/amendment tests passed. Synthetic CPU resume equivalence gave zero parameter differences, with selected epoch/threshold/metrics unchanged.
- [COMPLETION_AUDIT.json](audit/COMPLETION_AUDIT.json) confirms selected checkpoint hashes, source/protocol bindings, disjoint fold roles and full epoch counts. Recalculated summaries differ by at most 2.22e-16.
- Repeated native NumPy access violations delayed finalization. All training completed in the original environment. Finalization alone used an isolated NumPy 1.26.4 environment with unchanged PyTorch/sklearn/SciPy; the original environment was not modified. Synthetic metrics, threshold selection and sampling matched exactly, and a real validation forward comparison had maximum logit difference 0. See [FINALIZATION_RUNTIME_AUDIT.json](audit/FINALIZATION_RUNTIME_AUDIT.json) and [ENGINEERING_LEDGER.md](ENGINEERING_LEDGER.md).

Private runtime: `C:\ictal_onset_ssl_pr_e3_seed42_runtime\validation_amended`. The selected files are `fold_1_E3.pt` through `fold_5_E3.pt`; all optimizer/RNG resume checkpoints are retained privately. No outer evaluation or independent clinical/external test was run.
