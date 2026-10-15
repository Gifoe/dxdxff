# Five-fold Outer results: linear SVM and AdaBoost, DELTA14 ± PR

## Actual run and scope

Completed on the trusted server on 2026-10-16 (Asia/Shanghai), seed42, all original 80 patients / 7,635 canonical channels / 256 seizures. Fifty candidate-arm fits produced 20 selected new model-fold bundles (two classifiers × two PR settings × five folds). All 20 were frozen before new Outer inference and all predictions sealed before the new full evaluation-label read. Historical RBF scores were verified and reused, not rerun.

`14D` below means the exact physiological DELTA14 with no patient Z; `14D+PR` means the original full-patient population-Z transformation of those same 14 coordinates. It is not a raw encoder or a 14D concatenation with a PR vector. Scaling/calibration/classifiers use FIT only. Validation chooses thresholds; AdaBoost shares validation-selected hyperparameters across PR arms. Linear SVC preserves the old paired RBF C per fold and changes only the kernel family, with margins/calibrators necessarily refitted.

## Primary results: patient-macro means

Each patient has equal reporting weight. These values are not pooled-channel metrics or an unweighted mean of five folds.

| Classifier | Input | AUROC | AP | Macro-F1 | EZ-F1 | NEZ-F1 | Accuracy | EZ-frac. bias |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Linear SVM | 14D | 0.686394 | 0.487682 | 0.439358 | 0.201766 | 0.676951 | 0.645670 | +0.049166 |
| Linear SVM | 14D+PR | 0.743278 | 0.546308 | 0.543573 | 0.265254 | 0.821892 | 0.740973 | -0.070669 |
| AdaBoost | 14D | 0.648994 | 0.423426 | 0.467671 | 0.338173 | 0.597168 | 0.600608 | +0.193868 |
| AdaBoost | 14D+PR | 0.742809 | 0.527582 | 0.620479 | 0.405282 | 0.835675 | 0.763177 | -0.052556 |
| RBF SVM, historical frozen reference | 14D | 0.681944 | 0.480505 | 0.491935 | 0.342801 | 0.641069 | 0.638757 | +0.139967 |
| RBF SVM, historical frozen reference | 14D+PR | 0.734981 | 0.515032 | 0.621304 | 0.419839 | 0.822768 | 0.751726 | -0.013611 |

Bias is predicted EZ fraction minus true EZ fraction; -0.070669 means 7.0669 percentage points lower on average, not a relative percent change. The historical RBF primary results reproduced with maximum absolute error 0.

## PR effect: paired patient bootstrap

Difference = PR minus no PR; 10,000 paired resamples of the same 80 patients, seed42, percentile 95% intervals, no model/threshold refits.

| Classifier | ΔAUROC [95% CI] | ΔAP [95% CI] | ΔMacro-F1 [95% CI] | ΔEZ-F1 [95% CI] |
|---|---|---|---|---|
| Linear SVM | +0.056884 [0.015485, 0.100923] | +0.058626 [0.027521, 0.092528] | +0.104215 [0.067902, 0.142210] | +0.063488 [-0.005436, 0.130109] |
| AdaBoost | +0.093815 [0.050909, 0.137591] | +0.104156 [0.068840, 0.142480] | +0.152808 [0.105483, 0.201687] | +0.067109 [0.010165, 0.122713] |

## New classifiers versus RBF, all with PR

| Contrast, new minus RBF | ΔAUROC [95% CI] | ΔAP [95% CI] | ΔMacro-F1 [95% CI] | ΔEZ-F1 [95% CI] |
|---|---|---|---|---|
| Linear SVM − RBF SVM | +0.008298 [-0.004885, 0.021903] | +0.031276 [0.009299, 0.055934] | -0.077731 [-0.107112, -0.048319] | -0.154585 [-0.212800, -0.096813] |
| AdaBoost − RBF SVM | +0.007828 [-0.008872, 0.023476] | +0.012550 [-0.014538, 0.040647] | -0.000825 [-0.020044, 0.018227] | -0.014557 [-0.047348, 0.017792] |

Changing RBF to linear does **not** improve thresholded classification in this frozen comparison: its ranking/AP can improve while Macro-F1 and EZ-F1 worsen. AUROC/AP do not determine the calibrated probability scale or the validation-selected decision boundary. Both new PR models underpredict EZ more than RBF+PR on average. This observation does not authorize reselecting thresholds on Outer. AdaBoost+PR is close to RBF+PR on Macro-F1; these intervals do not support a general superiority claim.

## Verification and engineering ledger

- Synthetic SVC and AdaBoost API/calibration smoke checks passed before cohort training.
- Input gate reproduced exact D1 DELTA14 and original patient-Z / legal-bank scaled PR features in all five folds; no new extraction, raw processing, GPU use or patient exclusion.
- Independent sklearn confusion/F1/AUROC/AP/float64-logloss checks passed on all 480 private patient rows (320 new, 160 historical).
- All validation thresholds and paired candidate choices were independently reverified without Outer selection. Thirty SVM calibration subfold audits confirm patient exclusion and subfold-only preprocessing.
- Direct bootstrap resampling matched the count-matrix calculation for 128 draws to absolute tolerance 1e-12. All protected historical artifacts and new trained bundles/scores were unchanged.
- The initial run stopped only in the reporting phase after all 20 scores were sealed: the reused parent reporting helper retained its old default metric list, causing `KeyError: nez_f1`. `reporting/finalize.py` repaired only that reporting default. No training, prediction, cohort, model, calibration or threshold was changed or rerun. All original runtime logs remain private. The reporting recovery also verified the private cached frame against every sealed new score file.
- Candidate fits, calibration and validation took 67.833 recorded seconds in total (linear SVM 17.274 s; AdaBoost 50.559 s). This excludes imports, input hashing, Outer evaluation, verification, transfers and reporting; it is not a claimed full wall-clock benchmark.
- Freeze SHA256: `671c9afcb76589138e0d199bedcdc399e3373088429ac40c85421a9896302be4`.
- Prediction seal SHA256: `16b35eadc6dbaa79a56c473a1cdfba9cbb475ca9432f197902bdc320f411c97f`.
- Terminal states: `ONE_FROZEN_OUTER_PASS_COMPLETE`; `INDEPENDENT_VERIFICATION.json: PASS`; `REPORT_ONLY_RECOVERY_ALL_FROZEN_ARTIFACTS_UNCHANGED`.

## Detailed artifacts and caveats

`results/main_results.csv` contains exact patient means. Separate files contain pooled-channel metrics, per-fold patient-macro/pooled metrics, fold mean/SD, center-level aggregate results, ranking, confusion matrices, undefined-metric coverage, validation choices and paired bootstrap comparisons. A/B in those machine-readable tables denotes no-PR/PR respectively.

The cohort was previously viewed: this is **EXPLORATORY_REPEATED_TEST**, not a new blind confirmation. Only seed42 was trained. Bootstrap intervals condition on the fitted models and fixed thresholds, do not include training-seed variability and are not multiplicity-adjusted. SVM C is deliberately reused for a controlled kernel substitution, not optimized independently for the new linear model. AdaBoost's native probabilities are not guaranteed calibrated; weak F1 cannot be repaired by test-driven threshold search. PR uses unlabeled channels of the evaluated patient, not its labels. No raw arrays, identities, per-patient/channel results, fitted models or runtime logs are included in this repository.
