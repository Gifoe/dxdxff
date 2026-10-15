# M8 14D physiology baselines ± Patient Relative — completed seed42

**All ten groups and all fifty original Outer model-fold cells completed. Independent verification: PASS.** This is a comparison of conventional classifiers on fixed physiological features, **not an ablation of the complete M8 model**.

## Direct answer

Patient Z improves patient-average AUROC, sklearn Average Precision, ACC, positive EZ-F1 and class Macro-F1 in **all five tested classifiers**, under the same fixed folds, matched selected hyperparameters and validation-only operating points. It is not an MLP-specific gain. However, the effect is **not universal across every localization or probability metric**: SVM MRR/Hit@1 worsen, and some Log Loss/ECE values worsen. This is single-seed, exploratory repeated-test evidence, not a new blind or clinical confirmation.

## Table 1. Main classification — patient macro

Each patient contributes equally, once in its own Outer fold:80 patients,7,635 canonical channels,256 seizures;1,743 EZ and5,892 NEZ. EZ is the positive class. Version A uses original DELTA14 before PR; Version B uses the exact original Patient Z, still14D. Neither version adds Raw morphology, engineered ABS18, latent96 features or a28D concatenation.

| Model | Version | ACC | Precision | Recall | Specificity | EZ-F1 | Macro-F1 | AUROC | AP | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 14D (no PR) | 0.640687 | 0.431946 | 0.530705 | 0.649394 | 0.341309 | 0.495816 | 0.666886 | 0.470049 | 0.207110 |
| Logistic Regression | 14D + PR | 0.745668 | 0.472046 | 0.509983 | 0.839054 | 0.428868 | 0.623405 | 0.743762 | 0.544163 | 0.313462 |
| RBF SVM | 14D (no PR) | 0.638757 | 0.456576 | 0.522650 | 0.657401 | 0.342801 | 0.491935 | 0.681944 | 0.480505 | 0.205377 |
| RBF SVM | 14D + PR | 0.751726 | 0.477041 | 0.490914 | 0.841006 | 0.419839 | 0.621304 | 0.734981 | 0.515032 | 0.304461 |
| Random Forest | 14D (no PR) | 0.601841 | 0.405094 | 0.509037 | 0.632809 | 0.298753 | 0.464175 | 0.619541 | 0.427881 | 0.168690 |
| Random Forest | 14D + PR | 0.743540 | 0.455364 | 0.430056 | 0.849265 | 0.376708 | 0.597412 | 0.709770 | 0.506553 | 0.272174 |
| XGBoost | 14D (no PR) | 0.641606 | 0.396013 | 0.476563 | 0.684459 | 0.314338 | 0.499114 | 0.625431 | 0.443382 | 0.161940 |
| XGBoost | 14D + PR | 0.741911 | 0.470507 | 0.431019 | 0.851017 | 0.378188 | 0.597934 | 0.716844 | 0.516082 | 0.270167 |
| MLP | 14D (no PR) | 0.625437 | 0.385226 | 0.540055 | 0.646034 | 0.327393 | 0.487112 | 0.662600 | 0.464786 | 0.187711 |
| MLP | 14D + PR | 0.727635 | 0.466227 | 0.425909 | 0.849521 | 0.365902 | 0.588294 | 0.708938 | 0.512334 | 0.260719 |

Precision/MCC/NPV finite-patient counts are listed in the coverage table below. Macro-F1 is class-macro within each patient, then averaged across patients. Positive F1 always means EZ-F1. These are not pooled-channel or equal-fold averages.

## Table 2. PR contribution (B minus A)

| Model | ΔACC | ΔEZ-F1 | ΔMacro-F1 | ΔAUROC | ΔAP | ΔMCC | MF1 better patients | MF1 better folds |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | +0.104981 | +0.087559 | +0.127589 | +0.076876 | +0.074114 | +0.106352 | 60/80 | 5/5 |
| RBF SVM | +0.112968 | +0.077038 | +0.129369 | +0.053037 | +0.034527 | +0.099084 | 59/80 | 5/5 |
| Random Forest | +0.141699 | +0.077955 | +0.133237 | +0.090229 | +0.078672 | +0.103484 | 61/80 | 5/5 |
| XGBoost | +0.100305 | +0.063849 | +0.098820 | +0.091412 | +0.072700 | +0.108227 | 57/80 | 5/5 |
| MLP | +0.102198 | +0.038508 | +0.101182 | +0.046338 | +0.047548 | +0.073008 | 57/80 | 5/5 |

All five models improve every fold's mean Macro-F1, but not every individual patient. For metrics whose finite-patient coverage differs, this table subtracts the separately reported group means; paired-common-finite estimates/coverage are separately in paired_pr_intervals.csv. In particular, do not interchange the two MCC differences.

### Descriptive paired uncertainty

10,000 seed42 resamples of the same80 patient IDs; frozen per-fold models and numeric thresholds. All80 patients are valid for the three metrics below. Intervals are unadjusted95% percentile intervals, not multiple-testing-corrected confirmation.

| Model | ΔMacro-F1 [95% interval] | ΔAUROC [95% interval] | ΔAP [95% interval] |
|---|---|---|---|
| Logistic Regression | +0.127589 [+0.081251, +0.173335] | +0.076876 [+0.037270, +0.119159] | +0.074114 [+0.040009, +0.110600] |
| RBF SVM | +0.129369 [+0.086032, +0.173410] | +0.053037 [+0.014051, +0.091192] | +0.034527 [+0.000470, +0.067763] |
| Random Forest | +0.133237 [+0.094181, +0.173588] | +0.090229 [+0.044657, +0.137431] | +0.078672 [+0.042156, +0.116466] |
| XGBoost | +0.098820 [+0.062914, +0.135099] | +0.091412 [+0.046675, +0.136594] | +0.072700 [+0.030238, +0.116138] |
| MLP | +0.101182 [+0.058754, +0.144317] | +0.046338 [+0.013157, +0.077988] | +0.047548 [+0.010650, +0.084370] |

All fifteen listed intervals are above zero, although SVM's AP lower limit is only0.000470. The extensive historical cohort reuse, model/epoch/threshold selection and lack of training-seed replication are not represented in these conditional intervals. No post-Outer tuning followed these results.

## Table 3a. Extended classification — patient macro

| Model | Version | Weighted-F1 | Macro-Precision | Macro-Recall | Kappa | NPV | FPR↓ | FNR↓ |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 14D (no PR) | 0.613414 | 0.633937 | 0.590050 | 0.153852 | 0.820682 | 0.350606 | 0.469295 |
| Logistic Regression | 14D + PR | 0.746790 | 0.650517 | 0.674519 | 0.288482 | 0.828989 | 0.160946 | 0.490017 |
| RBF SVM | 14D (no PR) | 0.603729 | 0.633560 | 0.590026 | 0.160335 | 0.782366 | 0.342599 | 0.477350 |
| RBF SVM | 14D + PR | 0.753465 | 0.653733 | 0.665960 | 0.275480 | 0.825323 | 0.158994 | 0.509086 |
| Random Forest | 14D (no PR) | 0.579274 | 0.628087 | 0.570923 | 0.116451 | 0.805014 | 0.367191 | 0.490963 |
| Random Forest | 14D + PR | 0.737769 | 0.643544 | 0.639661 | 0.234803 | 0.810696 | 0.150735 | 0.569944 |
| XGBoost | 14D (no PR) | 0.628117 | 0.593927 | 0.580511 | 0.126846 | 0.784104 | 0.315541 | 0.523437 |
| XGBoost | 14D + PR | 0.735992 | 0.641821 | 0.641018 | 0.238528 | 0.812713 | 0.148983 | 0.568981 |
| MLP | 14D (no PR) | 0.601135 | 0.613959 | 0.593045 | 0.142039 | 0.806139 | 0.353966 | 0.459945 |
| MLP | 14D + PR | 0.721737 | 0.639391 | 0.637715 | 0.231491 | 0.812555 | 0.150479 | 0.574091 |

Macro-Recall is the requested finite-class mean recall. The explicitly excluded metric has no result column and was not calculated in this experiment. Accuracy is standard (TP+TN)/N, not class-averaged recall.

## Table 3b. Native/calibrated probability metrics — patient macro

| Model | Version | Brier↓ | Log Loss↓ | ECE↓ (10 bins) |
|---|---|---|---|---|
| Logistic Regression | 14D (no PR) | 0.234389 | 0.676231 | 0.288141 |
| Logistic Regression | 14D + PR | 0.225808 | 0.985054 | 0.294854 |
| RBF SVM | 14D (no PR) | 0.176955 | 0.537204 | 0.164712 |
| RBF SVM | 14D + PR | 0.166374 | 0.510849 | 0.181610 |
| Random Forest | 14D (no PR) | 0.211560 | 0.610990 | 0.235558 |
| Random Forest | 14D + PR | 0.188807 | 0.560308 | 0.217510 |
| XGBoost | 14D (no PR) | 0.246766 | 0.697332 | 0.288969 |
| XGBoost | 14D + PR | 0.212666 | 0.619069 | 0.253780 |
| MLP | 14D (no PR) | 0.273724 | 0.891006 | 0.323418 |
| MLP | 14D + PR | 0.244269 | 1.048834 | 0.292535 |

All three probability metrics are **lower is better**. ECE is positive-class reliability error in10 fixed uniform bins [0,0.1),...,[0.9,1]. Log Loss clips only at1e-15 and1−1e-15. SVM probabilities are calibrated exclusively on patient-held-out OOF margins within FIT; the other methods use native probabilities, not guaranteed calibrated clinical posteriors. Class weighting changes their probability interpretation.

PR lowers Brier in all five models, but **LR and MLP Log Loss increase**, and **LR and SVM ECE increase**. It is therefore incorrect to infer universally better calibration from better Macro-F1/AUROC. LR's improved ranking with worse Log Loss is retained, not corrected through test calibration.

## Table 4. Patient-local channel ranking

| Model | Version | MRR | Hit@1 | Hit@3 | Hit@5 | Precision@3 | Recall@3 | NDCG@5 |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 14D (no PR) | 0.648122 | 0.550000 | 0.675000 | 0.812500 | 0.487500 | 0.101952 | 0.498273 |
| Logistic Regression | 14D + PR | 0.737282 | 0.637500 | 0.812500 | 0.850000 | 0.612500 | 0.130995 | 0.599746 |
| RBF SVM | 14D (no PR) | 0.687438 | 0.575000 | 0.775000 | 0.812500 | 0.562500 | 0.112135 | 0.541959 |
| RBF SVM | 14D + PR | 0.648979 | 0.487500 | 0.787500 | 0.837500 | 0.558333 | 0.122331 | 0.545106 |
| Random Forest | 14D (no PR) | 0.631806 | 0.525000 | 0.675000 | 0.762500 | 0.470833 | 0.098829 | 0.472467 |
| Random Forest | 14D + PR | 0.703339 | 0.600000 | 0.750000 | 0.825000 | 0.562500 | 0.115491 | 0.559985 |
| XGBoost | 14D (no PR) | 0.695746 | 0.612500 | 0.750000 | 0.812500 | 0.487500 | 0.101594 | 0.508663 |
| XGBoost | 14D + PR | 0.751614 | 0.662500 | 0.825000 | 0.825000 | 0.604167 | 0.121407 | 0.574025 |
| MLP | 14D (no PR) | 0.670613 | 0.575000 | 0.712500 | 0.850000 | 0.520833 | 0.102186 | 0.522992 |
| MLP | 14D + PR | 0.711361 | 0.600000 | 0.762500 | 0.850000 | 0.570833 | 0.118634 | 0.576845 |

All ranking fields have80 valid patients in this actual cohort. Sorting uses continuous EZ probabilities within patient and canonical channel-name tie breaks; no pooled across-patient ranking is reported. Precision@3 divides by min(3,N), Recall@3 by total patient EZ count, and NDCG@5 uses binary relevance/log2 discount and ideal positive ordering.

SVM MRR falls0.687438→0.648979 and Hit@1 falls0.575000→0.487500 despite AUROC/AP improvements. Their paired intervals include zero; the point declines still prohibit claiming uniformly improved top-channel localization. Precision@3 also falls slightly. XGBoost+PR has the highest MRR/Hit@1 among feature baselines, but LR+PR has higher AUROC/AP/Macro-F1. Different ranking metrics emphasize different parts of the patient ordering.

## Table 5. Frozen D1/M8 reference comparison

| Model | Version | ACC | EZ-F1 | Macro-F1 | AUROC | AP | MRR | Hit@1 |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 14D (no PR) | 0.640687 | 0.341309 | 0.495816 | 0.666886 | 0.470049 | 0.648122 | 0.550000 |
| Logistic Regression | 14D + PR | 0.745668 | 0.428868 | 0.623405 | 0.743762 | 0.544163 | 0.737282 | 0.637500 |
| RBF SVM | 14D (no PR) | 0.638757 | 0.342801 | 0.491935 | 0.681944 | 0.480505 | 0.687438 | 0.575000 |
| RBF SVM | 14D + PR | 0.751726 | 0.419839 | 0.621304 | 0.734981 | 0.515032 | 0.648979 | 0.487500 |
| Random Forest | 14D (no PR) | 0.601841 | 0.298753 | 0.464175 | 0.619541 | 0.427881 | 0.631806 | 0.525000 |
| Random Forest | 14D + PR | 0.743540 | 0.376708 | 0.597412 | 0.709770 | 0.506553 | 0.703339 | 0.600000 |
| XGBoost | 14D (no PR) | 0.641606 | 0.314338 | 0.499114 | 0.625431 | 0.443382 | 0.695746 | 0.612500 |
| XGBoost | 14D + PR | 0.741911 | 0.378188 | 0.597934 | 0.716844 | 0.516082 | 0.751614 | 0.662500 |
| MLP | 14D (no PR) | 0.625437 | 0.327393 | 0.487112 | 0.662600 | 0.464786 | 0.670613 | 0.575000 |
| MLP | 14D + PR | 0.727635 | 0.365902 | 0.588294 | 0.708938 | 0.512334 | 0.711361 | 0.600000 |
| D1 | original frozen reference | 0.748950 | 0.408531 | 0.616556 | 0.732636 | 0.527175 | 0.689399 | 0.575000 |
| M8 | original frozen reference | 0.760166 | 0.434087 | 0.634097 | 0.762397 | 0.554067 | 0.749310 | 0.637500 |

D1/M8 use the exact same80 patients, original five TEST memberships,7,635 identities and metric definitions. All expanded reference metrics were genuinely recomputed from original hash-sealed scores/decisions; the four historical primary values match exactly. Their checkpoints, scalers and thresholds were neither modified nor neurally replayed. They retain their historical threshold selections; no new reference threshold is selected.

The strongest **no-PR Macro-F1** is XGBoost0.499114; the strongest **no-PR AUROC/AP** are SVM0.681944/0.480505. There is no unique best no-PR model across all metrics. With PR, LR is strongest on all four primary discrimination/classification metrics:AUROC0.743762,AP0.544163,Macro-F10.623405,EZ-F10.428868.

| LR+PR minus reference | ΔAUROC | ΔAP | ΔMacro-F1 | ΔEZ-F1 |
|---|---|---|---|---|
| D1 | +0.011127 | +0.016988 | +0.006849 | +0.020337 |
| M8 | -0.018634 | -0.009904 | -0.010692 | -0.005219 |

LR+PR exceeds D1's four point estimates but remains below complete M8 byAUROC0.018634,AP0.009904,Macro-F10.010692 andEZ-F10.005219. These best-versus-reference gaps are descriptive; no paired significance claim for this retrospectively best feature baseline is made. XGBoost+PR's MRR0.751614 and Hit@1=0.662500 are above M8's0.749310/0.637500 point estimates, despite lower overall AUROC/AP/F1. Do not declare complete M8 superior on every ranking metric.

**Scientific implication:** the strong linear PR baseline means the gain cannot be attributed solely to a specialized deep architecture. Future comparisons should include LR+PR, not only weak no-PR baselines. This does not isolate the complete M8 Raw-correction package or establish why its remaining gap exists.

## Pooled-channel results — separate estimand

| Model | Version | ACC | Precision | Recall | Specificity | EZ-F1 | Macro-F1 | AUROC | AP | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 14D (no PR) | 0.633137 | 0.327462 | 0.576018 | 0.650034 | 0.417550 | 0.574897 | 0.645571 | 0.331496 | 0.193550 |
| Logistic Regression | 14D + PR | 0.743811 | 0.439108 | 0.440620 | 0.833503 | 0.439863 | 0.636895 | 0.693916 | 0.378526 | 0.273792 |
| RBF SVM | 14D (no PR) | 0.640995 | 0.329809 | 0.554791 | 0.666497 | 0.413690 | 0.577491 | 0.617731 | 0.319777 | 0.190971 |
| RBF SVM | 14D + PR | 0.751277 | 0.453901 | 0.440620 | 0.843177 | 0.447162 | 0.643353 | 0.682811 | 0.393689 | 0.286804 |
| Random Forest | 14D (no PR) | 0.603667 | 0.291788 | 0.515777 | 0.629667 | 0.372720 | 0.541520 | 0.589300 | 0.296998 | 0.124433 |
| Random Forest | 14D + PR | 0.747348 | 0.439532 | 0.387837 | 0.853700 | 0.412069 | 0.625586 | 0.659954 | 0.376201 | 0.252771 |
| XGBoost | 14D (no PR) | 0.630124 | 0.305925 | 0.488812 | 0.671928 | 0.376325 | 0.556716 | 0.594354 | 0.324632 | 0.140159 |
| XGBoost | 14D + PR | 0.745383 | 0.436433 | 0.395869 | 0.848778 | 0.415162 | 0.626215 | 0.691653 | 0.405370 | 0.253415 |
| MLP | 14D (no PR) | 0.615717 | 0.303140 | 0.526104 | 0.642227 | 0.384648 | 0.552636 | 0.601534 | 0.314534 | 0.144455 |
| MLP | 14D + PR | 0.736739 | 0.411765 | 0.357430 | 0.848948 | 0.382678 | 0.607687 | 0.662320 | 0.350832 | 0.217308 |

All7,635 Outer channels are pooled once per group. Each channel's class decision uses its own fold's frozen threshold; no new global threshold is optimized. Pooled AUROC/AP use the saved continuous probabilities, not binary classes. Patient size and cross-patient calibration affect these values; they must not replace the primary patient-equal table. Full pooled extended statistics and TP/TN/FP/FN are in pooled_metrics.csv and confusion_matrices.csv.

## Five-fold mean ± sample SD

| Model | Version | Fold mean±SD AUROC | Fold mean±SD AP | Fold mean±SD Macro-F1 | Fold mean±SD EZ-F1 |
|---|---|---|---|---|---|
| Logistic Regression | 14D (no PR) | 0.666208 ± 0.039443 | 0.470630 ± 0.091400 | 0.495843 ± 0.054293 | 0.342154 ± 0.089002 |
| Logistic Regression | 14D + PR | 0.742360 ± 0.060432 | 0.543810 ± 0.064208 | 0.622687 ± 0.046989 | 0.427975 ± 0.089943 |
| RBF SVM | 14D (no PR) | 0.682458 ± 0.032556 | 0.481563 ± 0.073509 | 0.492162 ± 0.031234 | 0.343398 ± 0.089498 |
| RBF SVM | 14D + PR | 0.734094 ± 0.057490 | 0.515390 ± 0.063944 | 0.620831 ± 0.042751 | 0.419953 ± 0.079942 |
| Random Forest | 14D (no PR) | 0.618523 ± 0.041273 | 0.427347 ± 0.051896 | 0.463870 ± 0.025495 | 0.297935 ± 0.101790 |
| Random Forest | 14D + PR | 0.708116 ± 0.056922 | 0.506065 ± 0.085900 | 0.596166 ± 0.053659 | 0.375693 ± 0.102124 |
| XGBoost | 14D (no PR) | 0.624490 ± 0.066975 | 0.443143 ± 0.094618 | 0.499002 ± 0.029312 | 0.313392 ± 0.099858 |
| XGBoost | 14D + PR | 0.715396 ± 0.048353 | 0.515636 ± 0.066787 | 0.596562 ± 0.047450 | 0.376653 ± 0.083306 |
| MLP | 14D (no PR) | 0.660697 ± 0.067262 | 0.464009 ± 0.086380 | 0.486750 ± 0.026571 | 0.326354 ± 0.108393 |
| MLP | 14D + PR | 0.708151 ± 0.034841 | 0.512299 ± 0.086083 | 0.587584 ± 0.041595 | 0.365025 ± 0.086660 |

This section weights the five fold means equally; SD uses ddof=1. It is deliberately distinct from the80-patient mean because original fold sizes are16/16/17/15/16 (channels1,631/1,498/1,746/1,425/1,335). All individual fold values, both patient-macro and pooled, are in outer_fold_metrics.csv. Four-center aggregates are in center_metrics.csv; no source is selectively omitted.

## Undefined cases and coverage

| Model | Version | Precision n | MCC n | NPV n | ACC/F1/AUROC/AP/ranking n |
|---|---|---|---|---|---|
| Logistic Regression | 14D (no PR) | 73 | 71 | 78 | 80 |
| Logistic Regression | 14D + PR | 80 | 80 | 80 | 80 |
| RBF SVM | 14D (no PR) | 72 | 71 | 79 | 80 |
| RBF SVM | 14D + PR | 79 | 79 | 80 | 80 |
| Random Forest | 14D (no PR) | 69 | 67 | 78 | 80 |
| Random Forest | 14D + PR | 75 | 75 | 80 | 80 |
| XGBoost | 14D (no PR) | 75 | 73 | 78 | 80 |
| XGBoost | 14D + PR | 79 | 79 | 80 | 80 |
| MLP | 14D (no PR) | 72 | 71 | 79 | 80 |
| MLP | 14D + PR | 80 | 80 | 80 | 80 |

No patient was dropped. All80 patients actually have both truth classes, so AUROC/AP/positive F1 and ranking are available for every patient in all ten groups. Some models predict one class for individual patients: undefined precision, NPV or MCC remain NaN, never an arbitrary zero or0.5. Means use finite values with explicit denominators. Macro-Precision averages the defined class precisions; undefined class precision is not silently replaced by zero. Per-metric complete counts are in metric_coverage.csv; reference counts are in reference_metric_coverage.csv.

## Execution, training and leakage audit

- Cached DELTA14 exactly equals the original pre-PR D1 R14; no EEG/EDF processing or feature re-extraction occurred. Original source/cache/split hashes are recorded in the protocol and checked on server.
- The seven cached physiological families are beta log-bandpower, low-gamma log-bandpower, partial high-gamma power, RMS, variance, line length/sec and spectral entropy. The unchanged Raw250 definition has only partial80–125Hz high gamma and restricted10–125Hz entropy. Per-window descriptors subtract the same-channel reference from all genuine pre-reference windows (center<0), then use all genuine within-seizure windows and available-seizure mean/populationSD, in feature-major mean/std order. That is the fixed14D input; no complete80–150Hz band or new frequency content is claimed.
- PR calls the pinned original uas_core.patient_z. Float64 patient/channel population mean/SD and denominator protection are unchanged. PR plus original FIT scaling reproduces the original D1 legal bank elementwise. This is the physiological14D PR, not M8's separate96D Raw-latent PR.
- All versions use identical original FIT/validation/TEST roles. No TEST patient is used for fitting, hyperparameter choice or numeric threshold selection. Patient-local unlabeled TEST statistics are used only for PR at inference, as allowed by M8 conditions.
- Same moderate grids, selected hyperparameters, seed and procedure within each A/B pair. Common configuration maximizes mean validation patient Macro-F1 across A/B, then AP/AUROC and fixed grid order; each version independently selects its own validation EZ threshold. This is a prospectively fixed paired policy, not separately maximized AutoML. Only3 LR and4 candidates per other classifier are tested per fold/version:190 candidate fits,50 selected cells. No final refit on FIT+validation.
- FIT sample weights first equalize total patient weight, then apply original-FIT class-frequency weights. LR/SVM/MLP use FIT-only StandardScaler. RF/XGBoost retain native unscaled inputs. This is an algorithm-and-input-transform comparison, not proof of a unique physiological mechanism; representation units/numerical behavior can differ across classifiers.
- SVM uses three **patient-disjoint** calibration subfolds inside FIT; every subfold refits its imputer/scaler using only sub-FIT. Platt regression sees FIT OOF margins, never VAL/TEST labels. The final SVC fits FIT only. No test-dependent calibration or score orientation change.
- XGBoost3.0.5 was installed in an isolated D: environment; original runtime and unrelated GPU jobs were not modified. All five methods trained on CPU. NumPy2.3.5/SciPy1.16.3/sklearn1.7.2 versions are recorded.
- **Convergence limitation:**37/40 MLP candidate fits, including9/10 selected MLP cells, reached the predeclared400-iteration cap with ConvergenceWarning. They produce valid finite predictions and are fully reported, but are not certified optimizer-converged. No fit was removed, extended or rescued after Outer. Other classifier candidates emitted no captured convergence warning. Therefore these are fixed-budget MLP results, not the best achievable MLP claim.
- All50 selected models, imputers/scalers/calibrators and validation thresholds were hash-frozen before Outer inference. All50 probability/prediction ledgers were atomically sealed before the new full TEST-label read. Original labels were historically exposed elsewhere; this is not a claim of global label blindness.
- Eight synthetic metric/leakage contracts passed before training. Independent sklearn/confusion/calibration/ranking checks covered960 patient-model rows (800 new +160 frozen references), and public summaries/paired bootstrap were verified. Private prediction CSV round trip is exact for all76,350 scores/decisions. Train/Outer/verification stderr were empty; optimizer warnings are captured explicitly in VALIDATION_SEARCH.csv.

Freeze SHA256: 5d69a5a885e0e170d2e6253c53269d8853478e8f4720e9efa44abdfae08a06aa.

Prediction seal SHA256: 602ae16b25304267234f2f9e0a3220e391864c9529eb2ce3853a71abeddc405b.

Saved private predictions SHA256: f2af65de226cb481a37f5bfddd191a45951dbf1548b58fe1760c6fc1d07b1d33.

Saved private patient metrics SHA256: d830d102c6f1ae6e6466ec0242d51a9bc44f2905f1da1b08a9e8d976caa2063e.

## Selected matched configurations and numeric thresholds

| Model | Fold | Common grid config | A EZ threshold | B EZ threshold |
|---|---|---|---|---|
| Logistic Regression | 1 | {"C": 0.1} | 0.425000 | 0.515000 |
| Logistic Regression | 2 | {"C": 0.1} | 0.485000 | 0.625000 |
| Logistic Regression | 3 | {"C": 10} | 0.530000 | 0.575000 |
| Logistic Regression | 4 | {"C": 0.1} | 0.475000 | 0.575000 |
| Logistic Regression | 5 | {"C": 10} | 0.555000 | 0.515000 |
| RBF SVM | 1 | {"C": 1, "gamma": "scale"} | 0.280000 | 0.240000 |
| RBF SVM | 2 | {"C": 1, "gamma": 0.01} | 0.275000 | 0.440000 |
| RBF SVM | 3 | {"C": 10, "gamma": "scale"} | 0.290000 | 0.330000 |
| RBF SVM | 4 | {"C": 1, "gamma": 0.01} | 0.215000 | 0.300000 |
| RBF SVM | 5 | {"C": 1, "gamma": 0.01} | 0.240000 | 0.270000 |
| Random Forest | 1 | {"max_depth": null, "min_samples_leaf": 5} | 0.355000 | 0.395000 |
| Random Forest | 2 | {"max_depth": 8, "min_samples_leaf": 1} | 0.510000 | 0.680000 |
| Random Forest | 3 | {"max_depth": 8, "min_samples_leaf": 5} | 0.600000 | 0.525000 |
| Random Forest | 4 | {"max_depth": null, "min_samples_leaf": 5} | 0.415000 | 0.495000 |
| Random Forest | 5 | {"max_depth": null, "min_samples_leaf": 1} | 0.250000 | 0.410000 |
| XGBoost | 1 | {"max_depth": 2, "n_estimators": 500} | 0.505000 | 0.510000 |
| XGBoost | 2 | {"max_depth": 2, "n_estimators": 200} | 0.545000 | 0.750000 |
| XGBoost | 3 | {"max_depth": 2, "n_estimators": 200} | 0.550000 | 0.575000 |
| XGBoost | 4 | {"max_depth": 4, "n_estimators": 200} | 0.480000 | 0.670000 |
| XGBoost | 5 | {"max_depth": 4, "n_estimators": 500} | 0.580000 | 0.675000 |
| MLP | 1 | {"alpha": 0.0001, "hidden": [32]} | 0.420000 | 0.715000 |
| MLP | 2 | {"alpha": 0.01, "hidden": [32]} | 0.660000 | 0.760000 |
| MLP | 3 | {"alpha": 0.01, "hidden": [32]} | 0.590000 | 0.725000 |
| MLP | 4 | {"alpha": 0.01, "hidden": [32]} | 0.600000 | 0.695000 |
| MLP | 5 | {"alpha": 0.01, "hidden": [32]} | 0.535000 | 0.595000 |

All thresholds above apply to calibrated/native **pEZ ≥ threshold**. Frozen D1/M8 retain their original NEZ-oriented decisions, converted without altering predictions. Complete per-candidate validation results, iteration counts/warnings, seeds and model digests remain in the anonymous selection/search CSVs.

## Privacy and stop

The public directory contains source, protocol/hash records, anonymous aggregate outputs and this report only. Server-private private/results/predictions.csv contains labels, class decisions, continuous probability/decision scores, patient/channel identity, fold/model/version and model hash; private/results/per_patient_metrics.csv contains individual metrics. Both remain on the trusted server along with weights, preprocessors, calibration state and logs. They are not omitted scientifically and are not uploaded publicly. Expanded reference patient data likewise remain private.

The optional reference-coverage helper is a read-only post-inference reporting audit; it does not change the frozen training/evaluation source or predictions. No installed skill, additional seed/model family, post-Outer parameter adjustment or second scientific test is used. Work stops after these ten groups. Status:COMPLETE_VERIFIED. Conclusion:PR has broad average discrimination/classification benefit across these five classifiers, **not a guarantee of better top-channel ranking, calibration, every patient, or future independent cohorts**.
