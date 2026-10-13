# Conditional information and feature usefulness audit

All values below are from this executed FIT/validation study unless explicitly identified as historical. No outer outcomes, raw EEG or new windows were loaded. Nonzero residuals are not independent information and do not establish feature relevance.

## What is demonstrably redundant or unavailable?

- **12 retained constant-zero positions:** delta/theta frequency-band coordinates in the three retained views, at both seizure statistics. These supply no varying numeric information in this exact cache. This is not a conclusion that 1–8 Hz physiology is irrelevant.
- **Eight retained auxiliary DELTA duplicates:** the same high-gamma, RMS, variance and line-length values as their main coordinates. They provide no additional measurements. Removing them can nevertheless change LayerNorm weighting; this study preserves them.
- **26 ZDELTA coordinates:** intentionally masked under F0, not newly selected as useless by COVE.
- **Seven remaining feature types:** this study does not train feature-deletion models. It cannot declare their original ABS/DELTA/RATIO inputs useless merely because a correction group has little influence.

## Actual conditional predictability and group correction effects

R² values are five-fold means of FIT-patient-equal, patient-held-out OOF predictions, not in-sample fits. Negative R² is retained. The final column is full trained B3 AUROC minus the AUROC after disabling **that group's two corrections**, while preserving all original inputs. It is descriptive and is not a subset-selection rule.

| Physiological group | D from ABS: OOF R² | R from ABS+D: OOF R² | ΔAUROC from its COVE correction |
|---|---:|---:|---:|
| Beta mean | 0.707008 | 0.105870 | ≈0 |
| Beta std | 0.423836 | 0.056919 | ≈0 |
| Low-gamma mean | -0.290188 | -0.014834 | ≈0 |
| Low-gamma std | 0.263415 | 0.087813 | ≈0 |
| High-gamma mean | -0.033617 | -0.019700 | ≈0 |
| High-gamma std | 0.193745 | 0.003492 | ≈0 |
| RMS mean | 0.395863 | 0.853291 | +0.000021831 |
| RMS std | 0.518447 | 0.814034 | +0.000012327 |
| Variance mean | 0.721248 | 0.487064 | +0.000010842 |
| Variance std | 0.765020 | 0.335658 | +0.000010842 |
| Line length mean | 0.243521 | 0.680456 | -0.000021617 |
| Line length std | 0.238658 | 0.514236 | -0.000003034 |
| Spectral entropy mean | 0.220018 | 0.572008 | +0.000064713 |
| Spectral entropy std | 0.324911 | 0.337011 | +0.000066968 |

RMS RATIO is substantially linearly predictable, but remains nonzero after conditioning. Gamma RATIO is poorly predicted by these Ridge fits. None of the140 FIT fold/group/residual standard deviations falls below the registered1e-6 normalization floor. This only establishes nondegeneracy of the **FIT OOF** coordinates, not their utility or stability on target patients. All group AUROC correction effects are smaller than0.0001. There is no justified useful-subset selection from this table.

## Numerical scale and OOF-to-target instability

Additional label-free diagnostics inspect **legal original FIT/VAL banks only**. They do not change features, normalizers, thresholds or checkpoints. `FEATURE_GEOMETRY.csv` reports pre-Patient-Z cache magnitudes; `PATIENT_Z_FLOOR_AUDIT.csv` counts use of the **existing** pinned rule `denominator = SD if SD>1e-8 else1`, not a newly fitted cutoff. Five folds each contain13 validation appearances, so the following averages are also equal-65-appearance proportions.

| DELTA coordinate | Validation raw mean within-role SD | Validation appearances using original denominator1 |
|---|---:|---:|
| Beta mean / std | 4.74e-9 / 2.67e-9 | 93.85% / 96.92% |
| Low-gamma mean / std | 6.60e-9 / 8.11e-9 | 100% / 96.92% |
| High-gamma mean / std | 4.41e-10 / 5.06e-10 | 100% / 100% |
| RMS mean / std | 4.00e-5 / 1.57e-5 | 0% / 10.77% |
| Variance mean / std | 4.80e-8 / 1.83e-8 | 63.08% / 80% |
| Line length mean / std | 3.48e-3 / 3.12e-3 | 0% / 10.77% |
| Entropy mean / std | 5.43e-2 / 2.14e-2 | 0% / 10.77% |

These are statistics in each feature's own units, not a common biological noise threshold. Small raw values can be amplified or treated differently by the original normalization/scaling contract. The actual cause of the tiny band-DELTA cache values is not established by this study; raw/window provenance re-extraction was not authorized. Do not label them recovered physiological innovations.

In particular, validation-to-OOF residual-SD ratios for beta/gamma DELTA can collapse to roughly1e-10–1e-9; beta mean also has one fold with ratio7.67949. RATIO ratios are much less variable (beta/gamma roughly0.90–1.05). RMS, line length and entropy show more comparable residual scales, though not uniform center/patient stability. `OOF_FULLFIT_DRIFT.csv` separately compares full-FIT nuisance residuals on FIT and validation, to avoid attributing the entire shift to new-patient differences. Center distributions, patient-level SD summaries, means, correlations and extreme quantiles are in the registered audit CSVs. Near-zero group behavior is reported, not filtered away.

## FIT-only predictive probe

| Probe | AUROC | AUPRC | Log loss (lower is better) |
|---|---:|---:|---:|
| P0 original42 effective coordinates | 0.747010 | 0.559477 | 0.630872 |
| P1 plus14 correctly paired residual products | 0.745414 | 0.558259 | 0.630620 |
| P2 plus14 wrong-pair products | 0.747011 | 0.559746 | 0.629953 |

These are equal-patient inner-held-out **FIT** scores (255 FIT appearances across the original folds), not the65-appearance Development benchmark or outer TEST. The LR settings and feature families are matched. Nuisance/normalizer fitting is nested within each inner training subset; original outer-FIT feature preprocessing is fixed and shared by all probes, not refit in this diagnostic. No validation labels fit a probe or nuisance model. P1 provides no AUROC/AP benefit and its small log-loss change is not specific to correct matching. The probe is not conditional mutual information, and it did not control continuation: the numeric Stage A gate passed and all15 neural cells ran.

## What can be concluded about combinations?

The current measurements support neither the correct residual-product combination over concatenation, nor correct correspondence over wrong correspondence. Entropy/RMS/variance corrections cause tiny ranking changes; line-length correction trades some thresholded F1 for worse rank discrimination. This is not evidence that the underlying original features should be removed. The defensible outcome is **no demonstrated useful COVE interaction in the present interface**, plus a concrete cache-scale/normalization issue to audit independently before inventing a more complex interaction model. No feature subset is trained or recommended as a validated winner here.
