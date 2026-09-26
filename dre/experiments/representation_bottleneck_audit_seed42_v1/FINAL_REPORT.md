# Representation bottleneck audit: seed 42

Development-only on the historically viewed 80-patient cohort. No new outer-test loader, predictions or performance evaluation were used.
Frozen A1 source reproduction: **PASS**, 150 checkpoints, maximum validation-grid error 0.0; mean VLOO Macro-F1 0.6259962097.

| Mechanism | Baseline | Matched control | Candidate | Δ baseline | Δ matched | Positive folds baseline/control | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| temporal_order | 0.625996 | 0.616297 | 0.617843 | -0.008153 | +0.001546 | 1/5 / 2/5 | TEMPORAL_ORDER_NOT_SUPPORTED |
| recruitment_rank | 0.625996 | 0.623035 | 0.614271 | -0.011725 | -0.008764 | 0/5 / 1/5 | RECRUITMENT_RANK_NOT_SUPPORTED |
| cross_seizure_persistence | 0.625996 | 0.618165 | 0.620693 | -0.005303 | +0.002528 | 2/5 / 4/5 | CROSS_SEIZURE_PERSISTENCE_NOT_SUPPORTED |
| remove_ABS | 0.625996 | 0.625996 | 0.630672 | +0.004676 | +0.004676 | 3/5 / 3/5 | inconclusive |
| remove_RATIO | 0.625996 | 0.625996 | 0.627944 | +0.001948 | +0.001948 | 2/5 / 2/5 | inconclusive |
| delta_zdelta_only | 0.625996 | 0.625996 | 0.616871 | -0.009125 | -0.009125 | 1/5 / 1/5 | useful |

A/B/C support requires every prelocked mean, matched-control, fold-consistency, EZ-F1, AUPRC and absolute-performance check. D uses separate conservative removability/usefulness rules; it is not a new model-selection sweep.

## Prespecified questions

- Chronological temporal order beyond shuffled control: **TEMPORAL_ORDER_NOT_SUPPORTED**.
- Cross-channel recruitment rank beyond magnitude-only evidence: **RECRUITMENT_RANK_NOT_SUPPORTED**.
- Cross-seizure persistence beyond mean/std aggregation: **CROSS_SEIZURE_PERSISTENCE_NOT_SUPPORTED**. The >=2-seizure subset is diagnostic only.
- Four-view audit: **VIEW_REDUNDANCY_IDENTIFIED**. ABS removal and RATIO removal are individually inconclusive. Removing both to retain only DELTA+ZDELTA lowers Macro-F1 by 0.009125 in 4/5 folds; the pair is jointly useful, but neither view is individually proven necessary.
- Candidates reaching VLOO Macro-F1 >=0.640: none.
- Frozen probes with >=1 pp gain over A1: none.
- Frozen probes with positive matched-control mean: temporal_order, cross_seizure_persistence.
- Fold consistency is listed explicitly in the table; a positive mean alone is insufficient.

## Frozen A1 failure stratification

65 excluded-validation patient-fold cases; label-using variables are diagnostic only. Correlations are descriptive without multiplicity-based significance claims.
| Predictor | Pearson r with Macro-F1 | Spearman rho |
| --- | ---: | ---: |
| patient_ez_auprc | 0.7418744219958326 | 0.7445255481564289 |
| patient_ez_mrr | 0.6155085359663128 | 0.6276082248449543 |
| patient_ez_auroc | 0.7552465078295324 | 0.7947025664306792 |
| n_seizures | 0.0458509597127268 | -0.027185699333600405 |
| n_channels | -0.2733329089114663 | -0.21506065780240952 |
| true_ez_fraction | 0.010814267872262199 | 0.10783692786537903 |
| valid_window_count | 0.04950245908624923 | -0.025494582483974813 |
| missing_invalid_window_fraction |  |  |

Failure-type counts (fixed pooled-median definitions): Type-RANK=4, Type-DECISION=4, Type-BOTH=28, Type-NEITHER=29.
Among the 32 low-Macro-F1 cases, 28 also have poor ranking and 4 have good ranking; this is a descriptive ranking-associated failure pattern, not causal evidence or an independent prediction test.
Center descriptive aggregates (n patients, mean Macro-F1): hup (21, 0.662), lzu (11, 0.565), multicenter (9, 0.708), pediatric (6, 0.539). Small-n centers are not interpreted as causal effects.
Seizure count and true EZ fraction have near-zero rank correlation with Macro-F1 here; channel count is moderately negative. Window-mask missingness has zero variation, so window quality cannot be diagnosed from this cache field.

## Decision

Mechanisms passing their own frozen development gates: four_view_redundancy_or_necessity.
The identified D result is joint view necessity, not a successful replacement model: no frozen probe improved A1, no candidate reached 0.640 VLOO Macro-F1, and the individual ABS/RATIO roles remain unresolved. It does not by itself justify a constructive redesign.
These exploratory diagnostics alone are not sealed confirmation and do not authorize an outer-test claim.
Exact terminal: `REPRESENTATION_BOTTLENECK_IDENTIFIED`.
