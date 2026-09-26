# Frozen A1 validation failure stratification

Diagnostic only: 65 VLOO excluded-validation patient-fold cases. No outer test was run.
Types use pooled medians of an equal-weight AUPRC/AUROC/MRR percentile score and Macro-F1. Type-NEITHER is the nonfailure remainder.
No multiple-hypothesis significance claims are made. Small centers are not interpreted as effects.

| Predictor | Pearson r vs Macro-F1 | Spearman rho |
| --- | ---: | ---: |
| patient_ez_auprc | 0.7418744219958326 | 0.7445255481564289 |
| patient_ez_mrr | 0.6155085359663128 | 0.6276082248449543 |
| patient_ez_auroc | 0.7552465078295324 | 0.7947025664306792 |
| n_seizures | 0.0458509597127268 | -0.027185699333600405 |
| n_channels | -0.2733329089114663 | -0.21506065780240952 |
| true_ez_fraction | 0.010814267872262199 | 0.10783692786537903 |
| valid_window_count | 0.04950245908624923 | -0.025494582483974813 |
| missing_invalid_window_fraction |  |  |
| mean_prediction_entropy | -0.4614414567776577 | -0.4510489510489511 |
| median_probability_margin | 0.36659250096716245 | 0.33400349650349653 |

Missing/invalid-window fractions follow the cache window masks; if all recorded masks are valid, this variable has no variance and cannot diagnose signal quality.
Center, seizure-count and EZ-burden aggregates are in the accompanying CSVs. Patient IDs, labels and individual predictions remain private.

Terminal: `FAILURE_STRATIFICATION_COMPLETE`.
