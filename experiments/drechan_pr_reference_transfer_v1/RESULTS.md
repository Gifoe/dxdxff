# Results: patient reference transfer across disjoint channel sets

All 80 patients;  7,635 channels;  100 bidirectional partitions per patient. Twelve fixed physiological coordinates. No classification model or F1/AUROC/AP endpoint.

## Primary background forecasts

Lower CRPS is better. Losses are in the same TRAIN-standardized units; each patient has equal weight. Brackets are paired-stratum-bootstrap nominal 95% intervals.

| Probe | CRPS [95% CI] |
| --- | --- |
| GLOBAL | 0.450934 [0.379316, 0.537337] |
| SOURCE | 0.452609 [0.378295, 0.541441] |
| CENTER | 0.313221 [0.285844, 0.347953] |
| SCALE | 0.435714 [0.350676, 0.537191] |
| FULL | 0.173431 [0.131515, 0.226682] |

Primary GLOBAL-FULL gain: **0.277503 [0.221522, 0.351705]**. Positive-patient fraction: 0.9875.

## Prespecified pooled contrasts

Primary has nominal 95% bounds; all other rows show additional six-family adjusted bounds. Positive means lower loss for the right-hand method.

| Contrast | Gain [interval] |
| --- | --- |
| GLOBAL-FULL | 0.277503 [0.221522, 0.351705] |
| GLOBAL-CENTER | 0.137713 [0.0687818, 0.239759] |
| GLOBAL-SCALE | 0.0152206 [-0.014549, 0.0377846] |
| CENTER-FULL | 0.13979 [0.109881, 0.160299] |
| SCALE-FULL | 0.262283 [0.182984, 0.373999] |
| SOURCE-FULL | 0.279178 [0.204479, 0.384891] |

Full EZ-minus-NEZ absolute relative atypicality (six-family adjusted): **0.958464 [-0.2508, 2.56101]**.
Unsupported query-coordinate fraction: 0.04375 [0.01875, 0.075]. Unsupported-reference but query residual>1e-8 fraction: 0 [0, 0].

The support rule zeros unsupported relative coordinates, but CRPS retains every coordinate and treats SD0 as a point distribution. These are different declared rules.

## All four sources

| Source | Global CRPS | Full CRPS | Global-Full [95% CI] | EZ-NEZ gap [95% CI] |
| --- | --- | --- | --- | --- |
| hup | 0.43163 | 0.190817 | 0.240813 [0.20294, 0.282059] | 0.369851 [-0.765632, 1.10607] |
| lzu | 0.387252 | 0.183986 | 0.203266 [0.168916, 0.235753] | 1.57305 [-0.781031, 4.84644] |
| multicenter | 0.491288 | 0.176551 | 0.314736 [0.188589, 0.513559] | 1.93132 [0.245247, 4.99097] |
| pediatric | 0.629305 | 0.0616326 | 0.567673 [0.257842, 1.15875] | 0.169825 [-0.223165, 0.567281] |

## All12 coordinates

| Coordinate | Global-Full CRPS gain [95% CI] | EZ-NEZ relative gap [95% CI] |
| --- | --- | --- |
| beta_mean | 0.233167 [0.175068, 0.325795] | 2.11952 [0.715167, 4.61287] |
| beta_SD | 0.184417 [0.144991, 0.209969] | 0.883533 [0.380234, 1.56374] |
| low_gamma_mean | 0.305416 [0.172028, 0.488185] | 2.10447 [0.325946, 4.74856] |
| low_gamma_SD | 0.305033 [0.184121, 0.476918] | 0.533463 [-1.1876, 2.12403] |
| partial_high_gamma_mean | 0.165404 [0.090373, 0.216076] | 2.9178 [-0.087073, 7.36556] |
| partial_high_gamma_SD | 0.175186 [0.108254, 0.222883] | -0.471729 [-5.19574, 3.56338] |
| RMS_mean | 0.250646 [0.189121, 0.337366] | 0.587826 [0.388473, 0.81653] |
| RMS_SD | 0.220203 [0.168713, 0.27957] | 0.423588 [0.271175, 0.58054] |
| line_length_mean | 0.367793 [0.25099, 0.513556] | 0.929576 [0.549568, 1.45696] |
| line_length_SD | 0.512431 [0.260545, 0.871345] | 0.477372 [0.282563, 0.701677] |
| spectral_entropy_mean | 0.339653 [0.264696, 0.430696] | 0.591027 [0.392519, 0.836648] |
| spectral_entropy_SD | 0.270691 [0.209965, 0.337332] | 0.405113 [0.223089, 0.641123] |

## Conditional label-shuffle reference

| Scope | Observed gap | Null central95% | Plus-one upper fraction |
| --- | --- | --- | --- |
| ALL | 0.958464 | [-0.616506, 1.02952] | 0.037962 |
| hup | 0.369851 | [-0.58289, 1.60403] | 0.253746 |
| lzu | 1.57305 | [-1.59756, 2.16764] | 0.0959041 |
| multicenter | 1.93132 | [-0.247477, 1.31765] | 0.000999001 |
| pediatric | 0.169825 | [-0.201238, 0.372737] | 0.167832 |

This preserves within-patient EZ counts and frozen features/references. It is a conditional diagnostic, not an exact population-level test. It cannot replace the patient-bootstrap interval.

## Complete machine-readable results

BACKGROUND.csv,CONTRASTS.csv,ANOMALY.csv and SUPPORT.csv retain all five scopes and 13 coordinate summaries. PERMUTATION.csv retains all 1,000 draws per scope. Source/coordinate intervals are descriptive. The six pooled family slots were fixed before outcomes.

## Scope and uncertainty

This is historical-cohort exploratory assessment,conditional on fixed overlapping TRAIN banks. The 100 splits do not create 8,000 independent patients.79 assessments lie inside original FIT; one is the previously declared original-validation exception. All 80 remain absent from their own TRAIN/CAL. No strict79 complementary table is exported.
These Gaussian marginal probes measure reference transfer, not true Gaussianity,healthy baseline,technical nuisance,biological causality or new clinical effectiveness. Electrode sampling/anatomy can contribute to the patient reference. No spatial independence of channels is asserted; resampling operates on patients.
The prior mechanism study found no improvement in transferable standardized separation. That negative/mixed result remains unchanged and must accompany any broader PR narrative.
