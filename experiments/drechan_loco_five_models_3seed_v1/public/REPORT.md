# Five-model four-source LOCO: seeds42/43/44

All60 logical selections frozen before any new test; all60 sealed score cells before metrics.20 strict historical seed42 reuses,40 fresh fits.
Original seed42 patient roles held FIXED. Neural initialization/order/dropout and SVM OOF calibration vary with training seed.
Primary: mean3 endpoints for each patient, then mean80patients; NOT bestseed or a probability ensemble.
Exploratory repeatedly viewed historical cohort; three seeds do not confer external confirmation.

| Model | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-Frac. Bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OURS_RAW_FEATURE | 0.625048 | 0.745515 | 0.549025 | 0.430256 | 0.819839 | 0.737152 | 0.772382 | 0.675000 | -0.002879 |
| OURS_RAW | 0.589337 | 0.706683 | 0.517056 | 0.384000 | 0.794674 | 0.704760 | 0.760578 | 0.679167 | 0.014178 |
| OURS_FEATURE | 0.615259 | 0.750889 | 0.542931 | 0.419440 | 0.811079 | 0.730702 | 0.745588 | 0.637500 | 0.001129 |
| LINEAR_SVM_FEATURE_PR | 0.610443 | 0.738479 | 0.537266 | 0.409868 | 0.811018 | 0.727064 | 0.728211 | 0.637500 | -0.000358 |
| TIMECONVCNN_RAW | 0.446847 | 0.586908 | 0.376882 | 0.179052 | 0.714642 | 0.678147 | 0.492577 | 0.370833 | 0.001470 |
