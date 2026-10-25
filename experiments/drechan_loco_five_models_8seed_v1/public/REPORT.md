# Five-model four-source LOCO: fixed seeds42 through49

All160 logical selections frozen before any NEW test; all160 sealed score cells before metrics.60 historical seeds42/43/44 reuses,100 fresh fits.
Original seed42 patient roles held FIXED. Neural initialization/order/dropout and SVM OOF calibration vary with training seed.
Primary: mean8 endpoints for each patient, then mean80patients; NOT bestseed, selected subset or a probability ensemble.
Additional batch45-49 fixed AFTER viewing42-44. All8 retained; no outcome-dependent stopping. Exploratory repeated-test historical cohort, not confirmation.

| Model | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-Frac. Bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OURS_RAW_FEATURE | 0.623160 | 0.746108 | 0.548156 | 0.427220 | 0.819100 | 0.736048 | 0.772603 | 0.678125 | -0.002877 |
| OURS_RAW | 0.595271 | 0.708430 | 0.515347 | 0.389554 | 0.800988 | 0.711981 | 0.749152 | 0.657813 | 0.005688 |
| OURS_FEATURE | 0.615033 | 0.748471 | 0.541640 | 0.421405 | 0.808661 | 0.728729 | 0.739521 | 0.632812 | 0.008806 |
| LINEAR_SVM_FEATURE_PR | 0.610067 | 0.738479 | 0.537266 | 0.409231 | 0.810903 | 0.726921 | 0.728211 | 0.637500 | -0.000849 |
| TIMECONVCNN_RAW | 0.433001 | 0.591377 | 0.383493 | 0.198717 | 0.667284 | 0.629673 | 0.513736 | 0.389062 | 0.055255 |
