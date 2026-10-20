# Fixed no-Late-Gate DREChannNet: seed42 ablations

The main is the exact historical seed42 w/o Late Gate, selected following already viewed Outer results. This study is exploratory repeated-test, not independent confirmation. Variance and Head Interaction remain absent. No result in this run is used to redesign or reselect the main.

All16 current-architecture groups have Late Gate physically removed before training. PREVIOUS_FULL_WITH_LATE and B0_ORIGINAL are clearly labelled historical references and retain their old Gate only for comparison.18unique groups ×5folds =90cells:30 strict reuses,60 fresh fits. Original80patients/7635channels/256seizures/1471965validRaw250windows, seed42 and roles/optimizer/loss/FP32/TF32off/chunk4096/fullpatient/budgets are unchanged. No other seeds,33patient surrogate, extraction, hyperparameter search or performance gate.

## Equal-patient averages

| Group | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac. bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL | 0.653465 | 0.765286 | 0.593488 | 0.464629 | 0.842301 | 0.772424 | 0.846368 | 0.800000 | -0.034239 |
| PREVIOUS_FULL_WITH_LATE | 0.646339 | 0.765888 | 0.589527 | 0.447095 | 0.845582 | 0.774722 | 0.836034 | 0.775000 | -0.054110 |
| B0_ORIGINAL | 0.653828 | 0.752255 | 0.585240 | 0.467424 | 0.840232 | 0.770976 | 0.845674 | 0.800000 | -0.022662 |
| WO_FEATURE_PR | 0.640931 | 0.746600 | 0.577782 | 0.451313 | 0.830549 | 0.758802 | 0.825916 | 0.762500 | -0.017609 |
| WO_RAW_PR | 0.628524 | 0.761538 | 0.563517 | 0.439429 | 0.817620 | 0.750666 | 0.768532 | 0.675000 | -0.019540 |
| WO_BOTH_PR | 0.510425 | 0.755076 | 0.565205 | 0.365161 | 0.655689 | 0.659277 | 0.800646 | 0.712500 | 0.135118 |
| WO_WINDOW_PR | 0.648286 | 0.762667 | 0.590549 | 0.466199 | 0.830374 | 0.757839 | 0.839761 | 0.787500 | -0.005792 |
| WO_EARLY_FILM | 0.644071 | 0.766014 | 0.568578 | 0.448547 | 0.839594 | 0.764299 | 0.779291 | 0.675000 | -0.035998 |
| WO_PRE_REFERENCE | 0.620060 | 0.739474 | 0.558005 | 0.415426 | 0.824694 | 0.751424 | 0.802719 | 0.725000 | -0.024357 |
| EARLY_FILM_MISMATCH | 0.643104 | 0.768153 | 0.566495 | 0.450395 | 0.835813 | 0.761769 | 0.784725 | 0.700000 | -0.022285 |
| RAW_ONLY | 0.621288 | 0.727387 | 0.543166 | 0.403061 | 0.839516 | 0.761493 | 0.774164 | 0.700000 | -0.067481 |
| PHYSIOLOGY_ONLY | 0.627572 | 0.754357 | 0.547682 | 0.429154 | 0.825991 | 0.753284 | 0.743619 | 0.637500 | -0.019049 |
| DROP_BETA | 0.642504 | 0.768603 | 0.590108 | 0.449140 | 0.835868 | 0.760750 | 0.820121 | 0.750000 | -0.027654 |
| DROP_LOW_GAMMA | 0.647285 | 0.765184 | 0.589383 | 0.465280 | 0.829290 | 0.756767 | 0.829165 | 0.750000 | -0.002009 |
| DROP_PARTIAL_HIGH_GAMMA | 0.649565 | 0.767920 | 0.587071 | 0.454239 | 0.844891 | 0.772640 | 0.806864 | 0.725000 | -0.051029 |
| DROP_RMS | 0.646097 | 0.762832 | 0.588190 | 0.451423 | 0.840771 | 0.768513 | 0.832374 | 0.775000 | -0.039414 |
| DROP_LINE_LENGTH | 0.637995 | 0.761196 | 0.590936 | 0.430656 | 0.845334 | 0.771574 | 0.846799 | 0.787500 | -0.065653 |
| DROP_SPECTRAL_ENTROPY | 0.654035 | 0.766753 | 0.596977 | 0.467052 | 0.841019 | 0.769050 | 0.839933 | 0.775000 | -0.030430 |

Macro-F1 is the equal-patient average of within-patient EZ/NEZ F1. Pooled-channel metrics are separate. MRR/Top-1 use within-patient ranking with original channel-name ties. EZ-fraction bias is predicted minus observed EZ fraction: closeness to zero, not greater value, is desirable. [Folds](FOLD_RESULTS.csv), [fold sample SD](FOLD_SD.csv), [sources](SOURCE_RESULTS.csv), [pooled](POOLED_RESULTS.csv), [coverage](RAW_COVERAGE.csv).

## Aliases, not additional experiments

w/o Late Gate is the main itself. With Late fixed absent, w/o Both Guidance is exactly w/o Early FiLM, retaining direct FeaturePR. They are reported in ALIAS_RESULTS.csv but not double-counted in cells, rows or intervals. No Early/Late four-cell interaction is estimated from a matrix in which Late is fixed absent.

## Full-minus-control and paired uncertainty

10000seed42 draws resample80unique patients.18absolute results,17Full-minus controls and1PRinteraction across9metrics give324nominal95% percentile intervals.34additional Bonferroni-percentile intervals cover17Macro-F1/AUROC pairs (tail0.05/68). These approximate intervals are conditional on this seed and historical model choices, not training-seed robustness or independent confirmation. All favorable/unfavorable directions are retained.

| Contrast | Delta Macro-F1 | Delta AUROC | Folds both positive | Additional-family lower bounds (F1,AUC) |
| --- | ---: | ---: | ---: | --- |
| FULL_MINUS_PREVIOUS_FULL_WITH_LATE | +0.007126 | -0.000602 | 2/5 | -0.007750, -0.014317 |
| FULL_MINUS_B0_ORIGINAL | -0.000363 | +0.013031 | 2/5 | -0.015556, -0.002860 |
| FULL_MINUS_WO_FEATURE_PR | +0.012534 | +0.018686 | 3/5 | -0.012756, -0.004169 |
| FULL_MINUS_WO_RAW_PR | +0.024941 | +0.003748 | 3/5 | -0.011837, -0.040920 |
| FULL_MINUS_WO_BOTH_PR | +0.143040 | +0.010210 | 3/5 | +0.085259, -0.029825 |
| FULL_MINUS_WO_WINDOW_PR | +0.005179 | +0.002620 | 2/5 | -0.008275, -0.010995 |
| FULL_MINUS_WO_EARLY_FILM | +0.009394 | -0.000728 | 2/5 | -0.013684, -0.033741 |
| FULL_MINUS_WO_PRE_REFERENCE | +0.033405 | +0.025813 | 4/5 | +0.001899, -0.013019 |
| FULL_MINUS_EARLY_FILM_MISMATCH | +0.010362 | -0.002867 | 3/5 | -0.009412, -0.035452 |
| FULL_MINUS_RAW_ONLY | +0.032177 | +0.037899 | 5/5 | +0.004600, -0.007026 |
| FULL_MINUS_PHYSIOLOGY_ONLY | +0.025893 | +0.010929 | 4/5 | -0.005058, -0.038995 |
| FULL_MINUS_DROP_BETA | +0.010961 | -0.003317 | 1/5 | -0.006789, -0.018470 |
| FULL_MINUS_DROP_LOW_GAMMA | +0.006180 | +0.000102 | 1/5 | -0.007666, -0.010002 |
| FULL_MINUS_DROP_PARTIAL_HIGH_GAMMA | +0.003900 | -0.002633 | 0/5 | -0.019926, -0.018063 |
| FULL_MINUS_DROP_RMS | +0.007369 | +0.002454 | 3/5 | -0.007478, -0.010376 |
| FULL_MINUS_DROP_LINE_LENGTH | +0.015470 | +0.004090 | 4/5 | -0.001280, -0.008234 |
| FULL_MINUS_DROP_SPECTRAL_ENTROPY | -0.000570 | -0.001467 | 0/5 | -0.015620, -0.012412 |

[All intervals](INTERVALS.csv) · [34additional-family intervals](PRIMARY_FAMILY_INTERVALS.csv) · [every fold direction](FOLD_DIRECTIONS.csv)

## Operations and interpretation boundaries

- w/o Feature PR removes physiological summary and window-relative PR; w/o Raw PR removes only final96DpatientZ; w/o BOTH removes both, retaining waveform normalization/self pre-reference. Window-only deletion preserves corrected summary PR.
- Each feature deletion zeros both summary coordinates and its EarlyFiLM window coordinate before relative statistics, keeping width and other shared initialization. Late does not exist in new groups. All new fits initialize from complete untrained matched sources, not trained main weights.
- Early FiLM mismatch routes only its correct computed conditioning input within patient/seizure/exact whole-mask support, salt42; direct physiology remains correctly aligned. Singleton groups remain unchanged and coverage is reported. This tests context dependence, not physiological causality.
- Raw-only reads no explicit physiology; Physiology-only reads no Raw. Both use original64-head weight slices/scales without width compensation.
- Module/branch removals change capacity. Raw may compensate omitted explicit physiology; single deletions cannot prove global feature optimality or irreducibility.
- Full minus PREVIOUS_FULL_WITH_LATE isolates the fixed Gate removal in this inherited protocol. B0 also differs in summary preprocessing/window relative guidance, and is not solely a Late comparison.
- PR interaction is Full minus NoFeaturePR minus NoRawPR plus NoBothPR on patient metrics. No biological interaction follows from a performance interaction.
- All90models/preprocessors/epochs/thresholds froze before unfinished Outer; all90score cells sealed before metrics. Thirty reuses passed input/initialization/preprocessing/validation/epoch/threshold/checkpoint/score identity. Other with-Late PR/feature ablations were not reused.
- Small-source metric cells below5patients are suppressed, not removed from the main cohort. Five folds overlap in training patients; their SD and direction counts are descriptive. No one-seed result establishes multi-seed robustness.
- Independent audit: 1440patient rows, 12960metric values,324nominal/34additional intervals; maximum interval discrepancy 8.88e-16.
- Public artifacts contain only source/hashes/anonymous aggregates. Signals, identities, individual scores, weights, preprocessors, caches and runtime logs remain on the trusted server.
