# Fixed six-feature DREChannNet: complete three-seed ablation

All results are exploratory repeated-test, not independent confirmation. Current Full was fixed before these runs; B0 is a historical architecture reference only. No winning ablation is promoted. All285 selection cells froze before unfinished Outer inference, and all285 score cells sealed before this report.

19configurations × seeds42/43/44 × original5folds. Original80patients,7635channels,256seizures and1471965valid Raw windows, complete-patient updates and original FP32/TF32-off optimizer/loss/budgets are retained. Variance and head interaction remain absent. No new extraction, feature/hyperparameter search, Development performance gate or post-Outer tuning. The earlier33-patient internal three-seed experiment is not included.

Strict historical reuses: 50; fresh fits: 235. Mandatory45current seed42 cells plus only independently admitted historical Raw-only cells. Reuses preserve checkpoint/preprocessing/initialization/validation/threshold/prediction hashes. Old other-architecture deletion experiments are not reused.

## Equal-patient metrics, then equal-seed mean ± sample SD

| Group | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac. bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0_ORIGINAL | 0.641810 ± 0.010708 | 0.752568 ± 0.002498 | 0.583070 ± 0.005797 | 0.450019 ± 0.015183 | 0.833600 ± 0.006577 | 0.759424 ± 0.010921 | 0.820336 ± 0.022446 | 0.758333 ± 0.038188 | -0.017646 ± 0.005755 |
| FULL | 0.637641 ± 0.008321 | 0.764882 ± 0.005731 | 0.589071 ± 0.000626 | 0.439104 ± 0.008697 | 0.836179 ± 0.008341 | 0.761210 ± 0.011850 | 0.828035 ± 0.017842 | 0.762500 ± 0.033072 | -0.031579 ± 0.019522 |
| WO_FEATURE_PR | 0.628294 ± 0.007748 | 0.746395 ± 0.001872 | 0.573941 ± 0.006873 | 0.434592 ± 0.012480 | 0.821996 ± 0.004468 | 0.745766 ± 0.005226 | 0.804441 ± 0.025609 | 0.733333 ± 0.028868 | -0.007019 ± 0.007128 |
| WO_RAW_PR | 0.628900 ± 0.002447 | 0.758469 ± 0.005516 | 0.557424 ± 0.007938 | 0.438073 ± 0.008880 | 0.819727 ± 0.009376 | 0.750966 ± 0.010675 | 0.766750 ± 0.012946 | 0.675000 ± 0.021651 | -0.010041 ± 0.027695 |
| WO_BOTH_PR | 0.505288 ± 0.012843 | 0.753959 ± 0.005296 | 0.566120 ± 0.004960 | 0.383755 ± 0.011358 | 0.626822 ± 0.036694 | 0.637442 ± 0.023318 | 0.797823 ± 0.013959 | 0.716667 ± 0.014434 | 0.175725 ± 0.044811 |
| WO_WINDOW_PR | 0.641214 ± 0.011517 | 0.762460 ± 0.006763 | 0.588992 ± 0.002297 | 0.442666 ± 0.019638 | 0.839763 ± 0.004557 | 0.765923 ± 0.007885 | 0.826314 ± 0.022363 | 0.766667 ± 0.031458 | -0.042037 ± 0.009658 |
| WO_EARLY_FILM | 0.643034 ± 0.007285 | 0.762016 ± 0.008077 | 0.564690 ± 0.010599 | 0.451671 ± 0.012575 | 0.834396 ± 0.002093 | 0.759517 ± 0.004095 | 0.777860 ± 0.010235 | 0.683333 ± 0.007217 | -0.020727 ± 0.002869 |
| WO_LATE_GATE | 0.640839 ± 0.011029 | 0.764574 ± 0.006445 | 0.589688 ± 0.003386 | 0.446281 ± 0.015908 | 0.835397 ± 0.006341 | 0.760391 ± 0.010727 | 0.830804 ± 0.020391 | 0.775000 ± 0.033072 | -0.028204 ± 0.010209 |
| WO_PRE_REFERENCE | 0.620722 ± 0.004490 | 0.741152 ± 0.002263 | 0.557763 ± 0.001334 | 0.419123 ± 0.007698 | 0.822321 ± 0.006159 | 0.748373 ± 0.007101 | 0.794292 ± 0.004905 | 0.716667 ± 0.007217 | -0.015781 ± 0.012163 |
| WO_BOTH_GUIDANCE | 0.639654 ± 0.005623 | 0.761071 ± 0.004585 | 0.562253 ± 0.006927 | 0.444014 ± 0.007428 | 0.835295 ± 0.004198 | 0.759431 ± 0.004996 | 0.774254 ± 0.008058 | 0.679167 ± 0.007217 | -0.026954 ± 0.008350 |
| EARLY_FILM_MISMATCH | 0.641874 ± 0.005620 | 0.760118 ± 0.009130 | 0.557026 ± 0.009777 | 0.449821 ± 0.007963 | 0.833927 ± 0.003357 | 0.759570 ± 0.005095 | 0.772740 ± 0.002199 | 0.679167 ± 0.007217 | -0.019818 ± 0.003717 |
| RAW_ONLY | 0.614443 ± 0.006520 | 0.720207 ± 0.007754 | 0.530188 ± 0.011986 | 0.393623 ± 0.010151 | 0.835263 ± 0.003730 | 0.755728 ± 0.004993 | 0.763592 ± 0.013431 | 0.687500 ± 0.021651 | -0.061330 ± 0.006518 |
| PHYSIOLOGY_ONLY | 0.624910 ± 0.003641 | 0.751185 ± 0.002747 | 0.541121 ± 0.006463 | 0.424934 ± 0.006156 | 0.824886 ± 0.001174 | 0.751642 ± 0.001425 | 0.736679 ± 0.018753 | 0.633333 ± 0.031458 | -0.018284 ± 0.003089 |
| DROP_BETA | 0.635981 ± 0.008855 | 0.768765 ± 0.001593 | 0.589830 ± 0.003963 | 0.439355 ± 0.010548 | 0.832607 ± 0.008416 | 0.756911 ± 0.010116 | 0.801653 ± 0.015216 | 0.725000 ± 0.025000 | -0.024560 ± 0.015573 |
| DROP_LOW_GAMMA | 0.632562 ± 0.004871 | 0.763564 ± 0.004157 | 0.588601 ± 0.001679 | 0.433599 ± 0.010367 | 0.831525 ± 0.002305 | 0.755954 ± 0.003664 | 0.825693 ± 0.008585 | 0.754167 ± 0.014434 | -0.027848 ± 0.011454 |
| DROP_PARTIAL_HIGH_GAMMA | 0.638110 ± 0.007196 | 0.765020 ± 0.005404 | 0.588700 ± 0.005746 | 0.440104 ± 0.009187 | 0.836115 ± 0.005506 | 0.761145 ± 0.007721 | 0.811565 ± 0.007897 | 0.737500 ± 0.012500 | -0.033532 ± 0.012085 |
| DROP_RMS | 0.631593 ± 0.012144 | 0.759130 ± 0.004053 | 0.579324 ± 0.009230 | 0.433512 ± 0.027413 | 0.829674 ± 0.011911 | 0.753558 ± 0.013560 | 0.813152 ± 0.016357 | 0.745833 ± 0.031458 | -0.022172 ± 0.037850 |
| DROP_LINE_LENGTH | 0.630214 ± 0.003406 | 0.759559 ± 0.006280 | 0.585213 ± 0.002456 | 0.422940 ± 0.002373 | 0.837488 ± 0.004447 | 0.761293 ± 0.005333 | 0.833932 ± 0.024937 | 0.770833 ± 0.038188 | -0.046614 ± 0.012401 |
| DROP_SPECTRAL_ENTROPY | 0.638490 ± 0.011184 | 0.761188 ± 0.005137 | 0.585076 ± 0.006106 | 0.441317 ± 0.016312 | 0.835663 ± 0.006083 | 0.760574 ± 0.008484 | 0.821567 ± 0.016453 | 0.750000 ± 0.025000 | -0.032945 ± 0.007362 |

The primary summaries average metrics, not probabilities, and do not select a best seed. Macro-F1 first averages EZ/NEZ F1 within a patient, then patients equally. EZ-fraction bias is predicted minus observed EZ fraction. Pooled-channel values are separately labeled per seed and their metric means; MRR and Top-1 remain patient-local.

[Individual seeds](SEED_RESULTS.csv) · [seed sample SD](SEED_SD.csv) · [folds](FOLD_RESULTS.csv) · [fold sample SD](FOLD_SD.csv) · [sources](SOURCE_RESULTS.csv) · [source-by-seed](SOURCE_SEED_RESULTS.csv) · [pooled](POOLED_RESULTS.csv) · [coverage](RAW_COVERAGE.csv)

## Paired evidence and direction consistency

Each patient metric is first averaged over seeds42/43/44.10000seed42 draws resample the same80unique patients.351nominal95% percentile intervals cover19absolute results,18Full-minus controls and2factorial interactions, each across9metrics.36additional Bonferroni-percentile intervals cover18AUROC/Macro-F1 contrasts (two-sided tail0.05/72). Finite bootstrap-tail resolution and historical repeated testing limit interpretation; these intervals are conditional on the fitted models, not population-level estimates of training-seed uncertainty.

| Contrast | ΔMacro-F1 | ΔAUROC | Seeds both positive | Fold/seed cells both positive | Additional-family lower bounds (F1,AUC) |
| --- | ---: | ---: | ---: | ---: | --- |
| FULL_MINUS_B0_ORIGINAL | -0.004168 | +0.012313 | 0/3 | 8/15 | -0.020581, -0.002740 |
| FULL_MINUS_WO_FEATURE_PR | +0.009347 | +0.018487 | 3/3 | 11/15 | -0.013816, -0.000794 |
| FULL_MINUS_WO_RAW_PR | +0.008741 | +0.006413 | 3/3 | 8/15 | -0.015084, -0.033902 |
| FULL_MINUS_WO_BOTH_PR | +0.132353 | +0.010923 | 3/3 | 9/15 | +0.071662, -0.020470 |
| FULL_MINUS_WO_WINDOW_PR | -0.003573 | +0.002422 | 0/3 | 5/15 | -0.017407, -0.005511 |
| FULL_MINUS_WO_EARLY_FILM | -0.005392 | +0.002865 | 0/3 | 6/15 | -0.023900, -0.021894 |
| FULL_MINUS_WO_LATE_GATE | -0.003197 | +0.000307 | 1/3 | 6/15 | -0.012999, -0.004713 |
| FULL_MINUS_WO_PRE_REFERENCE | +0.016919 | +0.023730 | 3/3 | 11/15 | -0.001343, -0.000304 |
| FULL_MINUS_WO_BOTH_GUIDANCE | -0.002013 | +0.003811 | 0/3 | 7/15 | -0.020272, -0.022114 |
| FULL_MINUS_EARLY_FILM_MISMATCH | -0.004233 | +0.004764 | 0/3 | 4/15 | -0.022883, -0.021274 |
| FULL_MINUS_RAW_ONLY | +0.023198 | +0.044675 | 3/3 | 14/15 | -0.004630, +0.001010 |
| FULL_MINUS_PHYSIOLOGY_ONLY | +0.012732 | +0.013697 | 3/3 | 11/15 | -0.012527, -0.032854 |
| FULL_MINUS_DROP_BETA | +0.001660 | -0.003883 | 1/3 | 5/15 | -0.005342, -0.013048 |
| FULL_MINUS_DROP_LOW_GAMMA | +0.005079 | +0.001317 | 2/3 | 6/15 | -0.002517, -0.006425 |
| FULL_MINUS_DROP_PARTIAL_HIGH_GAMMA | -0.000468 | -0.000138 | 1/3 | 4/15 | -0.010018, -0.005725 |
| FULL_MINUS_DROP_RMS | +0.006048 | +0.005752 | 3/3 | 8/15 | -0.003672, -0.003476 |
| FULL_MINUS_DROP_LINE_LENGTH | +0.007428 | +0.005323 | 3/3 | 9/15 | -0.002216, -0.002119 |
| FULL_MINUS_DROP_SPECTRAL_ENTROPY | -0.000849 | +0.003694 | 1/3 | 6/15 | -0.009559, -0.003201 |

[All nominal intervals](INTERVALS.csv) · [36additional intervals](PRIMARY_FAMILY_INTERVALS.csv) · [every fold/seed contrast direction](FOLD_SEED_DIRECTIONS.csv)

## Interpretation boundaries

- Feature PR deletion removes physiological-summary and physiological-window PR; Raw PR deletion removes final Raw96D patient Z only. BOTH removes all those relative operations but retains waveform normalization and self pre-ictal subtraction. Window-only deletion preserves corrected summary PR. No control replaces legitimate inputs with arbitrary zeros except declared feature/branch removals.
- Both Guidance removes only Early FiLM and Late Gate, retaining direct physiological summary PR. Module removals change capacity; those contrasts do not by themselves separate information from optimization/capacity effects.
- Each feature deletion masks both summary coordinates and its early/late window coordinate before relative statistics. Other coordinates and shared initial tensors remain unchanged; every new cell starts from matched complete initialization, not trained Full weights.
- Early FiLM mismatch permutes only its already computed input within patient/seizure/exact full-mask groups, salt42 across all training seeds. Late/direct inputs stay correct. Singletons remain unchanged; the full support and actual changed/eligible/singleton window counts are reported. This is a context control, not proof of physiology or seizure propagation.
- Raw-only reads no physiological features; Physiology-only reads no Raw. Their64-wide heads use the original corresponding input-weight slice and scaling, without width compensation.
- PR interaction is Full − NoFeaturePR − NoRawPR + NoBothPR; guidance interaction is Full − NoEarly − NoLate + NoBothGuidance, computed per patient metric before averaging. They are statistical performance interactions, not biological interactions.
- Three seeds assess limited training variability; folds overlap in training patients. Neither three seeds nor paired intervals turn a historically reused Outer cohort into independent confirmation. Single deletions do not establish global optimality of the six-feature set.
- Small-source cells under five patients suppress source metric values only; all original patients/channels remain in the main results. No outcome-directed exclusions.
- Independent verification: 4560patient/seed rows, 41040metric values,351nominal and36additional interval checks; maximum interval discrepancy 5.55e-16.
- Public release includes source/hashes/anonymous aggregates only. Signals, identities, patient/channel predictions, checkpoints, preprocessors, caches and runtime logs stay on the trusted server.
