# Why similar AUROC does not imply similar F1

Post-completion explanation requested by the user. Frozen aggregates only: no additional inference, training, calibration, threshold sweep/oracle, optimization or prediction change.

| Published aggregate | Evaluation | AUROC | Macro-F1 | EZ-F1 | NEZ-F1 |
|---|---|---:|---:|---:|---:|
| Recent MAIN_B2 |65 Development appearances /47 identities|0.719570|0.642722|0.437631|0.847813|
| Historical source-free A0/F0 |80-person Outer|0.736977|0.624236|0.414671|0.833800|
| Current synchronous C1 LR |80-person Outer, separate FIT-only final split|0.711834|0.574056|0.345849|0.802264|

B2 values are from its published FINAL_REPORT; F0 from `a0_compact_rawfirst_ablation_seed42_v1/results/OUTER_ALL_MODELS_SUMMARY.csv`. These are **not new executed comparators or paired tests**; no historical individual predictions/labels are read. B2-versus-current mixes Development and Outer. F0-versus-current uses the same Outer cohort unit but different model/training/threshold protocols; it is still not a controlled causal attribution study.

C1 patient-equal sensitivity0.422034, specificity0.849173, EZ-F1 0.345849, NEZ-F1 0.802264. Macro-F1 is their average0.574056. Low EZ-F1 is the larger deficit; high NEZ-F1 does not compensate for missed EZ channels.

Historical F0 aggregate implies total TP729,FP799,TN5093,FN1014. CurrentC1 actual TP609,FP869,TN5023,FN1134:120 fewer detected EZ channels and70 additional false positives. Both kinds of error worsen. It is not established that adjusting a test threshold would solve this difference.

C1 selected EZ-probability thresholds per fold are0.285/0.305/0.335/0.270/0.390, selected only from10–11 FIT patients. All froze before Outer. No test-derived Youden or F1 oracle. Final LR TRAIN counts40/40/40/41/40 differ from original A0's larger original FIT model/original validation protocol. Small selection sets, score calibration, representation/regularization and operating ROC/PR regions are plausible contributors, **not separately identified causes**.

AUROC summarizes positive-versus-negative ordering across thresholds; a strictly increasing score transformation does not change it. F1 depends on actualTP/FP/FN at one operating cutoff and class composition. Similar AUROC does not imply equal calibration, equal ROC/PR shape near the selected point, equal threshold transfer or equal F1. Patient-equal averages also cannot be reconstructed from pooled confusion counts. The fixed linear mechanism probe is not an A0 replacement-performance claim.

The authorized hypothesis is C1 versus matchedC0/C2, not historical A0 F1 reproduction. Outer synchronization-specific AUROC CI crosses zero. No claim that calibration alone explains the difference, and no Outer threshold tuning is performed.
