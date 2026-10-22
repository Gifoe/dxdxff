# Four-source LOCO, seed42

multicenter/pediatric are source strata not proven single hospitals. Original TimeConv uses about -29..+29s windows; ours pre-reference and0..20s stages, different objectives/precision/capacity. Package comparison NOT architecture-only. One seed, four source strata, historically viewed cohort NOT independent external confirmation.

All20 models frozen before test, all20 score cells sealed before metrics. No old fitted model reuse.
Patient-equal primary; source-equal and channel-pooled are separate. Bootstrap is patient-paired within four fixed sources; no seed robustness or independent external confirmation.

| Model | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-Frac. Bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OURS_RAW_FEATURE | 0.627380 | 0.752589 | 0.557766 | 0.436358 | 0.818403 | 0.735118 | 0.782990 | 0.675000 | 0.001560 |
| OURS_RAW | 0.597093 | 0.710279 | 0.525347 | 0.378348 | 0.815839 | 0.728312 | 0.781137 | 0.700000 | -0.031678 |
| OURS_FEATURE | 0.613643 | 0.751334 | 0.543605 | 0.415435 | 0.811851 | 0.731076 | 0.734570 | 0.612500 | -0.002968 |
| LINEAR_SVM_FEATURE_PR | 0.607951 | 0.738479 | 0.537266 | 0.409190 | 0.806712 | 0.721983 | 0.728211 | 0.637500 | 0.007150 |
| TIMECONVCNN_RAW | 0.416030 | 0.574502 | 0.370946 | 0.176566 | 0.655494 | 0.629659 | 0.468280 | 0.350000 | 0.066937 |
