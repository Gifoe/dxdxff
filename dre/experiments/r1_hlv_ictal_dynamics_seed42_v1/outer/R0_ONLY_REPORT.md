# R0-only outer test (post-gate amendment)

The five validation-selected R0 checkpoints and thresholds were hash-frozen before this one-time test.
The original R1 validation gate failed and remains failed. R1 outer test was not run.
This is exploratory: prior outer outcomes on the same cohort were already viewed, and the R0-only test was requested after the gate failure.
No test outcome was used for training, selection, or tuning.

| Metric | 80-patient mean | Descriptive patient-bootstrap 95% CI |
| --- | ---: | ---: |
| patient_macro_f1 | 0.6225 | [0.5920, 0.6528] |
| patient_macro_ez_f1 | 0.4285 | [0.3797, 0.4755] |
| patient_macro_balanced_accuracy | 0.6773 | [0.6427, 0.7114] |
| patient_macro_auprc_ez | 0.5507 | [0.4948, 0.6058] |

Five-fold aggregate results: `R0_OUTER_RESULTS.csv`.
Amendment SHA-256: `5b45a59c77a83d3416693588fc39cab0dac7e55e2fe39c02772b105c5e2bbad4`.
