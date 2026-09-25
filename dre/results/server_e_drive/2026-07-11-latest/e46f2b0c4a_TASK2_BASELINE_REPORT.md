# Task 2 Baseline Report

Primary cohort: frozen Task 1 old-90 success patients plus all valid failure patients.

Positive class: success=1. Negative class: failure=0. Primary metric: patient-level macro-F1.

Completed model/seed rows: 1. Recorded failures: 0.

Patient bootstrap samples: 50. Shortcut risk: True.

| model | seed | macro_f1 | accuracy | brier | ece | sensitivity | specificity | ppv | npv | balanced_accuracy | auroc | auprc | TN | FP | FN | TP | n_patients |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| majority | 42 | 0.37656903765690375 | 0.6040268456375839 | 0.2392155610000651 | 0.00010609876727019074 | 1.0 | 0.0 | 0.6040268456375839 | nan | 0.5 | 0.4932203389830508 | 0.6008053691275168 | 0 | 59 | 0 | 90 | 149 |
