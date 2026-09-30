# Leakage audit

- PASS: dictionary weights, subsets, dilation and bias-quantile identities depend only on seed 42 and kernel ID.
- PASS: each fold's dictionary biases use only a deterministic label-free FIT sample.
- PASS: each 52-D robust scaler is fitted only on FIT patients.
- PASS: mRMR relevance and redundancy use only FIT patients; relevance is patient-balanced.
- PASS: patient rank and median deviation use signal-valid channels without labels.
- PASS: biomarker definitions were locked before development.
- PASS: validation is used only for the predeclared K/head/hyperparameter/epoch/threshold selection.
- PASS: no outer-test or Omni input is accepted by the Ictal runner.
- PASS: no test threshold or architecture tuning occurred.

`outer_test_accessed = false`; `test_used_for_tuning = false`.
