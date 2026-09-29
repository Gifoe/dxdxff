# Historical A1 metric unit and comparison guard

The supplied A1 ictal AUROC `0.746382` is **not** a five-fold outer-test
estimate. Its source table
`dre/experiments/unified_a1_tf_two_benchmark_seed42_v1/outputs/ICTAL_PRIMARY_METRICS.csv`
contains 65 fold-validation-patient cells, 47 unique validation patient IDs,
and 20 fixed-query repetitions per cell. The historical evaluator performs
leave-one-validation-patient-out (VLOO) epoch/threshold selection for each
target cell (`crst_ieeg_two_benchmark_seed42_v1/code/evaluate_ictal_vloo.py`).

A PC-CNN true outer-test estimate must not be subtracted from this development
VLOO number and called a matched paired effect. We will present any
development fixed-query/VLOO estimate and any later outer-test estimate as
different units. If Stage-C eligibility was determined on all validation
patients, a later VLOO readout cannot be called fully target-excluded; that
limitation must be reported. This provenance note does not amend the frozen
model, optimizer, split, checkpoint-selection rule, or test access policy.
