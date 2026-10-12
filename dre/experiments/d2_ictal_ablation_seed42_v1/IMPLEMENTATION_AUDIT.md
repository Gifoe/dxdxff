# Implementation identity and comparison boundaries

Reviewed historical original SourceA0/model.py, train.py, prepare.py, original
uas_core.py, frozen D2 outer evaluator/amendment, no-patient-z runner/protocol,
and source-rank-compression model/engineering/source audit. The exact original
feature_aggregation.py and evidence_views.py were reviewed against their
historical pinned hashes. This task uses no installed skill.

Full D2 is the original ictal rank4 source-conditioned patient-relative MLP,
not Omni-D2, GraphRole, a posterior decoder or a rank-compressed alternative.
The shared LayerNorm88/Linear88→96/GELU/Dropout0.15/Linear96→1 contains8817
parameters. Rank4 V contributes384, source u contributes16 and source b
contributes4, giving9221. Model outputs the NEZ logit.

## Source-free arm

`source_free_factory` constructs the exact original shared network, preserving
shared initialization and the original RNG path, but registers no V/u/b.
Its forward ignores source and is exactly `shared_MLP(x)`. Its regularizer is
zero and the original trainer is called with source penalty0. No data objective
or optimizer rule changes. This removes the entire source mechanism. Historical
rank-zero R0 has8821 parameters because it still contains four biases and is
explicitly not a valid source-free replacement.

An in-process constructor adapter changes only `train.SourceA0` during that
arm's call, then restores it. Historical source files are never edited.
Pre-training tests require exact original shared-state parity for all five
folds and source-invariant forward output.

## Feature-view arms

All four retain9221 parameters and the 88D original network. They receive
copies of the exact original transformed inputs with excluded columns set to
zero. All preprocessing parameters, retained values and cohort memberships
remain unchanged. Both main and auxiliary copies of an excluded view are
masked. The mask is applied before every scientific train/validation forward
and before outer inference. Retraining is from the original prescribed state,
not an inference-only intervention.

## Training, selection and replay

The hash-checked original trainer supplies FIT-only class weights, one update
per FIT patient, optimizer/LR/weight decay/gradient clipping, dropout, seeded
update order, validation-only threshold selection and checkpoint stopping.
Source-enabled arms retain the exact original normalized source L2 coefficient.
Only the source-free arm drops that nonexistent penalty.

Engineering uses synthetic CUDA runs to compare uninterrupted versus resumed
model tensors, Adam slots, training history, checkpoint selection and RNG for
full, feature-masked and source-free constructors. These are engineering tests,
not added scientific arms. Existing frozen references are unchanged; FIT/VAL
Full D2 logits are replayed under the original tolerance before new training.
Full-cohort labels are not materialized during auditing/training/selection.

All 35 selected checkpoint/preprocessor/mask/threshold bindings freeze before
new TEST inference; all35 prediction files seal before labels for new metrics.
The 10 existing cells reuse their private scores under original content hashes.
Canonical ascending NEZ probabilities and channel-string tie order are retained.

Headline results use 80 equally weighted patients, not unequal-fold mean or
pooled-channel metrics. Bootstrap effects are Full minus ablated, paired on
the same patients with frozen thresholds. Descriptive intervals are not
corrected for multiple exploratory comparisons. These correlated feature
ablations do not identify unique causal biological contributions, and repeatedly
viewed patient-held-out results do not establish unseen-center transfer.
