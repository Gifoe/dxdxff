# Implementation admission

Only W1/W2 train from scratch. Shared A0 helpers are hash-locked; source features
and split are SHA-gated. Development arrays are rebuilt and compared bitwise to
the original frozen A0 FIT/VAL arrays, including feature order, labels and
imputer/scaler states. Clinical fields in the all-cohort source export may be
materialized before sealing; no TEST predictions/performance are computed.

W2 uses isolated official EnsembleView/LinearBatchEnsemble/LinearEnsemble source
definitions. Only docstrings and unused package imports are removed. Verify AST,
initialization, forward and gradients against the hash-pinned full official file.
All shared/member parameters remain trainable. No A0 warmstart, auxiliary loss,
member pruning, deployed member selection or learned patient thresholds.

W2 loss=mean_channels(mean_members(BCE_NEZ)); score=mean_members(sigmoid(logit)).
W1 uses the original ordinary BCE and sigmoid. FIT EZ/NEZ ratio is positive weight
because NEZ=1. Patient-equal here means one mean-channel loss/update per FIT patient.
Sorted FIT IDs drive the exact original seed+epoch permutation in both arms.
Different model shapes prevent tensor-level initialization equality across arms;
each arm's initialization is deterministic and privately saved.

Original checkpoint/threshold lexicographic rules and epoch>=6 patience6 stopping
are reused, with30 epochs maximum. No zero-epoch fallback is introduced into
scratch training. AdamW/model/history/complete RNG states are atomically saved;
source, data and initial-state bindings reject incompatible resumes.

Member diagnostic F1 uses the frozen parent ensemble's numeric threshold; any
extra member-optimal diagnostic is explicitly posthoc and cannot change deployed
ensemble/threshold. Public per-patient improvements are quantiles/fractions only,
never identifiable rows. Bootstrap resamples47 unique patient IDs while retaining
all65 development appearances, not individual channels or appearance rows.

Six synthetic test groups cover required counts/state diversity, exact official
source parity, memberwise objective versus forbidden averaged-logit objective,
probability averaging/orientation, per-member shared/private gradients, unlabeled
FIT-only preprocessing/threshold ties, and exact epoch-boundary resume in both
arms. Hash-gated real data adds original identity/order/labels/A0 replay checks.
Three discarded real FIT patient updates per arm are mandatory before training.

Efficiency measures synchronized epoch optimization+validation/selection time,
excluding disk checkpoint serialization. Peak CUDA allocation includes inputs,
model and optimizer. Inference timing uses10 warmups/20 repeats on the same
whole validation batch per fold. Existing A0 training time/peak are unavailable,
not invented or measured by retraining. Diagnostics do not authorize tuning.

## Completed execution

A0 replay/array admission passed. Six pretraining synthetic groups and real
three-FIT-patient updates per arm passed under PyTorch2.11.0+cu128/NumPy2.4.3.
Ten scratch tasks completed in the prescribed W1-all-folds then W2-all-folds order.
Selected states/predictions and all 76 completed student epochs were verified;
no formal student was discarded or rerun after outcomes were observed.
Final scientific terminal is TABM_NOT_SUPPORTED; no outer evaluation occurred.

After training, two native aggregation crashes occurred; the second traceback
identified pandas repeated metadata-column insertion. Only member diagnostic
table assembly changed to concatenation, leaving metric calculations and all
science artifacts unchanged. A separate synthetic post-training regression
checks four individual/ensemble metrics against the original metric helper,
variance and non-deployment of member selection; 1/1 passed. This additional test
does not replace or edit the SHA-bound pretraining test file. The first hidden
launcher had exited with empty logs/no student artifacts; cause unconfirmed.
All prior source/logs are retained privately, and no outcomes were tuned.
