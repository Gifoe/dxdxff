# A0 / TabM source audit, before new training

Only W1 and W2 are new training arms. A0 is replayed from the existing PLLR
development checkpoints, not initialized into W2 or retrained. Matched A0
Macro-F1 is0.6380797828499001, 65 validation appearances /47 unique patients;
historical outer0.616167 is not its matched reference.

Reviewed the previous PLLR implementation, original PR-UAS `uas_core.py` and
frozen feature/split protocol. Reuse exact patientwise unlabeled z-score,
FIT-mean imputation/StandardScaler, NEZ orientation, patient-equal BCE updates,
threshold/checkpoint ties, RNG state and original epoch>=6 patience6 stopping.
Core source SHA256:1b1ccb0a361e6245cb2e6089347e147036aac90437c0992194f6db7bebf1503a.

Official reference pinned at28e47ae301c92ec37787dde1ce923a0793f405b4:

- [tabm.py](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/tabm.py)
- [README.md](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/README.md)

Inspected EnsembleView, LinearBatchEnsemble, LinearEnsemble, initialization and
independent member-loss / probability-average inference guidance. Full upstream
file SHA256:fc654af6a16bac53d893a8265c79d7af4ebddcb95ad0d600cc6b6bc6b7317ade.
Both local and trusted server environments lack rtdl_num_embeddings, an eager
full-package import dependency irrelevant to these three components. To avoid
changing a shared runtime, isolate the official classes and their helper functions
in `code/tabm_layers.py`. AST-based extraction removes documentation strings and
unrelated imports/modules only; shape guards, algebra, parameter names/layouts,
draw order, shared-bias initialization and random signs remain unchanged. No
deprecated tabm_reference implementation is used. Apache-2.0 attribution and
license are retained in the source and LICENSE_TABM.txt.

W2 uses these components with shared LN88, K4, one LinearBatchEnsemble88x96 with
scaling_init=('random-signs','ones'), no chunks, GELU/dropout.15, and independent
LinearEnsemble96x1 heads. Output[N,4]; loss averages independently evaluated
member BCE, inference averages sigmoid probabilities, not logits. W1 is ordinary
LN88/Linear88x111/GELU/dropout.15/Linear111x1. No additional representations,
ranking loss, label correction, temporal module or validation-selected members.

This is an A0-specific one-layer TabM adaptation, not a reproduction of full
published TabM architecture or benchmark findings. Parameter totals must be
A0=8817, W1=10167, W2=10132. Before training, compare isolated-class AST semantics,
initial tensors, outputs and gradients to the pinned full-source definitions
loaded privately without their unused embedding dependency.

No new branch is created, following the attachment's no-additional-branches
constraint. New files live solely in the requested experiment directory on the
current branch. No new outer predictions or performance are authorized. Original
source export contains all-cohort labels; they may be materialized for sealing,
but all fitting/selection/metrics are limited to original FIT/VAL membership.
