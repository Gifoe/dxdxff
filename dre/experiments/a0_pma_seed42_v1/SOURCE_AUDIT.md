# Source and historical evidence audit

The baseline is **w/o Source** from `d2_ictal_ablation_seed42_v1`, with
8,817 trainable parameters. It is not the 9,221-parameter SourceA0, rank-zero
R0 with source biases, older A0 development reference, A1 or Omni-D2.
The exact shared network is input LN(88), Linear(88,96), GELU,
Dropout(0.15), Linear(96,1). Its output is a NEZ logit, EZ label is zero.

Reviewed original `SourceA0`, `train.py`, `common.py`, `uas_core.py`,
preprocessing and frozen outer amendment. Hash checks pin the original
implementation, feature order, FIT-only preprocessing banks, partition and
five source-free checkpoints. The PMA runner imports the original training
function unchanged and replaces its model constructor in memory, with source
penalty zero. It does not edit historical code or artifacts.

## Prior studies reviewed

| Study | Relevant finding / limitation for this task |
|---|---|
| D2 complete ablation | Establishes the exact current source-free A0 and original 88-column interface. Patient Z is material; source conditioning lacks supported incremental gain. |
| D2 no Source + no ZDELTA | Joint deletion has not supported a primary gain. Its 62-column input is not a legal PMA baseline. |
| A0 TabM ensemble | Additional capacity did not establish Macro-F1 improvement. Older matched A0 numbers are not substitutes for the current frozen baseline. |
| A1 patient-conditioned low-rank adapter | Failed its declared improvement/context controls; patient-conditioned direction generation is not validated. A1 architecture/protocol differs. |
| Absolute-context complementarity probe | Context residual failed its development gate. Absolute patient context did not demonstrate useful incremental prediction. |
| Representation bottleneck audit | Temporal/recruitment/persistence explanations were not supported by their controls. Representation diagnostics are not new predictive results. |
| Cross-patient feature geometry audit | Label-local oracle geometry is non-deployable; patient direction instability does not establish a trainable magnitude mechanism. Older label-materialization limitation is not inherited here. |
| Ranking bottleneck/source audit | Proposed ranking changes did not pass their gates; no justification for adding ranking loss here. |

These historical studies are context, not paired comparator rows. Several use
A1 or different development definitions. PMA must succeed against the newly
matched B1/B2 controls rather than borrowing an unmatched historical score.

## Mechanistic distinction

PMA receives each channel's mean and log standard deviation from the exact
post-Patient-Z, post-FIT-preprocessed, pre-LayerNorm **88D vector**. Its
32D gate modulates a hidden bottleneck and returns a hidden residual to the
same classifier. It does not construct a shared patient context vector,
generate patient-specific classifier weights, use source identity or create
a second predictor. B2 has the same parameter count but receives only the
original 96D hidden representation, never the two magnitude statistics.

The frozen input contains 16 auxiliary duplicate dimensions and 16
constant-zero dimensions. Both remain unchanged. The statistic is therefore
an interface-dependent feature geometry measure, not 88 independent
physiological measurements or raw-amplitude recovery.
