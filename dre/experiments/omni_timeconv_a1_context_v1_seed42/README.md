# Frozen TimeConv R4 + A1 patient-level context, seed 42

This experiment evaluates a small patient-context residual head on top of a
frozen official Omni-iEEG TimeConv-CNN representation.  The prediction unit
remains the official EDF-channel unit.  The CNN was neither retrained nor
loaded into an optimizer.

The TRAIN representation source is the full-record R4 cache only.  It is
bound by its 296-file content digest and the pre-existing full-record gate:
316,364 segments, 145,052 labelled segments, 13,350 labelled EDF-channel
units, and frozen-CNN TRAIN AUROC 0.9586782931.  Historical five-clip TRAIN
features are not read.

The final, one-pass frozen TEST result is exploratory because Omni test
outcomes had been viewed historically.  V1 reached pooled AUROC 0.804692,
versus 0.798768 for frozen C0.  The patient-cluster bootstrap CI for V1-C0
AUROC is [-0.003066, 0.016994], so this does not isolate an A1-context gain
from the capacity-matched local residual.  The terminal conclusion is
`CONTEXT_GAIN_NOT_ISOLATED_FROM_LOCAL_CAPACITY`.

Only code and compact aggregate audits are versioned here.  Raw EEG, caches,
checkpoints, per-patient/channel predictions, and runtime logs remain private.

See [FINAL_REPORT.md](outputs/FINAL_REPORT.md) for the complete result and
[PROTOCOL_LOCK.json](outputs/PROTOCOL_LOCK.json) for the frozen test lock.
