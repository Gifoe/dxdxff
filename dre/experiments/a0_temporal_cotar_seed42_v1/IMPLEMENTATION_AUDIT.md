# Implementation and pretraining admission

Only T1 and T2 are trained. T0 is the exact previous matched A0, frozen and
evaluated on the same validation channels. Existing20 hard-label OOF teacher
checkpoints were recovered, not retrained. Their original private plan/identity,
checkpoint hashes, train-only imputer/scaler states and exclusion from queried
patient selection/training were independently checked. Fresh eval-mode logits
are used, never MC-averaged probabilities. These logits are detached, so A0 and
teacher parameters are not in the residual optimizer or gradient graph.

Each variant has5,797 trainable parameters; the frozen A0 has8,817. T1/T2 state
tensors and parameter names are identical at initialization, with mode as the
only nonparameter encoder difference. All remaining topology, inputs, continuous
time embedding, masks, pooling, optimizer and loss match. The four-linear core
and post-residual LayerNorm/2*d FFN follow the pinned official TeCh operator.

Stage pooling is independently masked. Across seizures, each stage's mean/std
uses only present stages, and std is population (not unbiased). A single-seizure
std is exactly zero. The sqrt backward is protected at zero without changing
the numerical population-std definition. All-missing softmax rows produce zero
weights/core, and invalid tokens are zeroed before encoding and after the wrapper.

Seven synthetic test groups passed locally and on the server. They cover exact
parameter/state equality and epoch0 identity; official copied-weight CoTAR parity;
local-core noninteraction versus global-core interaction; missing-token invariance
and all-missing finite gradients; stages and cross-seizure mean/std; electrode
permutation and name matching; label-blind median/MAD and constant dimensions;
FIT-only standardization/OOF exclusion; NEZ orientation; and interrupted/resumed
training state/history equality for each variant. Hash-verified live data audits
add cohort/fold/cache/feature-order/timing checks and exact A0 replay.

The real smoke uses only three FIT patients per arm, checking zero residual before
its first update, finite weighted BCE gradients, bounded outputs and absent A0
gradients. It is discarded smoke, not a scientific training result. Formal folds
reseed and use new private cell directories. Complete optimizer, model, RNG and
epoch history states are saved atomically with source/data bindings.

The original early-stop and threshold tie rules remain; epoch0 is an eligible
checkpoint with earlier-epoch precedence on ties. Selecting zero residual is an
honest fallback, not rejection or rerunning a bad training outcome. No temporal
permutation/T3 control, new outer evaluation, data exclusion or rescue model is
authorized. Patient-global constant residual components are diagnostic only.

The task did not quantify severe saturation; before training we froze abs(delta)
>=.49, severe if >10% of validation-channel appearances. The statistics script
uses that channel fraction, while displayed residual means are patient-equal.
No hyperparameter is chosen from these diagnostics. Source and protocol hashes
are immutable for all admitted formal training; finalizer has its own hash.

Execution completion and any engineering interruption are reported in FINAL_REPORT.

## Execution completed

All ten formal runs completed in one attempt (113 completed student epochs).
No completed epoch was discarded; no native crash, retraining or outcome-driven
restart occurred. Final aggregation and10,000 cluster draws completed in one
attempt. Selected checkpoints, histories, private predictions and source/protocol
bindings passed integrity checks; both stderr files are empty. The unexecuted
finalizer's saturation denominator was corrected before aggregate outcomes were
read to implement the already frozen channel-appearance definition. The executed
file's hash is stored separately in RUN_STATUS; no admitted training code changed.
Full results and failed gates are in FINAL_REPORT and results/. No outer scores
were computed. Clinical labels were present in original all-cohort artifacts
before FIT/VAL sealing, but no outer label was used in fitting or selection.
