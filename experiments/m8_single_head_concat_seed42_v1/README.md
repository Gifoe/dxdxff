# DREChannNet single-head concatenation, seed42

Fixed S0–S4 matrix,25 actual new fits on the original5folds. Raw96 plus
original preprocessed physiological14 slots feed one110→64→1GELU head.
S0 has zero physiology; S2/S4 mask variance after preprocessing; S3/S4
retain original1048-parameter Raw conditioning Gate. Identity models have9329
parameters; gated models10377, below original M8's11018. No D1 predictor,
trained D1 initialization, D1 logit, residual scalar or tanh output bound is
part of any new model.

Original unmodified Raw feature function preserves24D CNN, full masks,
pre-onset reference, early/late pooling, available-seizure mean/SD and
full-patient latentZ. A discarded **untrained** original constructor matches
CNN/Gate RNG consumption; new models copy only these untrained modules.
The independent110-input head initializer is paired across all5arms.

`PROTOCOL_LOCK.json` fixes training, validation-only architecture selection,
all80 main/diagnostic score cells, bootstrap and strict8-endpoint success.
Removing D1 also removes its classifier-specific LN88 readout; this is
disclosed rather than represented as a supervision-only intervention. S0's
14zero slots retain registered capacity/MAC, but those896input weights have
no active feature signal. Gate variants add1048 trainable parameters.

Historical fullM8, noVarianceM8, D1, Raw-only and LateFusion use hash-verified
sealed results, not reference retraining or Outer neural replay. New models
must all train/verify and freeze a validation-selected candidate before new
Outer inference. Scores are sealed before full-label reporting. The reused
Outer remains **EXPLORATORY_REPEATED_TEST**, not independent confirmation.

Execution: `execute.ps1` keeps a foreground SSH lifetime; private files/logs
stay on trusted D drive. Do not orphan detached jobs or duplicate running
workers. CPU reporting uses the existing audited environment with process-only
affinity; model training/inference retains original FP32 runtime and sources.

Expected delivery is source, hashes, anonymous aggregates and `REPORT.md`,
committed and pushed to `codex/m8-single-head-concat-seed42-v1`, with exact
remote-tip verification. Never publish raw EEG, identities, labels, patient/
channel predictions or embeddings, checkpoints, scalers or runtime logs.
