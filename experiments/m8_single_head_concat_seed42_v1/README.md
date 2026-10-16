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

## Completed result

All25 fits completed (334epochs); all validation predictions replayed exactly.
The validation-only rule selected **S4 before new Outer inference**.
The sealed fivefold80-patient result is **FAIL_STRICT_M8_TARGET**: S4 misses
ACC/AUROC/Macro-F1/NEZ-F1, despite higher AP/EZ-F1/MRR/Top-1 point estimates.
See [REPORT.md](REPORT.md), [all10models](results/MAIN_RESULTS.csv),
[paired10000-draw intervals](results/PAIRED_COMPARISONS.csv) and
[the unrounded decision](results/FINAL_DECISION.json).

Independent D1 branch cannot yet be removed without verified performance loss.
This is a failed strict point-estimate target under the tested protocol, not
proof of universal single-head inferiority or a statistically confirmed loss.
The eight selected-vs-M8 confidence intervals all include zero. Outer remains
exploratory repeated-test; no post-Outer architecture/threshold rescue was run.

The original frozen `complexity.py` and `execute.ps1` are retained unchanged.
After training, profiling encountered a historical Raw-only checkpoint key
mismatch (`state`, not `model`), before any new Outer. The audited
`tools/complexity_recovery.py` changes only that benchmark loader in memory;
`tools/resume_after_complexity.ps1` completed profiling/evaluation/reporting.
Do not blindly restart completed inference or overwrite preserved logs.
Additional read-only tools independently checked15 trained input masks,
five original initializers, analytical MACs and anonymous confusion counts.

Local aggregate/source release check (standard library only):

```powershell
python experiments/m8_single_head_concat_seed42_v1/tools/verify_release.py
```

Delivery contains source, hashes, anonymous aggregates and `REPORT.md`, pushed
to `codex/m8-single-head-concat-seed42-v1` with exact remote-tip verification.
Never publish raw EEG, identities, labels, patient/channel predictions or
embeddings, checkpoints, scalers or runtime logs.
