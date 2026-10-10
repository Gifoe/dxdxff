# Implementation audit

## Fixed model and optimization

- One 512-sample PatchTST-style backbone: 31 patches, d32, two four-head blocks,
  FF128, dropout0.10, fixed sine/cosine positional buffer. No window loss.
- Two raw-only energy context descriptors. Median/MAD normalization uses only
  measured within-channel context; no label/center/population fitting.
- Masked attention over up to12 local windows, masked mean across valid runs.
  All-padding handling returns zero raw representation and false availability;
  invalid weights exactly zero. S0 does not publish unsupported missing scores.
- D2 initialization uses exact frozen fold-specific state dicts. Gamma0 with
  a nonzero randomly initialized hidden projection gives exact initial D2
  identity and a gate gradient; raw encoder gradients are checked after movement.
- S1/S2 differ only in label-blind within-patient raw assignment, including
  auxiliary descriptors. Derangements respect identical seizure-availability
  signatures, so seizure membership stays fixed. Singleton groups are audited.
- Lower D2 LR1e-4, new LR3e-4, AdamW/wd1e-4, clip1, max30, min-stop6/patience6.
  Exactly one optimizer update per FIT patient. Channel microbatch8 gradients
  divide by patient channel count; source L2.001 is added once per patient.
- Exact original threshold selection is imported read-only from hash-locked
  `uas_core.py`. No 88D preprocessing is refit or modified.
- Float32, no AMP, TF32 disabled. Full original cohort and registered budget;
  no performance-driven architecture variants or extra training.

## Privacy and leakage

Original clinical labels are read only from frozen FIT/VAL development banks.
No outer TEST labels, scores or performance are used. Raw metadata auditing
retains original80 IDs privately without publishing them. Within-patient
unlabeled preprocessing is consistent with the requested patient-relative
transductive setting. The raw pickle contains unused clinical fields; their
presence is disclosed, not mislabeled as an inherently label-free source file.

D2 warm starts were already selected on the same VAL population. S1/S2 therefore
have repeated-validation selection bias. This is an exploratory development
study, not independent confirmation. S1_RAW_DISABLED is a post-training
intervention on fine-tuned weights, not exact archived D2.

## Engineering and execution

42 named checks and three real FIT-only smoke patients passed on the original
server GPU before formal training. Synthetic success is not predictive evidence.
The final code/protocol hash set was checked again immediately before training.
Original D0/D2 frozen metrics passed checkpoint/hash replay gates.

Additional read-only CPU checks ran while formal GPU training was active:
checkpoint serialization followed by exact model/Adam/dropout RNG replay, and
actual validation raw samples from all five folds at gamma0. All gave bitwise
identity. S0 feature/source independence was also verified. These supplemental
checks did not change formal parameters, clinical targets, selection or budget.

Epoch-boundary resumable state contains model, Adam slots, RNG, selected state,
patience, binding and history. A long-lived authenticated SSH session preserves
the server job; it does not rely on a detached child that Windows SSH may kill.
No unrelated GPU process was stopped. Existing raw/checkpoints are preserved.

Finalization requires15 complete run records with matching selected-checkpoint
and private prediction hashes. Only non-identifying aggregate artifacts may be
committed; raw windows, individual clinical targets, identities, per-channel
scores, checkpoints, optimizer states and runtime logs remain private.
