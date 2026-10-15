# Frozen Raw backbone + PR Linear Residual — seed42

**COMPLETE, 2026-10-16.** All20 residual heads and all five Outer folds for all
four backbones are finished. Eight baseline/residual configurations each cover
80 patients and7,635 channels. All640 patient metric rows,10,000 paired bootstrap
draws and20 epoch/threshold selections passed independent verification.
See [the complete report](report.md) and [patient-macro results](results/main_results.csv).
TimeConv-CNN+PR Linear Residual achieved AUROC0.646821, AP0.445751,
Macro-F10.548726 and EZ-F10.359109. All four improved Macro-F1 with paired95% CIs
excluding zero; only SEEG-Net's AUROC/AP paired intervals both excluded zero.

User request: freeze each of the preceding four **Original** raw encoders and
classifiers, reproduce their original Outer predictions, extract full-patient
channel embeddings/logits, and train a separate lightweight PR linear residual
under patient-disjoint inner validation, followed by all five Outer folds.

Parent: `codex/m8-raw-backbones-pr-seed42-v1`, commit
`5f660b5bb3861aa97c879122624132ce5d024171`. New branch:
`codex/m8-raw-frozen-pr-linear-residual-seed42-v1`.

## Fixed model and protocol

For a channel embedding `h` and frozen original NEZ logit `l`, define

`logit_EZ = -l + w^T PR(h) + b`.

Equivalently, the original NEZ logit receives the negative linear residual.
`PR(h)` is the exact parent's whole-patient, featurewise population Z convention:
float64 mean/SD, denominator SD if SD>1e-8 otherwise1, cast back to float32.
It uses the complete valid channel set, no labels or cross-patient statistics.
There is no MLP, second encoder, score router, additional engineered feature,
or logit Z normalization. The only trainable parameters are `w,b`.

All parent encoder/head parameters have `requires_grad=False`; all remain in
eval mode, so BN buffers and dropout state do not change. Original model and
prediction hashes are protected. Zero residual initialization exactly recovers
the original logit, and epoch0 is an eligible no-op validation candidate.

Reuse all original 80 patients, 7,635 channels, 256 seizures, 1,471,965 valid
250-Hz channel-windows and the exact original five FIT/validation/Outer splits.
Original chunks, full-window/seizure aggregation and FP32 inference are retained.
No raw extraction, window sampling, patient exclusion, encoder fine-tuning,
new backbone, new seed, or Outer-driven iteration is permitted.

The residual trains on frozen FIT embeddings only, with CPU float32 AdamW,
lr0.001, weight decay0.001, max200/min10/patience20, gradient cap1. Each epoch
is one complete FIT batch: mean weighted BCE within patient, then equal mean
over patients. The original NEZ class weight uses FIT labels only. No hyperparameter
search. Select epoch by validation patient Macro-F1, then EZ-F1, earlier epoch;
select its EZ threshold on the parent's0:0.005:1 grid and exact tie rules.
Parent encoders were themselves selected using these same legal inner validation
patients; the adapter never fits on validation or Outer patients. This is not
a new independent validation set for the parent encoder.

All twenty selected residual states and thresholds must be hash-frozen together
before any residual Outer predictions. Predictions are sealed before full Outer
labels are read for scoring. There is no test-time adaptation or threshold fitting.

## Reproduction and private representations

Every original Outer patient is replayed through its frozen **complete encoder
and classifier**. Predeclared maximum score/logit/embedding errors are
1e-6/1e-5/1e-4 respectively. Existing hash-bound Outer embeddings/logits/scores
are then reused exactly after successful parity, preserving the original anchor
and baseline metrics bit-for-bit. Fresh FIT/validation representations use the
same selected frozen checkpoint; validation scores must reproduce within1e-6.
No alternative source is selected based on performance.

Each representation cache contains the private channel embedding, original logit,
index alignment and source/checkpoint hashes; no outcome label. It is bound to
the experiment source and protocol. Completed caches and complete-epoch lightweight
head checkpoints support exact safe resume. A source/data/parity failure stops
the pipeline instead of changing its scientific choices.

## Execution and deliverables

Run `run.py` on the trusted Windows server with its pinned parent experiment
and existing environment. Set OMP/MKL/OPENBLAS threads to1, NumPy before Torch.
Keep the owning foreground SSH command alive; do not launch a detached orphan.
Do not run the parent experiment again. The script guards a completed final status
against reevaluation. No installed research or extraction skill is used.

Expected anonymous artifacts include representation-replay/zero-identity audits,
all20 validation selections and head histories; patient-macro, pooled, per-fold,
center, coverage, confusion and ranking summaries; 10,000 seed42 paired patient
bootstrap intervals; independent metric/baseline/hash verification and final status.
Raw, individual identities, embeddings/logits/predictions, labels, original and
residual weights, and runtime logs remain private. Only code, hashes and anonymous
aggregates may be published; this request does not itself create a GitHub PR.

Scientific status remains **EXPLORATORY_REPEATED_TEST**: the cohort and parent
Outer results were viewed previously. A held-out-patient fold is not a new blind
confirmation cohort. Do not tune this residual based on its Outer results.

## Engineering ledger

The first import-only launch found a namespace collision: wildcard import of the
parent support module replaced a local `ROOT` constant before opening the new
protocol file. No new input freeze, representation, head epoch or Outer prediction
had been produced. Renaming the local constant to `TASK_ROOT` fixes only path
resolution. The failed source and error log are preserved privately; the corrected
source is frozen before extraction. No scientific configuration changed.

All final values are actual completed Outer results, not smoke/admission values.
Independent validation-history verification is provided separately under
`tools/m8_raw_frozen_pr_linear_residual_selection_audit_v1/verify_selection.py`;
it verifies epoch0 plus every history argmax and re-computes validation thresholds
without fitting or Outer inference. Parent artifacts remain unchanged.
