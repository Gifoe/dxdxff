# E1 implementation and inference-scope audit

## Controlled switches

E1 is the supplied `IctalLocalization(use_pr=True, use_attention=False)`, **44,401 parameters**. Unlike E3, its supervised encoder is randomly initialized; the runner never calls SSL or loads an SSL encoder checkpoint. Both arms use the same fold-specific supervised seed `42 + 100*fold`, architecture/head capacity, raw preictal/ictal pairs, masks, patient-relative operator, NEZ logit orientation and evaluation score `sigmoid(-logit)`.

The real data path reuses E3's exact amended export. The original audit/manifest, common helper, model and original metric/training module are SHA-locked. Every loaded patient file is hash-checked. No raw waveform is re-extracted and no original source cache or E3 checkpoint is changed.

## Training and selection

The copied E1 loop matches the supplied `train_one(..., 'E1', ...)`: 30 epochs, AdamW LR1e-4/weight-decay1e-3, gradient clipping1, one weighted BCE optimizer step per patient, full available channel context, EZ:NEZ2:1. Checkpoints maximize validation mean patient EZ-AP, tie Macro-F1@0.5, then earlier epoch. The original 37-point validation-only threshold grid and tie rules are reused without edits.

The existing E3 `CachedPatientFiles` helper is reused verbatim; it permits only the current fold's FIT+VAL patients. Fold roles remain disjoint and outer patients are never scored in their held-out fold. Duplicate runner instances are rejected by an OS file lock.

Complete epochs are saved atomically with optimizer and Python/NumPy/Torch/CUDA RNG state. Resume rejects changed protocol/source/export/E3-reference bindings. E1 resume files are separate from all E3 files.

## Numerical tests

Synthetic CPU original-loop versus interrupted/resumed E1 training matches exactly: parameter difference0, identical selected epoch, threshold and metrics. Input caching is exact, held-out access is rejected and changed resume bindings fail. No SSL artifacts are created. These tests validate implementation, not scientific performance.

## Runtime disclosure

E1 uses the previously validated isolated NumPy1.26.4 / PyTorch2.8.0+cu128 runtime from the start. E3's training had used NumPy2.2.6 and its finalization used1.26.4 after native metric crashes. Torch/sklearn/SciPy, model/data and evaluation code are unchanged. The E3 audit found exact synthetic metric/threshold/sampling and real validation input/forward parity across those environments. This is not a claim of identical binary training environments or cross-runtime bitwise GPU gradient reproducibility.

## Comparison semantics

Primary summaries are the five patient-equal fold means: 65 matched validation cells, 47 unique patient IDs. The E3-minus-E1 bootstrap resamples 47 IDs with replacement, retains every matched validation appearance of each sampled ID and recomputes the 65-cell-style mean. This preserves the primary cell weighting; it differs from the single-arm descriptive bootstrap that collapses each ID's repeated appearances first.

Checkpoint/threshold selection used these validation outcomes. Paired intervals therefore remain development evidence, not a confirmatory held-out SSL effect. Independent EDF onset provenance is still unconfirmed. No E0/E2/E4/E5 or outer/external evaluation is authorized or run in this E1 task, even if the predeclared development eligibility gate passes.

Only source and aggregate results/audits may leave the private runtime. Patient/channel rows, waveforms, model checkpoints, logs and secrets must not be uploaded.
