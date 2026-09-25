# P2.1 V3-ASRR Codebase Audit

Audit date: 2026-07-18. The dirty local workspace, not the remote branch, is
the implementation baseline. No reset, checkout, or historical P2 deletion was
performed.

## Current P2 implementation

`NeuroEZCModel._forward_cane_path_cp_nez` runs the shared B0 encoder, Step4B
physics branch, temporal encoder, cross-seizure aggregator, and
`PatientChannelClassifier`. Its current P2 score path is:

```text
direct_nez_logit
  + CleanNEZPrototypeAnchor.anchor_residual       (individually bounded +/-0.20)
  + MultiSeizureNEZEvidenceResidual               (individually bounded +/-0.20)
  + CausalPropagationResidual                     (individually bounded +/-0.15)
  = final_nez_logit
```

The three bounds are independent, so the theoretical aggregate correction is
`+/-0.55`. There is no aggregate residual budget. The forward exports direct
logits/scores and contextual embeddings; prototype distance, z-distance,
assignment, utilization and residual; per-seizure logits/probabilities plus
mean/std/agreement and residual; six causal features, validity and residual;
and final NEZ/EZ scores.

P2 loss is in `neuroez_c/cane_path_cp_loss.py`. It combines patient-balanced
BCE, soft-worst center reweighting, soft macro-F1, prototype loss, fixed-fraction
high-confidence EZ ranking, and three-branch residual L2. The outer-only trainer
is `run_cane_path_cp_outer_only`; the historical nested PATH-head trainer remains
in `run_cane_path_cp_nested`. P2 checkpoints are
`outer_only_fold_<fold>_checkpoint.pkl`, with fold patient/channel ledgers and
`cane_path_fold_<fold>_seed_<seed>_audit.json` beside them.

Current outer-only checkpoint selection calls `ranking_validation_summary` on
every epoch. In F1 mode this searches a validation threshold per epoch and uses
thresholded F1 in the checkpoint key. In AUPRC mode the primary key is balanced
patient AUPRC harmonic mean. P2.1 therefore needs its own trainer path to enforce
checkpoint/threshold separation without changing P2 behavior.

## Data and causal cache

The patient dataset uses canonical channels from `patient_index`. Cache labels
are EZ-positive (`labels_ez: 1=observed EZ, 0=clean NEZ`) and the model/loss
derive NEZ-positive labels exactly once. `CausalPropagationFeatureStore` aligns
six ridge-VAR proxy features by exact case-folded subject plus canonical channel
key. The parquet builder already emits `cp_feature_valid`,
`cp_valid_seizure_count`, `cp_valid_window_fraction`, and
`cp_mean_var_stability`, but the current store/collator only forwards the six
features and validity. P2.1 must forward all four quality fields.

Centers are available in batches for sampling/loss diagnostics. They are not
inputs to the current P2 forward. P2.1 may use center only in the training-only
clean-NEZ alignment loss.

## V3 pipeline and ledger

The existing V3 TrueBest pipeline is the A9v3 clean configuration driven by
`scripts/run_v3_truebest_clean.ps1` and the shared `run_neuroez_c.py` training
entry. The local inspectable ledger is:

```text
A9v8_LCBO_stage1b_gamma010_mrr_all90_posEZ_standalone_nez_s5_8/
  reference/a9v3_oof_channel_scores.csv
```

It has 8811 rows and 90 patients with fields
`patient_id, subject_id, fold_id, center, channel_name, label_ez,
a9v3_oof_score`. Its score is explicitly an EZ probability, so P2.1 must use
`v3_score_nez = 1-a9v3_oof_score` and derive a clipped NEZ logit. The channel
key is exact canonical subject/channel matching using
`raw_brainbert_data.normalize_channel_name`; substring matching is prohibited.

This global OOF ledger lacks `split_role`, `inner_fold`,
`v3_fit_subject_hash`, and `v3_heldout_subject`. It can be filtered to the frozen
sensitivity80 cohort and used for `precomputed_oof_screening`, but cannot prove
stacked outer-test isolation. Screening must be marked
`PRECOMPUTED_GLOBAL_OOF_SCREENING_ONLY`, `formal_deployable=false`.

Strict `nested_fold_safe` input must instead provide, for every P2.1 outer fold,
outer-train inner-OOF V3 scores and outer-test scores from a V3 model fit only
on that outer train. The provider must reject missing provenance, fit/heldout
overlap, duplicate keys, patient mismatch, channel match below 0.99, nonfinite
scores, ambiguous orientation, or true-count use.

## P2.1 implementation boundary

Historical P2 remains selected only by `use_cane_path_cp_nez` and keeps its
existing forward, residuals, loss, trainer, CLI, checkpoint and ledgers.
P2.1 is selected by the independent `use_p21_v3_asrr_nez` switch. Shared
encoders and evidence modules are reused, with additive P2.1-only outputs and
heads. Profiles R0-R5 share one implementation and differ only through an
immutable profile configuration.

New modules are required for V3 orientation/alignment/robust standardization,
simplex selective fusion, reliability-weighted ranking/preservation, center
alignment, P2.1 training/output orchestration, profile validation, runners,
audits and reports. Existing model, dataset, causal store, CLI and experiment
dispatch require guarded P2.1 integration. No Task 2 code needs modification.
