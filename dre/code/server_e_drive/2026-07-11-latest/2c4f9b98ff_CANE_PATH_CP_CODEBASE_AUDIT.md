# CANE-PATH-CP Codebase Audit

Audit date: 2026-07-17. Base commit: `75952f6` from branch `upload-7-14`.

## Definitions and data flow

| Component | Actual location |
|---|---|
| `NeuroEZCModel` | `neuroez_c/model.py` |
| `PatientChannelClassifier` | `patient_channel_ranker.py` |
| `WindowGraphSpectralEncoder` | `graph_spectral_encoder.py` |
| `ChannelTemporalEncoder` | `temporal_encoder.py` |
| `CrossSeizureMILAggregator` | `seizure_aggregator.py` |
| `Exp_EZHybridLocalization` | `exp_ez_hybrid.py` |
| Outer folds | `data_factory.build_outer_splits` |
| Historical fit/validation split | `data_factory.split_train_val_subjects` |
| Patient dataset and collate | `neuroez_c/dataset.py` |
| Prediction summary and macro-F1 | `exp_ez_hybrid._summarize_prediction_records` |
| Balanced AUPRC harmonic mean | `exp_ez_hybrid._summary_score` |
| Patient/channel ledger output | `Exp_EZHybridLocalization._save_outputs` |
| Fixed protocol enforcement | `neuroez_c/protocol.py` |
| Existing raw/feature alignment | `neuroez_c/dual_view_data.py` |
| Step4B runner | `scripts/run_step4b_static_top20_all90_seed42.ps1` |
| N6 runner | `scripts/run_step4b_n6_dualview_ema_all90.ps1` |

The raw alignment code uses a compound record key consisting of subject, run,
sample and explicit seizure identity. Channels are matched with the existing
canonical channel normalizer. Feature windows are located from explicit
`window_relative_centers_sec`; record or file order is not used as alignment.

## Observed cache schema

The local fixed feature cache
`D:/DRE-Research/aaai/cache_task1_s5_8_local/all_window_cache.pkl` contains 90
patients and 281 run records. A record stores `window_features` as `[W,C,28]`,
`window_adjacency` as `[W,C,C]`, window centers as `[W]`, channel labels as
`[C]`, and canonical channel names as a length-`C` list.

The local raw cache `D:/DRE-Research/aaai/raw/all_window_cache.pkl` contains 155
patients and 501 records. Each sample stores `raw_waveform` as `[C,T]`,
`raw_temporal_sfreq`, `raw_temporal_duration_sec`, explicit feature-window
centers, and the same record/channel identity fields. The inspected first raw
record had shape `[88,15000]`; raw length is record dependent and must not be
hard-coded.

`collate_patient_ez_batch` produces:

- feature tensors `[B,S,W,C,F]`;
- physics tensors `[B,S,W,C,Fp]`;
- seizure mask `[B,S]`;
- seizure-channel mask `[B,S,C]`;
- window mask `[B,S,W]`;
- patient channel mask and labels `[B,C]`.

Raw tensors are only created for the historical N6 path as `[B,S,C,W,T]`.
CANE-PATH-CP does not consume this tensor; raw data is used only by the offline
ridge-VAR feature builder.

## Semantics and protocol

The cache field `labels_ez`/patient-index `labels` uses `1=observed EZ` and
`0=clean NEZ`. The NEZ-positive dataset path performs exactly one conversion:
`labels_nez = 1 - labels_ez`. CANE-PATH-CP logits represent `P(NEZ)` after a
sigmoid, and `score_ez` is generated as `1-score_nez`.

Fixed outer folds are generated patient-wise by `build_outer_splits`; the new
80-patient sensitivity cohort retains those original All90 assignments and
only removes the ten manifest subjects. The original manifest
`reference/all90_subjects.csv` is not modified.

The Step4B static feature order is:

1. `early_high_gamma_slope`
2. `early_line_length_slope`
3. `onset_latency_high_gamma`
4. `onset_latency_line_length`
5. `onset_rank_high_gamma`
6. `onset_rank_line_length`
7. `high_gamma_top20pct_mean`
8. `line_length_top20pct_mean`

Prediction ledgers use `(outer_fold, subject_id, channel_name, channel_id)` as
the ensemble key. Labels and center are alignment assertions, never model
inputs.

## Implemented files

Modified: `data_factory.py`, `patient_channel_ranker.py`, `neuroez_c/dataset.py`,
`neuroez_c/model.py`, `neuroez_c/protocol.py`, `run_neuroez_c.py`, and
`exp_ez_hybrid.py`.

New: cohort manifest/audit helpers, `cane_path_cp_heads.py`,
`causal_propagation_features.py`, `cane_path_cp_loss.py`,
`cane_path_threshold.py`, `cane_path_cp_trainer.py`, offline feature/audit
CLIs (including a read-only existing-cache audit), the Step4D runner, ensemble
and profile aggregators, 95 focused tests, and final method/report documentation.
