# Historical mechanism review, before Raw reconstruction

Actual source/protocol/reports were reviewed, including pinned live waveform alignment, physiology functions and original feature aggregation. Below numbers are historical repository aggregates, not new run outcomes or permitted Outer replays.

| Study | Relevant finding / mechanism | Boundary for this task |
|---|---|---|
| a0_compact_delta_ratio_only_seed42_v1 | DELTA_ONLY Development AUROC0.716600/AP0.504414/Macro-F10.642347/EZF10.443493;26 retained88-interface coordinates,14 unique varying,8aux duplicates,4zeros. Original source-free NEZ MLP and PatientZ. | Replay only Development from existing frozen checkpoints. No historical individual Outer files opened. |
| a0_compact_rawfirst_ablation_seed42_v1 | RawTiny32 pooled all windows/seizures, latentPatientZ and seeded original classifier: Development AUROC0.630896/MacroF10.567631 versus F0 0.716569/0.642067. | Direct raw pooled replacement lost useful engineered behavior; verify reconstruction first rather than repeat pooling-only architecture. Exact compound-key/window slice utilities reviewed. |
| a0_stable_feature_pruning_seed42_v1 | LN88-aware folding verifies float64 logit error1.33e-15 and float32 error8.34e-7; retraining not optimization-equivalent. Pruning gate failed. Spectral DELTA numerically tiny, not biological irrelevance. | Reuse exact folding algebra; matched E1 isolates compression/optimizer changes. Never replace LN88 withLN14 or drop duplicates without folding. |
| rawtiny_patient_relative_seed42_v1 | Raw noPR/PR AP0.336249/0.429936; hybrid0.557161 below A1 0.576743. Mean learned residual small. | ExistingRawTiny normalization/pooling loses amplitude/order and never recovered engineered features first. Proposed conditional latent pre-reference+GRU is different, not proof it will improve. |
| a0_sss_mil_seed42_v1 | S1 true raw/S2 shuffled MacroF10.650392/0.650461; disabling raw0.650355. CAPACITY_OR_FINETUNING_CONFOUNDED_GAIN. | Warm-started engineered fine-tuning/context capacity can mimic gain. Freeze D1 and compare equal-initialized time-order control with same windows. |
| a0_temporal_cotar_seed42_v1 | T2-T1 AUROC+0.002682 CIcrosses0; MacroF1-0.000429. Correct context query-included; [-10,0)stage baseline. | Temporal pooling/residual gate did not establish extra useful interaction; do not substitute CoTAR baseline for historical allcenter<0 DELTA. Reuse genuine identity/mask handling only where appropriate. |
| r1_hlv_ictal_dynamics_seed42_v1 | Learned state/source/damping: R1-R0 validationMacroF1+0.000985, AP-0.000196, MRR-0.019863; failedgate despite activatedbranch. | Nonzero physiology residual is not useful increment; fixedaux0.25 prevents zero-gate gradient starvation but does not justify success. |
| a0_cove_seed42_v1 | Correct residual products fail AUROC gate; B3 0.716295 versus F0 0.716569. Most gains survive disabling corrections; spectral DELTA suffers original SDfallback. | Conditional innovation is not independent information. Report everyfamily numerical fidelity and distinguish representation/optimization effects. |
| a0_fcpa_lf_seed42_v1 | LF_DATA_FIDELITY_GATE_FAILED: fullRaw extraction lineage/acquisition/EDF+D mapping unresolved. | Raw content may be measurable but native filtering and timing not independently certified. No newbands,EDF/raw acquisition or complete80-150Hz claim. |
| window_synchronous_patient_contrast_seed42_v1 | Inner context versus shifted AUROC+0.026080 positiveCI; requestedOuter contrast+0.013444 CI[-0.006018,0.034423] failed; shiftedcontrol bestOuter MRR/Top1. | Synchronous engineered window contrast does not confirm learned Raw order. This task does not reuse its models/predictions or reopen Outer. |

## Identified pinned implementation

- `b0_self_reference_features`:float32 prewindowmean, center<0, DELTA x-premean; its all-window fallback must never trigger in genuine reconstruction.
- `compute_spectral_channel_features`:Welch periodicHann,detrendFalse,2s,half overlap; original half-open band bins, float32 trapezoid and log1p. RMS includes1e-8 offset; entropy divides by PSDsum clipped1e-8, not a generic textbook entropy. All preserved explicitly.
- `task1_baselines/feature_aggregation.py`, SHA426344bfa6ef0dd4b1dc59fe267d7920cef020fdcb42e834efc9a9ae6b4ca273:profilep2_matched_simple averages finite original window evidence within a seizure, then mean/populationstd of available seizure means by canonical channel. No median/OLS/stage averaging replacement.
- `uas_core.patient_z`:float64 perpatient nanmean/nanstd; SD<=1e-8 uses1; original FIT-only SimpleImputer thenStandardScaler.
- Raw alignment/source hashes are independently pinned in PROTOCOL_LOCK.500samplewindows derive actual centers+valid ranges, not positional channels.
- Folding: absorbLN affine into first-linear weights, include multiplicity in88-position mean/variance and zero-position contribution, retain1537 trainables. Frozen identity and scratch optimizer parity are distinct.

## Decision

No reviewed study first reconstructed the exact14D physiologicalDELTA from the requiredRaw cache, passed a four-of-five-family fidelitygate, matched compactE1/D1, then frozeD1 and tested smallCNN+latentpre-reference+GRU against stage-local orderdestruction. This is a bounded overlap finding, not a claim of universal novelty. Historical failures require stagedstop, not a larger architecture rescue. OnlyD1–D3 plus requiredE1 andD0Development replay are in scope. No newOuter permitted.
