# Historical overlap audit (before current execution)

Scope: the ten named repository experiments and the actual NeuroEZC component implementations. Numbers below are **historical published repository aggregates**, not results of this audit, not new Outer TEST scores, and not matched comparators to the new FIT-only probe. Code, protocols, reports and negative controls were reviewed; historical private predictions/checkpoints were not used.

## Findings

| Historical experiment | Actual representation/control | Historical evidence | Difference from this audit |
|---|---|---|---|
| a0_temporal_cotar_seed42_v1 | Nine local descriptors; same-window channel median/MAD **includes query**; temporal pooling/residual model; optional temporal permutation was not run | T0/T1/T2 AUROC 0.710667/0.708445/0.711127; T2-T1 AUROC +0.002682, CI [-0.001405,0.009183]; Macro-F1 difference -0.000429 | Does not compare query-excluded pre-baseline physiological contrast to a matched stage-local shifted reference. Its time/name/genuine-window alignment is reused exactly, not its relative features or predictions. |
| r1_hlv_ictal_dynamics_seed42_v1 | Learned latent state, source/damping and same-window channel attention; six high-frequency dynamics descriptors | R0/R1 validation Macro-F1 0.650626/0.651611; AP -0.000196 and MRR -0.019863; prescribed +0.010 F1 gate failed | Learned dynamics are not the fixed seven-feature leave-query-out median. No equivalent time-reference shift test. |
| a0_sss_mil_seed42_v1 | Raw within-record robust normalization, query-included window energy contrast and sampled-window MIL; channel-assignment shuffle | S1/shuffled S2 AUROC 0.719545/0.719614 and Macro-F1 0.650392/0.650461; raw-disabled F1 0.650355; CAPACITY_OR_FINETUNING_CONFOUNDED_GAIN | Channel assignment shuffle is not same-query context time shift; negative evidence against attributing a small gain to raw evidence alone. |
| rawtiny_patient_relative_seed42_v1 | Temporal/seizure-pooled latent embeddings followed by patient Z/rank/attention | Raw no-PR/PR AP 0.336249/0.429936; hybrid AP 0.557161 below historical A1 0.576743 | PR can help a weak raw architecture but does not identify synchronous physiological correspondence. No matched reference time shift. |
| seizure_resolved_geometry_identifiability_seed42_v1 | Seizure-channel embedding PCA/distribution/persistence summaries; pooled pseudo-seizure shuffle preserving counts | STATIC_R4/SR_DIST/SR_PERSIST/combined AP 0.5335/0.4957/0.5264/0.4936; geometry gate failed | Seizure assignment and persistence are different from within-stage synchronous window contrast. Live server report/protocol/analyze/extract implementations were reviewed because this experiment is absent from this local branch. |
| patient_set_relational_ranking_seed42_v1 | Pooled R4 patient-set relations and wrong-patient pairing controls | Primary AP 0.571490 vs historical A1 0.576743; correct-wrong AP +0.001029, CI includes zero; R4_RELATIONAL_ZEROSHOT_NOT_SUPPORTED | Pairwise exclusion/attention is generic latent relational evidence, not same-time seven-descriptor leave-query-out physiological reference. |
| d2_graphrole_seed42_v1 | Raw correlation graphs, window graph-role summaries; channel-assignment shuffle | G1/G2 shuffled AUROC 0.747376/0.734149, AP 0.531924/0.532745, Macro-F1 0.649188/0.653198; GRAPHROLE_SHUFFLE_CONTROL_NOT_BEATEN | No new raw graphs or graph descriptors here. Removing a graph self-edge does not establish this physiological reference/time-shift hypothesis. |
| a0_cove_seed42_v1 | Aggregated ABS/DELTA/RATIO residual conditional innovations and products | F0/B3 AUROC 0.716569/0.716295 and AP 0.512349/0.508559; predictive gate failed | No actual window time matching or synchronized reference control. Near-zero aggregate spectral DELTA is not proof all spectral physiology is useless. |
| a0_stable_feature_pruning_seed42_v1 | Function-equivalent duplicate/constant compression and aggregate feature pruning | F0/full42/fixed six-spectral-DELTA deletion/FIT-selected AUROC 0.716569/0.715611/0.719570/0.713472; pruning stability gate failed | Aggregate compression/pruning neither tests nor identifies synchronization. No feature selection from these outcomes is permitted here. |
| a0_fcpa_lf_seed42_v1 | Low-frequency fidelity/lineage audit | LF_DATA_FIDELITY_GATE_FAILED; 162/256 continuous records numerically paired, 94 unresolved EDF+D records; exact raw-generation provenance unproven | The new audit uses only seven existing named window features; no delta/theta/alpha addition, no EDF re-extraction, no claim that 250-Hz raw preserves complete 80-150-Hz gamma. |

## Original components inspected

Live trusted-server source: WindowGraphSpectralEncoder (`graph_spectral_encoder.py`), ChannelTemporalEncoder (`temporal_encoder.py`), CrossSeizureMILAggregator (`seizure_aggregator.py`), PatientChannelClassifier (`patient_channel_ranker.py`), NeuralDynamicsResidualEncoder (`neuroez_c/physics_dynamics.py`) and the model wiring.

- WindowGraphSpectralEncoder explicitly discards adjacency and optionally applies same-window channel MHA. The query participates.
- ChannelTemporalEncoder supports masked mean and optional learned/gated temporal aggregations; none implements the specified physiological leave-query-out reference or its matched time shift.
- CrossSeizureMILAggregator uses masked mean/SD with a numerical SD floor. This audit instead requires exact population SD zero for one observed seizure.
- PatientChannelClassifier applies patient Z and optional channel attention to already pooled embeddings, not window-synchronous physiological contrasts.
- NeuralDynamicsResidualEncoder uses learned state/source/damping and same-window channel attention, with optional actual-time Euler next-state prediction. It does not implement the fixed median contrast/control.

## Decision

No equivalent **correctly synchronized, query-channel-excluded, own-preictal-baseline physiological reference versus matched stage-local circularly shifted reference** was found in the reviewed scope. This is a bounded implementation finding, not a universal novelty claim. Historical attention, PR, temporal pooling and shuffle controls are relevant negative evidence and cannot be ignored.

The audit proceeds only with a label-free achieved-disruption gate fixed before constructing new tables. Circular shift preserves context multisets and cyclic autocorrelation, but may alter ordinary boundary autocorrelation. Within-stage deviation means are algebraically unchanged by the shift; any identifiable contrast must come from variance/covariance and actual-time slopes rather than merely adding dimensions. A failed control stops before supervised probes; a predictive failure does not prove all biological cross-channel interaction irrelevant.

Historical provenance limits remain: annotation-relative cache time only; unknown acquisition filtering, incomplete EDF+D physical-clock/onset and raw-generation verification. No causal propagation or normal-tissue-reference interpretation is authorized.
