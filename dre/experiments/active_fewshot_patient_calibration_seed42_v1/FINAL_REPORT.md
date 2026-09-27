# Active few-shot patient calibration — development-only diagnostic

Exact A1 replay: 150/150 checkpoint grids within 1e-6. Exact R4 classifier replay and B=0 threshold/ranking identity passed. There are 65 VLOO cells, 47 unique patients, 20 fixed label-blind repetitions each. No outer predictions or metrics were computed. The legacy loader nevertheless materializes all 80 labels, so this is not strictly sealed.

All AP comparisons below are against frozen A1 on the identical fixed query channels; intervals use 10,000 patient-ID cluster resamples. Historical EZ labels simulate calibration and do not establish clinical label availability.

Full candidate-pool residual: AP 0.5929, delta +0.0161, 95% CI [+0.0085,+0.0252], folds 5/5.

| Policy | B=1 delta AP | B=2 | B=4 | B=8 | B=16 |
|---|---:|---:|---:|---:|---:|
| RANDOM | -0.0082 | -0.0008 | +0.0054 | +0.0106 | +0.0129 |
| UNCERTAINTY | -0.0155 | -0.0002 | +0.0135 | +0.0229 | +0.0254 |
| UNCERTAINTY_DIVERSITY | -0.0155 | +0.0051 | +0.0157 | +0.0192 | +0.0200 |

Required mean per-repetition positive-full-pool headroom recovery (unstable ratio; see warning below):

| Policy | B=1 | B=2 | B=4 | B=8 | B=16 |
|---|---:|---:|---:|---:|---:|
| RANDOM | -4.34 | -3.26 | -0.70 | +0.22 | +0.84 |
| UNCERTAINTY | -5.93 | -2.21 | -0.06 | +2.50 | +1.22 |
| UNCERTAINTY_DIVERSITY | -5.93 | +0.19 | -0.25 | +0.08 | +1.27 |

## Gate-level evidence and limitations

The few-shot gate passes for UNCERTAINTY at B=8: matched A1 AP 0.5767 to 0.5996, paired delta +0.0229 (95% patient-cluster CI [+0.0087, +0.0394]), positive 5/5 folds. At B=1 both active policies lower AP.
The predeclared active gate passes for UNCERTAINTY_DIVERSITY at B=4: versus matched RANDOM delta +0.0103 (95% CI [+0.0018, +0.0188]), positive 5/5 folds; both-class support 65.9% versus RANDOM 55.1%. Its absolute A1 delta is +0.0157, below the separate +0.020 few-shot threshold.
The two positive gates therefore occur at different policy/budget settings (UNCERTAINTY B=8; UNCERTAINTY_DIVERSITY B=4). No single setting is established as passing both gates. The literal predeclared decision logic still assigns the overall terminal below.
FULL_POOL improves A1 only +0.0161, below its required +0.030, despite 5/5 positive folds and a positive CI. This is a failed capacity gate; the stronger active-subset result does not license calling full-pool capacity supported.
Because adding a constant bias leaves ranking unchanged, BIAS_ONLY AP equals matched frozen-A1 AP by mathematical identity. Full-minus-bias AP is therefore identical to full-minus-A1 AP and is NOT an independent geometry check; the nonzero AP gain itself shows a channel-dependent score change, but oracle-direction alignment remains modest.
Only 786/1300 repetitions have positive FULL_POOL headroom. The required per-repetition recovery ratio divides by sometimes tiny positive headroom, so its mean can exceed 100% or be strongly negative; treat LABEL_EFFICIENCY's recovery thresholds as unstable descriptives, not additional success evidence.

The historical all-channel A1 AP (~0.557) is not the matched fixed-query baseline here (0.5767). No cross-protocol numerical comparison should be made between them.

## Required interpretation

1. A1 reproduction: yes, 150 checkpoints and R4 classifier replay within 1e-6.
2. B=0 reproduces the VLOO-selected A1 decision and ranking exactly on every fixed query split.
3–4. Full-pool residual: AP 0.5929, delta +0.0161, 95% CI [+0.0085,+0.0252], folds 5/5; this is the attainable 50%-candidate-pool capacity estimate for this fixed model, not a few-shot result.
5–7. Minimum budget with +0.01/+0.02/+0.03 AP and positive cluster CI is tabulated in LABEL_EFFICIENCY.csv; the best tested RANDOM/UNCERTAINTY/UNCERTAINTY_DIVERSITY rows are AP 0.5897, delta +0.0129, 95% CI [+0.0052,+0.0220], folds 5/5; AP 0.6021, delta +0.0254, 95% CI [+0.0118,+0.0406], folds 5/5; AP 0.5967, delta +0.0200, 95% CI [+0.0103,+0.0315], folds 5/5, respectively.
8. Both-class support at B=4 is RANDOM 55.1%, UNCERTAINTY 70.2%, UNCERTAINTY_DIVERSITY 65.9%; acquisition does not guarantee a balanced support.
9–10. Full residual versus bias-only matched AP/CI is in FULL_VS_BIAS.csv. Direction recovery is supported only if the predeclared full-minus-bias gate passes; a bias-only improvement is an offset, not geometry.
11. UNCERTAINTY residual cosine to the label-using patient oracle at B=1/2/4/8/16 is 0.100/0.155/0.194/0.235/0.283 (FIT-standardized coordinates). It rises with budget but remains modest; oracle alignment was never an acquisition or fitting input.
12. B=1/2/4/8/16 positive-full-pool-headroom recovery and cluster CIs are in HEADROOM_RECOVERY.csv; repetitions with nonpositive full-pool headroom are excluded, not silently set to zero.
13. UNCERTAINTY and UNCERTAINTY_DIVERSITY versus matched RANDOM are in ACTIVE_VS_RANDOM.csv; the latter is judged only by the locked active gate.
14. Five-fold mean signs and 47-patient clustered intervals are in FOLD_CONSISTENCY.csv and PATIENT_CLUSTER_BOOTSTRAP.csv; 20 repetitions are not independent subjects.
15. The experiment tests target-supervision identifiability but cannot prove zero-label failure has a single causal explanation; full-pool and bias-only controls constrain that interpretation.
16. A later actively selected few-shot model is justified only if both fixed gates pass. Retrospective labels are not automatically clinically obtainable.

`FULL_POOL_RESIDUAL_CAPACITY_NOT_SUPPORTED`
`FEWSHOT_PATIENT_GEOMETRY_CALIBRATION_SUPPORTED`
`ACTIVE_CHANNEL_SELECTION_SUPPORTED`
`ACTIVE_FEWSHOT_PATIENT_CALIBRATION_JUSTIFIED`
