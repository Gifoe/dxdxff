# Joint learned-response moments: actual full80 result

## Decision and metric contract

The target is **full80, equal-patient Macro-F1 >0.65**, computed by taking
each patient's mean of EZ-F1 and NEZ-F1, then averaging80patients. It is not
EZ-F1, pooled-channel F1, a fold-equal mean, an internal score, an oracle
threshold or an interval upper bound. **This experiment did not reach it.**

The prespecified primary is SELECTED: each original fold's procedure was
chosen only by legal ownFIT internal assessments, before original Development
or new Outer access. Its Macro-F1 is0.6264457691. No fixed secondary is
retrospectively adopted as the primary. All historical Outer has repeatedly
been viewed, so these are exploratory repeated-test results, not independent
confirmation. The research goal remains unachieved.

## Complete patient-equal fivefold results

Every row covers the same80patients/7,635channels. AP is patient average
precision; Top-1 is the fraction of patients whose highest-ranked channel is
EZ. F1 in the underlying CSV's `f1` field denotes EZ-F1. Bias is predicted
minus actual EZ fraction, averaged equally across patients.

| Procedure | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac. bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed Native | 0.630493 | 0.741296 | 0.550114 | 0.426475 | 0.834511 | 0.757014 | 0.784454 | 0.7000 | -0.041381 |
| Fixed precompand | 0.631432 | 0.759384 | 0.570215 | 0.420221 | 0.842644 | 0.766018 | 0.802328 | 0.7250 | -0.066138 |
| PRE_MEAN_DUP | 0.613641 | 0.749956 | 0.565860 | 0.399383 | 0.827899 | 0.748052 | 0.803120 | 0.7125 | -0.044790 |
| PRE_ENERGY_DUP | 0.619017 | 0.748655 | 0.558770 | 0.407830 | 0.830203 | 0.751633 | 0.788654 | 0.7000 | -0.036904 |
| PRE_JOINT | 0.632344 | 0.763467 | 0.569687 | 0.431127 | 0.833560 | 0.756896 | 0.800474 | 0.7125 | -0.039736 |
| SELECTED (primary) | 0.626446 | 0.759603 | 0.569824 | 0.413132 | 0.839759 | 0.762137 | 0.792933 | 0.7125 | -0.063222 |

The joint point is only0.001851 above Native (0.185percentage points), not
two points. The primary is0.004047 below Native. The preceding compact
PRE_MEAN experiment's full80 point was0.634936, also approximately0.63 and
still below0.65. That older configuration differs in head shape and is
descriptive context, not an equal-capacity comparison or a new selection input.

## Paired uncertainty and stability

All intervals below are nominal95% percentile intervals from10,000 paired
patient bootstrap draws. They condition on these fitted models and reused
data; they are not multiplicity-adjusted or independent-test guarantees.

| Macro-F1 contrast | Difference | Paired95% interval |
|---|---:|---|
| Fixed precompand - Native | +0.000939 | [-0.009937, 0.012318] |
| MEAN_DUP - Native | -0.016852 | [-0.042634, 0.008995] |
| ENERGY_DUP - Native | -0.011476 | [-0.035318, 0.012273] |
| JOINT - Native | +0.001851 | [-0.021141, 0.025299] |
| SELECTED - Native | -0.004047 | [-0.023319, 0.015530] |
| JOINT - MEAN_DUP | +0.018703 | [0.002624, 0.035223] |
| JOINT - ENERGY_DUP | +0.013327 | [0.000317, 0.027561] |
| ENERGY_DUP - MEAN_DUP | +0.005376 | [-0.008491, 0.018863] |

Joint responses outperform the duplicated controls at the nominal interval
level in this reused final test, but not Native. This is evidence of a
conditional moment-combination effect, not evidence of reaching the target
or overall superiority. The internal comparisons to both duplicated controls
had intervals crossingzero. Joint also failed to improve upon the stronger
preceding compact PRE operators in matched post-hoc internal comparisons;
all those results are retained in INTERNAL_REPORT.md.

| Procedure | Absolute Macro-F195% interval | Fivefold Macro-F1 sampleSD |
|---|---|---:|
| Native | [0.601968, 0.659418] | 0.035327 |
| Fixed precompand | [0.602277, 0.660779] | 0.037602 |
| MEAN_DUP | [0.583880, 0.643598] | 0.061419 |
| ENERGY_DUP | [0.590279, 0.647753] | 0.052987 |
| JOINT | [0.603724, 0.661595] | 0.048771 |
| SELECTED | [0.595559, 0.658531] | 0.063367 |

Neither the joint arm nor the primary improves fold stability over Native.
Upper confidence limits above0.65 do not make any point result exceed0.65.
Full nine-metric intervals, per-fold results and finite coverage are published,
not only the favorable contrasts.

## Sources and aggregation are not interchangeable

| Procedure | HUP (36) | LZU (21) | Multicenter (15) | Pediatric (8) |
|---|---:|---:|---:|---:|
| Native Macro-F1 | 0.651040 | 0.551194 | 0.698488 | 0.618699 |
| Fixed precompand | 0.665682 | 0.548155 | 0.686961 | 0.591797 |
| MEAN_DUP | 0.671538 | 0.529112 | 0.640020 | 0.525533 |
| ENERGY_DUP | 0.668246 | 0.536814 | 0.646665 | 0.561430 |
| JOINT | 0.686011 | 0.551745 | 0.656177 | 0.557725 |
| SELECTED | 0.668164 | 0.526145 | 0.698082 | 0.567684 |

The joint increase is not uniform across sources: its higher HUP point is
offset by lower multicenter and pediatric points than Native. These small,
exploratory strata do not establish source-specific mechanisms or justify
choosing different architectures from Outer labels. No patient is removed.

Pooled-channel Macro-F1 is0.633552 Native,0.640932 JOINT and0.629220 SELECTED;
these differ from the patient-equal primary because channel counts vary.
The complete pooled metrics/confusion totals remain in OUTER_POOLED.csv,
clearly separate from OUTER_PATIENT_MEAN.csv. No aggregation is substituted
to make the target appear achieved.

## What was actually trained and frozen

One CNN (1to12,k9/s4;12to24,k7/s4;GELU) produces learned response mean `mu`
and `log1p(mean(h*h))`. Three random-start networks concatenate(mu,mu),
(energy,energy) or(mu,energy), preserving an identical192to64to1head and
14,577parameter positions. Each fold starts with exactly matching CNN/head
weights across the three arms. Duplicating a statistic matches parameter
positions, not effective information capacity. This generic moment pool is
not claimed novel or interpreted as measured physiological band power.

All fresh models retain pre-only waveform normalization, pre-latent reference,
0to8/8to20phase means, available-seizure mean/populationSD and complete-patient
Z. There are no manual physiological predictor features, independent D1
branch, old logits, prior embeddings, warm-start weights or window subsampling.
FP32/TF32off, optimizer/loss/budgets and the201threshold grid were frozen.
CAL selected internal epochs/thresholds; internal HELD selected procedures
within each ownFIT. Original Development selected final epochs/thresholds.
Outer did neither. Full configurations and rules are in the two protocol locks.

The internal program actually performed39freshfits/1,061epochs. Fifteen final
FIT/Development fits then performed346epochs, totaling1,250.033964epochseconds
(mean3.612815s/epoch), excluding inference, replay, audits and orchestration.
No completed epoch was discarded. All retained final epochs were >=1.
Fivefold primary choices were FROZEN_PRE/NATIVE/JOINT/ENERGY_DUP/JOINT.
SELECTED is therefore a fold-specific policy, not one uniform new CNN.

All30models/thresholds were frozen before the new Outer pass. All30fold score
cells (references reused exactly, fresh models actually forwarded) were sealed
before full Outer truth was opened. One sealed pass covered the original
fivefold Outer counts16/16/17/15/16. No best-Outer arm, threshold or patient
subset was selected afterward.

## Independent checks, Raw support and engineering events

All480patient metric rows/9metrics,126absolute and paired interval rows,
15Development prediction replays and all201grid decisions passed independent
verification. Maximum Development replay error was0; maximum metric error
1.1102230246251565e-16; maximum bootstrap difference2.220446049250313e-16.
All three random initial states per fold matched exactly; frozen references
and selected aliases were unchanged. Source/fold/coverage checks passed.

The separate Raw census passed80patients/7,635channels/256seizures and
1,471,965valid500sample windows. All valid samples/times were finite. The
18,632invalid padding slots include2,584nonfinite time slots, all outside the
unchanged valid mask. Coverage is not reduced to remove them. Counts are
724,672pre,199,960at0to8,299,940at8to20 and247,393at>=20seconds. All valid
windows are encoded; only the fixed pre/two-phase support (1,224,572windows)
is claimed to influence the classifier. No later-window contribution or
low-frequency provenance is inferred from250Hz sampling alone.

The first final admission had an incorrect FIT-union=80assertion. Actual
unchanged original roles have79unique FIT patients but80Outer patients;
no role or data changed. The exact-role admission repair passed before
any final fit. Its original code/logs and failed admission are preserved.

After writing and printing metric verification PASS, the verifier process
had a Windows Application Error1000: PID27612, torch_cpu.dll,0xc0000005.
The external coordinator correctly stopped before the Raw census. A native
teardown problem is consistent with the timing but the underlying defect
was not established. NewRunTag03 skipped all finished stages, performed
only the missing census, and exited0 with
`JOINT_RAW_ALL_SIX_FINAL_PROCEDURES_VERIFIED_EXACT_ORIGINAL_ROLES`.
No training, prediction or metric computation was rerun in this recovery.

Key SHA256 locks:

- Final input: `cd08eeec8272affa4ac7dac448b47b206a725f4ec0c74abe00faf27c25b0d6f2`
- Final model/threshold freeze: `85453fe0914bc93f6b9bd41f553932d2c6fc1870b6e4ab19e3e8666350b88266`
- Final score seal: `476a96a1b4d200002238f345f6a716e46ef0be955e1369ea2aaf6d50a3193cc1`
- Whitelisted17file delivery: `52b144f50cacccd883bc6b272ff46fca8132c39f75003a36a436e671dbe58072`

Only source, hashes and anonymous aggregates are uploaded. Raw, truth,
identities, weights, individual scores/latents, runtime logs and private
operational records remain on the trusted server. No claim of goal completion
or independent confirmation is made. Future candidate selection must use
legal internal data, not the source/fold breakdowns in this report.
