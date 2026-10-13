# Executed model identity, feature isolation and leakage audit

PASS: all five authoritative legal FIT/VAL banks, original feature/order/split
hashes and original preprocessing are verified. Prior pruning input freeze,
prepared banks, selected checkpoints, prediction seal and independent verification
are hash-locked. The main is exactly the prior B2, not F0 or the automatic B3.
Every main checkpoint initializer hash and selected checkpoint/prediction binding
replays; frozen main probabilities reproduce with **max drift0** on CUDA.
Original thresholds, decisions and patient metrics are identical. Main is not
retrained, recalibrated or substituted by new predictions.

The authoritative manifest is copied unchanged and hash-verified. B2 removes
three named spectral DELTA groups, six coordinates. Single-view sets are a
disjoint partition of B2's36 canonical inputs: ABS14, DELTA8, RATIO14. DELTA
contains only RMS, variance, line-length/sec and spectral entropy mean/std.
Delta the frequency band is not DELTA the reference-difference feature view.

Original preprocessing first performs Patient Z within each patient's unlabeled
channels and the original FIT-fitted full88D imputer/scaler. No scaler is refit
on validation; no new raw data or feature construction. ABS_ONLY still includes
the original Patient Z, so it is **not** a patient-context-free absolute model.

Each compact graph gathers only its canonical retained coordinates. Deleted
positions are zeros in reconstructed original88-position normalization; original
duplicate multiplicities and epsilon1e-5 are preserved. Main has42 virtually
nonzero positions, the single views14 each, despite DELTA having only8 independent
inputs. This is not native LayerNorm(k), score fusion or a trainable mask.

Actual legal FIT/VAL untrained-network folding parity: max float64 logit drift
5.274e-16; max float32 drift2.683e-7. Arbitrary-LN-affine synthetic double/float
and actual-manifest tests pass. Changing any excluded input leaves outputs
bitwise identical on all tested actual banks and every final validation model.
Trained frozen-main parity and untrained folding parity are separate tests.

All15 new model-fold cells independently fold the original **untrained** seeded
A0 initializer; they never warm-start from trained B2 or another view. Physical
first-layer input widths are14/8/14, parameters1537/961/1537. Hidden96/GELU/
Dropout0.15/final classifier topology matches the main. AdamW0.001, decay0.0001,
clip1, FIT-only NEZ-positive BCE weighting, one FIT-patient update, max30/min6/
patience6 and validation Macro-F1/EZ-F1/earlier-epoch selection with original
0.005 NEZ-threshold grid are unchanged. Same fold seed42+1009*fold. Losses
and thresholds are not adjusted after comparative outcomes.

All15 selections and thresholds freeze before comparative Development summary.
New predictions are atomically saved and hash-sealed. Independent sklearn
score-only verification PASS for260 patient/model metric cells, including65
frozen-main appearances. All66 paired metric/interval calculations are checked
by an independently implemented cluster-multiplicity bootstrap. No verifier
model inference, new threshold search or outer outcome evaluation.

All15 cells completed137 epochs; measured original-trainer invocations total
11.715 seconds, excluding admission/tests/inference/statistics/transfer/reporting.
No completed-task reuse or clinical-training crash in this run; stderr0. The
previous study's file-lock/native/transport history is not attributed to this run.

The original80-patient/256-seizure/7635-channel contract is inherited and checked
by original archive/order/partition/bank provenance; only legal FIT/VAL arrays
are used. Outer membership metadata excludes test patients, but no outer features,
scores or fresh TEST evaluation are opened. Historical results remain unchanged.

The user selected B2 after seeing prior Development results. Such adaptive reuse
is disclosed, not relabeled validation-only architecture selection or independent
confirmation. Input capacity and normalization geometry change with view deletion;
these are model ablations, not causal physiological irrelevance tests. No extra
model, view combination, ranking rescue, seed, threshold tuning or installed skill.
