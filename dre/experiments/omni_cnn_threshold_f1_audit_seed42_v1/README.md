# Omni CNN threshold-vs-F1 post-hoc audit (seed 42)

This audit explains why the frozen Omni CNN has AUROC 0.798767 but a
test-derived Youden Macro-F1 of 0.599267, versus 0.659754 at a fixed 0.5
threshold. It **does not** revise the source experiment's failed
`RAW_ENCODER_REPRODUCTION_FAILED` gate.

The original evaluator retained test probabilities only in memory, so there
was no persisted prediction file to consume. The user explicitly authorized
one deterministic replay with the already frozen checkpoint and test NPZ.
`code/replay_frozen_predictions.py` must reproduce all three previous
aggregate metrics within the locked tolerance *before* writing private
segment- and EDF-channel-level prediction tables. Checkpoint, source NPZ,
labels and model are never changed. These patient/channel tables stay on the
server and must never be committed.

`code/audit_thresholds.py` uses only that verified, frozen channel table to
produce threshold sweep, operating points, F1 plateaus, 10,000 fixed-threshold
patient-cluster bootstrap draws, PDF figures and `FINAL_AUDIT.md`. Its
post-hoc F1-maximizing threshold is diagnostic, never a formal benchmark
performance. There is no additional training, variant selection or test
prediction manipulation.
