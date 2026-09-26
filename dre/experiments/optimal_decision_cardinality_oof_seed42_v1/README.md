# Optimal decision-cardinality OOF, seed 42

This exploratory experiment tests whether a small label-blind patient head can correct the number of channels selected by the frozen historical B0 ranking. It reuses the audited 25-cell outer-fit patient cross-fit and five historical B0 checkpoints from `direct_patient_shift_oof_seed42_v1`. The historical outer outcomes were already viewed; this is not fresh confirmation.

`PROTOCOL_LOCK.json` was fixed before new optimal-K targets or validation outcomes were computed. The server-only cache contains patient/channel predictions, raw features, thresholds and models. Public `A/` contains aggregate audits and validation results only. `C3` is the predeclared primary candidate, with a five-condition gate before any new outer-test reading.
