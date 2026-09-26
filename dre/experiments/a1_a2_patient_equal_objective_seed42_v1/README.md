# A1/A2 patient-equal objective experiment

This is a fixed-80-patient Task-1 exploratory study of **training objective only**. A0 is a newly trained matched R0 control; A1 changes only the BCE aggregation to an equal mean over patients while retaining EZ=2/NEZ=1 channel weights; A2 adds the locked soft patient Macro-F1 term. All three use the identical R0 inference architecture and exact 30-epoch trajectories.

The primary development estimate is leave-one-validation-patient-out epoch/threshold selection (VLOO), not the apparently selected 13-patient score. The historical R0 validation score `0.650626` is not substituted for A0. Current outer-test outcomes and prior outer result files are not read by development code. The 80-patient cohort has been historically viewed, so any eventual test would be exploratory, not fresh sealed confirmation.

`PROTOCOL_LOCK.json` fixes objectives, seeds, selection, candidate and test gates before training. Private runtime holds per-patient rows and all checkpoints. Only aggregate results and code may be committed.

## Completed seed-42 result

Five folds × three objectives × 30 epochs completed. The matched VLOO Macro-F1 means were A0 `0.615099`, A1 `0.625996`, A2 `0.617376`. A1's gain over A0 was `+0.010897`, but only `3/5` folds were positive. A2 did not meet its replacement criteria. The locked candidate was A1; its test-readiness gate failed on fold consistency and both apparent full-validation thresholds. The exact terminal is `A1_A2_DEVELOPMENT_GATE_FAILED`. No seed52/62 or current outer test was run. See `FINAL_REPORT.md` and `development/TEST_READINESS_GATE.json`.
