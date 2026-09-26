# A1/A2 patient-equal objective experiment

This is a fixed-80-patient Task-1 exploratory study of **training objective only**. A0 is a newly trained matched R0 control; A1 changes only the BCE aggregation to an equal mean over patients while retaining EZ=2/NEZ=1 channel weights; A2 adds the locked soft patient Macro-F1 term. All three use the identical R0 inference architecture and exact 30-epoch trajectories.

The primary development estimate is leave-one-validation-patient-out epoch/threshold selection (VLOO), not the apparently selected 13-patient score. The historical R0 validation score `0.650626` is not substituted for A0. Current outer-test outcomes and prior outer result files are not read by development code. The 80-patient cohort has been historically viewed, so any eventual test would be exploratory, not fresh sealed confirmation.

`PROTOCOL_LOCK.json` fixes objectives, seeds, selection, candidate and test gates before training. Private runtime holds per-patient rows and all checkpoints. Only aggregate results and code may be committed.
