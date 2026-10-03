# Official Omni TimeConv-CNN under an A1-style patient-equal protocol

This is an evaluation-only, exploratory repeated-test analysis. It reads the
already frozen official TimeConv-CNN test representation cache, hard-replays
the official EDF-channel result, and then averages EDF scores within a fixed
`(patient, channel)` identity. It does not import or execute the neural model.

The program keeps patient/channel rows in a private server directory. The
tracked output files are aggregate-only and include the official pooled metric
beside the additional patient-equal protocol. A train/validation threshold is
reported unavailable unless a separately frozen, patient-held-out validation
prediction artifact is supplied; in-sample TRAIN predictions are explicitly
not a legal threshold source.

Run `code/evaluate_frozen_patient_equal.py` only against the hash-bound
frozen representation cache listed in `PROTOCOL_LOCK.json`.
