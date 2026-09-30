# Event source audit

Terminal: **`EVENT_EVIDENCE_UNAVAILABLE`**

The official inference implementation at revision `57c22a75a59b5c3a98006806ad42000f6a3fa5b6` requires an external
`--model_path`; the repository does not bundle that event checkpoint. The server audit found the
following allowed checkpoint candidates:

- None found.

Existing result files containing `3label_pred` or `pathological_probability`:

- None found.

The dataset does contain 1014 HFO candidate CSVs with columns
`detector, end, file_name, name, participant, session, start` and 54 expert-annotation parquet files. These are
candidate timestamps and a limited event-classification training/evaluation corpus, respectively;
they are not frozen three-class inference outputs covering official Task 2 TRAIN/TEST records.

Under section 6 of the locked ECEA protocol, training a replacement Task-2 event detector is
forbidden. Therefore adapter CV, parameter fitting, freeze, and official TEST evaluation were not
run. This is an availability terminal, not evidence that ECEA helps or fails scientifically.
