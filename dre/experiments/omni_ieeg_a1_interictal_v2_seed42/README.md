# Omni-iEEG A1 interictal v2, seed 42

This experiment starts from frozen v1 and changes exactly three components:
official channel ground truth, contemporaneous within-EDF cross-channel
reference for the original 9×4 views, and an official-train-validation-selected
classification threshold. It does not change the 27,713-parameter A1 model,
weighted patient-equal BCE, AdamW settings, seed, official split, or signal
preprocessing. See `PROTOCOL_LOCK.json` for the exact numerical rules.

The ordered official evaluation condition is important: successful-outcome
unresected `good=1` channels are **normal before SOZ is checked**. This differs
from a common verbal summary of the benchmark. The source-line audit and the
number of affected channels are in `OFFICIAL_LABEL_DEFINITION_AUDIT.md`.

All recordings and native-signal cache remain private on the original server.
`code/audit_official_labels.py` reconstructs the eligible metadata cohort from
the frozen Omni source and cache. `code/extract_cross_channel_features.py`
builds resumable private 36D features. `code/run_pipeline.py` waits for official
train features, then trains/selects/refits/freezes, and only then launches the
official-test extraction and single inference. Its D1 v1 rescore explicitly
reports the prediction-overlap denominator; v1 did not score every channel in
the v2 official cohort, so a full-cohort label-only attribution is unavailable.

Local/server numerical tests:

```text
python code/test_cross_channel_reference.py
python code/test_v2_metrics.py
```

The server-specific `run_pipeline.py` is a resume-only coordinator. Train
feature shards must be started separately and never duplicated. Runtime NPZ,
checkpoints, per-patient prediction resume files, and logs are excluded from
Git. Compact outputs are committed only after they are validated.
