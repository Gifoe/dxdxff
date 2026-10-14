# Temporal alignment audit — PASS within cache representation

The historical CoTAR alignment implementation was copied byte-identically and verified against live source SHA256 `3ec0f20ec33e9ce443e07a6443f0c74ce8ed989c7f1d06ec5c553a1134ad2dfd`. Its old relative features/model/predictions are discarded. New physiological values retain original float64 source values, not its clipped relative tensors.

Actual checks:80 patients;256 records;7,635 unique canonical pairs;253 records with59 windows and one each with51/40/56;2-second genuine windows; actual spacing0.9999990463–1.0000009537 seconds;27 padded-boundary windows rejected;0 invalid-time windows;0 nonfinite genuine feature-channel windows.

Canonical names are matched by name, never array position. There are25,267 expected record-channel incidences,24,995 genuinely observed and272 missing. No missing channel or padded zero is used as observed physiology. All7,635 canonical channels have at least two genuine early-stage observations in at least one seizure. Missing incidences remain masked, not interpolated or imputed before summary construction.

All timestamp operations use original onset-relative window centers, stable chronological ordering, and left-closed/right-open stages: pre[-10,0),early[0,8),later[8,20). Stage slopes use actual timestamps, not an assumed59-window grid. Source nine-feature finite masks are reused for validated genuine-window alignment; only the seven specified feature values enter any predictor.

Per-record/channel/feature availability and count ledgers remain private. Single observed time point has slope0 with explicit ledger; single observed seizure has population SD0. Missing summaries remain NaN until TRAIN-only preprocessing. Canonical permutation, missing-channel exclusion, padding and stage separation tests passed.

This verifies annotation-relative cache alignment only. Independent acquisition/EDF physical-clock or onset verification remains incomplete.
