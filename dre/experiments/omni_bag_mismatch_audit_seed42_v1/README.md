# Omni Train-Inference Bag Mismatch Audit

This directory contains a pre-registered, evaluation-only audit of whether the
historical five-window TRAIN representation materially differs from frozen
full-record inference. It does not fit a model or select a threshold.

The private segment-level caches are intentionally kept off Git because they
contain channel identities. Public outputs contain aggregate metrics, runtime,
and cryptographic provenance. Run order is:

1. `code/preflight_replay.py` — verify the frozen TEST and historical TRAIN-5
   AUROCs before any new inference.
2. `code/extract_train_full.py` — apply the exact frozen TEST windowing,
   preprocessing, CNN, and probability semantics to every official TRAIN EDF.
3. `code/run_audit.py` — compute the locked comparisons, subsampling audits,
   source/duration checks, and final report.

See `PROTOCOL_LOCK.json` for all fixed seeds, thresholds, and decision rules.
