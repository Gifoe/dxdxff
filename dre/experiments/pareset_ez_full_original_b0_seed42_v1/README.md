# PaReSet-EZ full with historical B0 evidence (seed 42)

This is the later, **full-model-only** run requested after auditing the historical CDEL pipeline. It is distinct from the earlier matched 36-D adapted-record experiment in [`../pareset_ez_original_inner_outer_seed42_v1`](../pareset_ez_original_inner_outer_seed42_v1) and from the abandoned no-inner/threshold-0.5 run on branch `codex/pareset-ez-flat5-noinner-seed42-v1`. Do not merge their fold metrics as though they were one protocol.

The five frozen fit/validation/test folds cover 80 unique test patients (16/16/17/15/16). The full PaReSet-EZ architecture and objective are unchanged. The nine source descriptors and 36-D abs/delta/zdelta/ratio evidence are built by the exact archived historical B0 implementation; the normalizer is fitted only on each fold's fit patients. Historical validation patient Macro-F1 selects the checkpoint, historical CDEL's validation-only threshold selector chooses the test threshold, and the historical patient-equal evaluator computes the reported metrics. The experimental model keeps its own optimizer and loss. CDEL is an ensemble with component-specific training and potentially different input-feature settings, so this is **not** an identical-optimization or identical-input-head-to-head comparison; the patient membership and final evaluation semantics are matched.

| Patient-equal metric, seed 42 | PaReSet full | Historical CDEL | Difference |
| --- | ---: | ---: | ---: |
| Macro-F1 | 0.616170 | 0.644589 | -2.842 pp |
| Balanced accuracy | 0.669786 | 0.688029 | -1.824 pp |

Macro-F1 was lower for full in all five folds. See [`results/REPORT.md`](results/REPORT.md) and [`results/FULL_FOLD_RESULTS.csv`](results/FULL_FOLD_RESULTS.csv). This one-seed result is **exploratory**: the 80 patients' historical outer results had already been inspected. It is not a fresh sealed confirmation or a multiseed stability claim.

Provenance:

- Server run: original Windows server, output `D:\nips-temp\pareset_ez_v1\full_original_b0_seed42_v1`.
- Protocol lock: [`results/PROTOCOL_LOCK.json`](results/PROTOCOL_LOCK.json), SHA-256 `61997ec1f2ba16628adfe957c9f1996eb1cbb22e945c028ccfa906bd0393d8b9`.
- The source cache/adapted records and frozen partition manifest remain private. Their hashes are in the lock. No cache, patient-level predictions, checkpoint, or runtime log is published.
- The historical B0 evidence and CDEL reporting code in `code/historical/` match the lock's source hashes. The full-model code, adapter, and runner also match the lock hashes.
- A read-only real-cache parity check compared 991,908 values across three patients and found maximum absolute difference zero. This is a sampled parity check, not an exhaustive 80-patient equivalence proof.

To rerun where the private input files are available, invoke `code/train_full_original_protocol.py` with `--supplement-root ../pareset_ez_original_inner_outer_seed42_v1/code`, `--production-root code/historical/P23_TRN_NEZ_80`, `--model-root code/source`, `--historical-cdel-overall results/HISTORICAL_CDEL_OVERALL.csv`, plus the private `--data`, `--manifest`, and a new `--output-root`. The runner records source hashes in a lock before any test evaluation and rejects a different lock on resume. `code/test_full_original_b0_parity.py` provides the sampled source-cache parity check.
