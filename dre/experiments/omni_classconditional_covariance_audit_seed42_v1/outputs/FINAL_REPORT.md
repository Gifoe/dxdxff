# Omni class-conditional covariance audit

**Terminal status:** `BASELINE_REPLAY_FAILED`.

| Gate | Expected AUROC | Replayed AUROC | Result |
|---|---:|---:|---|
| TRAIN full-record | 0.9586782931 | 0.9571564413 | failed |
| TEST official EDF-channel | 0.7987673466 | 0.7987677712 | passed |

The requested hard stop was enforced. The test replay is within the stated `1e-5` tolerance, but the train replay misses its binding reference by approximately `0.001522`. The expected train reference was not replaced with the observed result.

The accompanying training-side aggregation audit tested the official EDF-channel probability mean plus mean-logit, first/last-segment, all-segment, and patient-channel aggregations. None yields the stated `0.9586782931`. Thus the discrepancy cannot be honestly resolved by selecting a convenient aggregation after observing the value.

No class-conditional covariance, Fisher direction, target Fisher, UTC, diagonal UTC, CORAL, Mahalanobis prototype, bootstrap, center, or sample-size result was generated. No covariance adapter or neural model was created. The known Omni test outcome remains exploratory/repeated-test context, and it was not used for tuning.
