# Seed-42 outer-validation result and frozen gate

The 8D PCA on 176 label-blind, pre-patient-z raw-feature summaries was fitted independently on outer-fit patients in each fold; the eight components explained 87.86% of outer-fit standardized-descriptor variance on average. No validation or test patient fitted the imputer, scaler, PCA or cardinality heads.

Patient-equal Macro-F1 by fold:

| Fold | C0 formal B0 | C1 constant correction | C2 score context | C3 score + absolute context | C3 − C0 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.641188 | 0.634188 | 0.625716 | 0.609564 | −0.031624 |
| 2 | 0.641978 | 0.633919 | 0.575447 | 0.551038 | −0.090940 |
| 3 | 0.627280 | 0.615867 | 0.565705 | 0.558458 | −0.068822 |
| 4 | 0.639903 | 0.643023 | 0.621328 | 0.634265 | −0.005638 |
| 5 | 0.591553 | 0.590264 | 0.559972 | 0.569570 | −0.021983 |
| Mean | **0.628380** | 0.623452 | 0.589634 | 0.584579 | **−0.043801** |

C1 was positive on 1/5 folds and averaged −0.493 pp versus C0. C2 and C3 were positive on 0/5 folds, averaging −3.875 pp and −4.380 pp. C3 decreased mean EZ-F1 by 7.140 pp and balanced accuracy from `0.666140` to `0.627357`. Relative to C0, C3 improved 24/65 patients, degraded 40/65, and left 1 unchanged. Its mean distance to the diagnostic optimal-K interval was `0.169440` of channel count versus `0.142317` for C0. Rank-cut thresholding reproduced exact Top-K classification without any score/classification mismatches in the validation outputs.

Locked C3 gate: mean Macro-F1 gain ≥ +1 pp; ≥3/5 positive folds; mean EZ-F1 change ≥ −1 pp; positive mean fold correlation with diagnostic oracle `delta_q`; and ≥15% of oracle headroom recovered. C3 met **0/5 conditions**: gain −4.380 pp, 0/5 folds, EZ-F1 −7.140 pp, correlation `−0.0060`, recovered-headroom fraction `−0.588` (the oracle gap was +7.446 pp). `VALIDATION_GATE.json` records `STOP` and `outer_test_arrays_read=false`.

The earlier raw-logit A1 achieved `0.591867` under its own fixed procedure, also below C0. C3 (`0.584579`) is lower, but this is a within-development comparison, not a new outer-test claim. The selected classifier remains frozen B0 ranking plus a small decision head; no new B0 training, BCR distillation, or model search was performed.
