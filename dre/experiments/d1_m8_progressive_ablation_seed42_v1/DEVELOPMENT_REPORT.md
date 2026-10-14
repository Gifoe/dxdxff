# Development —65 appearances,47 unique patients

| Model | AUROC | AUPRC | Macro-F1 | EZ-F1 | MRR | Top1-EZ | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|---|
| A0 Raw ABS | 0.684288 | 0.463550 | 0.555853 | 0.323305 | 0.669599 | 0.569231 | 0.412047 | 0.809406 |
| A1 Raw DELTA | 0.681432 | 0.470875 | 0.566046 | 0.332951 | 0.665033 | 0.584615 | 0.387291 | 0.834094 |
| A2 frozen D1 | 0.725978 | 0.503601 | 0.644546 | 0.443537 | 0.658502 | 0.523077 | 0.506245 | 0.866074 |
| A3 frozen M8 | 0.756633 | 0.542226 | 0.666795 | 0.477249 | 0.699708 | 0.569231 | 0.544976 | 0.876867 |

These are actual A0/A1 validation outputs plus original hash-verified D1/M8 validation ledgers. They are not Outer values. All models have identical canonical patient/channel/fold identities. D1/M8 checkpoints, scalers and numeric thresholds were reused unchanged. A0/A1 undergo independent validation-only checkpoint/threshold selection under the original policy. No architecture selection followed these results.

| Model | Fold | Selected epoch | Frozen NEZ threshold | New epochs run |
|---|---|---|---|---|
| A0 | 1 | 23 | 0.535 | 29 |
| A1 | 1 | 7 | 0.415 | 13 |
| A0 | 2 | 2 | 0.465 | 8 |
| A1 | 2 | 2 | 0.520 | 8 |
| A0 | 3 | 3 | 0.400 | 9 |
| A1 | 3 | 19 | 0.375 | 25 |
| A0 | 4 | 2 | 0.450 | 8 |
| A1 | 4 | 8 | 0.425 | 14 |
| A0 | 5 | 2 | 0.500 | 8 |
| A1 | 5 | 1 | 0.520 | 7 |
| A2 | 1 | 2 | 0.435 | reused |
| A2 | 2 | 19 | 0.260 | reused |
| A2 | 3 | 5 | 0.300 | reused |
| A2 | 4 | 3 | 0.320 | reused |
| A2 | 5 | 5 | 0.400 | reused |
| A3 | 1 | 4 | 0.410 | reused |
| A3 | 2 | 1 | 0.255 | reused |
| A3 | 3 | 10 | 0.295 | reused |
| A3 | 4 | 13 | 0.365 | reused |
| A3 | 5 | 15 | 0.395 | reused |

A0→A1 Development AUROC decreases0.002856 even though its Macro-F1 rises0.010192. The desired progressive trend is therefore not uniformly present across scopes/metrics. No model was altered in response. See the separate Outer report for the authorized80-patient evaluation.
