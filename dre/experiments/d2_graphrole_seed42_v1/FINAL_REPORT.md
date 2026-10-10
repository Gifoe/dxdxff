# D2-GraphRole seed42 — completed exploratory development experiment

| Model | Macro-F1 | EZ-F1 | NEZ-F1 | BA | EZ-AP | EZ-AUROC | MRR | Top1 | Sensitivity | Specificity | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D2 | 0.647926 | 0.443199 | 0.852653 | 0.669549 | 0.519936 | 0.718221 | 0.719610 | 0.615385 | 0.463999 | 0.875099 | 0.790571 |
| G1 | 0.649188 | 0.451017 | 0.847360 | 0.676590 | 0.531924 | 0.747376 | 0.761840 | 0.676923 | 0.494371 | 0.858808 | 0.782107 |
| G2 | 0.653198 | 0.442994 | 0.863402 | 0.664488 | 0.532745 | 0.734149 | 0.768563 | 0.692308 | 0.437049 | 0.891927 | 0.802216 |
| G3 | 0.645147 | 0.443346 | 0.846948 | 0.668910 | 0.506121 | 0.719322 | 0.722786 | 0.630769 | 0.479811 | 0.858009 | 0.787108 |
| G1_GRAPH_ZERO | 0.637575 | 0.443224 | 0.831926 | 0.674325 | 0.512918 | 0.729320 | 0.726622 | 0.646154 | 0.520770 | 0.827881 | 0.763073 |

Terminal: `GRAPHROLE_SHUFFLE_CONTROL_NOT_BEATEN`. Full continuation gate: False; 0.700 target: False.

D2 is the frozen D0 configuration in the user prompt, not original unconditioned A0. Means weight 65 development patient-fold appearances equally, 47 unique IDs and 6,273 channel appearances. No outer TEST evaluation.

## Paired unique-ID cluster bootstrap

| Contrast | Macro-F1 delta | 95% percentile interval | Positive folds |
|---|---:|---|---:|
| G1-D2 | +0.001262 | [-0.015543, +0.020438] | 2/5 |
| G2-D2 | +0.005272 | [-0.005372, +0.017204] | 4/5 |
| G3-D2 | -0.002779 | [-0.013909, +0.008550] | 1/5 |
| G1-G2 | -0.004009 | [-0.019826, +0.013470] | 2/5 |
| G1-G3 | +0.004042 | [-0.013202, +0.022161] | 2/5 |

10,000 draws, seed42; all repeated appearances retained inside unique-ID clusters. Thresholds remain fixed within draws. All11 metric contrasts are in PAIRED_BOOTSTRAP.csv.

## Required answers

1. Original D2 replay passes all bank/checkpoint/hash and metric gates. D2 was not retrained. Fresh float32 replay logit differences are at most 9.54e-7; after that numerical parity gate, evaluation exactly reuses the original serialized frozen scores. All 11 aggregate metrics reproduce exactly, not a claim of bitwise identical fresh CUDA logits.
2. All80 patients,256 runs and7635 canonical channel identities are retained with a strict one-to-one join.
3. Cached adjacency is nonzero, finite and symmetric but measured-window construction provenance cannot be confirmed: full raw reconstruction parity fails. It is not claimed to be a fabricated zero placeholder.
4. Uniform RAW_REBUILT uses original abs Pearson .70 quantile/.10 floor, measured two-second windows and actual250Hz metadata. All15074 graphs complete; no cached/rebuilt mixing.
5. Structurally problematic dimensions: []. Missing temporal/rank quantities remain missing until FIT-only imputation; see quality and missingness audits.
6. No graph descriptor has strong linear overlap (absolute Pearson >=0.80) with any of the 88 existing inputs in any FIT fold. The strongest observed relationship is PageRank window mean versus absolute spectral-entropy seizure mean, r=-0.422209 in fold1. Degree mean has r=-0.404386 with the same descriptor. This rules out strong marginal linear redundancy, not nonlinear redundancy. No descriptor is selected or removed.
7. G1-D2 Macro-F1 delta +0.001262; interval above.
8. G1-D2 EZ-F1 delta +0.007817.
9. G1-D2 EZ-AP delta +0.011989.
10. G1-D2 EZ-AUROC/MRR/Top1 deltas +0.029155/+0.042231/+0.061538. Ranking movement alone is not improvement.
11. G1-G2 Macro-F1 delta -0.004009; interval above. G2 is the stronger correspondence control.
12. G1-G3 Macro-F1 delta +0.004042; interval above.
13. Correct channel-specific graph benefit is not established by the full predeclared gate.
14. Across the 6,273 repeated development channel appearances, G1 gains 99 EZ true positives but loses 75 (net +24); it adds 215 false positives and removes 129 (net +86). It corrects 228 D2 errors but spoils 290 correct predictions (net 62 additional errors). Patient-equal sensitivity increases by 0.030372 while specificity declines by 0.016290. Recovering EZ channels does not by itself establish a better operating point. These counts include repeated appearances and are not independent channel observations.
15. Positive folds: G1-D2 2/5; G1-G2 2/5.
16. Benefits are not consistent across sources. G1-D2 Macro-F1 deltas: HUP -0.009463, LZU +0.007440, Multicenter +0.043276, Pediatric -0.019431. G1-G2 deltas: HUP +0.001183, LZU -0.015710, Multicenter +0.019399, Pediatric -0.023931. The largest apparent gain is concentrated in Multicenter. No center-specific thresholds or new source mechanism were fitted.
17. G1 measurably uses graph input under the locked numerical criterion: True. Neutralization mean absolute probability change 0.065080; changed decision fraction 0.064549; Macro-F1 delta +0.011613. Large input weights do not prove improvement.
18. Exact source metric parity and deterministic replay tests passed; solver failures do not silently become zeros. Of 15,074 measured graphs, 12,248 are disconnected. When the largest eigenvalue has a unique dominant component, the nonnegative principal eigenvector is computed deterministically; tied dominant eigenspaces would be explicitly missing (none occurred here). This avoids the original exception-to-zero fallback. Unlabeled patient normalization and FIT-only missing-value imputation are retained.
19. Larger-than-retraining/feature-capacity gain is not established; G2/G3 and intervals prevent relying only on G1-D2.
20. +0.015 G1-D2 criterion: False.
21. G1>=0.700: False.
22. Dynamic GNN follow-up is not justified by this registered experiment. No GNN is implemented here.
23. Stop this GraphRole seed42 approach without changing graph definitions or tuning to these outcomes.

## Limitations and control interpretation

G3 has neutral zero graph inputs, but LayerNorm104 can center these positions to nonzero values; its graph-column weights can receive gradients. The prompt statement that zeros cannot activate first-layer weights is not true for this architecture. The requested architecture was retained; G3 is a neutral-input/retraining control, not a complete feature-capacity test. G2 remains primary for correspondence.
G1_GRAPH_ZERO is an inference-only intervention at the selected G1 threshold, not retrained G3 or frozen D2. Graph descriptors use unlabeled within-patient context; no population scaler is fit on VAL. The original80 cohort is not reduced for missing rank stability.
Undirected correlation measures functional coupling, not directional causal spread. Clinical onset independently verified in0/256 records. Development patients have been reused in prior studies, and model/threshold selection reuses VAL; bootstrap is descriptive uncertainty, not independent clinical confirmation.
Private waveforms, graph matrices/vectors, IDs, labels, prediction ledgers, checkpoints and logs remain on the original server.

## Execution verification

Protocol SHA256 `1421ff533e6b951a288fc15e1da46add5f56c50f0f5e443881593b6b39af122b`. All15 formal runs completed, 196 epochs; epoch-body time 29.80s, excluding preparation/tests/reporting. Initial states are identical among G1/G2/G3 within each fold. All required engineering checks passed before formal training.
