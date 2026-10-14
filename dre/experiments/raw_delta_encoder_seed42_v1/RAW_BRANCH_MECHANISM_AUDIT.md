# Raw branch mechanism

All D2/D3 folds completed; no unfavorable fold was omitted. Initial full states matched and each arm used the same frozen D1 checkpoint. All49,990 eligible record-channel-stage permutations changed order while retaining the stage multiset, reference, valid support, identity and destination time. Moving complete embeddings is equivalent to moving complete windows through this independent shared CNN; within-window samples are unchanged.

Selected D2 effective coefficients, folds1–5: 0.063576, 0.001839, 0.003495, 0.202459, 0.196288. Mean absolute final-logit residuals: 0.035337, 0.000620, 0.000993, 0.138900, 0.133877. Folds2–3 are nearly inactive at the decision level: disabling the residual changes no binary prediction. Nonzero gamma alone does not prove useful information.

Every intervention retains the selected **D2** threshold:

| Intervention | AUROC | AP | Macro-F1 | EZ-F1 | MRR |
|---|---:|---:|---:|---:|---:|
| D2 full | 0.732515 | 0.515027 | 0.650238 | 0.451028 | 0.684968 |
| gamma=0 | 0.725978 | 0.503601 | 0.642297 | 0.439911 | 0.658502 |
| Stage-local shuffle | 0.732724 | 0.516315 | 0.648890 | 0.449246 | 0.685936 |
| Absolute latent, no subtraction | 0.727811 | 0.502535 | 0.641246 | 0.436943 | 0.661887 |

Gamma-zero logits exactly equal the embedded D1 logits; maximum probability drift versus the separate frozen D1 replay is1.192093e-7, with identical decisions at D1's original threshold. Its Macro-F1 differs from D1's0.644546 because the intervention keeps D2's threshold, not D1's independently selected threshold. This is not broken D1 identity.

Across65 Development appearances (6,273 channel appearances, not unique cohort channels), full versus gamma-zero corrected66/spoiled25 binary decisions: TP526→539, FP728→700, TN4341→4369, FN678→665. Both EZ detection and false-positive reduction contribute. There are13,970 strict channel-pair order reversals out of352,239 pairs. Full versus shuffled inference corrected11/spoiled5 decisions and caused4,453 reversals; versus absolute inference corrected45/spoiled12 and caused14,875 reversals. Individual records remain private.

The auxiliary Raw head alone, at diagnostic threshold0.5, has patient AUROC0.674862/AP0.468361/Macro-F1 0.535197; it is not a competitive independent Raw model. `RESIDUAL_DISTRIBUTION_SHIFT.csv` reports actual FIT/Development raw-logit distributions and residual magnitudes, without retuning.

Separately trained D2−D3 AUROC is+0.004539, exploratory interval[0.000473,0.008685], but fails the locked+0.005 minimum. D2 exceeds D3 AUROC in only three folds; most of the difference is in fold5, with very different selected epochs/gates. Inference shuffling slightly increases D2 AUROC/AP. This is weak, qualified evidence, not robust identification of chronological-order utility. Pre-subtraction helps the frozen trained model, but the absolute intervention is not a retrained absolute-Raw baseline.
