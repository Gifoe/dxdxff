| Layer | Dim | Domain AUROC | Center predictability | Median |SMD| | MMD | Train probe AUROC | Test probe AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Input summary | 453 | 0.453598 | 0.817252 | 0.029947 | 0.001667 | 0.672455 | 0.662558 |
| TimeConv-1 | 32 | 0.442107 | 0.777198 | 0.039547 | 0.001801 | 0.714604 | 0.733126 |
| TimeConv-2 | 64 | 0.428002 | 0.744846 | 0.054283 | 0.002007 | 0.737425 | 0.731642 |
| ResNet | 512 | 0.488142 | 0.889615 | 0.042458 | 0.003128 | 0.926675 | 0.722631 |
| 32D embedding | 32 | 0.425785 | 0.863675 | 0.031377 | 0.003459 | 0.955934 | 0.762401 |
| logit | 1 | 0.407024 | 0.698383 | 0.035557 | 0.003849 | 0.957602 | 0.791272 |

# Omni Representation Shift Audit

Baseline replay passed: TRAIN full-record AUROC `0.9586782931` and TEST AUROC `0.7987673466`. All representations came from the locked evaluation-mode forward pass.

## Direct answers

1. **No layer reaches even moderate TRAIN/TEST separability.** Domain AUROC is below the locked 0.65 boundary at every layer.
2. Domain AUROC does **not** increase with depth: R0=0.454, R1=0.442, R2=0.428, R3=0.488, R4=0.426, R5=0.407. Scores below 0.5 are reported with the locked TEST-positive orientation and are not post-hoc flipped; they show unstable cross-patient ranking, not useful domain discrimination.
3. By the primary domain-AUROC criterion, R3 is numerically largest at 0.488142, but this is still indistinguishable from useful separability. Other metrics disagree on the maximum: median |SMD| peaks at R2 (0.054283), while MMD peaks at R5 (0.003849). There is no defensible single "strongest-shift layer."
4. TRAIN center identity is most predictable at **R3 (ResNet)**, macro OVR AUROC 0.889615.
5. There is no strong pooled marginal shift to explain by center composition. Center fractions differ modestly (for example, Open-iEEG channel units 50.9%→45.2% and `Other` 32.8%→39.0%), while pooled domain AUROC remains below 0.5 at every layer. The encoder strongly records center identity, but composition imbalance does not turn into a stable TRAIN/TEST separator.
6. Within reliable named centers, TRAIN→TEST representation separability also remains low: the maximum AUROC over layers is 0.510 for HUP, 0.515 for Open-iEEG, and 0.472 for SourceSink. `Other` is an aggregate rather than one acquisition center and reaches only 0.577. Thus within-center **marginal** shift is not obvious, even though localization AUROC drops by 0.271 (HUP), 0.156 (Open-iEEG), and 0.097 (SourceSink).
7. R4 pathological-minus-normal centroid direction cosine is **0.990948**.
8. The R4 TRAIN-only linear probe falls from 0.955934 OOF AUROC to 0.762401 on TEST, a gap of 0.193533.
9. True classifier-input task-direction AUROCs are parallel=0.407024 and orthogonal=0.432057. Both are low and differ by only 0.025, so the data do not support either task-parallel or task-orthogonal nuisance concentration. This uses the real 16D `fc_out` input because no final 32D linear classifier exists.
10. The classifier task coordinate is not strongly shifted under the locked 0.80 threshold.
11. R4 center-specific label directions are highly consistent: TRAIN off-diagonal cosines are 0.986–0.996 and diagnostic-only TEST cosines are 0.937–0.999. This does not rescue the classifier: the TRAIN-only R4 linear-probe weight has only 0.234 cosine with the diagnostic TEST-probe weight, so higher-order/covariate effects remain.
12. Supported audit labels: **NO_SIMPLE_REPRESENTATION_SHIFT**.
13. The locked mapping selects **stop normalization adapter route** as the next research direction. This audit does not implement it.

## Composition, performance, and limits

The pooled TRAIN-to-TEST AUROC loss is 0.159911. Only three named centers have estimable AUROC in both splits, so the shift/performance Spearman correlation is not estimated. The evidence is consistent with conditional shift, label/prevalence differences, or hidden patient/acquisition confounding; it is not evidence for a correctable global mean/covariance offset.

The prompt's 32D classifier-direction premise is false for this checkpoint. R4 is followed by a nonlinear 32→32→16→1 head, so projecting R4 with an invented 32D weight would be invalid. `TASK_DIRECTION_SHIFT.csv` uses the actual 16D final linear weight and labels it `P16_TRUE_CLASSIFIER_INPUT`.

PCA figures are explanatory only. TEST-label probes are marked `DIAGNOSTIC_ONLY_NOT_DEPLOYABLE`. No adapter, normalization, CORAL, whitening, fine-tuning, or test-time adaptation was performed.
