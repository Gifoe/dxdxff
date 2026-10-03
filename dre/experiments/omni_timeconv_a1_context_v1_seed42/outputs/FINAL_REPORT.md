# Frozen TimeConv R4 + A1 patient-level context on Omni-iEEG

**Terminal: `CONTEXT_GAIN_NOT_ISOLATED_FROM_LOCAL_CAPACITY`.** This is one exploratory/repeated official TEST pass: Omni outcomes were historically viewed. TimeConv was never retrained or loaded into the optimizer; C1/V1 were selected entirely with TRAIN-only patient CV before the final head hashes were frozen.

| Model | TEST pooled AUROC | AP | Macro-F1 @0.5 | Patient-equal AUROC | Patient AP | MRR | Top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 frozen TimeConv | 0.798768 | 0.330301 | 0.659754 | 0.817186 | 0.748585 | 0.831977 | 0.772727 |
| C1 local residual | 0.798752 | 0.330271 | 0.659509 | 0.817164 | 0.748581 | 0.831977 | 0.772727 |
| V1 A1-context residual | 0.804692 | 0.340926 | 0.657703 | 0.825038 | 0.742935 | 0.813032 | 0.738636 |

## Direct answers

1. Frozen TimeConv replayed `0.7987677712` on 8,104 labeled EDF-channel units: **yes**.
2. Zero-init identity passed: maximum absolute score difference `0` (<1e-7).
3. Context coverage is in `CONTEXT_CHANNEL_COVERAGE_AUDIT.csv`; it counts all cached good channels, labelled and `-1` context-only, separately.
4. Yes: context uses every cached observable good channel before any label filter; `-1` channels have no BCE term.
5. C0 TRAIN-CV pooled AUROC: `0.952520`.
6. C1 TRAIN-CV pooled AUROC: `0.952992`.
7. V1 TRAIN-CV pooled AUROC: `0.955166`.
8. V1 nonnegative versus C0 folds: `5/5`.
9. V1 TRAIN-CV versus C1: `+0.002175`.
10. Final V1 alpha: `0.134174`.
11. Delta distribution is recorded in `ALPHA_DELTA_AUDIT.csv`; it is not called collapsed unless its observed spread is near zero.
12. Context-zero, cross-patient-context, and no-attention interventions are reported without retraining in `PATIENT_CONTEXT_INTERVENTIONS.csv`.
13. C0 official TEST AUROC: `0.798768`.
14. C1 official TEST AUROC: `0.798752`.
15. V1 official TEST AUROC: `0.804692`.
16. V1−C0 `+0.005924`, patient-cluster bootstrap 95% CI [-0.003066, +0.016994]. V1−C1 `+0.005940`, CI [-0.003028, +0.016997].
17. V1 exceeds published 0.8061: `False`.
18. V1 reaches 0.815 / 0.820 / 0.830: `False` / `False` / `False`.
19. Center changes are listed in `TEST_CENTER_METRICS.csv`; Zurich remains not-estimable where it has one label class.
20. The evidence supports the claimed transfer **only** if the terminal states `A1_CONTEXT_TRANSFER_SUPPORTED`; otherwise this V1 does not justify that claim.

## Scope distinction

This is not A1-interictal v1/v2 (new 36-D handcrafted descriptors), A1-TF fusion, PC-CNN raw re-encoding, or PR-Residual's small per-EDF relative head. V1 uses the immutable TimeConv R4 anchor, mean-EDF patient-channel anchors, exact A1 patient-relative z and two-head channel attention, then adds one shared residual correction back to the original EDF-channel TimeConv evidence.

The residual-head development split is patient-disjoint, but the frozen TimeConv encoder had already been trained on the official TRAIN population. Therefore this is not an independently held-out backbone-validation estimate. No test result changed architecture, hyperparameters, epoch rule, checkpoint, threshold, or intervention.
