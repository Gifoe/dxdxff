# Omni class-conditional covariance audit

**Status:** `COVARIANCE_SHIFT_NOT_ZERO_SHOT_ACTIONABLE`. This is exploratory/repeated-test evidence because the Omni test outcome was historically viewed; no test result was used to change a model, covariance formula, representation layer, or shrinkage.

| Method | Target labels used to construct? | Target unlabeled stats used? | AUROC | AP | ΔAUROC vs CNN |
|---|---|---|---:|---:|---:|
| Frozen TimeConv-CNN | NO | NO | 0.798768 | 0.330301 | +0.000000 |
| Source Fisher | NO | NO | 0.799422 | 0.314087 | +0.000654 |
| Mean correction | NO | YES | 0.799422 | 0.314087 | +0.000654 |
| UTC diagonal | NO | YES | 0.801970 | 0.325843 | +0.003202 |
| UTC full covariance | NO | YES | 0.786162 | 0.299096 | -0.012605 |
| CORAL diagnostic | NO | YES | 0.800895 | 0.319705 | +0.002127 |
| UTC Mahalanobis | NO | YES | 0.786162 | 0.299096 | -0.012605 |
| Target Fisher diagnostic | YES_DIAGNOSTIC_ONLY | NO | 0.805698 | 0.343707 | +0.006931 |

| Representation | cos(mean direction) | covariance shift normal | covariance shift pathological | cos(Fisher direction) |
|---|---:|---:|---:|---:|
| R4_32D | 0.992003 | 0.672613 | 0.180774 | 0.278152 |
| P16_16D | 0.987300 | 0.574715 | 0.281732 | 0.752650 |

## Direct answers

1. The frozen CNN replay passed: TRAIN full AUROC `0.9586782931` and TEST AUROC `0.7987677712`.
2. R4 mean-direction cosine is `0.992003`; P16 is `0.987300`.
3. R4 normal/pathological covariance shifts are `0.672613` / `0.180774`.
4. R4 source-versus-target Fisher cosine is `0.278152`.
5. Target Fisher is diagnostic only. UTC uses source class means and target **unlabeled** covariance, not target pathology labels.
6. The primary UTC-CNN patient-cluster bootstrap ΔAUROC is `-0.012605` (95% CI `-0.050473`, `+0.031241`; Pr(Δ>0) `0.2987`).
7. The historical target logistic-probe vector was not available as a frozen artifact; it was not retrained, so the requested probe-direction comparison is explicitly unavailable rather than fabricated.
8. The sample-size table separates held-out-patient estimates from the all-target non-held-out reference.

No covariance adapter was developed. Stop after this audit.
