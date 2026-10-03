# Omni Cross-Center Ranking Decomposition Audit

**Terminal:** `CROSS_CENTER_MISALIGNMENT_NOT_ZERO_SHOT_ACTIONABLE`. This is a pure exploratory/repeated-test audit: the test set was historically viewed, but no model, score, threshold, representation, covariance procedure, or alignment parameter was selected from test labels.

| Method | Pooled AUROC | Within-center weighted AUC | Cross-center weighted AUC | Delta vs CNN |
|---|---:|---:|---:|---:|
| Frozen CNN | 0.798768 | 0.288887 | 0.509881 | +0.000000 |
| Source Fisher | 0.799422 | 0.296667 | 0.502755 | +0.000654 |
| CNN + UCS | 0.733622 | 0.288887 | 0.444735 | -0.065146 |
| Fisher + UCS | 0.763687 | 0.296667 | 0.467020 | -0.035081 |
| Fisher + CPA | 0.764648 | 0.296667 | 0.467981 | -0.034120 |
| Fisher + Gaussianization | 0.764648 | 0.296667 | 0.467981 | -0.034120 |

## Exact pair decomposition

| Positive center | Negative center | Pair weight | CNN AUC | Fisher AUC | Delta | Weighted Delta |
|---|---|---:|---:|---:|---:|---:|
| HUP | HUP | 0.011577 | 0.613096 | 0.714238 | +0.101141 | +0.001171 |
| HUP | Open-iEEG | 0.046797 | 0.435832 | 0.720609 | +0.284777 | +0.013327 |
| HUP | SourceSink | 0.003696 | 0.582713 | 0.744647 | +0.161934 | +0.000598 |
| HUP | Zurich | 0.034585 | 0.708050 | 0.719854 | +0.011804 | +0.000408 |
| Open-iEEG | HUP | 0.085342 | 0.894305 | 0.821313 | -0.072992 | -0.006229 |
| Open-iEEG | Open-iEEG | 0.344980 | 0.801224 | 0.819610 | +0.018386 | +0.006343 |
| Open-iEEG | SourceSink | 0.027243 | 0.901636 | 0.843522 | -0.058114 | -0.001583 |
| Open-iEEG | Zurich | 0.254951 | 0.912017 | 0.821303 | -0.090714 | -0.023128 |
| SourceSink | HUP | 0.022857 | 0.752273 | 0.748752 | -0.003522 | -0.000080 |
| SourceSink | Open-iEEG | 0.092395 | 0.613992 | 0.756072 | +0.142080 | +0.013127 |
| SourceSink | SourceSink | 0.007296 | 0.737793 | 0.774240 | +0.036447 | +0.000266 |
| SourceSink | Zurich | 0.068283 | 0.813026 | 0.760805 | -0.052222 | -0.003566 |

The pairwise weighted sums reproduce pooled AUROC to `<1e-10`. Source Fisher's weighted within-center change is `+0.007779674` and its weighted cross-center change is `-0.007125537`. The three largest negative cells are: Open-iEEG positive vs Zurich negative (-0.023128); Open-iEEG positive vs HUP negative (-0.006229); SourceSink positive vs Zurich negative (-0.003566).

## Direct answers

1. Source Fisher's pooled gain is `+0.000654137`. It is decomposed rather than inferred from center-average AUROCs.
2. Weighted within-center improvement: `+0.007779674`.
3. Weighted cross-center change: `-0.007125537`.
4. `Delta W_within > 0`: `True`.
5. Cross-center ranking cancels the within gain: `True`.
6. Negative pair cells are listed above in descending absolute contribution.
7. Zurich negative pairs account for `35.781828%` of all positive-negative ranking pairs.
8. The largest pathological location offsets are HUP versus Open-iEEG: CNN `-0.315509` and Fisher `-2.291846`. Normal Fisher medians also span HUP `-6.848270` to Zurich `-7.652201`; the full non-rounded table is emitted separately.
9. Scale mismatch is material: CNN normal Open-iEEG/Zurich IQR is `40.58x`, and Fisher pathological Open-iEEG/HUP IQR is `1.92x`.
10. Removing Zurich is diagnostic only: Fisher minus CNN becomes `+0.041950` on `5493` units, versus `+0.000654` on the official population.
11. TRAIN five-fold pseudo-target OOF results:
- Frozen CNN UCS: delta `-0.075284`, nonnegative folds `0/5`.
- Frozen CNN CPA: delta `-0.056312`, nonnegative folds `0/5`.
- Frozen CNN Gaussianization: delta `-0.056313`, nonnegative folds `0/5`.
- Source Fisher UCS: delta `-0.033920`, nonnegative folds `0/5`.
- Source Fisher CPA: delta `-0.036878`, nonnegative folds `0/5`.
- Source Fisher Gaussianization: delta `-0.036875`, nonnegative folds `0/5`.
12. The preregistered viability rule is `delta AUROC >= +0.003` and `>=4/5` nonnegative folds for a Source Fisher alignment. It passed: `False`.
13. Best Fisher alignment exploratory TEST AUROC: `0.764648`; exceeds 0.8061: `False`.
14. Every within-center alignment cell was asserted unchanged to `<1e-10`; only cross-center pair rankings can move.
15. Final classification: `CROSS_CENTER_MISALIGNMENT_NOT_ZERO_SHOT_ACTIONABLE`.

No center-specific affine adapter, calibration model, threshold, fusion, covariance correction, or neural model was developed. Stop after this audit.
