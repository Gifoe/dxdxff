# Task 1 CANE-PATH-CP Sensitivity80 Report

## Analysis Status

POST-HOC 80-PATIENT SENSITIVITY ANALYSIS. NOT THE FROZEN PRIMARY OLD-90 COHORT.

No full nested experiment has been run in this code-delivery stage, so this report contains no claimed model result. Formal execution replaces this skeleton with an observed, non-fabricated report generated from the fixed P2 ensemble.

## Cohort

The exclusion manifest contains ten unconfirmed post-hoc review cases. The retained center counts are HUP 36, LZU 21, multicenter 15, and pediatric 8.

## Method And Leakage Controls

The implementation uses NEZ=1/EZ=0, three bounded residuals, label-free offline ridge-VAR proxies, four-fold outer-train cross-fitting, a patient-adaptive threshold head, and equal averaging of fixed model seeds 42/43/44. Outer-test labels are unavailable to model selection, PATH training, profile selection, and seed weighting.

## Limitations

This post-hoc cohort is not a primary analysis. Observed labels are imperfect. Ridge-VAR is a directed-influence proxy rather than validated causal identification. No metric threshold, including macro-F1 0.70, is guaranteed.
