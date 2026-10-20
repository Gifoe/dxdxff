# Findings: no Early FiLM, seeds52/62

The exact existing no-Early/no-Late architecture was freshly trained at the two requested seeds across all five original patient-held-out folds. Variance and Head Interaction remain absent; direct Feature PR, final Raw PR and own pre-reference remain active. The output is not Raw-only or a PR deletion. Parameters remain9329, head64, original pooling and all training/validation rules unchanged.

## Actual new results

| Metric | Seed52 | Seed62 | Two-seed mean | Seed sample SD |
| --- | ---: | ---: | ---: | ---: |
| Patient Macro-F1 | 0.633450 | 0.645675 | 0.639563 | 0.008645 |
| Patient AUROC | 0.762518 | 0.773726 | 0.768122 | 0.007925 |
| Patient AP | 0.561388 | 0.580476 | 0.570932 | 0.013497 |
| EZ-F1 | 0.428970 | 0.458311 | 0.443640 | 0.020747 |
| NEZ-F1 | 0.837930 | 0.833040 | 0.835485 | 0.003458 |
| ACC | 0.760888 | 0.757673 | 0.759280 | 0.002273 |
| MRR | 0.753653 | 0.805291 | 0.779472 | 0.036514 |
| Top-1 | 0.637500 | 0.725000 | 0.681250 | 0.061872 |
| EZ-frac. bias | -0.046641 | -0.017085 | -0.031863 | 0.020899 |

The two-seed average's nominal patient-bootstrap95%interval is[0.611955,0.667838]for Macro-F1 and[0.733257,0.803145]for AUROC. Resampling uses10000seed42draws of the same80unique patients after within-patient averaging of seed metrics. It does not estimate a population of training seeds, and two seeds are insufficient to establish broad seed robustness. All27intervals are in public/INTERVALS.csv.

The seed-to-seed Macro-F1 spread is1.22percentage points; the AUROC spread is1.12points. There is no license to select seed62 merely because its point estimates are larger. Neither the direct comparison of two seeds nor an unmatched historical Full score isolates the causal effect of removing FiLM.

## Historical references, not new runs

These are the already completed old three-seed WO_BOTH_GUIDANCE cells, which are graph-equivalent to current WO_EARLY_FILM because the current main already omits Late Gate. They are not old WO_EARLY_FILM cells that still retained Late Gate.

| Historical seed | Macro-F1 | AUROC | AP | EZ-F1 |
| --- | ---: | ---: | ---: | ---: |
| 42 | 0.644071 | 0.766014 | 0.568578 | 0.448547 |
| 43 | 0.633324 | 0.756958 | 0.554850 | 0.435442 |
| 44 | 0.641568 | 0.760241 | 0.563333 | 0.448053 |

Source: experiments/drechan_f0_complete_ablation_3seed_v1/public/SEED_RESULTS.csv at the immutable parent tree. These three historical seeds are not merged into the primary new-two-seed estimate. This extension does not train matched Full models at52/62, so it cannot determine a new-seed Full-minus-noEarly effect or justify replacing the fixed main. Existing matched42/43/44 comparisons remain the relevant earlier evidence.

## Evidence and limits

All ten new selections froze before any new Outer inference and all predictions sealed before metric reporting. Independent verification reconstructed160patient-seed rows,1440metrics, all tables/coverage and27intervals. Per-fold results/sample SD, pooled-channel metrics and cross-source metrics are separate from equal-patient results. Centers with fewer than5patients have suppressed metrics; no patient or window was removed to improve outcomes.

All results are exploratory repeated-test on the historically viewed Outer cohort. Fresh seeds do not restore independence, and the findings do not imply physiological modulation is necessary or useless under every architecture. No post-Outer model, threshold, seed, feature or budget selection was made.
