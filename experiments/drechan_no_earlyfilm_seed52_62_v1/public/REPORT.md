# No Early FiLM and no Late Gate: seeds52 and62

Ten freshly initialized fold models on the original80patients/7635channels/256seizures/1471965validRaw250windows. Same five patient-held-out folds, corrected direct FeaturePR, final RawPR, own pre-ictal latent reference, original temporal/seizure pooling and head64. Variance and Head Interaction remain absent;9329parameters. No attention, widened head, mismatch control, new data or Full model at these two seeds.

All10models,preprocessors,epochs and validation-only thresholds froze before any new Outer inference; all10score cells sealed before metric reporting. No performance gate, budget or threshold tuning. This is exploratory repeated-test, not independent confirmation. Neither seed is selected as best.

| Seed/estimate | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac. bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 52 | 0.633450 | 0.762518 | 0.561388 | 0.428970 | 0.837930 | 0.760888 | 0.753653 | 0.637500 | -0.046641 |
| 62 | 0.645675 | 0.773726 | 0.580476 | 0.458311 | 0.833040 | 0.757673 | 0.805291 | 0.725000 | -0.017085 |
| Two-seed metric mean | 0.639563 | 0.768122 | 0.570932 | 0.443640 | 0.835485 | 0.759280 | 0.779472 | 0.681250 | -0.031863 |
| Seed sample SD (n=2) | 0.008645 | 0.007925 | 0.013497 | 0.020747 | 0.003458 | 0.002273 | 0.036514 | 0.061872 | 0.020899 |

Primary endpoints are equal-patient averages, not pooled-channel metrics, a score ensemble or the best fold/seed. EZ-fraction bias is predicted minus observed EZ fraction; closer to zero is desirable.

[Ten folds](FOLD_RESULTS.csv), [fold sample SD](FOLD_SD.csv), [sources](SOURCE_RESULTS.csv), [per-seed sources](SOURCE_SEED_RESULTS.csv), [pooled metrics separately](POOLED_RESULTS.csv), [coverage](RAW_COVERAGE.csv). Centers with fewer than5patients have suppressed metrics.

The27nominal95%patient-bootstrap intervals cover9endpoints for each seed and for the per-patient two-seed metric average. Resampling uses10000seed42draws of the same80unique patients, not160independent seed-patient rows. No superiority testing or multiplicity-adjusted comparison family is defined. Two seeds provide limited information about seed variability.

No matched Full models were run at52/62; these results alone cannot estimate a seed-matched Early FiLM contribution. Existing42/43/44 results are historical references and are not merged into the primary two-seed estimate.

Independent reconstruction passed160patient rows,1440metric values and27intervals; maximum interval difference5.55e-16. All scientific sources are hash-bound to pre-fit admission. Signals,identities,individual scores,weights,preprocessors,caches and runtime logs remain private.
