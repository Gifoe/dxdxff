# No Early FiLM: seed52/62 extension

Only the exact existing six-feature no-Early/no-Late DREChannNet is trained at seeds52 and62 (five original folds each). Variance and Head Interaction remain absent. Direct Feature PR, final Raw PR and own pre-reference remain enabled. No model widening, attention or mismatch experiments.

Ten fresh initializations use the unchanged original full initializer before module deletion; no pretrained checkpoint initialization. All ten selections freeze before Outer, and all ten scores seal before reporting. Original80patients/7635channels/256seizures/1471965validRaw250windows and legal internal selection remain fixed. See [pre-fit lock](PROTOCOL_LOCK.json).

Run on the trusted server with `run_all.ps1 -StopAfter Admission`, then only after source+admission are pushed, run `run_all.ps1 -SourceCommit <verified40hex> -StopAfter All` under a sustained foreground SSH owner. New RunTag required for a diagnosed engineering resume. No parent sources are modified.

Completed: ten fresh models,122actual epochs, all five folds and80Outer patients per seed. Independent reconstruction passed160patient rows,1440metric values and27nominal intervals. All13admitted scientific source hashes remain unchanged. No worker failure, restart or completed epoch discard occurred.

| Seed | Patient Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac. bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 52 | 0.633450 | 0.762518 | 0.561388 | 0.428970 | 0.837930 | 0.760888 | 0.753653 | 0.637500 | -0.046641 |
| 62 | 0.645675 | 0.773726 | 0.580476 | 0.458311 | 0.833040 | 0.757673 | 0.805291 | 0.725000 | -0.017085 |
| Metric mean | 0.639563 | 0.768122 | 0.570932 | 0.443640 | 0.835485 | 0.759280 | 0.779472 | 0.681250 | -0.031863 |
| Seed sample SD | 0.008645 | 0.007925 | 0.013497 | 0.020747 | 0.003458 | 0.002273 | 0.036514 | 0.061872 | 0.020899 |

[Full report](public/REPORT.md), [findings and historical context](FINDINGS.md), [per-seed results](public/SEED_RESULTS.csv), [all folds](public/FOLD_RESULTS.csv), [fold SD](public/FOLD_SD.csv), [sources](public/SOURCE_RESULTS.csv), [per-seed sources](public/SOURCE_SEED_RESULTS.csv), [pooled separately](public/POOLED_RESULTS.csv), [27intervals](public/INTERVALS.csv), [coverage](public/RAW_COVERAGE.csv), [verification](public/INDEPENDENT_OUTER_VERIFICATION.json), [execution audit](RUN_AUDIT.md).

The mean averages patient metrics across the two requested seeds, not prediction probabilities. No best seed is selected. Existing42/43/44 results are historical, not newly rerun or merged into the primary estimate. These new runs have no seed-matched Full control and cannot alone quantify Early FiLM's contribution.

All results remain exploratory repeated-test on historically viewed Outer patients. Publish code/hash proofs/anonymous aggregates only; raw signals, identities, weights, preprocessors, per-patient predictions, caches and logs stay private on serverD.
