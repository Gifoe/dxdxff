# No Early FiLM: seed52/62 extension

Only the exact existing six-feature no-Early/no-Late DREChannNet is trained at seeds52 and62 (five original folds each). Variance and Head Interaction remain absent. Direct Feature PR, final Raw PR and own pre-reference remain enabled. No model widening, attention or mismatch experiments.

Ten fresh initializations use the unchanged original full initializer before module deletion; no pretrained checkpoint initialization. All ten selections freeze before Outer, and all ten scores seal before reporting. Original80patients/7635channels/256seizures/1471965validRaw250windows and legal internal selection remain fixed. See [pre-fit lock](PROTOCOL_LOCK.json).

Run on the trusted server with `run_all.ps1 -StopAfter Admission`, then only after source+admission are pushed, run `run_all.ps1 -SourceCommit <verified40hex> -StopAfter All` under a sustained foreground SSH owner. New RunTag required for a diagnosed engineering resume. No parent sources are modified.

Outputs pending actual execution. Every new seed and the two-seed metric mean/sample SD will be reported; no best-seed selection. Existing42/43/44 results are historical, not newly rerun or merged into the primary estimate. These new runs have no seed-matched Full control and cannot alone quantify Early FiLM's contribution.

All results remain exploratory repeated-test on historically viewed Outer patients. Publish code/hash proofs/anonymous aggregates only; raw signals, identities, weights, preprocessors, per-patient predictions, caches and logs stay private on serverD.
