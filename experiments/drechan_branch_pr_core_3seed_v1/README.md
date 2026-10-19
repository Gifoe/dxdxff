# Same-candidate core-module audit, three seeds

**Completed and independently verified:75fresh core fits/916epochs plus15exact parentFULL reuses.** All90selections replayed/frozen and all90score cells sealed before metrics;720patient-fold-seed rows,6480endpoint values,108nominal/eight additional intervals passed. Fixed qualification FAIL; no newOuter or main-model replacement. See [actual findings and all nine endpoints](FINDINGS.md), [full-precision results](results/MAIN_RESULTS.csv), [paired intervals](results/INTERVALS.csv) and [execution audit](RUN_AUDIT.md). This is part of the active six-feature DREChannNet goal, not a different easier model. Negative findings are retained.

The parent PR/context study has now completed with qualification FAIL. This additional core diagnostic was frozen/launched before those outcomes and continues the complete matrix; it does not cancel that failure, promote the parent or authorizeOuter. A possible future simplification requires a separately frozen comparison, not selecting an Outer winner.

| Arm | Change | Parameters |
| --- | --- | ---: |
| FULL | exact parent candidate;15 selected cells/scores reused after verification | 11553 |
| WO_EARLY_FILM | unconditioned copy of same CNN stem/pool | 10505 |
| WO_LATE_GATE | all1lateGate, remove its parameters | 10505 |
| WO_BOTH_GUIDANCE | only both guidance operations removed; direct physiology remains | 9457 |
| WO_RAW_PRE_REFERENCE | skip only raw latent pre-reference subtraction | 11553 |
| EARLY_FILM_MISMATCH | fixed within-patient/seizure/identical-support donor for EarlyFiLM only | 11553 |

Both PR operations, new branch-relative residuals, six-feature masks, full pooling, one64-hidden head and all source/data/optimizer/threshold settings remain unchanged. All75new fits use matching untrained initializers across seeds42/43/44 and original five within-FIT TRAIN/CAL/ASSESS paths. No originalOuter or Development selection. Neither0.62-style internal results nor this audit replace the historical0.653828Outer point.

Train mode can run independently while the parent study continues, only after numerical/resource admission and with a separate D-drive namespace. It trains75new controls, not duplicates of the parent's fits. Finish mode requires the parent all-model freeze, sealed predictions and independent verification; only then join15exact FULL references, replay/freeze90selected cells and seal all new scores before metrics. Keep each foreground SSH owner alive; detached children can be reaped at SSH exit on this server.

Commands after source push:

`run_all.ps1 -Mode Train -SourceCommit <verified40hex> -RunTag <fresh>`

`run_all.ps1 -Mode Finish -SourceCommit <same source40hex> -RunTag <new fresh>`

Report all nine endpoints,108nominal/8additional paired intervals, guidance interaction, parameter counts, mismatch coverage and seed/fold/source/pooled summaries. Positive points are not statistical proof. Do not infer information-only effects from module deletions that also change capacity. The capacity-matched mismatch tests early-conditioning specificity but is not a winner to promote.

Actual findings: EarlyFiLM and correct-donor specificity fail the joint AUROC/Macro point rule. LateGate and raw pre-reference have positive points in all3seeds, but both-metric intervals still do not simultaneously exclude zero. Pre-reference AUROC has a positive additional-family interval; Macro-F1 does not. Removing both guidance operations has higher AUROC/Macro points than FULL but lower AP, without an established paired advantage. This is evidence for a separately frozen simplification study, not permission to pick the best ablation after viewingOuter. The unsupported128-gain residual parent also remains unqualified.

`python verify_public.py` checks the anonymous release, bound scientific sources, aggregate arithmetic, controls and fixed gate without loading private data. It does not independently recreate patient-level bootstrap draws; the separate server verifier does that and reports PASS.

No newOuter is run here. Final performance and retained-module claims require actual original-protocol final training/test and paired evidence, still unfinished. Only code/hashes/anonymous aggregate material is public; patient data, scores, weights, preprocessing and logs remain private.
