# SCM-EZ final report

SCM-EZ failed the locked Ictal development gate by a large margin. See [ICTAL_REPORT.md](ICTAL_REPORT.md) for the full fold, diagnostic, bootstrap, leakage and runtime evidence.

| Model | AUROC | AP | Macro-F1 | MRR | Top1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical A1 | 0.746382 | 0.576743 | 0.620810 | 0.740038 | 0.654771 |
| SCM-EZ Full | 0.497126 | 0.339976 | 0.438187 | 0.462141 | 0.362296 |

SCM-EZ exceeds A1 in only 1/5 folds. Patient-reference shuffle changes AUROC by only +0.000084, while temporal-state permutation improves AUROC by +0.019711. The proposed patient-relative reference is unused and the true cross-state ordering is harmful in this implementation/data combination.

No Omni file exists because the gate did not authorize Omni access.

**Terminal: `STOP_SCM_ICTAL_GATE_FAILED`.**
