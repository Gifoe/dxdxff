# Main-table-matched PR Linear Residual: A-C

This independent exploratory analysis replaces only the five TimeConv PR score cells with the completed, frozen-backbone PR Linear Residual from the main table. All 80 patients and 7,635 channels remain included. No model is trained. SVM, DREChanNet, TimeConv No PR and Score-Z remain unchanged.

The original end-to-end TimeConv A-D results remain in `../drechan_pr_ranking_robustness_v1/`. Its seizure-sufficiency experiment D does not become a residual experiment.

Execution: `python scripts/run.py --source-commit <frozen commit>` on the trusted server; inspect anonymous exports with `python scripts/release_check.py --package <archive>`.

Completed 2026-10-29: `ABC_PR_LINEAR_RESIDUAL_COMPLETE_VERIFIED`. All 45 cells,
720 patient-condition endpoint rows and 4,320 budget rows passed verification;
1,203 intervals were reconstructed independently. All 80 patient fixed-head
replays matched their sealed probabilities exactly (five fold-specific
heads, evaluated on 80 patients). Forty unchanged cells matched 640 historical
patient rows. No models were fitted.

- [Chinese results and interpretation](report.md)
- [Protocol and source pins](PROTOCOL_LOCK.json)
- [Anonymous complete tables](results/public/)
- [Verification audit](logs/SANITY_CHECKS.md)
- [Patient-gain figure](figures/Figure_A_patient_gain_distributions.pdf)
- [EZ coverage figure](figures/Figure_B_EZ_coverage.pdf)

TimeConv PR denotes **PR Linear Residual** in every public result. SVM and
DREChanNet PR retain their original Full PR implementations. Source grouping
uses patient-held-out outer predictions and is not a leave-one-source-out test.
