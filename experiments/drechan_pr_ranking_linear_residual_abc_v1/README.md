# Main-table-matched PR Linear Residual: A-C

This independent exploratory analysis replaces only the five TimeConv PR score cells with the completed, frozen-backbone PR Linear Residual from the main table. All 80 patients and 7,635 channels remain included. No model is trained. SVM, DREChanNet, TimeConv No PR and Score-Z remain unchanged.

The original end-to-end TimeConv A-D results remain in `../drechan_pr_ranking_robustness_v1/`. Its seizure-sufficiency experiment D does not become a residual experiment.

Execution: `python scripts/run.py --source-commit <frozen commit>` on the trusted server; inspect anonymous exports with `python scripts/release_check.py --package <archive>`.
