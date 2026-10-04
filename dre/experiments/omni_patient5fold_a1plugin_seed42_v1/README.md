# Omni patient-level 5-fold A1-context audit

This experiment compares a TimeConv interictal baseline against the matched
A1-style patient-context residual under a frozen patient-level five-fold OOF
protocol (seed 42).  The primary result is **insufficient plugin gain**:
patient-equal AUROC is 0.7622 for the baseline and 0.7630 for the plugin;
paired delta +0.00110, 95% patient-bootstrap CI [-0.02935, +0.03149].

See [FINAL_REPORT.md](FINAL_REPORT.md) and the non-identifying tables in
`results/`.  Raw recordings, cached data, checkpoints, patient/channel
predictions, patient metrics, runtime logs, and secrets are deliberately
excluded.
