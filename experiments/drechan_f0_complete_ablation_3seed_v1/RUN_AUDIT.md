# Execution audit (in progress; not an outcome report)

- Scientific source pushed and remote-verified before any new fit: `86039bce7ad5e8f0990d39061667b5cae12b7f1f`.
- Real-server admission proof pushed and remote-verified before any new fit: `02bfef51cf2ae630898509e43f9a613aa7a3c6b3`.
- Admission:11 executable tests passed;285 matched initialization cells checked;45 current seed42 cells and5 historical Raw-only cells passed all strict reuse gates.235 fresh fits,285 logical cells.
- All3 seeds matched the original unchanged no-PR forward/loss/all gradients/clipped Adam/RNG update.19 complete-patient paths passed two discarded admission updates each. These discarded checks are not formal fits or performance estimates.
- Foreground-owned sequential production started at2026-10-20T22:25:27+08:00, RunTag `production01`. No Raw copying, subsampling, reduced patient count, reduced budget, simultaneous model training or changes to frozen scientific sources.
- Training is in progress. All285 selections must freeze before unfinished Outer inference; all285 score cells must seal before reporting. No new three-seed metrics are asserted at this stage.
- Completion criteria:19groups ×3seeds ×80patients,4560 patient/seed rows,41040 metric values,351 nominal and36 additional-family paired interval checks. Final anonymous aggregate archive is accepted only after independent verification and release hash/schema checks.
- All conclusions remain exploratory repeated-test. No ablation is required to deteriorate; adverse results are retained and do not trigger changes to the fixed Full or its thresholds.
- Release-only `release_check.py` does not participate in training, preprocessing, scoring or source bindings; it checks the17 immutable scientific-file hashes, executable tests and the final explicit anonymous-export schema.
