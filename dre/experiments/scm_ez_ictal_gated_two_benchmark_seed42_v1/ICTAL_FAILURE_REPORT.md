# Ictal failure report

SCM-EZ produced AUROC 0.497126 and AP 0.339976 versus A1 0.746382 and 0.576743. All five locked gate conditions failed; only one fold had a positive AUROC delta. Reference shuffling caused no loss, and temporal-state permutation improved AUROC. The patient-reference and cross-state mechanisms therefore failed to produce stable discrimination.

No post-result tuning, retrained ablation, outer test, or Omni run was performed.

**Terminal: `STOP_SCM_ICTAL_GATE_FAILED`.**
