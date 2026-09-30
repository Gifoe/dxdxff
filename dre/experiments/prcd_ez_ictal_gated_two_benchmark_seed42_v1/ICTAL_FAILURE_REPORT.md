# PRCD-EZ Ictal gate failure

- Absolute fixed dictionary did not exceed A1: CD-EZ AUROC 0.680595 versus 0.746382.
- Relative coordinates improved AUROC by 0.049468 over CD-EZ, but PRCD-EZ still trailed A1 by 0.016319.
- Biomarkers were not materially useful: removing all 30 changed AUROC by -0.002963 and AP by -0.000438.
- Failures were not confined to one metric: AUROC, AP, fold consistency and MRR all failed. Macro-F1 alone passed.
- PRCD-EZ exceeded the historical A1 fold AUROC in only 2/5 folds.
- The model therefore does not support progression to Omni.

**Terminal: `STOP_ICTAL_GATE_FAILED`.**
