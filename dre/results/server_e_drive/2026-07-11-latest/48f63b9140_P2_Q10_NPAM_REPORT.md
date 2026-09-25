# P2-Q10-NPAM Report

Profile: `M5_LIMITED_FINETUNE`
Protocol: `quick`
Patients: 3

## Protocol checklist
1. success=1
2. failure=0
3. Engel I maps to success
4. Engel II-IV map to failure
5. single P2 backbone
6. patient-wise outer folds
7. fixed fold manifest
8. no inner CV
9. no validation selection
10. fixed epochs
11. no early stopping
12. fixed threshold 0.5
13. no calibration
14. outer-train normalizer only
15. P2 train/test leakage checked
16. checkpoint fold metadata checked
17. three raw-network phases
18. declared exclusion applied
19. center excluded from model
20. coordinates excluded
21. SOZ/resection excluded
22. true-K excluded
23. patient-equal loss weighting
24. final checkpoint saved
25. one OOF prediction per evaluated patient

QUICK_SCREENING_NOT_PAPER_VALID

FULL_OUTER_CV_NOT_RUN

OUTER_CV_NOT_PAPER_VALID
