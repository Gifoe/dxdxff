# B=8 Teacher Ceiling Study — development-only

65 VLOO target cells, 47 unique patient IDs, 20 fixed-query repetitions each. All values are exploratory matched-query patient-level analyses, not outer-test or prospective results.

1. Original B8 replay: 0.599632 EZ-AP; previous 0.599632, max private per-repetition AP error 0.
2. Retuned B8: 0.605291, delta +0.005659 AP; FIT-only regularization selection.
3. FULL_POOL anomaly: current B8/FULL 0.599632/0.592852; retuned 0.605291/0.692464, paired delta -0.087173 [-0.137911,-0.039183]. Terminal `PREVIOUS_B8_FULLPOOL_REVERSAL_EXPLAINED_BY_REGULARIZATION`.
4. PCA B8: d4 0.594094, d8 0.595757, d16 0.599522; all dimensions and lambda values were FIT-only selected.
5. Episodic meta B8: d4 0.592822, d8 0.595022, d16 0.598035. Paired intervals/fold signs in accompanying CSVs.
6. Highest observed deployable B8 Teacher: `RETUNED_64D_B8`. This is descriptive winner selection across predeclared variants, not independent validation.
7. Best Teacher: AP 0.605291, MRR 0.766719, Top1 0.683476, Macro-F1 0.656353, EZ-F1 0.429842, BA 0.681546; delta AP vs current B8 +0.005659 [-0.001972,+0.014412], positive folds 3/5. Its Macro-F1 and EZ-F1 are lower than current B8 (0.662126 and 0.450731), despite slightly higher AP.
8. Milestones: TEACHER_AP_062_REACHED=False, TEACHER_AP_065_REACHED=False, TEACHER_AP_068_REACHED=False, TEACHER_AP_070_REACHED=False; B8_TEACHER_IMPROVEMENT_SUPPORTED=False.
9. Oracle-direction cosine increases from 0.235240 (current B8) to 0.265004 (best B8), alongside the small AP increase. Retuned FULL_POOL rises to cosine 0.563352 and AP 0.692464. The broad direction is concordant, but variants are not strictly monotone; cosine was target-label diagnostic only, not a selection input. See the aggregate plot.
10. B=8 bottleneck: simply reducing dimension or episodically learning a low-dimensional projection did **not** improve the current B8 teacher; the largest B8 gain is only +0.005659 AP with a CI crossing zero. Retuned FULL_POOL, using on average 48.05 candidate labels, reaches 0.692464 AP; B8 support is one-class in 15.15% of repetitions. This weighs against the claim that the main bottleneck was merely 8 labels fitting a 64D residual, and points to supervision amount/acquisition as plausible constraints. It does not isolate a causal eight-label information ceiling: these retrospective controls differ in both label count and selection mechanism.

No Student distillation. No new outer-test metrics. The legacy loader still materializes all 80 labels; literal strict sealing is therefore false. The retrospective best-variant comparison is uncorrected for multiple comparisons. Projection-optimizer meta training and validation are patient-disjoint, but the earlier global FIT lambda search included those meta-validation patients.
