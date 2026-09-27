# Cross-patient feature geometry audit — seed 42

Development FIT + validation only. The within-validation-patient probes use that patient's labels for inner training and are NONDEPLOYABLE. No new outer-test evaluation was run.

Protocol caveat: the inherited source cache initializer loaded the full 80-patient pickle and materialized outer-patient labels in memory before FIT/validation filtering. No outer-test predictions or metrics were computed and no outer-test loader was constructed, but the literal "do not read outer-test labels" requirement was not met. Treat this branch as a development diagnostic with a sealing-process deviation, not a cleanly sealed audit.

Overall terminal: `FEATURE_INFORMATION_PRESENT_BUT_PATIENT_GEOMETRY_UNSTABLE`.

## Frozen-source and controls

A1 replay: 150 checkpoints, maximum validation-grid error 0, mean Macro-F1 0.625996.
Channel-permutation maximum metric error: 0. Repeated extraction matched: True.
One within-patient FIT-label permutation: validation AP 0.340 versus source 0.539 and prevalence 0.215. It dropped substantially but did NOT reach prevalence; this weakens a clean chance-control claim and is reported without reinterpretation.

## What the diagnostic distinguishes

The mean source-feature validation sign-agreement fraction is 0.623. By view: DELTA 0.631, ABS 0.627, ZDELTA 0.617, RATIO 0.615. A large absolute effect with low sign agreement is direction-unstable, not information-free.
Patient-relative raw72 shared-probe AP changes versus global: patient_z +0.006 (3/5 positive folds), patient_robust +0.014 (3/5 positive folds), patient_rank +0.005 (3/5 positive folds). None reaches the fixed simple-alignment gate. Rank Gaussianization does not beat ordinary patient-z on AP.
The validation median FIT-consensus cosines are R1 0.758, R3 0.674, R4 0.671. Pairwise and transfer aggregate CSVs give full distribution summaries. FIT-to-validation source-direction reversed-AUROC fractions are R1 0.281, R3 0.203, R4 0.180.
Matched heldout-channel AP (shared → patient-specific linear): R1 0.593 → 0.758; R3 0.620 → 0.803; R4 0.600 → 0.816. All three satisfy the preregistered geometric gate; see 10,000-resample patient bootstrap CIs in GEOMETRY_GAP_BOOTSTRAP.csv.
The same within-patient MLP AP values are R1 0.665, R3 0.679, R4 0.691. They do not surpass the local linear control, so this audit does not identify patient-specific nonlinearity.
RICH9 shared AP 0.539; RICHALL 0.545 (Δ +0.006). PCA72-matched Δ +0.006. Additional cached descriptors do not meet the richer-feature gate; AUROC and Top1 also fail nondecrease.

## Decision and limits

The strongest supported next *class* of experiment is FIT-only learned patient-conditioned coordinate alignment or adaptation, evaluated prospectively on development data before any sealed test. This audit does not implement it. A patient-specific CV oracle cannot be deployed without that patient's EZ labels; its AP is also measured on smaller heldout subsets than a whole-patient AP, so the matched shared scorer on the identical subsets is the required comparison.
This is not an information-theoretic claim about EEG. The source cache loader materializes the 80-patient cache at initialization, but the analysis constructs no outer-test loader and computes no outer-test metric. The cache-initialization behavior should be fixed in a future strict sealed-state audit if even in-memory test-label materialization is disallowed.

Individual terminals: `CROSS_PATIENT_GEOMETRY_INSTABILITY_SUPPORTED`, `PATIENT_SPECIFIC_NONLINEARITY_NOT_SUPPORTED`, `SIMPLE_FEATURE_ALIGNMENT_NOT_SUPPORTED`, `RICHER_HANDCRAFTED_FEATURE_HEADROOM_NOT_SUPPORTED`.
