# Implementation audit

- Frozen lock SHA-256 `43b7680413d360163eb9428bc03f0816c46e8a83fbc93465a10abd131aafbcb9` was pushed before new Teacher outcomes.
- Prior exact A1 150-checkpoint/R4 replay and B0 identity were reused and hash-checked. Prior matched fixed-query retuned 64D FULLPOOL AP was replayed from the audited reference; no new target FULLPOOL fit was run.
- 869 FIT patient-contexts across 17 exact A1 contexts; 4345 convex heads. Five deterministic label-blind channel folds per patient. Each channel's score came from a head fitted on the other four folds only.
- The 21-point regularization grid used three deterministic FIT half-split episodes per patient. For each OOF-scored patient, lambda was selected only from other FIT patients in that context. This is stricter than the prior global FIT-lambda selection and prevents the scored patient's label from choosing its own lambda.
- FIT OOF metrics weight contexts by inherited target-cell multiplicity and then average five outer folds equally. The supplementary 10,000-draw bootstrap clusters repeated FIT appearances by exact patient ID.
- The prespecified Gate 0 fails on MRR and Top1 despite a large AP gain. No Student, continued-training control, target scores, KD tuning, or target outcome evaluation were executed. CSV files for these stages explicitly say NOT_RUN_GATE0.
- Legacy A1 VLOO uses cross-target labels and its loader materializes all 80 labels; literal strict target-label sequencing is false.
