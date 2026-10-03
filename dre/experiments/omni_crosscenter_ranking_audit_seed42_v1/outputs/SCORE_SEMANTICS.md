# Score semantics

Both scores use **higher = more pathological**. Frozen CNN uses the official normal-logit orientation followed by `1 - sigmoid(logit)` and an EDF-channel segment mean. Source Fisher is the frozen R4 patient-equal TRAIN Fisher direction `(pooled source covariance)^-1 (mean_pathological - mean_normal)`. The exact replay AUROCs below exceed 0.5 in this orientation; neither sign nor any other score definition was selected from this audit.

- Frozen CNN TEST AUROC: `0.7987677712`
- Source Fisher R4 TEST AUROC: `0.7994219077`
