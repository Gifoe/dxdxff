# D2 source-rank compression: source review before training

Authoritative D2: commit `394dbff2085114d5f2d162cc7802384c95f0ef8d`,
`dre/experiments/a0_source_posterior_decoder_seed42_v1/`.
Read model.py, train.py, prepare.py, protocol, final report and actual source,
fold and correction CSVs. The classifier uses hidden output after GELU/dropout,
then shared Linear96->1 plus `(h@V.T)*u[g]` summed plus source bias.
The rank4 4x96 effective correction has no stricter algebraic bound than the
four-source matrix itself. This does not prove excess flexibility or overfit.

Original `pr_uncertainty_aware_supervision_seed42_v1/code/uas_core.py` was read
in full. Its SHA256 is `1b1ccb0a361e6245cb2e6089347e147036aac90437c0992194f6db7bebf1503a`.
Authoritative original helpers, hash-checked at import, supply patient_z,
FIT-only prepare, seeding/RNG restore, sorted patient groups and exact threshold
selection. No teacher, posterior or soft-label objective is called.
The shared network initializer is the same PRMLP construction, independently
seeded each fold. Dedicated source RNG does not consume dropout training RNG.

Original labels are EZ0/NEZ1. Positive BCE weighting is FIT EZcount/NEZcount,
not a reversed EZ-positive objective. One patient is one optimizer update.
The original normalized source L2 is preserved; R0 removes absent terms only.
Different ranks imply different factorized parameter spaces and mean-square
penalties even under the same coefficient; this is not an isolated causal test
of rank divorced from initialization, joint training and regularization.

Frozen D2 Macro-F1=.6479260773776448, EZF1=.44319946896815243,
EZAP=.5199357062538446, EZAUROC=.7182207455375411. Actual center Macro-F1:
HUP .6808180054701995, LZU .6159564944753768,
Multicenter .7244327938091477, Pediatric .5383449351521187.
Old correction audit effective rank4 and magnitude heterogeneity are observed
properties, not evidence that source specialization causes errors.

Historical reports reviewed:

- A0 robust posterior identifiability v2: `NUMERICAL_REPAIR_WITHOUT_DECISION_GAIN`;
  bounded Gaussian/MAP repair did not pass FIT-only predictive admission; no
  complete posterior validation advantage is available.
- A0-SSS-MIL: S1 Macro-F1 .650392 vs shuffled S2 .650461, S1-S2 -.000070;
  `CAPACITY_OR_FINETUNING_CONFOUNDED_GAIN`. Raw assignment benefit unestablished.
- D2-GraphRole published commit `1a66d792fabece8aec2085508db318aaa0ce0267`:
  G1 .649188 versus shuffled G2 .653198, G1-G2 -.004009;
  `GRAPHROLE_SHUFFLE_CONTROL_NOT_BEATEN`. Correct graph correspondence benefit
  unestablished. Neither addition enters this rank-compression experiment.

The original 80/7635/88D export and frozen five FIT/VAL banks are authoritative.
All 101 archived reference artifacts/checkpoints are hash-verified. The original
88 names/order and bitwise preprocessed tensors are verified by independently
calling original prepare on unlabeled raw engineered features, not re-extracting
EEG. All-cohort NPZ y is never accessed; labels come only from sealed original
FIT/VAL banks. Outer split IDs may be checked for exclusion, not their labels,
scores or performance. No clinical labels become model input.

R4 compatibility preserves exact state names/shapes and original arithmetic.
Fresh forward parity precedes exact archived prediction reuse. R0/R1/R2 train
from fresh identical shared initial parameters, never historical final weights.
Exactly15 new formal cells are authorized. No rank3, alternative lambda, extra
seed, feature selection, source-specific validation threshold or outer TEST.

Mechanism audit uses U@V, not nonidentifiable factors as biomarkers. Rank is
computed by float64 SVD with standard size-scaled machine-epsilon tolerance.
Unknown source returns the shared head. A within-patient source-constant bias
cannot change mathematical ranking; finite precision ties are separated from
directional ranking changes in inference-only interventions.

Before outcomes, the unspecified severe-source-collapse rule is operationalized
as a >.050 absolute source Macro-F1 decline relative to frozen R4, with no
posthoc exemption. R2 remains the primary hypothesis even if a secondary rank
has the highest point estimate. Reused development bootstrap is descriptive,
not independent testing or evidence of unseen-center transfer.
