# A2 implementation

Frozen D1 is constructed directly; RawDeltaEncoder/D2/GRU is never instantiated. Original CNN2160 parameters, head6273 and scalaralpha are initialized before the physiology gate consumes RNG. Gate7→32→24 has1048 parameters with zero final weight/bias; step0 multiplier exactly1, final NEZ logit exactlyD1. Total11019, trainable9482, frozen1537.

CNN sees per-window mean/popSD-normalized Raw (floor1e-6, clip[-8,8]); physiology sees unnormalized Raw. Same genuine center<0 latent reference. Multiplicative-only bounded[.75,1.25] guidance, no physiology additive prediction path. Means over[0,8)/[8,20), available-seizure mean/populationSD; singletonSD0 with finite zero-variance gradients. PatientZ across valid canonical channels uses float64 populationSD and original1e-8 floor semantics.

Historical overlap: original RawDelta uses the same CNN/head but a GRU; this A2 drops recurrent/order modeling and adds physiology-conditioned statistical morphology. Prior SSS-MIL attention pooling, COVE engineered-feature residuals and HLV dynamic gates are not evidence for this proposed mechanism. FiLM/gating itself is not claimed novel. Multiplicative structure alone cannot establish information beyond physiological DELTA; trained matched controls remain necessary and unexecuted in this A2-only scope.

Mean pooling is invariant to jointly permuting Raw and physiology within a stage. This is not a within-stage chronology learner. Clinical onset provenance and source filters are not newly established.
