# PRCD-EZ feature audit

- The 36 physiological inputs are read unchanged from the validated A1 Ictal cache.
- The added 16 coordinates are fixed Hann/rFFT/log1p physical-band powers over 2-s windows with 1-s hop.
- Every fold fits its 52-D median/IQR scaler from FIT patients only.
- Dictionary subsets, temporal patterns, dilations and quantile identities depend only on seed 42 and kernel ID.
- Dictionary biases are fitted from a deterministic label-free sample of FIT sequences only.
- Patient-relative coordinates are computed only after record-level motif statistics and cross-seizure pooling.
- Median deviation subtracts the within-patient median and never divides by MAD, standard deviation or variance.
- Percentile ranks use average ties and map to [-1,1]; a one-channel group maps to zero.
- Spectral burden, PAC and AEC definitions are fixed before development. Missing physical bands are represented by zero plus availability bits where specified.
- mRMR relevance is estimated from per-patient EZ-minus-non-EZ contrasts. Redundancy uses a deterministic, class-balanced sample of FIT channel representations.
- Validation and test identities are excluded from scaler, bias and mRMR fitting. This experiment exposes no outer-test argument.
