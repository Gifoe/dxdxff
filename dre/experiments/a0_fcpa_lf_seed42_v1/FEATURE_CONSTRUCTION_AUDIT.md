# Not constructed: mandatory source gate failed

No LF24 bank, imputer/scaler, Patient Z LF transform, reliability q tensor,
new feature vector, development bank or model input was constructed.
`LF_FEATURE_COVERAGE.csv` reports source/support candidates, explicitly
`LF24_feature_bank_extracted=False`; it is not an extracted-feature coverage
claim. No missing evidence was copied from neighbors or filled with labels.

The fixed but unexecuted specification is3 bands×4 views×2 seizure statistics
=24D,4s primary Welch windows with fully preictal reference supports and no
padding/phase crossing. Original A0 view formulas and FIT-only preprocessing
would have to be reused after source admission. This report does not claim
that those computations, feature-finite checks or2s/4s stability ran.

The original88D A0 input remains unchanged, including16 constant LF columns
and16 duplicates. Alpha is not in the primary PRUNED list. The historical
named Window alpha numerically corresponds to the retained10–13Hz subset,
not full8–13Hz. New full-alpha exposure would be a supplementary input, not
a renaming or repair of the frozen A0 representation.

The intended HF24 input must come from name-indexed canonical B0 beta/
low-gamma/high-gamma columns in frozen A0.250Hz Raw has125Hz Nyquist and
cannot supply full80–150Hz high gamma. No such re-extraction, mixing or
original-HF preprocessing change was performed.

Within/between-patient LF24 variation, redundancy with A0, predictive
complementarity, preictal-reference stability, quality-stratum performance
and LF-Direct information checks are **NOT_RUN_STAGE_A_FAILED**. No claim
of incremental information follows from the measured Raw PSD.
