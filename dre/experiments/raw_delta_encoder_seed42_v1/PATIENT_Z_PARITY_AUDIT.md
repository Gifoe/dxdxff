# Patient Z and FIT-only preprocessing

The pinned original operator converts to float64 and computes per-patient, per-feature population nanmean/nanstd across canonical channels. SD greater than 1e-8 is used; otherwise the denominator is exactly one. An independent loop reproduced Patient Z exactly in all ten E1/D1 cells.

Original SimpleImputer (mean) and StandardScaler are fitted only on each original FIT set. E1 and D1 fit them independently on their respective E14/R14 measurements. E1's final fourteen normalized columns were bit-identical to the named columns of the historical masked DELTA bank. Patient roles, feature order and known bank hashes were asserted; no own-fold TEST identity or label appears in the legal bank.

Stage C applies the corresponding differentiable population-SD patient normalization to the 96D latent summaries across valid canonical channels, with the same 1e-8 denominator rule. This is hidden-representation normalization, not test-fitted population scaling. Genuine available seizures and stages are masked; missing stages cannot be silently invented. D1's already fitted preprocessing is not refit during Stage C.

See actual `PREPROCESSING_PARITY.json`, compact parity and independent verification. All scalers, patient membership and feature banks remain private.
