# Leakage audit

- `train_prism.py` exposes only frozen fit/validation loaders. It has no outer
  or official-test cache argument.
- The raw-token store is built explicitly from the current fit and validation
  patient IDs. Test identities cannot enter token construction or scaler fit
  before a separate post-freeze evaluator is introduced.
- The 68-D extractor and empirical rank functions accept waveform/features and
  signal-valid masks only; they do not accept labels, outcome, resection, or
  supervision availability.
- The robust median/IQR scaler is fitted only from feature masks belonging to
  the fit IDs. Validation/test observations are never appended to its samples.
- Omni clip selection hashes seed, patient, EDF, and cached clip index only.
  It is independent of labels.
- Checkpoint selection is validation AUROC-led; the Omni classification
  threshold is selected afterwards on validation only. No threshold belongs to
  training loss or checkpoint ranking.
- Any formal test remains blocked by `PROTOCOL_LOCK.json` until all development
  selections are frozen. Previous Omni test access makes any future test result
  exploratory rather than blind.
