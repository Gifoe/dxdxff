# A1 feature source audit

- Spectral source: `historical_a1_ez_features.py`; function `compute_spectral_channel_features`; SHA-256 `1b046b542b791f2416467582fe95e2dfa991d04ebb535bf188e04b9a1a1439d6`.
- Interictal comparison source: `historical_a1_dataset.py`; function `_interictal_comparison_features`; SHA-256 `9826059603242eeb686c4e896b42d2c3c5a33d5860830f58b3df8e94487863a2`.
- Selected ABS descriptor order: `log_bp_delta, log_bp_theta, log_bp_beta, log_bp_low_gamma, log_bp_high_gamma, rms, variance, line_length_per_sec, spectral_entropy`.
- Historical bands: delta 1-4 Hz, theta 4-8 Hz, beta 13-30 Hz, low-gamma 30-80 Hz, high-gamma 80-150 Hz.
- Historical spectral implementation: Welch (`nperseg=min(samples, round(2*fs))`, 50% overlap, no detrending), integration by trapezoid, and `log1p` for all band powers.
- Numerical constants: RMS uses `+1e-8`; entropy normalizes PSD with floor `1e-8` and uses `log(p+1e-8)/log(n_bins+1e-8)`; interictal robust reference uses median and `max(1.4826*MAD, 1e-5)`.
- Line length is `sum(abs(diff(signal)))/duration_seconds`.
- F36 exactly requests historical `[ABS, DELTA, ZDELTA, LOGR]`; percentile and missing-indicator extensions are explicitly disabled.  With no external interictal baseline supplied for an Omni clip, the historical function's fallback uses its 30 2-second windows as the robust reference.
