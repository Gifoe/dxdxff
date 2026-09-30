# Feature contract

Each token is `[A1-36D, spectral-32D]`. The physiology portion exactly retains
the existing A1 descriptor meanings: delta/theta/beta/low-gamma/high-gamma
log-bandpower, RMS, variance, line-length per second, and spectral entropy in
ABS/DELTA/ZDELTA/LOGR views. Ictal uses the historical same-channel pre-onset
reference; Omni uses its existing within-EDF contemporaneous cross-channel
median/MAD reference.

The new spectral portion is deterministic and label-free: a 2-s Hann window,
rFFT power, integration in 32 fixed log-spaced 1--300 Hz physical bins, and
`log1p`. Bins completely beyond `0.45 * fs` are set to zero and masked. No
upsampling is permitted. Robust median/IQR normalization is fitted on training
patients only, then frozen for validation and any later formal test.

Ranks use all and only signal-valid channels of the same record, window, and
feature. Equal values receive average ranks; an only channel receives zero.
Neither labels nor supervision availability participate in feature extraction,
rank construction, scaler fitting, or clip selection.
