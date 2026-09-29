# Reference operator audit (pre-test)

Ictal uses the historical same-channel pre-onset mean and standard deviation.
For each feature/window, DELTA is `x - mean_pre`, ZDELTA is DELTA divided by
`max(std_pre, 1e-5)`, and LOGR is
`log((abs(x)+1e-5)/(abs(mean_pre)+1e-5))`. All 256 raw records have matched
historical feature records; the 80-patient private extraction completed with
three partial source records explicitly masked. A synthetic elementwise parity
test against the historical A1 implementation passed.

Omni has no onset reference. Within each synchronized patient/EDF/2-s window,
the nine-feature cross-channel median and `1.4826*MAD + 1e-5` are computed
over all good channels. DELTA/ZDELTA/LOGR use those contemporaneous values.
No cross-record normalization, patient ID, center ID, or label is supplied to
the conditioner. The same 36-dimensional neural module processes both
benchmarks.

The reference operators are data-context constructions. Their difference
must not be described as proof that one identical preprocessing operation was
used for two physiologically different benchmarks.
