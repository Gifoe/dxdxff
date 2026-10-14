# Genuine-window DELTA reconstruction

New measurement recomputation reads only original hash-verified waveform banks whose every500-sample slice was verified against the original Raw source in the historical Raw alignment audit. This run freshly verifies whole source/cache/archive/split hashes and every raw-bank hash and compound channel/window masks; it does not falsely claim a second source-slice scan.

Pinned ez_features functions operate on original unnormalized float32 genuine Raw250 windows. Same-channel pre-reference is the float32 mean of all actual centers<0. No invalid/padded observations, filters, new bands or fallback references. Seven families: beta, lowgamma, partial80–125Hz highgamma, RMS, variance, line length/sec, entropy10–125Hz.

Both the original full-source-view reduction layout and the actual new7D window tensor must reaggregate within original tolerances to existing R14 for80 patients/256 records/7635 channels/1471965 genuine channel-windows. Never relax tolerances or substitute contaminated padded support. See actual `WINDOW_DELTA_PARITY.json`.

Scale per family: median of each FIT patient's median absolute window DELTA, then median across FIT patients; floor1e-6; clip[-8,8]. Each FIT patient contributes one estimate. No clinical labels, validation scaling or individual-feature performance selection. FIT/Development clipping is descriptive, not a fitting rule.

Executed outcome: exact80/256/7635/1471965 coverage and zero maximum R14 reaggregation error. Both numerical reconstructions passed without changing original tolerances. The scale floor binds beta/lowgamma/partialhighgamma/variance in all folds; it suppresses their typical normalized amplitudes, so this implementation is not a strong test of balanced seven-family guidance. Details and the distinction between numerical integrity and scientific adequacy are disclosed in FINAL_REPORT. No post-outcome scale correction or rerun was performed.
