# Raw ABS14 / DELTA14 parity

**VERIFIED:** both new arms use the exact same pinned Raw250/500-sample seven-family measurements, all genuine windows, canonical channels and available seizures. No engineered ABS18 values are used.

ABS: per-seizure/channel mean of absolute descriptor levels over all genuine windows, then masked available-seizure mean/population SD. DELTA: subtract same-channel float32 reference from **all genuine center<0 windows**, then original genuine within-seizure mean and across-seizure mean/populationSD. Both outputs are14D in feature-major mean/std order. “ABS” means absolute physiological levels, not a new absolute-value operation on the descriptor.

This run recomputed both tensors from existing pinned waveform banks without re-acquiring/filtering/resampling EDF. Full original feature-view reduction and actual seven-dimensional DELTA reduction both passed the original numerical tolerance; the actual DELTA14 additionally equals original D1 R14 **elementwise, maximum error0**. Both arrays are finite. Missing seizure channels were unavailable rather than padded observations. Single-seizure SD is exactly zero. No clinical label entered feature extraction.

Raw250 Nyquist125Hz does not restore full80–150Hz high gamma. The unchanged descriptors are beta,low-gamma,partial80–125Hz high-gamma,RMS,variance,line-length/sec and original restricted10–125Hz spectral entropy. No new band/filter/PSD definition was introduced. See results/FEATURE_PARITY.json for genuine-window counts and hashes.
