# A1 feature reuse audit

The nine inputs are selected, without redefining their formulas, from the
historical `ez_features.compute_spectral_channel_features` implementation
(server source SHA-256 `18ec360e9d9ec7e5b56b05208c0ef6575c13769ab6866ef241e4d65e8a554c72`).
The selected historical `PRUNED_SPECTRAL_FEATURE_NAMES` order is:

1. `log_bp_delta`
2. `log_bp_theta`
3. `log_bp_beta`
4. `log_bp_low_gamma`
5. `log_bp_high_gamma`
6. `rms`
7. `variance`
8. `line_length_per_sec`
9. `spectral_entropy`

The original implementation uses Welch spectral density with a two-second
`nperseg`, one-second overlap, no detrending, and bands defined in the same
source file. Its values depend on the effective sampling rate; Omni signals
are notched at 60 Hz and anti-aliased/resampled to 300 Hz before the original
function is called, consistently for every train/test record. No new raw or
STFT representation is supplied to A1.

On a 63-s real cached train signal, direct per-channel MNE notch/resample
matched the official-style MNE `RawArray.notch_filter().resample()` path to
`1.21e-13` microvolts maximum absolute difference (18,900 output samples).

The four-view construction changes only the temporal reference because an
interictal segment has no pre-onset baseline. Across 59 windows for each
channel/descriptor, `r=median(f)` and `s=1.4826*MAD(f)+1e-5`; the views are
`f`, `f-r`, `(f-r)/s`, and the historical safe ratio
`log((abs(f)+1e-5)/(abs(r)+1e-5))`. Therefore the model still receives 36D.

The train-side numerical audit in `audit/FEATURE_DISTRIBUTION_AUDIT.json`
checks 5 patients, up to 10 channels each, multiple windows and all four
views. It requires finite 36D tensors. No test outcome enters this check.

Official Omni segment-selection code is from commit
`57c22a75a59b5c3a98006806ad42000f6a3fa5b6`. Its training extractor draws
up to five random 60-s segments per *channel*, whereas A1 channel attention
requires channels aligned in time. This adaptation draws the same frozen
random starts once per EDF and shares them across its channels; the count and
start-index algorithm remain the official one. Test uses every complete
nonoverlapping 60-s segment, after the official one-second edge exclusion.
This synchronized-channel adaptation must be disclosed when comparing with
the official independent-channel model.
