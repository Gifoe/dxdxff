# Extraction provenance — do not infer preprocessing from defaults

## Exact artifacts identified

| Alias | SHA256 | Role |
|---|---|---|
| A0 export | `1bdcc479a2bd148bdc94614c8253bea67e4008bd98bc6dad9e8030df827db9bd` | Frozen88D; identity/order metadata only, not model input regeneration |
| A0 Window Cache | `9b5bb58a0175aba494ad79e30c5a65c7cee33c7d464273f9bcf62b9b280b0087` |80-patient/256-record intersection; named historical window features |
| Main Raw Cache | `011d7ffaa55469c9f34259be04b7136a987d6d95f0ce9cc443a06751557db733` |155-patient/501-record waveform superset |
| Other Raw Cache | `2d11efe0eb261ed71e9bfbaf7edaa2e689a88a533c0d083b52aeedaa9b09d297` |149-patient/476-record superset; main256 signals exactly equal |
| Original five-fold manifest | `fd897fa7eed2c521fd5b14c1ae95d91b5d43b2ae85822317dcda08a878f58278` | Original memberships, not refitted splits |

Recorded and relocated EDF paths, source-channel metadata, onset details and
EDF whole-file hashes are held only in the trusted private ledger. No
individual EDF path or patient/channel identity is published.

## Current source logic that was actually reviewed

- `neuroez_c/evidence_views.py`, SHA
  `201f888411a8f45fa363cfb2ac11c165a04ae2eef5abcd3aad64aedfe3098b03`:
  alpha is in BASE spectral names, not the A0 PRUNED nine-feature list.
- `ez_features.py`, SHA
  `18ec360e9d9ec7e5b56b05208c0ef6575c13769ab6866ef241e4d65e8a554c72`:
  Welch PSD is first restricted by `spectral_min_freq`; band integration
  then uses half-open band masks and `log1p(power)`. With a10Hz restriction,
  delta/theta have no bins and return exact zero; named alpha then contains
  only retained10–13Hz bins, not the full8–13Hz range.
- `module2_preprocessing.py`, SHA
  `099f3e9bbf03204e63f4d216ce81ec1249cf9dd332d40d901173be92ce891ac9`:
  its EDF loader can high-pass at `bandpass_low` (default10Hz), notch, resample,
  globally detrend and median-center. This is a possible source path, not
  automatically the actual generator of these exact caches.
- The current `build_neuroez_c_four_center_caches.py` delegates to data_readers.
  The reader has **independent** `reader_bandpass_low` default1Hz and
  target512Hz; its cache-stage `spectral_min_freq` default10Hz separately
  restricts Window PSD. `_seizure_to_run_record` constructs the centered raw
  from the reader's already-preprocessed signal. Current1Hz reader and10Hz
  Window cutoff can therefore coexist. Reader defaults do not prove which
  settings generated the actual stored signals.
- `_build_centered_raw_waveform` resamples the extracted valid segment to250Hz,
  places it in a60-second zero-padded array and returns valid-start/length.
  Resampling cannot restore frequencies removed upstream.

All reviewed entry-point, reader, feature, padding and alignment source hashes
are in the locked audit metadata. Archived module2 source copies reviewed in
this repository have default10Hz, while the current data_reader config has
default1Hz. These are distinct source paths, not evidence that one exact
cache used a particular high-pass.

## What is not proven

The merged raw payload contains no full preprocessing configuration, exact
CLI or generating source SHA. Its current source file hash is not a embedded
historical build hash. Existing raw-build logs are UTF-16 BOM text and were
decoded accordingly. They mention the main cache and include source-checkpoint
reuse, but the inventory did not recover explicit filter arguments or an
exact generating-source hash. The `reader1/reader3` filenames identify build
worker variants; they are not interpreted as1Hz/3Hz filter settings.

**Exact generation provenance: `PROVENANCE_NOT_PROVEN`.** In particular,
the reused source checkpoint's prior filtering cannot be established from
the final pickle or current command defaults. Absence of a CLI from the
recovered logs is not evidence that no filtering occurred.

## Measured diagnostics versus provenance

The frozen0.5Hz reference,1Hz historical-reader candidate and10Hz negative
control are diagnostic transformations of matched accessible EDF windows,
not reconstructed claims about the actual generating CLI. EDF hardware
prefilter fields are inspected separately; unknown/blank header information
does not mean unfiltered recording. Raw EDF is not called a low-frequency
ground truth merely because it exists or its sampling rate is high.

Welch finite/nonzero band power can include transition-band attenuation,
spectral leakage and hardware-processed content. Diagnostic nonzero values
alone never promote a source through the data-availability gate.
