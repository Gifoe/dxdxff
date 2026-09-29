# Physiological descriptor audit (pre-test)

- The nine feature names and extraction implementation are pinned to the
  server's historical `ez_features.py` SHA-256
  `18ec360e9d9ec7e5b56b05208c0ef6575c13769ab6866ef241e4d65e8a554c72`.
- Features are computed per 2-s window with 1-s stride. The 59 windows provide
  nine absolute features and three nine-feature reference views: ABS, DELTA,
  ZDELTA, LOGR (36 coordinates total).
- Ictal descriptors come from the existing frozen feature cache. The
  synthetic parity test compares `views_exact_a1` element-for-element with the
  historical `b0_self_reference_features`; it passes.
- Omni descriptors are computed from the frozen native-signal HDF5 using the
  same pinned feature function, but with a contemporaneous cross-channel
  median/MAD reference, as specified by this experiment. This is a different
  reference operator, not a different neural model topology.
- The Ictal safe upper frequency is 112.5 Hz at 250-Hz sampling; all four
  high-gamma view coordinates are masked. The first fold's train-only moment
  fit observed 32/36 coordinates. No spectral bin above 112.5 Hz is invented.
- The fit-fold descriptor mean/standard deviation is estimated exclusively
  from fit patients and applied unchanged to validation/test. Missing
  coordinates are zero after masking; they do not enter moment estimation.
- Three shorter Ictal source records preserve their actual observed sample
  spans and descriptor-window masks. Zero-padding is not interpreted as EEG.

This audit does not claim PC-CNN performance or final-test access.
