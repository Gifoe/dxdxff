# Raw input and provenance

The prespecified contract below has now passed actual admission:
`results/RAW_ALIGNMENT_AUDIT.json` confirms80 patients/256 records/7635
channels/24995 run-channel incidences and1471965 valid channel-windows;
all source slice parity checks pass. These are actual new audit counts,
not assumed historical coverage. Physical-clock/filter limitations remain.
The historical raw cache has155 patients/501 records, but only the fixed80
patients/256 seizure records are in scope. Raw and feature-window cache SHA
must match the old RawTiny protocol. Compound run/seizure keys and normalized
canonical channel names, not positional guesses, determine matching.

Only complete500-sample2s windows at the stored feature centers may enter the
model. Bounds must lie within both the15000-sample array and raw valid-start/
valid-samples interval. No missing seizure/window is an observation, duplicate
rounded window is accepted, or padded value participates. Expected unchanged
coverage is7635 canonical channels,24995 run-channel incidences,15074 valid
record-windows and1471965 channel-windows; failures stop before training.

The feature-cache pickle is reduced to alignment metadata. The raw cache is
reduced to waveform plus metadata. Engineered feature values and labels are
not passed to R0. No global A0 imputer/scaler is used by the learned64→88 latent.
Per-window temporal normalization remains the exact original RawTiny frontend.

250Hz cache Nyquist125Hz cannot provide the full80–150Hz high-gamma band
available to some historical engineered features. The raw baseline therefore
differs in available bandwidth and architecture, not merely a controlled
deletion of handcrafted features. Poor performance cannot establish that raw
EEG is intrinsically inadequate.

Historical FCPA-LF established strong numerical correspondence for certain
continuous EDF supports but did not prove exact cache-generation CLI/source,
acquisition filtering or all EDF+D physical clocks. Stored onset-relative
centers are source-code supported, not independently certified for every run.
This task uses that existing cache as an explicitly qualified input baseline,
not as newly verified low-frequency ground truth or onset propagation proof.

No new filter, resampling, spectral extraction, EDF acquisition or patient
quality exclusion is allowed. All original patients remain in the denominator.
