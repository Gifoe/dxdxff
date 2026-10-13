# Measurable power is not completed LF fidelity

Actual Raw250Hz valid-window audit:1,471,965 channel-windows,100% finite
delta/theta/alpha powers, median fractions0.509464/0.202113/0.086877 of
total1–40Hz. These are waveform observations, not localization metrics.

The completed EDF diagnostics are separated by timeline/acquisition evidence.
Use `LF_PSD_COMPARISON_QUALIFIED_V2.csv`, not the unqualified all256 aggregate
as a physical-time or device-fidelity claim. The latter is retained to show
what was actually computed before the full EDF+D qualification.

| Continuous-header diagnostic | Paired records | Raw/original-EDF median ratio | Spearman |
|---|---:|---:|---:|
| Delta1–4 |162|0.983282|0.996938|
| Theta4–8 |162|0.997716|0.999451|
| Alpha8–13 |162|0.995868|0.999698|
| Beta13–30 |162|0.997941|0.999948|

These use894,570 channel-windows on physically complete guarded supports
under the continuous-record assumption. The45 continuous-header records
with declared bandwidth metadata have163,914 paired windows; their
delta/theta/alpha median ratios are0.986627/0.997846/0.995961 and
Spearman0.993402/0.997729/0.998891. Missing acquisition evidence for HUP
is not filled by these high correlations.

The applied10Hz negative control has far lower delta/theta power: Raw/control
median ratios1.5832e6/1704.83 on continuous-header records. Thus these Raw
spectra are not equivalent to that specific control. The fixed1Hz candidate
is close, but similarity does not recover the historical processing CLI.
Filtering transitions and2s spectral leakage remain relevant; neither nonzero
leakage nor high Spearman alone proves reliable1Hz physiology.

Complete valid intervals exclude cached padding. Original FFT integration
was checked with deterministic2/6/11/20Hz synthetic tones. These checks reduce
padding/integration mistakes; they do not rule out all earlier normalization,
resampling, reference or acquisition artifacts. Exact anti-aliasing and
referencing provenance remains unknown for the frozen cache.

Native EDF Welch uses round(2fs) samples; unusual noninteger header rates
produce grids not exactly aligned to every band edge. Reported masked-bin
comparisons are descriptive. The total-minus-low residual is marked a
**proxy**, not universally an exact13–40 integral. Dense frequency-by-frequency
PSD-shape/transfer-function proof was not established. No favorable diagnostic
was used to promote the failed source gate or select a source/filter.

Primary4s LF feature extraction and same-support2s/4s stability were **not run**:
Stage A failed first. There is no claim that high delta fraction proves useful
LF evidence, that all data were10Hz high-passed, or that all Raw LF is artifact.
