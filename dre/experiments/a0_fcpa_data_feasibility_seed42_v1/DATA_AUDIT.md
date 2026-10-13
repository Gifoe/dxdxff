# Completed strict no-training low-frequency data audit

**Decision: STOP_DATA_NOT_IDENTIFIABLE.** This means the required availability
evidence is incomplete, not that all low-frequency samples are absent.
No model was trained, A0 feature regenerated, threshold optimized,
classification performance evaluated or new outer outcome accessed.

## Checklist and evidence origin

| Item | Status | Evidence actually established |
|---|---|---|
| Frozen A0 archive, feature order, original5 folds | VERIFIED | Content hashes and unlabeled identity/order/split metadata; x/y not indexed. |
|80 patients,256 seizures,7,635 canonical channels | VERIFIED | Exact intersection,24,995 seizure-channel incidences; no exclusions by spectrum or labels. |
| Other Raw Cache as independent source | FAILED | Its256 main-cohort waveform hashes and time/channel metadata exactly match the main Raw Cache. |
| Raw250Hz/60s and valid intervals | VERIFIED | All256 records;125Hz Nyquist. No80–150Hz high-gamma re-extraction. |
| Raw delta/theta/full-alpha numerical power | VERIFIED | Actual Welch run on1,471,965 complete channel-windows; finite rate100%. |
| Actual raw-generation CLI/source/filter chain | UNVERIFIED | No embedded complete configuration or generating SHA; recovered logs do not establish filter arguments or reused checkpoint provenance. |
| EDF paths/header-channel identities | VERIFIED | All256 resolved and mapped; physical units uV, converted by EDF scaling. Individual paths/hashes stay private. |
| Independent event-time corroboration | PARTIAL |162 records with matched accessible event sidecars;94 without corroboration. Header TAL existence is not an onset verification. |
| Acquisition filter information | PARTIAL |45 Multicenter records declare HP0 and LP values; other211 records lack usable HP/LP fields. Header declarations are not independently certified device calibration logs. |
| Physical-time EDF/cache parity | PARTIAL |162 EDF+C records permit record-clock diagnostics;72 LZU and22 pediatric are EDF+D and require TAL clock verification. Index replay is not physical-time proof. |
| Named historical alpha meaning | VERIFIED on continuous-header diagnostic support | Source restriction and name-indexed replay support10<=f<13, not full8–13. |
| Historical27-window mask identity | UNVERIFIED | Exact old mask ledger not recovered. Measured nominal59-grid has30 invalid windows:27 LZU plus3 Multicenter, over3 padded records. Count similarity does not recover the old mask. |
| FCPA availability gate | FAILED | Provenance, complete independent time/acquisition coverage and exact historical-mask reconciliation are incomplete. |

The16 constant delta/theta export columns and16 exact auxiliary duplicates
are **historical manifest findings**, hash-bound here, not newly recomputed
from full A0 x. Alpha is absent from the PRUNED primary feature list.
The current Window Cache was actually indexed by feature name: delta/theta
power is zero throughout, while `log_bp_alpha` is nonzero. These window
features must not be confused with the low-frequency content of Raw samples.

## Actual Raw measurements

Fixed2s periodic-Hann Welch, nperseg500 at250Hz, constant detrend, density,
closed-edge trapezoids. No padded or incomplete samples contribute.
The15,074 retained record-windows yield1,471,965 channel-windows. All three
bands are finite100%; they are correlated windows, not independent patients.

| Band | Median power fraction of1–40Hz | Median power, assumedV² |
|---|---:|---:|
| Delta1–4 |0.509464|1.204472e-9|
| Theta4–8 |0.202113|5.253513e-10|
| Alpha8–13 |0.086877|2.091132e-10|

Full pooled/equal-patient/center quantiles, maxima and operational-floor rates
are in `results/RAW_CACHE_BANDPOWER_SUMMARY.csv`. The fixed1e-6 relative
and1e-18 assumedV² floors are diagnostics, **not measured equipment noise**.
Raw absolute calibration remains provenance-dependent. Median mean-PSD8–10
over10–13 is1.57689;0.24654% fall below the fixed0.05 diagnostic floor.
There is measurable8–10 power, not a recovered full-alpha Window feature.

## EDF comparison qualifications

The original all256 tables preserve the executed **record-index diagnostics**.
Do not interpret their EDF+D rows or aggregate as independently verified
physical-time correspondence. See the companion FCPA-LF
`results/LF_PSD_COMPARISON_QUALIFIED_V2.csv` for timeline-qualified summaries.
The162 EDF+C diagnostic records contain894,570 paired channel-windows.
Raw/original-EDF median ratios are0.983282(delta),0.997716(theta),
0.995868(alpha), with Spearman0.996938/0.999451/0.999698.

The fixed10Hz negative control produces vastly less delta/theta power:
Raw/control median ratios1.5832e6 and1704.83 on continuous-header supports.
The observed Raw is not numerically equivalent to this particular control.
This does **not** identify the historical generating CLI, rule out other
filtering/artifacts, certify acquisition bandwidth or confirm clinical timing.
Closer correspondence to the fixed1Hz candidate is likewise not a proven
historical filter setting. PSD integrals alone are not a dense transfer-function
or reliable single1Hz physiological measurement.

For noninteger EDF sampling rates, round(2fs) Welch grids do not universally
land exactly on every band edge. Native masked-bin ratios are descriptive;
total-minus-low is only a residual proxy, not an exact13–40 integral in all
files. The V2 qualification preserves all original numeric observations and
corrects their interpretation without tuning a gate or recomputing signals.

Name-indexed alpha replay on HUP gives median old-alpha/native10–13 ratio
0.995622 and Spearman0.9999997; the old-alpha/full8–13 ratio is0.405204.
This is strong numerical support for a truncated alpha feature, not full alpha.

## Why stop despite high numerical correspondence

Finite PSD and near-unit index-based ratios do not substitute for auditable
hardware filtering, exact build provenance or EDF+D clinical-time mapping.
None of those gaps was filled with a code default, a filename or a favorable
spectrum. Original A0 input/preprocessing and sealed TEST stay unchanged.
All per-record identities, waveform/EDF paths and paired power arrays remain
on the trusted server. No LF efficacy or localization claim is made.
