# Measurement catalogue and eligibility interpretation

One measurement group contributes two DELTA coordinates: the mean and population
SD across available seizure summaries. Thus1–9 groups mean2–18 active numerical
coordinates, not a parameter-compressed network. All neural routes have32 fixed
global slots; inactive coordinates are zeroed after common FIT preprocessing.

DELTA subtracts each channel/seizure's available pre-reference window mean,
averages all valid cache windows within that seizure, then summarizes seizures.
Absolute phase measurements are used for descriptive biology, not silently
added as new model inputs. The Raw model's original latent pool uses early0–8
and late8–20 seconds; a separately registered Raw-only8–30-second control
addresses the longer support of physiological summaries.

## All32 prospective groups

All32 passed the predeclared finite/own-subFIT nonconstant gates in every
assessment and final path. No prospective group was removed using an Outer
outcome or a correlation cutoff. `FIT_ELIGIBILITY.csv` contains640 actual
gate decisions. A nonselected group is **not** a demonstrated useless biomarker.

| Group | Named measurement | Definition/source distinction | Reason to consider; interpretation boundary |
|---:|---|---|---|
|0|log_bp_beta|Pinned legacy `log1p` band power13–30 Hz|Amplitude recruitment; numerical units matter|
|1|log_bp_low_gamma|Pinned legacy30–80 Hz power|Faster recruitment; not HFO|
|2|partial_high_gamma_80_125|Pinned legacy partial80–125 Hz power|Restricted high-frequency activity; not full80–150 Hz or PLHG|
|3|rms_legacy|`sqrt(mean(x²)+1e-8)`|Amplitude; floor is part of the historical formula|
|4|variance|Pinned variance formula|Dispersion; may overlap RMS but conditional utility requires testing|
|5|line_length_per_sec|Sum of absolute adjacent differences divided by duration|Combined amplitude/roughness; not a pure frequency measure|
|6|spectral_entropy_10_125|Pinned normalized PSD entropy over retained bins|Spectral distribution, not permutation entropy|
|7|log_variance_stable|`log(max(var(x),1e-30))`|Dynamic-range control; not algebraically equivalent preprocessing|
|8|hjorth_mobility_per_second|`250 sqrt(var(diff(x))/max(var(x),1e-30))`|Time-scale/roughness with physical sample-rate factor|
|9|hjorth_complexity_stable|Ratio of derivative mobility to signal mobility, stabilized floors|Shape complexity, not clinical excitation/inhibition|
|10|permutation_entropy_m3_tau1|Normalized ordinal entropy, length3, delay1, stable tie order|Ordinal complexity; not the published full postictal PEI|
|11|spectral_centroid_13_95|PSD-weighted frequency in13–95 Hz|Spectral center of mass; retained-band descriptor|
|12|log_gamma_beta_ratio|Log power30–80 minus log power13–30 Hz|Relative spectrum, not an absolute fast-activity measure|
|13|partial_high_gamma_fraction_80_125|Power80–125 divided by13–125 Hz power|Restricted-band fraction; not full broadband fraction|
|14|spectral_flatness_13_95|Geometric/arithmetic mean of normalized retained-bin PSD|Tonal versus broadband spectral shape|
|15|spectral_slope_proxy_20_95_notchgap|OLS log-PSD/log-frequency slope over20–45 and65–95 Hz bins|Not FOOOF decomposition, nor a validated EI estimate|
|16|normalized_line_length|Line length divided by signal SD, floor1e-15|Amplitude-normalized roughness; different from legacy line length|
|17|clipped_excess_kurtosis|Centered SD-normalized samples clipped to±8, standardized fourth moment minus3|Tail/spike-shape proxy, not a detector with event labels|
|18|stable_log_beta_sameband|Log of inverse-`log1p` original beta power, floor1e-30|Same-band numeric control; not recovered physiology|
|19|stable_log_low_gamma_sameband|Same transformation for original low gamma|Dynamic-range control|
|20|stable_log_partial_high_gamma_sameband|Same transformation for original partial high gamma|Dynamic-range control|
|21|partial_alpha_10_13_legacy|Original alpha definition restricted by the10 Hz spectral minimum|Only10–13 Hz, not complete8–13 Hz alpha|
|22|log_total_power_10_125_legacy|Pinned total power on retained spectral support|Broad retained-band amplitude; not complete1–125 Hz power|
|23|hjorth_mobility_legacy|Original epsilon-regularized Hjorth source formula|Keep distinct from stable per-second implementation|
|24|hjorth_complexity_legacy|Original epsilon-regularized Hjorth source formula|Units/floors can dominate; do not merge by name|
|25|rebuilt_degree_norm|Degree of the existing verified Raw-rebuilt window graph|Network topology; legacy cached graph is a different source|
|26|rebuilt_strength_norm|Weighted node strength in that graph|Connectivity magnitude; not channel band power|
|27|rebuilt_clustering_coeff|Node clustering in that graph|Local neighborhood topology|
|28|rebuilt_eigenvector_centrality|Principal-eigenvector node centrality|Dependent on exact adjacency and eigensolver conventions|
|29|rebuilt_pagerank|PageRank in that graph|Global network prominence|
|30|rebuilt_kcore_norm|Normalized core number in that graph|Network embedding depth|
|31|rebuilt_local_efficiency|Efficiency among a node's neighbors|Graph-defined local communication proxy|

`features.py` from the preceding feature-discovery experiment is the executable
source for groups7–17. Its Welch uses250 Hz,500-sample Hann windows, overlap250,
no detrending, density scaling, explicit half-open band masks and trapezoidal
integration. Groups0–6 and21–24 use the separately pinned original extractor;
their numeric definitions are not silently replaced by the candidate Welch.
Groups18–20 invert the original float32 `log1p` values before applying log.
Graph source files and metadata must match the previous Raw-rebuild gate and
join exact record/sample/time/channel/mask identities. No graph is recomputed
from test labels or a different signal source.

## Historical28: what was excluded, and why

The actual legacy window cache has13 spectral/morphological fields,7 graph
fields and8 onset/burst fields. This run scanned all1471965 valid channel-windows
per field, not padded samples. Detailed counts are in
`results/HISTORICAL28_ELIGIBILITY.csv`.

- **Delta/theta power:** both fields are exactly zero in all valid windows.
  Historical LF fidelity/provenance gates also failed. This is unavailable
  trustworthy measurement, not evidence that low-frequency physiology has no
  predictive value. The present run does not read new EDF or rescue LF.
- **Other11 spectral/morphological fields:** recomputed with pinned formulas;
  alpha is explicitly partial10–13 Hz, high gamma partial80–125 Hz. They are
  represented in the32-group pool, not excluded on a historical ablation score.
- **Seven legacy cached graph fields:** inherited source-mismatch finding
  prevents calling their cached adjacency equivalent to the raw-derived graph.
  Seven separately verified Raw-rebuilt counterparts are nevertheless admitted.
  This distinguishes source failure from a claim against network physiology.
- **Eight onset/burst fields:** `early_high_gamma_slope`,
  `early_line_length_slope`, `onset_latency_high_gamma`,
  `onset_latency_line_length`, `onset_rank_high_gamma`,
  `onset_rank_line_length`, `high_gamma_top20pct_mean`,
  `line_length_top20pct_mean` are constant across time within every one of the
  24995 record/channel units examined. They are broadcast nonlocal descriptors,
  not validated2-second-window measurements. Their source/mask replay is not
  established for this DELTA construction. Subtracting the broadcast baseline
  would erase them by construction; that would be a representation error,
  not a negative biomarker experiment. They require a separate event-level
  design and provenance audit before supervised use.

The historical88D A0 export's16 constant-zero and16 exact copied coordinates
are a separate representation audit, not32 distinct ineffective physiological
attributes. Exact duplicates and highly correlated groups must not be
conflated. This run found no exact duplicate prospective group under its
own-subFIT gate; correlations are reported without automatic pruning.

## Numerical and clinical limits

`FIT_NUMERICAL_QUALITY.csv` is a label-blind **pre-normalization summary-scale
diagnostic**: finite/constant fractions and the fraction of patient coordinate
SDs below1e-8. It is not an independent clinical calibration or a new scaler.
`FIT_REDUNDANCY.csv` reports patient-wise Spearman correlations separately for
mean/mean and SD/SD coordinates. High correlation does not imply zero conditional
information, and different log/floor transformations do not restore lost signal.

The250 Hz Nyquist limit is125 Hz. Full80–150 Hz HFO/PLHG, complete alpha,
delta/theta and source-unproven event descriptors are not claimed as tested.
Acquisition filtering, clinical onset alignment and independent surgical EZ
truth remain incompletely identifiable from these operational caches.
