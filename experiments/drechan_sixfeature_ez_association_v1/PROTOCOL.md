# Task A protocol: six-feature EZ/NEZ association

Fixed before computing new80-patient association results. No classifier is fitted,
no neural model is imported, and no waveform/feature extraction is performed.
The six-feature set and this historical cohort were selected/viewed during model
development. This is exploratory, descriptive association, not external validation.

## Exact measurement and aggregation

Use the current model's frozen R14, not the engineered E14 or patient-Z/FeaturePR.
Indices0–7 and10–13 retain exactly12 active coordinates; Variance indices8/9 excluded.
Source audit pins legacy spectral kernel, raw reconstruction and aggregation hashes.
Raw cache windows have500samples at250Hz. Welch uses nperseg500,noverlap250,
detrend=False and the legacy default window. Actual spectral bins are spaced0.5Hz.
Beta is13–29.5Hz; lowgamma30–79.5Hz; the old80–150Hz mask has only80–125Hz
support at this sampling rate and includes the Nyquist bin. Never call it full HFO.
Band power uses log1p of trapezoidal PSD integration. RMS retains sqrt(mean(x²)+1e-8);
Line Length is sum(abs(diff(x)))/2seconds; entropy retains the original10–125Hz
PSD normalization and1e-8 constants. No definitions, floors, units or bands change.

For each channel and seizure, subtract the mean feature over that channel's valid
negative-time windows. Average DELTA over ALL valid seizure-record windows, including
pre and post windows, then compute mean and populationSD(ddof=0) across finite
available seizures. The observed valid-window centers span approximately-29 to+29s
(small original timestamp deviations retained); this is NOT a new0–20s feature.
No waveform alignment or temporal segmentation is recomputed. Seven patients have
one seizure: their across-seizure SD coordinates are degenerate zeros and remain
in the primary analysis. Seizure counts are not treated as independent patients.

The kernel uses the original raw-cache amplitude units without V/µV conversion.
The frozen summaries do not certify a harmonized physical unit across all sources;
do not relabel native values as measured µV. log1p and RMS/entropy numerical floors
are unit-sensitive; they are preserved exactly, not assumed negligible.
Feature-relative ranking is computed within each patient on these original values,
not by comparing magnitudes across patients. Upstream safe-float/nan_to_num behavior
belongs to the frozen representation; no additional imputation is introduced.

## Statistical analysis

Archived labels are NEZ=1,EZ=0; convert EZ=(y==0). Verify the exact7635 patient/channel
alignment against the label archive before accessing labels. For each patient and
coordinate, finite EZ and NEZ channels produce D=2AUC-1 using average tied ranks
(pairwise ties0.5). Missing class gives NA, not zero. Constant values give D0.
Keep raw signs; positive means larger DELTA-summary measurements in EZ.

Average patients equally. Report mean,median,10000seed42 patient-bootstrap percentile
95%CI, positive/negative/zero percentages, finite n and constant-patient counts.
Wilcoxon is two-sided,zero_method=wilcox,method=approx,continuity correction=True,
explicit tie-corrected normal approximation; all-zero vectors receive p1.
This fixes behavior across SciPy versions and does not mislabel tied data as exact.
The p-value tests signed location under its assumptions; it is not a direct test
of the arithmetic mean. Apply BH to exactly12 primary p-values. CIs are nominal,
not simultaneous. No feature is removed or sign-flipped based on these results.

Four-source descriptions retain actual n36/21/15/8. Source-specific means,CIs and
direction percentages are descriptive, with no new source-wise significance family.
Direction consistency is reported quantitatively; no all-six-significant requirement.

Negative control:10000 joint within-patient label permutations, preserving each
patient's EZ count and using the same permutation across all12 coordinates.
Report null mean,SD,central95%reference interval,Monte Carlo SE and supplementary
two-sided permutation p-values (numerical equality tolerance1e-14).
Do not rerun seeds to make controls or associations pass. Null intervals are not
patient-bootstrap uncertainty intervals. They test this statistical pipeline,
not the validity of clinical EZ labels.

## Prior work, independence and privacy

The old atlas's12new endpoints are slope/crossing/top20 descriptors in51 original
FIT patients, not these six-feature12D coordinates. Its old160window/phase family
and overlapping five FIT cohorts cannot replace a new80-patient12-test family.
Reuse only the immutable summary cache, mapping, label convention and reviewed
rank/BH/bootstrap logic. Old reports/data/models remain unmodified.

No univariate association establishes conditional independence, added predictive
utility, indispensability, a causal EZ mechanism, or reproducibility in new hospitals.
Across-seizure SD reflects variability, not evidence of repeated-seizure reliability.

Individual960patient-coordinate rows remain private on the trusted server.
Public patient_effect_summary.csv contains12coordinate-level quantile/count rows,
and a40bin histogram supports all-patient distribution plots without exact
individual effects or stable patient identifiers. Only explicit anonymous
aggregate allowlists leave the server. Execution logs remain private.

Independent verification recomputes all960 effects by direct pairwise signs and
sklearn ROC AUC, all60 mean-bootstrap intervals by count matrices, all12
Wilcoxon tests by tie-corrected variance, BH by a separate SciPy implementation,
and all9.6million permuted patient-coordinate effects by pairwise dominance scores.

Method references: [SciPy Wilcoxon](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats.wilcoxon.html)
and [Mann–Whitney U](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html).

## Figure contract

Quantitative grid; Python/matplotlib only. Figure A shows ALL12 signed patient
mean effects with nominal95%CI and BH q, without predetermined signs or significance.
English text,editable SVG/PDF,178mm double-column width,readable7–9pt labels,600dpiPNG.
Supplementary panels show all12 binned patient-effect distributions and all4
source-specific mean/CIs. Mean/SD are distinguished by marker shape and restrained
color,not color alone. Caption includes patient n,zero handling,CI method and FDR family.
