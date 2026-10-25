# Task A results: physiological feature association with EZ/NEZ

## Conclusion

All 12 fixed coordinates showed positive patient-average signed rank-biserial
effects and BH q < 0.05 in this 80-patient historical cohort. All 12 nominal
patient-bootstrap mean intervals excluded zero. There were **no nonsignificant
coordinates in the prespecified overall 12-test family**; none was dropped,
reoriented or substituted. This is univariate association, not evidence that
each feature provides independent conditional information to a classifier.

Line Length mean had the largest observed mean effect: D = 0.504
[0.424, 0.583], q = 1.34e-11; 69/80 patients (86.25%) had D > 0.
Beta mean, partial high-gamma mean, low-gamma mean and entropy mean followed
with D = 0.451, 0.434, 0.422 and 0.420. These are descriptive effect rankings,
not statistical tests of one feature outperforming another.

Within-patient directions were not universal: positive effects occurred in
81.25–88.75% of patients for the six mean coordinates and 71.25–77.50% for SD
coordinates. Every coordinate also had patients with a negative effect.
Ten of 12 coordinates had positive mean effects in all four sources.
Pediatric RMS mean and partial high-gamma SD had negative source-average
effects; their intervals included zero. Do not describe all six features as
uniformly reproducible across patients or sources.

## Table A: complete overall results

D = 2 AUC(EZ versus NEZ) − 1, with tied pairs worth 0.5 and raw directions
retained. Positive D means a higher **baseline-subtracted summary**, not
necessarily higher absolute physiological amplitude or power, in EZ channels.
Each of the 80 patients contributes one effect per coordinate; channels and
seizures are not independent replicates. All intervals below are nominal 95%
percentile intervals from 10,000 seed42 patient bootstrap draws. Wilcoxon
tests are two-sided, zero-dropping, continuity-corrected, tie-corrected normal
approximations; they test signed location, not directly the arithmetic mean.
BH correction uses exactly 12 tests. Percentages include zero-effect patients.

| Coordinate | Mean D [95% CI] | Median D | Wilcoxon p | BH q | Positive % | Negative % | Zero % | Valid n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Beta power — mean | 0.451 [0.372, 0.526] | 0.526 | 3.14e-12 | 1.88e-11 | 88.75 | 11.25 | 0.00 | 80 |
| Beta power — seizure SD | 0.358 [0.273, 0.440] | 0.419 | 2.8e-09 | 4.2e-09 | 77.50 | 13.75 | 8.75 | 80 |
| Low-gamma power — mean | 0.422 [0.329, 0.511] | 0.517 | 2.8e-10 | 6.72e-10 | 83.75 | 16.25 | 0.00 | 80 |
| Low-gamma power — seizure SD | 0.325 [0.244, 0.404] | 0.328 | 1.48e-09 | 2.96e-09 | 76.25 | 13.75 | 10.00 | 80 |
| Partial high-gamma power — mean | 0.434 [0.356, 0.513] | 0.490 | 7.81e-12 | 3.13e-11 | 87.50 | 12.50 | 0.00 | 80 |
| Partial high-gamma power — seizure SD | 0.223 [0.137, 0.308] | 0.217 | 2.89e-06 | 2.89e-06 | 71.25 | 20.00 | 8.75 | 80 |
| RMS — mean | 0.327 [0.235, 0.415] | 0.389 | 1.16e-08 | 1.55e-08 | 81.25 | 18.75 | 0.00 | 80 |
| RMS — seizure SD | 0.275 [0.190, 0.358] | 0.250 | 1.67e-07 | 1.82e-07 | 71.25 | 18.75 | 10.00 | 80 |
| Line Length — mean | 0.504 [0.424, 0.583] | 0.546 | 1.12e-12 | 1.34e-11 | 86.25 | 13.75 | 0.00 | 80 |
| Line Length — seizure SD | 0.331 [0.248, 0.410] | 0.331 | 1.93e-09 | 3.31e-09 | 77.50 | 13.75 | 8.75 | 80 |
| Spectral Entropy — mean | 0.420 [0.332, 0.504] | 0.463 | 9.33e-11 | 2.8e-10 | 85.00 | 15.00 | 0.00 | 80 |
| Spectral Entropy — seizure SD | 0.307 [0.220, 0.394] | 0.308 | 2.91e-08 | 3.49e-08 | 75.00 | 16.25 | 8.75 | 80 |

Machine-readable full precision: [feature_association.csv](results/feature_association.csv)
and [statistical_tests.csv](results/statistical_tests.csv).
The [LaTeX table](results/feature_association_table.tex) uses the same verified
numbers; retain this caption and the exploratory interpretation when reusing it.

## Table S1: four-source descriptions

Values are patient-average D [nominal 95% within-source patient-bootstrap CI].
The actual patient counts are retained; the primary estimate above weights
patients equally, not sources equally. Source intervals are descriptive and
are not separately multiplicity-adjusted significance claims.

| Coordinate | HUP (36) | LZU (21) | Multicenter (15) | Pediatric (8) |
|---|---:|---:|---:|---:|
| Beta power — mean | 0.506 [0.396, 0.614] | 0.269 [0.144, 0.388] | 0.577 [0.419, 0.729] | 0.450 [0.067, 0.753] |
| Beta power — seizure SD | 0.411 [0.273, 0.541] | 0.257 [0.137, 0.373] | 0.401 [0.213, 0.581] | 0.307 [-0.059, 0.634] |
| Low-gamma power — mean | 0.513 [0.406, 0.613] | 0.304 [0.151, 0.442] | 0.459 [0.188, 0.685] | 0.256 [-0.197, 0.678] |
| Low-gamma power — seizure SD | 0.320 [0.203, 0.435] | 0.282 [0.165, 0.395] | 0.429 [0.275, 0.586] | 0.267 [-0.163, 0.664] |
| Partial high-gamma power — mean | 0.521 [0.422, 0.615] | 0.247 [0.106, 0.377] | 0.466 [0.238, 0.684] | 0.472 [0.154, 0.758] |
| Partial high-gamma power — seizure SD | 0.282 [0.165, 0.395] | 0.204 [0.071, 0.328] | 0.247 [0.082, 0.420] | -0.041 [-0.455, 0.404] |
| RMS — mean | 0.535 [0.438, 0.627] | 0.139 [0.007, 0.260] | 0.312 [0.078, 0.537] | -0.087 [-0.405, 0.204] |
| RMS — seizure SD | 0.328 [0.183, 0.469] | 0.216 [0.107, 0.326] | 0.327 [0.174, 0.485] | 0.089 [-0.217, 0.418] |
| Line Length — mean | 0.622 [0.533, 0.704] | 0.264 [0.126, 0.388] | 0.644 [0.494, 0.779] | 0.344 [-0.074, 0.739] |
| Line Length — seizure SD | 0.397 [0.280, 0.513] | 0.224 [0.111, 0.333] | 0.368 [0.192, 0.546] | 0.250 [-0.154, 0.635] |
| Spectral Entropy — mean | 0.519 [0.399, 0.627] | 0.191 [0.074, 0.301] | 0.495 [0.259, 0.698] | 0.439 [0.036, 0.786] |
| Spectral Entropy — seizure SD | 0.382 [0.235, 0.520] | 0.142 [0.006, 0.276] | 0.404 [0.261, 0.556] | 0.218 [-0.020, 0.495] |

All 12 source-mean effects were positive in HUP, LZU and multicenter. In
pediatric, 10/12 means were positive, but only beta mean, partial high-gamma
mean and entropy mean had nominal intervals wholly above zero (3/12).
The remaining nine pediatric intervals included zero. Pediatric RMS mean
was −0.087 [−0.405, 0.204], and partial high-gamma SD was −0.041
[−0.455, 0.404]. Pediatric RMS SD had a positive mean (0.089) but negative
median (−0.058), illustrating skew and patient heterogeneity. The n = 8
pediatric sample does not establish equivalence or absence of association.

The full 48-row [source table](results/source_association.csv) also gives
medians, direction percentages and nonzero-patient counts.

## Table S2: within-patient label-permutation negative control

For each of 10,000 seed42 permutations, shuffle EZ/NEZ labels within each
patient, preserve that patient's EZ count and use the same label shuffle for
all 12 coordinates. The null means ranged from −0.000181 to 0.000387;
all 12 were within four Monte Carlo standard errors of zero. The null
reference intervals were approximately ±0.04, unlike the observed positive
effects. All observed absolute means exceeded every corresponding sampled
null absolute mean; the plus-one two-sided empirical p was 1/10001 =
0.00009999 for each coordinate, the finite Monte Carlo resolution, not zero.
These supplemental p-values do not replace or expand the primary Wilcoxon/BH
family. Null reference intervals are not bootstrap confidence intervals.

| Coordinate | Null mean D | Central 95% null reference interval | Supplementary permutation p |
|---|---:|---:|---:|
| Beta power — mean | 0.000043 | [-0.0419, 0.0421] | 0.0001000 |
| Beta power — seizure SD | 0.000072 | [-0.0393, 0.0389] | 0.0001000 |
| Low-gamma power — mean | 0.000038 | [-0.0421, 0.0425] | 0.0001000 |
| Low-gamma power — seizure SD | -0.000181 | [-0.0395, 0.0392] | 0.0001000 |
| Partial high-gamma power — mean | 0.000387 | [-0.0421, 0.0415] | 0.0001000 |
| Partial high-gamma power — seizure SD | -0.000009 | [-0.0389, 0.0393] | 0.0001000 |
| RMS — mean | 0.000010 | [-0.0421, 0.0419] | 0.0001000 |
| RMS — seizure SD | -0.000011 | [-0.0394, 0.0400] | 0.0001000 |
| Line Length — mean | 0.000031 | [-0.0415, 0.0416] | 0.0001000 |
| Line Length — seizure SD | 0.000027 | [-0.0389, 0.0403] | 0.0001000 |
| Spectral Entropy — mean | 0.000121 | [-0.0415, 0.0424] | 0.0001000 |
| Spectral Entropy — seizure SD | -0.000040 | [-0.0391, 0.0387] | 0.0001000 |

Full precision and Monte Carlo errors: [permutation_control.csv](results/permutation_control.csv).
This control validates chance-level behavior of the calculation; it does not
validate the biological correctness of clinical EZ labels.

## Measurements and coverage

- 80/80 patients; 7,635 channels (1,743 EZ, 5,892 NEZ); 256 seizures;
  1,471,965 valid channel-windows; all original sources and masks retained.
- All 91,620 selected channel-coordinate values and all 960 patient effects
  were finite. No new imputation, patient exclusion or feature extraction.
- Input was the exact frozen float32 R14, before Feature PR/patient-relative
  standardization, selecting indices 0–7 and 10–13. All 80 per-patient summary
  caches matched this input byte-exactly. Variance was not analyzed.
- Each coordinate is computed from the original DELTA windows: same-channel,
  same-record valid pre-onset mean is subtracted; average across **all** valid
  record windows, including pre and post; then across-seizure mean or population
  SD (ddof = 0). Valid window centers span −29.000954 to +29.000938 seconds;
  724,672 pre-onset and 747,293 nonnegative-time valid channel-windows.
  These are not newly defined 0–20 s ictal summaries.
- Seven single-seizure patients have constant zero across-seizure SD coordinates
  and remain included as zero effects. There were seven zero effects in beta,
  partial high-gamma, Line Length and entropy SD, and eight in low-gamma and
  RMS SD. The additional zero effects need not mean constant channel values.
  Wilcoxon excludes zero effects; the mean, bootstrap and percentages retain them.
- Beta uses 13 ≤ f < 30 Hz; low gamma 30 ≤ f < 80 Hz; the legacy high-gamma
  mask has only 80–125 Hz support at 250 Hz sampling, including the Nyquist bin.
  Power is log1p integrated Welch PSD. Entropy uses the original 10–125 Hz
  kernel. RMS/entropy floors, Line Length definition and native cache units
  are unchanged. Physical V/µV harmonization across sources is not certified;
  do not attach invented µV units. See [PROTOCOL.md](PROTOCOL.md) and
  [feature definitions](results/feature_definitions.json) for exact formulas.

## Figures and direct manuscript use

- Figure A: [PDF](figures/feature_association.pdf),
  [SVG](figures/feature_association.svg), [600 dpi PNG](figures/feature_association.png).
- All-patient effect distributions: [PDF](figures/patient_effect_distributions.pdf),
  [SVG](figures/patient_effect_distributions.svg),
  [PNG](figures/patient_effect_distributions.png).
- Four-source descriptions: [PDF](figures/source_association.pdf),
  [SVG](figures/source_association.svg), [PNG](figures/source_association.png).

Figure captions are in [FIGURE_CAPTIONS.md](FIGURE_CAPTIONS.md). The figures
contain all 12 coordinates, including source reversals, not a favorable subset.

Suggested preliminary-analysis paragraph (not a claim of pre-model collection):

> We examined the six physiological features independently of fitted classifiers
> by comparing EZ and NEZ channels within each of 80 patients. For the original
> baseline-subtracted across-seizure mean and SD summaries, signed rank-biserial
> effects were computed per patient and averaged with equal patient weight.
> All 12 coordinates showed positive aggregate effects (mean D = 0.223–0.504;
> BH q ≤ 2.90 × 10⁻⁶ across 12 two-sided Wilcoxon tests), with Line Length mean
> showing the largest observed effect (D = 0.504, 95% patient-bootstrap CI
> 0.424–0.583). Nevertheless, effects varied across patients and sources,
> including reversed source-average directions for pediatric RMS mean and
> partial high-gamma SD. These findings provide exploratory within-patient
> association evidence for the historically selected summaries, not independent
> external validation, conditional feature utility or a causal EZ mechanism.

## Interpretation boundaries and numerical verification

The current features and cohort had already been used in historical model
development and viewed Outer analyses. Placement before the model section does
not make this a prospectively independent feature validation. No classifier
was trained, but feature selection was not independent of historical outcomes.
Across-patient directional prevalence is not test–retest reliability. Positive
entropy DELTA association does not establish that absolute ictal entropy is
higher, and positive SD association is not proof of seizure-to-seizure stability.
No multivariate/conditional analysis was performed; redundancy is unresolved.

Independent verification passed 960 effects (primary average ranks versus direct
pairwise signs and sklearn ROC AUC), all 60 mean-bootstrap intervals via count
matrices, all 12 Wilcoxon tests via tie-corrected variance and normal tails,
BH via an independent implementation, all 48 source rows and all 9.6 million
permuted patient-coordinate effects via pairwise dominance. Maximum observed
effect discrepancy was 1.11e-16. Source inputs remained unchanged; classifiers
trained = 0; GPU used = false. See [independent proof](results/INDEPENDENT_VERIFICATION.json),
[input audit](results/INPUT_AUDIT.json), [prior-work audit](PRIOR_WORK_AUDIT.csv)
and [execution audit](RUN_AUDIT.md).

Individual patient effects, labels, raw inputs and complete runtime logs remain
on the trusted server. Public patient_effect_summary.csv contains only 12
coordinate-level count/quantile rows, not identifiable patient rows. Distribution
figures use fixed 0.05-wide aggregate bins, not exported individual effects.
