"""Render the complete report and LaTeX table from verified aggregate CSVs only."""
import json
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
DATA=HERE/'results'
NAMES=['Beta power','Low-gamma power','Partial high-gamma power','RMS','Line Length','Spectral Entropy']


def main():
    assert json.loads((DATA/'STATUS.json').read_text(encoding='utf-8'))['status']=='COMPLETE_INDEPENDENTLY_VERIFIED_SIXFEATURE_EZ_ASSOCIATION'
    defs=json.loads((DATA/'feature_definitions.json').read_text(encoding='utf-8'))
    order=[r['coordinate'] for r in defs]
    rows=pd.read_csv(DATA/'feature_association.csv').set_index('coordinate').loc[order]
    sources=pd.read_csv(DATA/'source_association.csv')
    null=pd.read_csv(DATA/'permutation_control.csv').set_index('coordinate').loc[order]
    assert len(rows)==12 and len(sources)==48 and rows.valid_patients.eq(80).all()
    labels=[f'{name} — {summary}' for name in NAMES for summary in ['mean','seizure SD']]
    table=['| Coordinate | Mean D [95% CI] | Median D | Wilcoxon p | BH q | Positive % | Negative % | Zero % | Valid n |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    source_table=['| Coordinate | HUP (36) | LZU (21) | Multicenter (15) | Pediatric (8) |',
                  '|---|---:|---:|---:|---:|']
    null_table=['| Coordinate | Null mean D | Central 95% null reference interval | Supplementary permutation p |',
                '|---|---:|---:|---:|']
    latex=[r'\begin{tabular}{llrrrr}',r'\hline',
           r'Feature & Summary & Mean $D$ [95\% CI] & Median $D$ & BH $q$ & $D>0$ (\%) \\',r'\hline']
    for j,(key,r) in enumerate(rows.iterrows()):
        table.append(f'| {labels[j]} | {r.mean_effect:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}] | {r.median_effect:.3f} | {r.pvalue:.3g} | {r.BH_q:.3g} | {r.positive_percent:.2f} | {r.negative_percent:.2f} | {r.zero_percent:.2f} | {int(r.valid_patients)} |')
        s=sources[sources.coordinate==key].set_index('source')
        cells=[f'{s.loc[z,"mean_effect"]:.3f} [{s.loc[z,"ci_low"]:.3f}, {s.loc[z,"ci_high"]:.3f}]' for z in ['hup','lzu','multicenter','pediatric']]
        source_table.append('| '+labels[j]+' | '+' | '.join(cells)+' |')
        n=null.loc[key]
        null_table.append(f'| {labels[j]} | {n.null_mean:.6f} | [{n.null_ci_low:.4f}, {n.null_ci_high:.4f}] | {n.empirical_two_sided_p:.7f} |')
        mantissa,exponent=f'{r.BH_q:.2e}'.split('e')
        qtex=rf'${mantissa}\times10^{{{int(exponent)}}}$'
        latex.append(f'{NAMES[j//2]} & '+('Mean' if j%2==0 else 'Seizure SD')+f' & {r.mean_effect:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}] & {r.median_effect:.3f} & '+qtex+f' & {r.positive_percent:.2f}'+r' \\')
    latex += [r'\hline',r'\end{tabular}',
        '% n=80 patients for every coordinate. Mean/SD summarize baseline-subtracted seizure-record summaries.',
        '% Nominal percentile 95% patient-bootstrap CI, 10,000 seed42 draws; BH family=12 two-sided Wilcoxon tests.',
        '% Use table* in a two-column manuscript; do not remove the historical-selection qualification.']
    (HERE/'results'/'feature_association_table.tex').write_text('\n'.join(latex)+'\n',encoding='utf-8')
    text="""# Task A results: physiological feature association with EZ/NEZ

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

{{OVERALL}}

Machine-readable full precision: [feature_association.csv](results/feature_association.csv)
and [statistical_tests.csv](results/statistical_tests.csv).
The [LaTeX table](results/feature_association_table.tex) uses the same verified
numbers; retain this caption and the exploratory interpretation when reusing it.

## Table S1: four-source descriptions

Values are patient-average D [nominal 95% within-source patient-bootstrap CI].
The actual patient counts are retained; the primary estimate above weights
patients equally, not sources equally. Source intervals are descriptive and
are not separately multiplicity-adjusted significance claims.

{{SOURCE}}

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

{{NULL}}

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
"""
    text=text.replace('{{OVERALL}}','\n'.join(table)).replace('{{SOURCE}}','\n'.join(source_table)).replace('{{NULL}}','\n'.join(null_table))
    (HERE/'RESULTS.md').write_text(text,encoding='utf-8')
    print('VERIFIED_AGGREGATE_REPORT_COMPLETE 12 primary,48 source,12 null rows; no individual data')


if __name__=='__main__':main()
