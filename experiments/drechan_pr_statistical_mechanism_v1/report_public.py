"""Generate complete human-readable reports from authenticated aggregate tables."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
NAMES={'G0_GLOBAL':'Global','G1_CENTER':'Center only','G2_SCALE':'Scale only','G3_FULL':'Full PR',
       'DONOR_FULL':'Matched donor','OP_GLOBAL':'Operational global','OP_PR':'Operational PR'}


def load(n):return pd.read_csv(HERE/'results'/(n+'.csv'),float_precision='round_trip')


def main(df):return df[(df.estimator=='DIRECT')&np.isclose(df.factor,1e-8,rtol=1e-12,atol=0)]


def f(v):
    if pd.isna(v):return '—'
    if abs(v)<1e-10:return '0 (numerical)'
    return f'{v:.5f}'


def ci(r,adj=False):
    a='adjusted_ci_low' if adj else 'ci_low';b='adjusted_ci_high' if adj else 'ci_high'
    get=lambda k:r[k] if isinstance(r,pd.Series) else getattr(r,k)
    if pd.isna(get(a)):return '—'
    return '['+f(get(a))+', '+f(get(b))+']'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'


def run():
    manifest=json.loads((HERE/'results/PACKAGE_MANIFEST.json').read_text())
    for n,h in manifest['files'].items():assert hashlib.sha256((HERE/'results'/n).read_bytes()).hexdigest()==h
    t=load('transferable_snr');a=load('representation_alignment');b=load('boundary_drift')
    p=load('paired_intervals');fac=load('factorial_contrasts');src=load('source_results')
    n=load('negative_control_summary');donor=load('support_and_donor_audit')
    privacy=json.loads((HERE/'results/PUBLIC_PRIVACY_REVIEW.json').read_text())
    assert privacy['strict79_qualitative_conclusion']=='NO_SUPPORTED_IMPROVEMENT_IN_TRANSFERABLE_T'
    tp=main(t);ap=main(a);bp=main(b);pp=main(p)
    out=['# Results: patient-relative statistical mechanism\n',
'''## Main result

The primary endpoint does not support improved transferable standardized separation.
Full PR minus Global T is **−0.06909**, 95% CI **[−0.13636, 0.00063]**;
median paired change is −0.06911 and 29/80 patients (36.25%) improve. This interval
includes zero. There is no supported improvement, or statistically resolved
decrease, in the80-person primary analysis.

Direct mean-difference direction alignment increases by 0.07922 (nominal95% CI
[0.04169, 0.11778]; additional25-slot family interval[0.02230, 0.14198]).
Centering reduces midpoint displacement without changing T or alignment, as required
by the algebra. Fisher sensitivity reverses the alignment effect. Overall this is
**Outcome D: mixed, estimator-dependent effects**, not a uniformly beneficial PR mechanism.

## Cohort, partition and uncertainty

All80 patients receive one held-out assessment: HUP36, LZU21, multicenter15,
pediatric8; 7,635 channels, 1,743 EZ and5,892 NEZ,256 seizures. The12 active
original R14 DELTA coordinates are unchanged. No classifier or neural fits were run.

The old internal split assessed33 unique patients across40 appearances. It could
not be presented as80 evaluations. A label-free assignment was frozen before any
new endpoints. Metadata admission found one pediatric patient never in original
FIT. Consequently79 assessments lie within original FIT and one explicit exception
uses its original-validation path, never its original-test path. All80 are absent
from their corresponding TRAIN/CAL; TRAIN uses only original FIT and CAL is unused.
This is not an80-person pure original-FIT-internal evaluation. A predeclared strict79
sensitivity is reported below without replacing the primary80.

Intervals use10,000 seed42 paired unique-patient bootstrap draws, conditional on
the fixed cohort, overlapping TRAIN paths and frozen shared directions. They do not
include direction-estimation/retraining uncertainty. Absolute/source intervals
are nominal and descriptive. The secondary family contains25 declared slots,
including conservative algebraic aliases; adjusted percentile intervals use
coverage1−0.05/25. Extreme tails contain about10 resamples and have limited
Monte Carlo precision. No p-values are assigned to mathematical identities.

## Primary representation arms

T is signed standardized EZ-minus-NEZ separation; A is cosine alignment. MAE is
mean absolute class-midpoint displacement in the prespecified TRAIN-noise units;
SD is the sample SD of signed midpoint displacements. Smaller MAE/SD is better.
All means weight patients equally, not channels. Intervals below are nominal95%.
''']
    rows=[]
    for arm,name in NAMES.items():
        rt=tp[tp.arm==arm].iloc[0];ra=ap[ap.arm==arm].iloc[0];rb=bp[bp.arm==arm].iloc[0]
        rows.append([name,f(rt['mean'])+' '+ci(rt),f(ra['mean'])+' '+ci(ra),
            f(rb['mean'])+' '+ci(rb),f(rb.SD)+f' [{rb.SD_ci_low:.5f}, {rb.SD_ci_high:.5f}]'])
    out.append(table(['Representation','T','A','Midpoint MAE','Midpoint SD'],rows))
    out.append('''
Operational PR reproduces the actual14D Feature PR pipeline, including its own
legal TRAIN-only final imputer/scaler and FP32 rounding, then selects the12active
coordinates. It is separate from the common-coordinate2×2 factorial. Its T change
is −0.06932[−0.13608,0.00038], yielding the same conclusion for T.

## All primary-estimator paired comparisons

For T/A, effect=target−control. For MAE/SD, effect=control−target, so positive
means less drift. Identity rows are floating-point zero, not hypothesis tests.
SD contrasts do not have individual paired medians or positive-patient fractions.
''')
    out.append(table(['Target / control','Endpoint','Effect','95% CI','25-slot interval','Median paired','Positive patients'],
        [[NAMES[r.target]+' / '+NAMES[r.control],r.metric,f(r.effect),ci(r),ci(r,True),f(r.median_paired),
          '—' if pd.isna(r.positive_fraction) else f'{100*r.positive_fraction:.2f}%'] for r in pp.itertuples()]))
    out.append('''
## Complete2×2 factorial decomposition

Main effects average the two corresponding simple effects; interaction is
G3−G2−G1+G0, with drift signs reversed so positive means reduction. T/A scaling
effects alias already reported pairs and are not additional independent hypotheses.
The large drift interaction is dominated by the failure of scaling without
centering: Scale-only MAE=19.20898 and SD=39.24242 versus Global1.32975/1.61097.
It is not evidence of a new physiological interaction. Full PR adds only0.03357
MAE reduction over center-only; its adjusted interval[−0.05975,0.13668] includes zero.
''')
    out.append(table(['Component','Endpoint','Effect','95% CI','25-slot interval','Identity'],
        [[r.component,r.metric,f(r.effect),ci(r),ci(r,True),str(r.mathematical_identity)] for r in main(fac).itertuples()]))
    out.append('''
## Four-source descriptive results

Each source retains its actual patient count. These are not four independent
hospital-validation studies. Full-minus-Global T is negative in every source;
direct A increases in every source. Source differences are descriptive, not
multiplicity-adjusted subgroup discoveries. Intervals apply to each arm, not to
the difference of two arm means.
''')
    ss=main(src);rows=[]
    for source in ['hup','lzu','multicenter','pediatric']:
        g=ss[(ss.source==source)&(ss.arm=='G0_GLOBAL')].iloc[0]
        q=ss[(ss.source==source)&(ss.arm=='G3_FULL')].iloc[0]
        rows.append([source,int(g.patients),f(g.T_mean),f(q.T_mean),f(q.T_mean-g.T_mean),
                     f(g.A_mean),f(q.A_mean),f(q.A_mean-g.A_mean),f(q.MAE_mean)])
    out.append(table(['Source','n','Global T','Full T','ΔT','Global A','Full A','ΔA','Full MAE'],rows))
    out.append('\nAll seven arms, including center-only, scale-only, donor and operational probes:\n')
    out.append(table(['Source','n','Arm','T [95% CI]','A [95% CI]','MAE [95% CI]','SD [95% CI]'],
        [[r.source,int(r.patients),NAMES[r.arm],f(r.T_mean)+f' [{r.T_ci_low:.5f}, {r.T_ci_high:.5f}]',
          f(r.A_mean)+f' [{r.A_ci_low:.5f}, {r.A_ci_high:.5f}]',
          f(r.MAE_mean)+f' [{r.MAE_ci_low:.5f}, {r.MAE_ci_high:.5f}]',
          f(r.SD)+f' [{r.SD_ci_low:.5f}, {r.SD_ci_high:.5f}]'] for r in ss.itertuples()]))
    out.append('''
## Negative controls

All100 prespecified within-patient TRAIN-label permutations were retained.
Representations remain fixed; each repeat reconstructs TRAIN directions,
covariances, references and noise scales. True ASSESS labels are used for C2.
The intervals below are central95% reference ranges across100 repeats, NOT
patient-bootstrap CIs or exact population-level permutation tests.

Observed Global T=0.94424 exceeds its shuffled-TRAIN reference upper endpoint0.61807;
Full T=0.87515 likewise exceeds0.63883. Both observed A values exceed their null
reference ranges. This supports a label-dependent transferable direction within
this historical probe, but does not show a benefit of PR over Global.
''')
    nn=n[(n.estimator=='DIRECT')&(n.diagnostic=='TRUE_ASSESS')&(n.control=='ABSOLUTE')]
    out.append(table(['Arm','Endpoint','Observed','Null mean','Null sample SD','Null95% reference'],
        [[NAMES[r.target],r.metric,f(r.observed),f(r.null_mean),f(r.null_sample_SD),
          f'[{r.null_reference_low:.5f}, {r.null_reference_high:.5f}]'] for r in nn.itertuples()]))
    out.append('''
The separate MATCHED_H3_LABEL_NULL also shuffles ASSESS labels within patients,
preserving prevalence, solely for its diagnostic class midpoint. It is not a
predictive evaluation. Generic centering still reduces MAE under shuffled labels.
For center-only, observed MAE reduction1.03174 exceeds the matched-null central
range[0.28400,0.69397]; Full1.06530 exceeds[0.23237,0.67875]. Center-only SD
reduction1.14969 remains inside[0.43415,1.24332], whereas Full1.24615 lies only
slightly above[0.38996,1.22552]. These reference comparisons are not adjusted tests
of an observed-minus-null population effect. They do not establish a clinical threshold.

For a patient with EZ fraction π and projected all-channel mean m, the class
midpoint satisfies b=m+(0.5−π)(mu_EZ−mu_NEZ). Removing m therefore changes b even
without creating EZ-specific ranking information. This identity explains why
the generic label-null centering effect matters.
''')
    nn=n[(n.estimator=='DIRECT')&(n.diagnostic=='MATCHED_H3_LABEL_NULL')]
    out.append(table(['Target / control','Endpoint','Observed','Null mean','Null95% reference'],
        [[NAMES[r.target]+' / '+('Absolute' if r.control=='ABSOLUTE' else NAMES[r.control]),r.metric,
          f(r.observed),f(r.null_mean),f'[{r.null_reference_low:.5f}, {r.null_reference_high:.5f}]'] for r in nn.itertuples()]))
    out.append('''
## Correct versus matched-donor reference

All80 donor pairs match source and74/80 match the floor-log2 channel-count bin.
Correct-reference Full PR exceeds donor T by0.44513 (adjusted interval
[0.16912,0.75817]) and A by0.42862[0.24338,0.61390]. Drift also improves.
However, mismatching causes severe input-variance distortion: fold-average donor
variance ranges from149.91 to618,239.51 versus correct-reference0.90625–1.0.
This control supports operational reference matching under this construction;
it cannot distinguish physiological specificity from artificial distribution shift.
The donors are source/channel-count matched, not variance-distribution matched.
''')
    out.append(table(list(donor.columns),[[f(v) if isinstance(v,float) else v for v in row] for row in donor.itertuples(index=False,name=None)]))
    out.append('''
## Direction-estimator and epsilon sensitivities

The Fisher estimator does not support improved T either: Full−Global0.03595,
nominal95% CI[−0.04665,0.12386]. Its alignment changes by−0.11456
[−0.16547,−0.06170], opposite to the primary direct estimator. Thus angular
alignment improvement is probe-dependent, not a universal geometric property.
Changing the prespecified epsilon factor to1e−6 or1e−10 leaves the conclusions
unchanged. Fisher results do not replace the primary direct result.
''')
    selected=p[(p.target=='G3_FULL')&(p.control=='G0_GLOBAL')]
    out.append(table(['Estimator','Epsilon factor','Endpoint','Full−Global effect (drift reduction)','95% CI'],
        [[r.estimator,f'{r.factor:.0e}',r.metric,f(r.effect),ci(r)] for r in selected.itertuples()]))
    out.append('\nAll absolute arms for both estimators and all three numerical factors:\n')
    rows=[]
    for rt in t.itertuples():
        qa=a[(a.estimator==rt.estimator)&(a.factor==rt.factor)&(a.arm==rt.arm)].iloc[0]
        qb=b[(b.estimator==rt.estimator)&(b.factor==rt.factor)&(b.arm==rt.arm)].iloc[0]
        rows.append([rt.estimator,f'{rt.factor:.0e}',NAMES[rt.arm],f(rt.mean)+' '+ci(rt),
            f(qa['mean'])+' '+ci(qa),f(qb['mean'])+' '+ci(qb),f(qb.SD)+f' [{qb.SD_ci_low:.5f}, {qb.SD_ci_high:.5f}]'])
    out.append(table(['Estimator','Factor','Arm','T [95% CI]','A [95% CI]','MAE [95% CI]','SD [95% CI]'],rows))
    out.append('''
## Prespecified strict79 sensitivity

The never-original-FIT validation-only exception remains in the primary80 analysis.
The prespecified79-person sensitivity was fully computed and independently checked
without refitting or changing directions. It does not support improved transferable
T and does not replace the primary80 inference. Its32 exact statistics/intervals
remain trusted-side: combining exact80 and79 means reveals the omitted patient's
endpoint by subtraction. Publishing those complementary aggregates would violate
the individual-statistics privacy boundary. The public privacy review commits to
the verified private table and retains this qualitative negative conclusion.
''')
    out.append('''
## Identities, numerical verification and complete files

G0/G1 and G2/G3 have equal class differences, directions, T and A within floating
point error. This is an algebraic consequence, not a zero-effect discovery.
Single-coordinate rank-biserial effects remain unchanged because their AUC is
unchanged under positive affine transformations. Supported Full PR is likewise
affine-invariant; admission maximum error3.53×10⁻11, no support crossings.
All80 patients and all12 coordinates remain;42/960 patient-coordinate supports
are absent (seven single-seizure patients, six SD coordinates each), with the
prespecified safe-scale fallback. These patients are not dropped.

Independent arithmetic reconstructs7,070 TRAIN directions and179,360 patient-direction
rows, covering717,440 core endpoint values. All1,112 nominal interval checks and
25 additional-family interval checks pass; maximum endpoint discrepancy
1.14284×10⁻10. Nine synthetic/isolation tests also pass under both local and
trusted-server runtimes. No new classifier metrics were calculated.

The18 privacy-filtered aggregate/verification payloads plus manifest are in `results/`.
`paired_intervals.csv` contains all168 paired rows; `factorial_contrasts.csv`
contains all72 factorial rows, including sensitivities and identities.
`negative_controls.csv` contains every one of the2,200 null-aggregate rows;
`negative_control_summary.csv` contains all136 summaries, including Fisher.
`source_results.csv` contains all168 source/arm/estimator/factor rows.
`patient_distribution_bins.csv` publishes only fixed anonymous bins, not individual
effects. Exact individual results, the complementary79-person table and runtime
logs remain on the trusted server. The publication-only privacy filter changes no
scientific result, endpoint, direction, source hash or primary80 analysis.

The cohort and feature set were used in historical model development. Findings
are exploratory associations conditional on this protocol, not external
validation, clinical causality, a proof of method novelty or PR indispensability.
''')
    text='\n'.join(out)
    spacing={'All80':'All 80','all80':'all 80','All100':'All 100','all100':'all 100',
        'All7':'All 7','all7':'all 7','all12':'all 12','all three':'all three',
        'The12':'The 12','the12':'the 12','12active':'12 active','actual14D':'actual 14D',
        'and5,892':'and 5,892','NEZ,256':'NEZ, 256','assessed33':'assessed 33','across40':'across 40',
        'Consequently79':'Consequently 79','an80-person':'an 80-person','strict79':'strict 79',
        'primary80':'80-person primary','the80-person':'the 80-person',
        'nominal95%':'nominal 95%','additional25-slot':'additional 25-slot','family interval[':'family interval [',
        'use10,000':'use 10,000','contains25':'contains 25','about10':'about 10',
        'common-coordinate2×2':'common-coordinate 2×2','Complete2×2':'Complete 2×2','complete2×2':'complete 2×2',
        'contains all168':'contains all 168','contains all72':'contains all 72','contains all136':'contains all 136',
        'contains all158':'contains all 158','contains all179':'contains all 179','contains all42':'contains all 42',
        'all168':'all 168','all72':'all 72','all136':'all 136','the2,200':'the 2,200',
        'The18':'The 18','All1,112':'All 1,112','and717,440':'and 717,440','reconstructs7,070':'reconstructs 7,070',
        'and179,360':'and 179,360','variance ranges from149.91':'variance ranges from 149.91',
        'Global1.32975':'Global 1.32975','Full1.06530':'Full 1.06530','reduction1.03174':'reduction 1.03174',
        'reduction1.14969':'reduction 1.14969','Full1.24615':'Full 1.24615','donor T by0.44513':'donor T by 0.44513',
        'and A by0.42862':'and A by 0.42862','correct-reference0.90625':'correct-reference 0.90625',
        'reduction over center-only;':'reduction over center-only;', 'benefit0.03357':'benefit 0.03357',
        'reduction1.06530':'reduction 1.06530','additional reduction':'additional reduction',
        'Full−Global0.03595':'Full−Global 0.03595','coordinate supports':'coordinate supports'}
    for old,new in spacing.items():text=text.replace(old,new)
    (HERE/'RESULTS.md').write_text(text,encoding='utf-8',newline='\n')


if __name__=='__main__':run()
