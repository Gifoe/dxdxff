"""Full report, single-column booktabs table and bounded manuscript paragraph."""
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import core as c

HERE=Path(__file__).resolve().parent
LABELS={'BETA':'Beta','LOW_GAMMA':'Low Gamma','HIGH_GAMMA':'Partial High Gamma',
 'RMS':'RMS','LINE_LENGTH':'Line Length','SPECTRAL_ENTROPY':'Spectral Entropy','ALL_SIX':'All Six'}


def label(arm):
    pr=not arm.endswith('_NO_PR');name=arm.removesuffix('_NO_PR').removesuffix('_PR')
    return LABELS[name]+(' (+PR)' if pr else ' (w/o PR)')


def cell(row,m):return f"{row[m]:.3f} [{row[m+'_ci_low']:.3f}, {row[m+'_ci_high']:.3f}]"


def metric_table(frame,metrics):
    out=['| Input | '+' | '.join(m.replace('_','-') for m in metrics)+' |',
         '|---|'+'---:|'*len(metrics)]
    for r in frame.to_dict('records'):out.append('| '+label(r['arm'])+' | '+' | '.join(cell(r,m) for m in metrics)+' |')
    return '\n'.join(out)


def main():
    data=HERE/'results';assert c.jr(data/'STATUS.json')['status']=='COMPLETE_VERIFIED_SIXFEATURE_EZ_PREDICTION'
    main=pd.read_csv(data/'main_results.csv');allr=pd.read_csv(data/'all_results.csv');single=pd.read_csv(data/'individual_feature_results.csv')
    pair=pd.read_csv(data/'paired_comparisons.csv');pr=pd.read_csv(data/'pr_improvement.csv')
    null=pd.read_csv(data/'negative_control_summary.csv').set_index('metric')
    folds=pd.read_csv(data/'fold_results.csv');sources=pd.read_csv(data/'source_results.csv');selection=pd.read_csv(data/'model_selection.csv')
    proof=c.jr(data/'INDEPENDENT_VERIFICATION.json');internal=c.jr(data/'INTERNAL_VERIFICATION.json')
    assert main.arm.tolist()==c.MAIN and len(allr)==14 and len(single)==12 and len(pair)==56
    full=main.set_index('arm').loc['ALL_SIX_PR'];without=main.set_index('arm').loc['ALL_SIX_NO_PR']
    tables=HERE/'tables';tables.mkdir(exist_ok=True)
    tex=[r'\begin{table}[t]',r'\centering',
        r'\caption{Predictive Performance of Individual and Combined Physiological Features. Patient-average results over the original five held-out folds ($n=80$).}',
        r'\label{tab:physiological-prediction}',r'\scriptsize',r'\setlength{\tabcolsep}{2pt}',
        r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrr@{}}',r'\toprule',
        r'Input & AUROC & AP & Macro-F1 & Top-1 \\',r'\midrule']
    names=['Beta','Low Gamma','High Gamma','RMS','Line Length','Spectral Entropy','All Six (w/o PR)','All Six (+PR)']
    best={m:float(main[m].max()) for m in c.PRIMARY};best_rows={}
    for j,r in enumerate(main.to_dict('records')):
        vals=[]
        for m in c.PRIMARY:
            value=f"{r[m]:.3f}";is_best=abs(r[m]-best[m])<1e-12
            vals.append(r'\textbf{'+value+'}' if is_best else value)
            if is_best:best_rows.setdefault(m,[]).append(r['arm'])
        tex.append(names[j]+' & '+' & '.join(vals)+r' \\')
    tex += [r'\bottomrule',r'\end{tabular*}',r'\par\vspace{2pt}',
       r'\parbox{\columnwidth}{\scriptsize All rows use PR except the explicitly marked row. High Gamma is the available 80--125\,Hz band. Bold denotes the best unrounded mean in each column. Confidence intervals are reported with Fig.~B and in the supplementary tables.}',r'\end{table}']
    fragment='\n'.join(tex)+'\n';(tables/'feature_prediction_single_column.tex').write_text(fragment,encoding='utf-8')
    standalone=r'\documentclass{article}'+'\n'+r'\usepackage[paperwidth=93mm,paperheight=88mm,margin=2mm]{geometry}'+'\n'+r'\usepackage{booktabs}'+'\n'+r'\pagestyle{empty}'+'\n'+r'\begin{document}'+'\n'+fragment+r'\end{document}'+'\n'
    (tables/'feature_prediction_preview.tex').write_text(standalone,encoding='utf-8')
    compile_path=tables/'TABLE_COMPILE_AUDIT.json';render_status='PENDING_LATEX_COMPILE'
    if compile_path.exists():
        compile_proof=json.loads(compile_path.read_text(encoding='utf-8'))
        if compile_proof.get('preview_sha256')==hashlib.sha256((tables/'feature_prediction_preview.tex').read_bytes()).hexdigest():
            render_status=compile_proof['status']
    (tables/'TABLE_AUDIT.json').write_text(json.dumps(dict(rows=8,metrics=c.PRIMARY,bolded_best_rows=best_rows,
        order=c.MAIN,booktabs=True,columnwidth=True,selection_from_unrounded_values=True,render_status=render_status),indent=2)+'\n',encoding='utf-8')
    lines=['# Experiment B results: physiological feature predictive informativeness','',
      f"Completed: 14 configurations, 80 patients, 7,635 channels (1,743 EZ / 5,892 NEZ), 256 seizures. All original five-fold roles were retained. There were 1,410 unique formal LR fits: 840 inner-CV candidates, 70 selected final models and 500 models across 100 label-permutation repetitions. No historical classifier model/score was reused.",'',
      '## Main eight configurations','',
      'Values are equal-patient means [nominal 95% patient-bootstrap CI, 10,000 seed42 draws]. Every row covers all 80 patients. C was selected inside FIT by patient-held-out AUROC; CAL selected the Macro-F1 threshold. Models were not refitted on FIT+CAL.','',
      metric_table(main,c.PRIMARY),'',
      'Within the eight prespecified main rows, Full+PR has the highest AUROC/AP point estimates, but Line Length+PR has the highest Macro-F1 and Top-1. All four Full+PR-minus-Line-Length+PR intervals include zero, even before multiplicity correction. These results do not establish that the joint feature set outperforms the strongest single-feature control. Across all 14 configurations, supplementary Line Length NoPR has a numerically higher AUROC (0.760) than Full+PR (0.757); it is not substituted for a prespecified main control.','',
      'Secondary endpoints:','',metric_table(main,['EZ_F1','NEZ_F1','Accuracy','MRR']),'',
      '## Supplement: every individual feature with and without PR','',
      metric_table(single,c.PRIMARY),'',metric_table(single,['EZ_F1','NEZ_F1','Accuracy','MRR']),'',
      '## Seven prespecified Full+PR-minus-control comparisons','',
      'Differences and interval limits are absolute percentage points. Ordinary intervals are paired 95% percentile intervals. The four primary endpoints across all seven contrasts form one 28-comparison family; additional Bonferroni percentile intervals use confidence level 1−0.05/28. Secondary endpoints have nominal intervals only. The extreme adjusted limits use approximately nine bootstrap order statistics, so their Monte Carlo precision is limited. No comparator was selected after viewing these results.','',
      '| Control | Endpoint | Difference (pp) | Nominal 95% CI (pp) | 28-family interval (pp) |',
      '|---|---|---:|---:|---:|']
    for r in pair.to_dict('records'):
        adj=f"[{r['adjusted_ci_low_pp']:.2f}, {r['adjusted_ci_high_pp']:.2f}]" if r['primary_family_size']==28 else 'Not in primary family'
        lines.append(f"| {label(r['control'])} | {r['metric']} | {r['difference_pp']:+.2f} | [{r['ci_low_pp']:.2f}, {r['ci_high_pp']:.2f}] | {adj} |")
    lines += ['','## PR improvements for all seven feature sets','',
       'Values are PR minus NoPR, percentage points [nominal paired 95% CI]. Only the Full comparison on the four primary endpoints additionally belongs to the 28-comparison family above. Single-feature PR differences are descriptive. PR/NoPR use the same C grid and budget, with separately selected legal C values.','',
       '| Features | '+' | '.join(c.PRIMARY)+' |','|---|'+'---:|'*4]
    for name in c.FEATURES+['ALL_SIX']:
        rr=pr[pr.feature==name].set_index('metric')
        vals=[f"{rr.loc[m,'difference_pp']:+.2f} [{rr.loc[m,'ci_low_pp']:.2f}, {rr.loc[m,'ci_high_pp']:.2f}]" for m in c.PRIMARY]
        lines.append('| '+LABELS[name]+' | '+' | '.join(vals)+' |')
    lines += ['','## Fold-level performance','',
       'Each cell is AUROC / Macro-F1, patient means within that Outer fold. Fold sizes are 16,16,17,15,16; the overall estimate weights patients, not the five fold means equally. All eight endpoints for all 14 groups are in results/fold_results.csv.','',
       '| Input | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |','|---|---:|---:|---:|---:|---:|']
    for arm in c.MAIN:
        vals=[]
        for f in range(1,6):
            rr=folds[(folds.arm==arm)&(folds.fold==f)].set_index('metric')
            vals.append(f"{rr.loc['AUROC','mean']:.3f} / {rr.loc['Macro_F1','mean']:.3f}")
        lines.append('| '+label(arm)+' | '+' | '.join(vals)+' |')
    lines += ['','## Four-source performance','',
       'Each cell is AUROC / Macro-F1. These are descriptive source subsets of the original patient-held-out evaluation, not a new LOCO study. Sources retain HUP36,LZU21,multicenter15,pediatric8. Source-level 95% intervals and all eight endpoints are provided for every group in results/source_results.csv.','',
       '| Input | HUP (36) | LZU (21) | Multicenter (15) | Pediatric (8) |','|---|---:|---:|---:|---:|']
    for arm in c.ARMS:
        vals=[]
        for source in ['hup','lzu','multicenter','pediatric']:
            rr=sources[(sources.arm==arm)&(sources.source==source)].set_index('metric')
            vals.append(f"{rr.loc['AUROC','mean']:.3f} / {rr.loc['Macro_F1','mean']:.3f}")
        lines.append('| '+label(arm)+' | '+' | '.join(vals)+' |')
    lines += ['','## Training-label permutation negative control','',
       f"All 100 fixed full12D+PR permutations were retained. Across repetitions, patient-average AUROC was {null.loc['AUROC','null_mean']:.3f} (sample SD {null.loc['AUROC','null_sample_SD']:.3f}; central 95% reference range {null.loc['AUROC','null_reference_low']:.3f}–{null.loc['AUROC','null_reference_high']:.3f}), compared with {full.AUROC:.3f} for true-label training. The chance AUROC reference is 0.5.",'',
       f"Null AP averaged {null.loc['AP','null_mean']:.3f}; patient-equal EZ prevalence was {null.loc['AP','patient_equal_EZ_prevalence']:.3f}, not 0.5. Null Top-1 averaged {null.loc['Top_1','null_mean']:.3f}. This conditional control produces learned random linear projections of the fixed physiological features, not independent uniform channel rankings. Feature geometry and finite patient channel counts can therefore yield AP/Top-1 distributions different from a uniform-ranking prevalence reference. Do not interpret their null values as 0.5 or equate them exactly with prevalence. The full 100-repetition aggregate distribution is in results/negative_control.csv, with all eight endpoint summaries in negative_control_summary.csv.",'',
       'Only FIT labels were shuffled within each patient; EZ counts were preserved and sample weights were recomputed. Each fold used the legally selected true-FIT full+PR C and fixed label-free FIT preprocessing, with its own true-CAL threshold. Outer labels remained real. This conditional control distribution is not presented as an exact permutation-test p-value or a patient confidence interval.','',
       '## Research questions','',
       '### Individual-feature prediction','']
    for r in main[~main.arm.str.startswith('ALL_SIX')].to_dict('records'):
        verdict='above 0.5 throughout its nominal interval' if r['AUROC_ci_low']>.5 else 'not separated from 0.5 by its nominal interval'
        lines.append(f"- {label(r['arm'])}: AUROC {cell(r,'AUROC')}; {verdict}. These are descriptive absolute intervals, not six additional corrected hypothesis tests.")
    lines += ['','### Combining features','']
    for metric in c.PRIMARY:
        rr=pair[(pair.metric==metric)&(pair.control!='ALL_SIX_NO_PR')]
        higher=rr[rr.adjusted_ci_low>0];lower=rr[rr.adjusted_ci_high<0];uncertain=rr[(rr.adjusted_ci_low<=0)&(rr.adjusted_ci_high>=0)]
        lines.append(f"- {metric}: Full+PR has a positive 28-family interval versus {len(higher)}/6 single-feature controls, a negative interval versus {len(lower)}/6, and an interval crossing zero versus {len(uncertain)}/6. Positive: {', '.join(label(x) for x in higher.control) or 'none'}. Negative: {', '.join(label(x) for x in lower.control) or 'none'}.")
    lines += ['','### Patient-relative normalization','',
       'PR increases Macro-F1 point estimates for all seven feature sets, but is not a uniform improvement in ranking metrics. Line Length AUROC decreases from 0.760 to 0.748, and Low Gamma AP decreases from 0.507 to 0.499. These supplementary PR differences remain descriptive; the full primary comparison is reported below.','']
    for metric in c.PRIMARY:
        r=pair[(pair.control=='ALL_SIX_NO_PR')&(pair.metric==metric)].iloc[0]
        lines.append(f"- Full+PR versus FullNoPR {metric}: {r.difference_pp:+.2f} pp; nominal CI [{r.ci_low_pp:.2f}, {r.ci_high_pp:.2f}], 28-family CI [{r.adjusted_ci_low_pp:.2f}, {r.adjusted_ci_high_pp:.2f}].")
    lines += ['','### Linear localization without raw neural representations','',
       f"The all-six+PR Logistic Regression achieved AUROC {cell(full,'AUROC')}, AP {cell(full,'AP')}, Macro-F1 {cell(full,'Macro_F1')} and Top-1 {cell(full,'Top_1')} without raw iEEG neural embeddings. The supported claim is predictive information in held-out patients of this historical cohort, not external clinical validation.",'',
       '## Connection to Experiment A','',
       'Experiment A measured signed within-patient association; B fits a shared cross-patient decision function. Below, A effect sizes retain their original direction, while B uses only legal FIT labels to learn coefficients. Association strength does not by itself fix cross-patient scaling, redundancy or a linear boundary.','',
       '| Feature | A mean-coordinate D | A SD-coordinate D | B +PR AUROC | B NoPR AUROC |','|---|---:|---:|---:|---:|']
    ar=pd.read_csv(HERE.parent/'drechan_sixfeature_ez_association_v1/results/feature_association.csv')
    for j,name in enumerate(c.FEATURES):
        aa=allr.set_index('arm')
        lines.append(f"| {LABELS[name]} | {ar.iloc[2*j].mean_effect:.3f} | {ar.iloc[2*j+1].mean_effect:.3f} | {aa.loc[name+'_PR','AUROC']:.3f} | {aa.loc[name+'_NO_PR','AUROC']:.3f} |")
    lines += ['',
       'All six +PR single-feature AUROC intervals are above 0.5, consistent with A\'s positive patient-average associations. Nevertheless, RMS has weaker cross-patient discrimination (AUROC 0.663) than Line Length (0.748), and combining all features does not establish an improvement over Line Length. Patient heterogeneity, correlated coordinates and the shared linear boundary are possible explanations, not mechanisms isolated by this analysis. Association in A does not demonstrate independent conditional contribution in B.']
    selected='; '.join(f"fold{f}: "+', '.join(f"{label(r.arm)} C={r.C:g},t={r.threshold:.3f}" for r in selection[selection.fold==f].itertuples()) for f in range(1,6))
    variation=pd.read_csv(data/'coordinate_variation.csv')
    lines += ['','## Coordinate variation and PR numerical support','',
       'All six Mean coordinates had nonzero within-patient variation in 80/80 patients (100%). Each SD coordinate had nonzero variation and numerical PR support in 73/80 patients (91.25%). The remaining seven patients had one seizure and constant SD coordinates; all stayed in the evaluation. Support counts were identical under all five FIT-specific unit scales. Full 60-row audit: results/coordinate_variation.csv.','',
       '## Convergence, coverage and interpretation','',
       f"All 1,410 unique fits are recorded in fit_audit.csv. Nonconverged fits: {internal['nonconverged_fits']}; maximum observed iterations: {internal['max_iterations_observed']}. The iteration budget was not increased. Every main/supplementary group has 80 patients and 7,635 channels, with no undefined endpoint exclusions. Seven single-seizure patients were retained; their SD-coordinate variation/support is explicitly audited in coordinate_variation.csv.",'',
       'Original DELTA definitions and numerical floors were preserved. PR matches current DREChannNet feature-wise patient normalization; NoPR also has FIT-only global scaling. Historical LR results are excluded because their weighting, selection and legacy PR policies differ.','',
       'The six features and this cohort were used in historical model development. This is exploratory repeated-cohort cross-patient prediction, not independent external validation. Bootstrap intervals condition on fitted models and fixed splits, and do not capture retraining uncertainty. Predictive information does not establish causal mechanisms or irreplaceable conditional contribution of every feature.','',
       f"Independent verification passed {proof['patient_rows']} main/supplementary patient rows plus {proof['null_patient_rows']} null patient rows, {proof['patient_endpoint_values']} endpoint values, all {proof['score_cells']} score cells, 560 absolute/source intervals, 112 paired intervals and 28 additional-family intervals. Selection replay covered all 840 candidate models and 570 CAL thresholds. See INDEPENDENT_VERIFICATION.json and INTERNAL_VERIFICATION.json for numerical details.",'',
       '## Reproduction and artifacts','',
       'Source/input hashes, versions and original role counts: results/INPUT_AUDIT.json. Actual dates, source commits, the metadata-only engineering repair and private log locations: RUN_AUDIT.md. Complete selected C/threshold records: results/model_selection.csv; all candidate scores: results/inner_cv.csv.','',
       'Server execution uses run_repaired.ps1 with the original source commit, separately verified engineering commit and a new run tag. It never overwrites completed logs/models. Public archive validation must run before explicit allowlist extraction. Local regeneration:','',
       '```powershell','python experiments/drechan_sixfeature_ez_prediction_v1/release_check.py --package <anonymous-archive.zip>',
       'python experiments/drechan_sixfeature_ez_prediction_v1/render_figures.py',
       'python experiments/drechan_sixfeature_ez_prediction_v1/publish_report.py','```','',
       'Figure B: [PDF](figures/feature_prediction.pdf), [SVG](figures/feature_prediction.svg), [600 dpi PNG](figures/feature_prediction.png).',
       'Single-column table: [LaTeX](tables/feature_prediction_single_column.tex).',
       'All eight endpoints/intervals: [all_results.csv](results/all_results.csv). Individual predictions, models/scalers, identities and complete execution logs remain on the trusted server.','']
    (HERE/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    delta=pair[(pair.control=='ALL_SIX_NO_PR')&(pair.metric=='Macro_F1')].iloc[0]
    ll=main.set_index('arm').loc['LINE_LENGTH_PR']
    paragraph=(f"We assessed the predictive information of the six physiological features using Logistic Regression in the original five patient-held-out folds. "
       f"Regularization was selected by patient-isolated cross-validation within FIT, and CAL patients determined the classification threshold. "
       f"Combining all six features with patient-relative normalization yielded patient-average AUROC {full.AUROC:.3f} (95% CI {full.AUROC_ci_low:.3f}–{full.AUROC_ci_high:.3f}), "
       f"AP {full.AP:.3f}, Macro-F1 {full.Macro_F1:.3f} and Top-1 {full.Top_1:.3f} across 80 patients. "
       f"Among the eight main configurations, the joint model had the highest AUROC and AP point estimates, whereas Line Length alone had higher Macro-F1 ({ll.Macro_F1:.3f}) and Top-1 ({ll.Top_1:.3f}). "
       "The four paired intervals against Line Length all included zero, so a joint-feature advantage over this single feature was not established. "
       f"Compared with the globally standardized NoPR model, PR increased joint-model Macro-F1 by {delta.difference_pp:.2f} percentage points "
       f"(28-comparison-adjusted interval {delta.adjusted_ci_low_pp:.2f}–{delta.adjusted_ci_high_pp:.2f}). The adjusted AUROC and AP differences spanned zero. "
       f"Within-patient training-label permutations produced mean AUROC {null.loc['AUROC','null_mean']:.3f} across 100 repetitions. "
       "Together with the within-patient associations in Experiment A, these results support physiological feature information for cross-patient EZ localization without raw neural representations, but not the irreplaceability of each feature. "
       "This historical-cohort reanalysis provides exploratory evidence, not independent external validation.")
    (HERE/'PAPER_PARAGRAPH.md').write_text('# Preliminary Physiological Feature Analysis\n\n'+paragraph+'\n',encoding='utf-8')
    captions="""# Figure and table captions

## Figure B. Physiological feature predictive performance

Dots show equal-patient mean AUROC across 80 Outer patients in the original
five patient-held-out folds; horizontal lines show nominal 95% percentile
patient-bootstrap intervals from 10,000 seed42 draws. The dashed line marks
AUROC0.5. All inputs use current feature-wise patient-relative normalization
except All Six w/o PR (orange open square), which retains FIT-only global
standardization. Other rows are blue circles. Each individual feature contributes
the original across-seizure mean and population SD, and All Six uses all12
coordinates. Partial high gamma covers80–125Hz. C is selected only within FIT;
CAL determines thresholds. Primary paired comparisons use the separately reported
28-comparison family intervals, not these absolute nominal intervals. This
historical-cohort reanalysis is exploratory cross-patient prediction.

## Supplement. Training-label permutation control

The histogram retains all100 fixed all-six+PR training-label permutation
repetitions, evaluated on the true Outer labels of all80 patients. The grey
dashed line marks chance AUROC0.5 and the blue line true-label full+PR training.
FIT labels were shuffled within patients with class counts preserved and sample
weights recomputed. C and label-free preprocessing follow the legal true-FIT
selection; each null model has its own true-CAL threshold. The distribution
describes this conditional negative control, not patient-bootstrap uncertainty.
"""
    (HERE/'FIGURE_CAPTIONS.md').write_text(captions,encoding='utf-8')
    print('PREDICTION_REPORT_AND_SINGLE_COLUMN_TABLE_WRITTEN 8main,6supp,all7comparisons')


if __name__=='__main__':main()
