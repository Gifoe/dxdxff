"""Aggregate-only report and three fixed uncertainty figures; no private model imports."""
import csv
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PUB=Path(r'D:\m8_6d_no_variance_ablation_baselines_seed42_v1_runtime\public')
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']
LABELS={'FULL_F0':'Full six-feature F0','WO_EARLY_FILM':'w/o early FiLM','WO_LATE_GATE':'w/o late Gate','WO_BOTH_GUIDANCE':'w/o both guidance',
'WO_BOTH_PR':'w/o both feature and Raw PR','WO_RAW_PRE_REFERENCE':'w/o Raw pre-reference','WO_HEAD_INTERACTION':'w/o head interaction',
'EARLY_FILM_MISMATCH':'Early FiLM channel mismatch','RAW_ONLY':'Raw-only','PHYSIOLOGY_ONLY':'Physiology-only',
'DROP_BETA':'Drop beta','DROP_LOW_GAMMA':'Drop low gamma','DROP_PARTIAL_HIGH_GAMMA':'Drop partial high gamma','DROP_RMS':'Drop RMS','DROP_LINE_LENGTH':'Drop line length','DROP_SPECTRAL_ENTROPY':'Drop spectral entropy'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(name):
    with (PUB/(name+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def output_csv(name,rows):
    with (PUB/(name+'.csv')).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def forest(ids,labels,title,filename,ci,pr_pair=False):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(13.5,max(4.5,.37*len(ids)+1.7)),sharey=True)
    for ax,metric,heading in zip(axes,['macro_f1','auroc','auprc'],['Macro-F1','AUROC','AP']):
        rows=[next(r for r in ci if r['contrast']==key and r['metric']==metric) for key in ids]
        lo=np.array([float(r['low']) for r in rows]);hi=np.array([float(r['high']) for r in rows]);means=np.array([float(r['mean']) for r in rows]);yy=np.arange(len(ids))[::-1]
        ax.hlines(yy,lo,hi,color='#666666',lw=1.5,zorder=2)
        for y,m,key in zip(yy,means,ids):ax.plot(m,y,marker='D' if key=='Early_by_late_guidance' else 'o',color='#0072B2',ms=5,zorder=3)
        ax.axvline(0,color='#333333',linestyle='--',lw=.8,zorder=1);ax.grid(axis='x',color='#D9D9D9',lw=.5)
        span=max(float(hi.max()),0)-min(float(lo.min()),0);pad=max(span*.08,.002)
        ax.set_xlim(min(float(lo.min()),0)-pad,max(float(hi.max()),0)+pad);ax.set_title(heading,fontweight='bold')
        ax.set_xlabel('Metric difference');ax.set_yticks(yy);ax.set_yticklabels(labels)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        for spine in ['top','right','left']:ax.spines[spine].set_visible(False)
        ax.tick_params(axis='y',length=0);ax.tick_params(axis='x',labelsize=9)
    fig.suptitle(title,fontsize=13,fontweight='bold',y=.985)
    caption=('PR minus no PR' if pr_pair else 'Full minus ablation')+'; nominal 95% paired intervals, 10,000 draws (n=80). Exploratory repeated-test.'
    if 'Early_by_late_guidance' in ids:caption='Full minus ablation; final row is the factorial interaction. Nominal 95% paired intervals (80 patients; 10,000 draws). Exploratory repeated-test.'
    fig.text(.5,.035,caption,ha='center',fontsize=9,color='#444444')
    fig.subplots_adjust(left=.255,right=.985,top=.87,bottom=.19,wspace=.32)
    dest=PUB/'figures';dest.mkdir(exist_ok=True)
    for ext in ['svg','pdf','png']:fig.savefig(dest/(filename+'.'+ext),dpi=180)
    plt.close(fig)

def main():
    status=json.loads((PUB/'OUTER_STATUS.json').read_text(encoding='utf-8'))
    check=json.loads((PUB/'INDEPENDENT_OUTER_VERIFICATION.json').read_text(encoding='utf-8'))
    assert status['status']=='COMPLETE_VERIFIED_F0_6D_30_GROUP_FIVE_FOLD_OUTER' and check['status']=='PASS'
    lock=json.loads((HERE/'PROTOCOL_LOCK.json').read_text(encoding='utf-8'));arms=lock['arms'];deep=lock['deep_arms'];classifiers=lock['classifiers']
    points={r['arm']:r for r in table('MAIN_RESULTS')};ci=table('INTERVALS');fam=table('MACRO_FAMILY_INTERVALS')
    assert len(points)==30 and len(ci)==603 and len(fam)==37
    readable={'LogisticRegression':'Logistic regression','Linear_SVM':'Linear SVM','RBF_SVM':'RBF SVM','RandomForest':'Random forest','XGBoost':'XGBoost','MLP':'MLP','AdaBoost':'AdaBoost'}
    for c in classifiers:
        LABELS[c+'_NO_PR']=readable[c]+' without PR';LABELS[c+'_PR']=readable[c]+' with PR'
    output_csv('CORE_ABLATION_TABLE',[points[a] for a in deep[:10]])
    output_csv('FEATURE_ABLATION_TABLE',[points[a] for a in deep[10:]])
    output_csv('CLASSIFIER_TABLE',[points[a] for a in arms[16:]])
    output_csv('PR_PAIRED_INTERVALS',[r for r in ci if r['kind']=='PR_pair'])
    forest(['FULL_minus_'+a for a in deep[10:]],[LABELS[a].replace('Drop ','') for a in deep[10:]],'Six-feature model: conditional feature contribution','FEATURE_CONTRIBUTION',ci)
    forest(['FULL_minus_'+a for a in deep[1:10]]+['Early_by_late_guidance'],[LABELS[a] for a in deep[1:10]]+['Early x late interaction'],
        'Six-feature model: core mechanisms and branch comparison','CORE_PAIRED_INTERVALS',ci)
    forest([c+'_PR_minus_NO_PR' for c in classifiers],[readable[c] for c in classifiers],'Six-feature classifiers: patient-relative normalization','CLASSIFIER_PR_INTERVALS',ci,True)
    md=['# F0 without Variance: complete ablations and classical classifiers','',
        'Actual completed experiment: sixteen neural configurations plus seven classifiers in two PR settings, each over all five original patient-held-out folds. The cohort remains80patients/7635channels/256seizures. All150selected models froze before newOuter; all150score cells sealed before metric reporting.','',
        'The full six-feature model and Raw-only reuse ten exact previously frozen cells;70neural fits and70classical models are genuinely new. The main model was chosen by the user after historical Outer viewing: every conclusion is exploratory repeated-test, not independent confirmation. No new model or threshold was chosen from this Outer pass.','',
        '## Patient-average primary results','',
        'Each row is an equal mean of80patient metrics. Macro-F1 is the mean of EZ-F1 and NEZ-F1, not pooled-channel F1. EZ-fraction bias is predicted minus true fraction; closer to zero is preferable.','']
    def add_table(selected):
        md.extend(['| Configuration | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac. bias |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
        for a in selected:md.append('| '+LABELS[a]+' | '+' | '.join(f'{float(points[a][m]):.6f}' for m in METRICS)+' |')
    md+=['### Neural mechanisms and six feature deletions',''];add_table(deep)
    md+=['','### Other classifiers: exact same12physiological summary coordinates',''];add_table(arms[16:])
    warnings=[r for r in table('TABULAR_TRAIN_CELLS') if r['warning_categories']]
    md+=['',f"Training warnings: {len(warnings)}/70 classical fits recorded warning categories. See [training records](TABULAR_TRAIN_CELLS.csv); the fixed MLP400-iteration budget was not enlarged after outcomes. Reaching the budget is not evidence of optimization convergence.",'','Unrounded [main results](MAIN_RESULTS.csv); separate [pooled](POOLED_RESULTS.csv), [folds](FOLD_RESULTS.csv), [fold SD](FOLD_SD.csv), [sources](SOURCE_RESULTS.csv) and [coverage](METRIC_COVERAGE.csv). All group rows, including worse outcomes, remain reported.','',
        '## Paired uncertainty','',
        'Same80patient resampling indices,10000draws,seed42.603nominal intervals cover30absolute estimates,29Full-minus-other comparisons,one guidance interaction and7classifier PR differences across9metrics.37additional Macro-F1 intervals use Bonferroni-level percentile tails .05/(2x37); these are approximate exploratory summaries, not exact family-error guarantees or correction for the entire historical research sequence.','',
        '| Contrast | Macro-F1 difference | Nominal95% interval | Additional37-comparison interval |',
        '| --- | ---: | --- | --- |']
    for r in ci:
        if r['kind']=='absolute' or r['metric']!='macro_f1':continue
        f=next(q for q in fam if q['contrast']==r['contrast'])
        md.append('| '+r['contrast']+f" | {float(r['mean']):+.6f} | [{float(r['low']):.6f}, {float(r['high']):.6f}] | [{float(f['low']):.6f}, {float(f['high']):.6f}] |")
    md+=['','[All603nominal intervals](INTERVALS.csv) and [all37Macro-F1 family intervals](MACRO_FAMILY_INTERVALS.csv). Positive Full-minus-other favors Full; positive classifier PR-minus-noPR favors PR. Guidance interaction is Full+Neither-NoEarly-NoLate.','',
        '## Core mechanisms','',
        '![Core paired intervals](figures/CORE_PAIRED_INTERVALS.png)','',
        'NoBothPR simultaneously removes direct physiological-summary patient Z and final96D Raw-summary patient Z. It does not disable per-window waveform normalization, the pre-reference, either conditioning module or the physiological DELTA definition. There are no separate single-branch PR ablations in this new matrix. Other switches preserve their original operation boundaries and full initializer before deletion.','',
        'The early FiLM mismatch remains within patient/seizure/identical valid-window-support groups. Only earlyconditioning receives donor trajectories; correctly aligned direct physiology and late Gate stay intact. Variance is masked for both recipient and donor. Counts are in [RAW_COVERAGE.csv](RAW_COVERAGE.csv). Same parameter count is a control for capacity, not proof against every alternative architecture.','',
        '## Remaining six explicit inputs','',
        '![Conditional feature contribution](figures/FEATURE_CONTRIBUTION.png)','',
        'Variance is masked permanently in every neural physiology path. Each additional deletion is trained from the matched fresh initialization and removes the corresponding two summary coordinates and early/late window coordinate; widths remain14/7. Thus single deletions have five active measurements, not six plus hidden Variance. Raw can still encode correlated properties; no single-deletion result proves absence of the physiology or optimality of the retained subset. Partial high gamma is80–125Hz and entropy10–125Hz, never full80–150Hz or recovered low frequencies.','',
        '## Classical PR comparison','',
        '![Classical classifier PR intervals](figures/CLASSIFIER_PR_INTERVALS.png)','',
        'Classifiers actually receive12columns: the two summaries of each remaining measurement. NoPR uses the original DELTA summary values, PR uses original label-free within-patient population Z, followed by unchanged FIT-only preprocessing. Shared model/fold hyperparameters come from the old legally validation-selected7-feature configurations and are frozen across both12D arms; there is no new search. This is a fixed-hyperparameter transfer comparison, not proof that every classifier is optimally tuned for12D. All70models are newly fitted on legalFIT only; no trained7Dclassifier is relabeled as12D.','',
        'LR, SVM and MLP use FIT-only scaling; tree models retain native unscaled inputs. SVM probabilities use3patient-held-out subfolds strictly insideFIT for Platt calibration. Validation only selects each numeric threshold; it never calibrates SVM or refits the model. Classifiers have no Raw branch, so their PR switch refers only to the12D physiological input. Their original EZ>=threshold decision rule and the neural NEZ<threshold rule are preserved without rounding scores.','',
        '## Completeness and boundaries','',
        f"- Independently checked {check['patient_rows']} patient rows and {check['patient_metric_checks']} endpoints,603nominal and37additional family intervals. Maximum interval difference {check['maximum_interval_difference']:.3g}.",
        '- All groups retain80patients and7635channels. Each of13new Raw-consuming arms retains1471965valid windows. Physiology-only and tabular models consume no Raw during inference; no fabricated Raw counts are assigned.',
        '- The same seed42, five folds, source roles, full-patient neural updates, FP32 settings, optimizer/loss and epoch/threshold rules were preserved. Every configuration completes Outer regardless of validation performance.',
        '- One fixed seed does not establish seed robustness; patient bootstrap does not cover researcher selection, the main model choice or training randomness. Cross-source tables are descriptive on the same historical cohort.',
        '- Results may be negative or metric-dependent; no reduced subset or updated method is selected from this pass. Full-versus-branch differences can include parameter-capacity/optimization differences, not only pure information content.',
        '- Only source/hashes and anonymous aggregate tables/figures are released. Signals, identities, individual predictions, weights, preprocessors, maps and logs stay on the trusted server.','']
    (PUB/'REPORT.md').write_text('\n'.join(md),encoding='utf-8')
    manifest={str(p.relative_to(PUB)).replace('\\','/'):sha(p) for p in (PUB/'figures').iterdir() if p.is_file()}
    manifest.update({n+'.csv':sha(PUB/(n+'.csv')) for n in ['MAIN_RESULTS','INTERVALS','MACRO_FAMILY_INTERVALS','TABULAR_TRAIN_CELLS']})
    (PUB/'RENDER_MANIFEST.json').write_text(json.dumps(dict(status='GENERATED_PENDING_VISIBLE_QA',renderer_sha256=sha(__file__),files=manifest),indent=2)+'\n',encoding='utf-8')
    print('SIXD_VERIFIED_30_GROUP_REPORT_AND_THREE_FIGURES_RENDERED',flush=True)
if __name__=='__main__':main()
