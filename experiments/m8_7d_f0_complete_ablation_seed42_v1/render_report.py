"""Render only already-verified anonymous aggregates; no model or private data imports."""
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;PUB=Path(r'D:\m8_7d_f0_complete_ablation_seed42_v1_runtime\public')
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']
LABELS={'FULL_F0':'Full F0','WO_EARLY_FILM':'w/o early FiLM (C2)','WO_LATE_GATE':'w/o late Gate','WO_BOTH_GUIDANCE':'w/o both guidance','WO_FEATURE_PR':'w/o feature PR','WO_RAW_PR':'w/o Raw PR',
 'WO_RAW_PRE_REFERENCE':'w/o Raw pre-reference','WO_HEAD_INTERACTION':'w/o head interaction','EARLY_FILM_MISMATCH':'Early FiLM channel mismatch','RAW_ONLY':'Raw-only','PHYSIOLOGY_ONLY':'Physiology-only',
 'DROP_BETA':'Drop beta','DROP_LOW_GAMMA':'Drop low gamma','DROP_PARTIAL_HIGH_GAMMA':'Drop partial high gamma','DROP_RMS':'Drop RMS','DROP_VARIANCE':'Drop variance','DROP_LINE_LENGTH':'Drop line length','DROP_SPECTRAL_ENTROPY':'Drop spectral entropy'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(name):
    with (PUB/(name+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def write_csv(name,rows):
    with (PUB/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def forest(ids,labels,title,filename,ci):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
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
        for spine in ['top','right','left']:ax.spines[spine].set_visible(False)
        ax.tick_params(axis='y',length=0);ax.tick_params(axis='x',labelsize=9)
    fig.suptitle(title,fontsize=13,fontweight='bold',y=.985)
    caption='Full minus ablation; nominal 95% paired intervals, 10,000 patient draws (n=80). Exploratory repeated-test.'
    if 'Early_by_late_guidance' in ids:caption='Full minus ablation; final row is the factorial interaction. Nominal 95% paired intervals (80 patients; 10,000 draws). Exploratory repeated-test.'
    fig.text(.5,.035,caption,ha='center',fontsize=9,color='#444444')
    fig.subplots_adjust(left=.255,right=.985,top=.87,bottom=.19,wspace=.32)
    dest=PUB/'figures';dest.mkdir(exist_ok=True)
    for ext in ['svg','pdf','png']:fig.savefig(dest/(filename+'.'+ext),dpi=180)
    plt.close(fig)

def main():
    status=json.loads((PUB/'OUTER_STATUS.json').read_text(encoding='utf-8'));verification=json.loads((PUB/'INDEPENDENT_OUTER_VERIFICATION.json').read_text(encoding='utf-8'))
    assert status['status']=='COMPLETE_VERIFIED_F0_18_ARM_FIVE_FOLD_OUTER' and verification['status']=='PASS'
    protocol=json.loads((HERE/'PROTOCOL_LOCK.json').read_text(encoding='utf-8'));arms=protocol['arms'];points={r['arm']:r for r in table('MAIN_RESULTS')};ci=table('INTERVALS');family=table('MACRO_FAMILY_INTERVALS')
    assert len(points)==18 and len(ci)==324 and len(family)==18
    feature=[]
    for arm,name in zip(arms[11:],protocol['features']):
        rr={'feature':name,'arm':arm}
        for m in METRICS:
            r=next(r for r in ci if r['contrast']=='FULL_minus_'+arm and r['metric']==m)
            for k in ['mean','low','high']:rr[m+'_'+k]=r[k]
        feature.append(rr)
    write_csv('FEATURE_CONTRIBUTION',feature)
    write_csv('CORE_ABLATION_TABLE',[points[a] for a in arms[:11]])
    write_csv('RAW_COMPLEMENTARITY',[points[a] for a in ['FULL_F0','RAW_ONLY','PHYSIOLOGY_ONLY']])
    write_csv('GUIDANCE_INTERACTION',[r for r in ci if r['kind']=='interaction'])
    forest(['FULL_minus_'+a for a in arms[11:]],[LABELS[a].replace('Drop ','') for a in arms[11:]],'Conditional contribution of the seven physiological inputs','FEATURE_CONTRIBUTION',ci)
    forest(['FULL_minus_'+a for a in arms[1:11]]+['Early_by_late_guidance'],[LABELS[a] for a in arms[1:11]]+['Early x late interaction'],
        'Core mechanisms and Raw-physiology complementarity','PAIRED_CORE_INTERVALS',ci)
    out=['# F0 single-head complete ablation: seed42 five-fold Outer','',
         'All eighteen configurations completed the original patient-held-out five folds: 80 patients, 7,635 channels and256seizures. Sixteen arms were freshly trained in80fold cells; frozen F0/C2 contributed ten exact reused cells. No Development eligibility screen or Outer-driven tuning was applied.','',
         'The primary endpoint is the equal-patient mean binary Macro-F1 over all80patients, not pooled-channel F1 or an equal-weight five-fold average. The same seven measurements are retained for the full model; explicit leave-one-out masks act in all three physiology input paths.','',
         '## All fixed configurations','',
         '| Configuration | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-frac. bias |',
         '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for a in arms:out.append('| '+LABELS[a]+' | '+' | '.join(f'{float(points[a][m]):.6f}' for m in METRICS)+' |')
    out+=['','Unrounded values: [MAIN_RESULTS.csv](MAIN_RESULTS.csv). All80patients contribute to each arm; undefined-value coverage is explicitly audited, not replaced by favorable zeros. Separate [pooled](POOLED_RESULTS.csv), [fold](FOLD_RESULTS.csv), [fold SD](FOLD_SD.csv) and [source](SOURCE_RESULTS.csv) tables preserve their distinct aggregation units.','',
          '## Paired differences from the complete F0','',
          'Positive Full-minus-ablation means the complete model scores higher. These are fixed-model, seed42 patient-paired percentile intervals; they do not include model-selection or seed uncertainty.','',
          '| Comparison | Macro-F1 difference | Nominal95% CI | Additional18-comparison family interval |',
          '| --- | ---: | --- | --- |']
    for a in arms[1:]:
        r=next(r for r in ci if r['contrast']=='FULL_minus_'+a and r['metric']=='macro_f1');f=next(z for z in family if z['contrast']==r['contrast'])
        out.append(f"| Full minus {LABELS[a]} | {float(r['mean']):+.6f} | [{float(r['low']):.6f}, {float(r['high']):.6f}] | [{float(f['low']):.6f}, {float(f['high']):.6f}] |")
    interaction=next(r for r in ci if r['kind']=='interaction' and r['metric']=='macro_f1');interaction_family=next(r for r in family if r['kind']=='interaction')
    out+=['','The extra family intervals use Bonferroni-level percentile tails .05/(2x18). They are approximate exploratory multiplicity summaries, not exact guarantees and not a repair for the broader repeated-test research history.','',
          '## Early and late guidance','',
          'The four matched configurations are complete F0, no early FiLM (C2), no late Gate, and neither guidance. The predefined metric interaction is Full + Neither - NoEarly - NoLate. It is a metric-scale statistical interaction, not proof of a causal physiological interaction.',
          f"Macro-F1 interaction: {float(interaction['mean']):+.6f}, nominal95% paired CI [{float(interaction['low']):.6f}, {float(interaction['high']):.6f}]; additional family interval [{float(interaction_family['low']):.6f}, {float(interaction_family['high']):.6f}]. All nine interaction endpoints are in [GUIDANCE_INTERACTION.csv](GUIDANCE_INTERACTION.csv).",'',
          'The early-FiLM mismatch control keeps the same parameter count and input distribution/support within patient/seizure groups but removes correct channel alignment only for the early conditioner. The direct14and late Gate remain correctly aligned. Its eligible/singleton/change counts are in [RAW_COVERAGE.csv](RAW_COVERAGE.csv); the control is not described as a proof against all alternative same-capacity architectures.','',
          '## Conditional feature contribution','',
          '![Feature contribution](figures/FEATURE_CONTRIBUTION.png)','',
          'Points and horizontal lines show Full-minus-drop and nominal95%paired patient intervals for Macro-F1, AUROC and AP. Feature order is fixed, not sorted by favorable outcomes. The partial high-gamma measurement is80–125Hz, not full80–150Hz. Spectral entropy covers the existing10–125Hz definition, not recovered low-frequency physiology.','',
          'Deletion removes the explicit measurement from the direct summary pair, early FiLM and late Gate simultaneously throughout training and inference. Raw can still carry correlated information. A small leave-one-out effect may reflect redundancy or compensation; it does not prove that the physiological property is absent, useless universally, or that this is the globally optimal feature set. No subset was selected from these Outer outcomes.','',
          '## Core mechanisms and branch complementarity','',
          '![Core paired intervals](figures/PAIRED_CORE_INTERVALS.png)','',
          'Raw-only removes all explicit physiology routes rather than only the direct14coordinates. Physiology-only is a freshly trained matched single head on the original14summary coordinates and consumes no Raw. Both share their corresponding slices of the full initialization and the original fixed branch scaling; they are not substituted by historical residual-M8 or differently bounded-logit baselines. See [RAW_COMPLEMENTARITY.csv](RAW_COMPLEMENTARITY.csv).','',
          'Feature PR means summary14 patient Z; Raw PR means only final96latent-summary patient Z. Removing Raw PR does not remove per-window waveform standardization. Removing Raw pre-reference does not redefine the seven physiological DELTA measurements. The no-interaction arm disables only the bounded hidden Raw-by-physiology term.','',
          '## Integrity and interpretation limits','',
          f"- Independently verified {verification['patient_rows']} patient rows, {verification['patient_metric_checks']} endpoints, {verification['nominal_interval_checks']} nominal intervals and {verification['macro_family_interval_checks']} additional family intervals. Maximum interval discrepancy: {verification['maximum_interval_difference']:.3g}.",
          '- All90checkpoints, epochs, thresholds, preprocessors and sources were frozen before new Outer inference. All90score cells were sealed before full metric reporting. Exact reused reference scores remain unchanged.',
          '- Each of the15new Raw-consuming arms preserves all1,471,965valid windows. Physiology-only has zero Raw inference windows, explicitly reported rather than assigned fabricated counts.',
          '- Original patient roles, FIT-only preprocessing, class weights, optimizer, loss, stopping budget and threshold grid/ties remain fixed. No test-driven model, threshold or selection change was made.',
          '- F0 itself was chosen after historical Outer viewing. All evidence here is exploratory repeated-test, not independent confirmation or a reason to silently nominate an improved ablation as a new confirmed model. One seed does not establish training-seed robustness.',
          '- Full-model performance and every negative or mixed ablation remain in the report. The fixed matrix supports conditional mechanism comparisons, not a claim of globally optimal7D selection.',
          '- Only code, hashes and anonymous aggregates/figures are public; raw signals, identities, private predictions, mappings, weights and runtime logs stay on the trusted server.','']
    recovery=PUB/'NATIVE_RESUME_AUDIT.json'
    if recovery.exists():
        r=json.loads(recovery.read_text(encoding='utf-8-sig'))
        assert r['backup_hash_verified'] and r['all_completed_hashes_verified'] and not r['scientific_source_changed']
        out+=['## Engineering recovery','',
              f"A Windows native access violation ({r['exception']}) occurred during atomic PyTorch LAST serialization in {r['arm']}, fold{r['fold']}. The last complete epoch{r['last_complete_epoch']} checkpoint, all optimizer/RNG state and all{r['completed_fresh_cells_preserved']} completed-cell hashes were validated. Its exact checkpoint and failed temporary file were preserved before a fresh-process resume at epoch{r['resume_epoch']}. No model/data/loss/selection/hyperparameter or scientific-source change was made. The root cause of the native exception is not established; it is not described as a Python-level training error or as a failure-free run. See [NATIVE_RESUME_AUDIT.json](NATIVE_RESUME_AUDIT.json).",'']
    (PUB/'REPORT.md').write_text('\n'.join(out),encoding='utf-8')
    manifest={str(p.relative_to(PUB)).replace('\\','/'):sha(p) for p in (PUB/'figures').iterdir() if p.is_file()}
    manifest.update({n+'.csv':sha(PUB/(n+'.csv')) for n in ['MAIN_RESULTS','INTERVALS','MACRO_FAMILY_INTERVALS']})
    (PUB/'RENDER_MANIFEST.json').write_text(json.dumps(dict(status='GENERATED_PENDING_VISIBLE_QA',renderer_sha256=sha(__file__),files=manifest,source='verified anonymous aggregates'),indent=2)+'\n',encoding='utf-8')
    print('VERIFIED_AGGREGATE_REPORT_AND_FIGURES_RENDERED',flush=True)

if __name__=='__main__':main()
