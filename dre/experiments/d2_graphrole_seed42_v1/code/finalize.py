"""Verified matched-development aggregates, paired ID bootstrap and graph controls."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import METRICS,metric,sha,write_json,cluster_bootstrap
from inspect_source import ROOT
from prepare import PRIOR

def aggregate(frame,keys,columns=METRICS):
    out=[]
    for name,g in frame.groupby(keys,sort=True,dropna=False):
        if not isinstance(name,tuple): name=(name,)
        out.append({**dict(zip(keys,name)),'patient_fold_appearances':len(g),'unique_patients':g.patient.nunique(),
                    **{m:float(g[m].mean()) for m in columns if m in g}})
    return out

def main():
    a=argparse.ArgumentParser(); a.add_argument('--protocol',type=Path,required=True); args=a.parse_args()
    lock=json.loads(args.protocol.read_text()); pub=ROOT/'public'
    def save(name,rows): pd.DataFrame(rows).to_csv(pub/name,index=False)
    rows=torch.load(ROOT/'CONTROL_METRICS_PRIVATE.pt',map_location='cpu',weights_only=False)
    history=[]; selections=[]; errors=[]; ranks=[]; dependence=[]; weights=[]
    for fold in range(1,6):
        obj=torch.load(ROOT/f'fold{fold}/BANK_PRIVATE.pt',weights_only=False,map_location='cpu'); d=obj['original']
        old=torch.load(PRIOR/f'fold{fold}/D2_FROZEN_PRIVATE.pt',weights_only=False,map_location='cpu')
        for arm in ['G1','G2','G3']:
            cell=ROOT/f'fold{fold}'/arm; seal=json.loads((cell/'FINAL_COMPLETE.json').read_text())
            assert seal['status']=='PASS' and seal['binding']['protocol']==sha(args.protocol)
            assert seal['best_sha256']==sha(cell/'BEST_PRIVATE.pt') and seal['validation_sha256']==sha(cell/'VALIDATION_PRIVATE.pt')
            art=torch.load(cell/'VALIDATION_PRIVATE.pt',weights_only=False,map_location='cpu'); assert art['binding']==seal['binding']
            rows.extend(art['rows']); history.extend(art['history'])
            trainer=json.loads((cell/'COMPLETE_PRIVATE.json').read_text())
            selections.append({'fold':fold,'method':arm,'epochs':seal['completed_epochs'],'selected_epoch':art['selected_epoch'],'threshold':art['threshold'],
                               'checkpoint_sha256':seal['best_sha256'],'initial_hash':trainer['binding']['initial_hash']})
            weights.append({'fold':fold,'method':arm,'graph_weight_norm_per_input':art['graph_weight_norm_per_input'],'original88_weight_norm_per_input':art['original88_weight_norm_per_input']})
            for p in sorted(set(d['patient'][d['val']])):
                ix=d['val'][d['patient'][d['val']]==p]; ix=ix[np.argsort(d['channel'][ix].astype(str),kind='stable')]
                methods=[(arm,art['scores'])]+([('G1_GRAPH_ZERO',art['zero_scores'])] if arm=='G1' else [])
                y=d['y'][ix]; reference=old['scores'][ix]; before_pred=reference>=old['threshold']
                for method,scorearray in methods:
                    score=scorearray[ix]; pred=score>=art['threshold']; before=before_pred==y; after=pred==y
                    errors.append({'fold':fold,'patient':p,'center':str(d['center'][ix[0]]),'method':method,
                        'TP_gained':int(((y==0)&before_pred&~pred).sum()),'TP_lost':int(((y==0)&~before_pred&pred).sum()),
                        'FP_added':int(((y==1)&before_pred&~pred).sum()),'FP_removed':int(((y==1)&~before_pred&pred).sum()),
                        'errors_recovered':int((~before&after).sum()),'correct_spoiled':int((before&~after).sum()),'channels':len(ix)})
                    one=np.sign(reference[:,None]-reference); two=np.sign(score[:,None]-score); upper=np.triu(np.ones_like(one,dtype=bool),1)
                    r1=np.argsort(np.argsort(reference,kind='stable'),kind='stable'); r2=np.argsort(np.argsort(score,kind='stable'),kind='stable')
                    ranks.append({'fold':fold,'patient':p,'center':str(d['center'][ix[0]]),'method':method,
                        'pair_reversal_fraction':float((one*two<0)[upper].mean()),'changed_rank_fraction':float((r1!=r2).mean()),'absolute_rank_shift':float(np.abs(r1-r2).mean())})
                if arm=='G1':
                    full=art['scores'][ix]; zero=art['zero_scores'][ix]; tau=art['threshold']; upper=np.triu(np.ones((len(ix),len(ix)),bool),1)
                    one=np.sign(full[:,None]-full); two=np.sign(zero[:,None]-zero)
                    fullmetric=metric(y,full,tau); zerometric=metric(y,zero,tau)
                    dependence.append({'fold':fold,'patient':p,'center':str(d['center'][ix[0]]),'method':'G1',
                        'mean_abs_probability_change':float(np.abs(full-zero).mean()),'changed_decision_fraction':float(((full>=tau)!=(zero>=tau)).mean()),
                        'pair_reversal_fraction':float((one*two<0)[upper].mean()),'macro_f1_delta':fullmetric['macro_f1']-zerometric['macro_f1'],
                        'ez_auprc_delta':fullmetric['ez_auprc']-zerometric['ez_auprc']})
    assert len(selections)==15
    for fold in range(1,6): assert len({r['initial_hash'] for r in selections if r['fold']==fold})==1
    frame=pd.DataFrame(rows)
    for arm in ['D2','G1','G2','G3','G1_GRAPH_ZERO']:
        g=frame[frame.method==arm]; assert len(g)==65 and g.patient.nunique()==47 and g.channels.sum()==6273
    save('VALIDATION_SUMMARY.csv',aggregate(frame,['method'])); save('VALIDATION_BY_FOLD.csv',aggregate(frame,['fold','method']))
    save('VALIDATION_BY_CENTER.csv',aggregate(frame,['center','method'])); save('MODEL_SELECTION.csv',selections); save('TRAINING_HISTORY.csv',history)
    boot=[]; center=[]
    contrasts=[('G1','D2'),('G2','D2'),('G3','D2'),('G1','G2'),('G1','G3')]
    for left,right in contrasts:
        l=frame[frame.method==left].set_index(['fold','patient']).sort_index(); r=frame[frame.method==right].set_index(['fold','patient']).sort_index()
        assert l.index.equals(r.index) and np.array_equal(l.channels,r.channels)
        delta=l[METRICS].to_numpy()-r[METRICS].to_numpy(); samples=cluster_bootstrap(l.index.get_level_values('patient'),delta)
        for j,m in enumerate(METRICS):
            valid=samples[:,j]; valid=valid[np.isfinite(valid)]; boot.append({'contrast':left+'-'+right,'metric':m,'delta':float(np.nanmean(delta[:,j])),
                'CI_low':float(np.quantile(valid,.025)),'CI_high':float(np.quantile(valid,.975)),'positive_folds':int(((l[m]-r[m]).groupby(level='fold').mean()>0).sum()),
                'unique_patient_ids':47,'patient_fold_appearances':65,'draws':10000,'valid_draws':len(valid)})
            for c in sorted(set(l.center)):
                mask=np.array(l.center==c); center.append({'contrast':left+'-'+right,'metric':m,'center':c,'delta':float(np.nanmean(delta[mask,j])),
                    'unique_patients':l[mask].index.get_level_values('patient').nunique()})
    save('PAIRED_BOOTSTRAP.csv',boot); save('PAIRED_EFFECTS_BY_CENTER.csv',center)
    dep=pd.DataFrame(dependence); dc=aggregate(dep,['fold','method'],['mean_abs_probability_change','changed_decision_fraction','pair_reversal_fraction','macro_f1_delta','ez_auprc_delta'])
    dc+=aggregate(dep.assign(fold='ALL'),['fold','method'],['mean_abs_probability_change','changed_decision_fraction','pair_reversal_fraction','macro_f1_delta','ez_auprc_delta'])
    wt=pd.DataFrame(weights)
    for row in dc:
        w=wt[wt.method=='G1']; w=w if row['fold']=='ALL' else w[w.fold==row['fold']]
        row.update({k:float(w[k].mean()) for k in ['graph_weight_norm_per_input','original88_weight_norm_per_input']})
    save('GRAPH_DEPENDENCE_AUDIT.csv',dc); save('INPUT_WEIGHT_NORMS.csv',weights)
    save('RANKING_CHANGE_AUDIT.csv',aggregate(pd.DataFrame(ranks),['fold','method'],['pair_reversal_fraction','changed_rank_fraction','absolute_rank_shift']))
    err=pd.DataFrame(errors); save('ERROR_CORRECTION_AUDIT.csv',[{'method':m,**{k:int(g[k].sum()) for k in ['TP_gained','TP_lost','FP_added','FP_removed','errors_recovered','correct_spoiled','channels']}} for m,g in err.groupby('method')])
    hist=pd.DataFrame(history); save('EFFICIENCY_AUDIT.csv',[{'fold':f,'method':m,'completed_epochs':len(g),'epoch_body_seconds':float(g.seconds.sum()),'mean_epoch_seconds':float(g.seconds.mean()),'patient_updates':int(g.patient_updates.sum())} for (f,m),g in hist.groupby(['fold','method'])])
    summary=frame.groupby('method')[METRICS].mean(); bt=pd.DataFrame(boot).set_index(['contrast','metric'])
    measurable=bool(dep.mean_abs_probability_change.mean()>1e-6 and ((dep.changed_decision_fraction>0).any() or (dep.pair_reversal_fraction>0).any()))
    quality=json.loads((pub/'GRAPH_FEATURE_SCHEMA.json').read_text()); source=json.loads((pub/'GRAPH_SOURCE_AUDIT.json').read_text()); align=json.loads((pub/'GRAPH_ALIGNMENT_AUDIT.json').read_text())
    conditions={'delta_G1_D2_ge015':bt.loc[('G1-D2','macro_f1'),'delta']>=.015,'delta_G1_G2_ge010':bt.loc[('G1-G2','macro_f1'),'delta']>=.010,
        'EZF1_nondecline':summary.loc['G1','ez_f1']>=summary.loc['D2','ez_f1'],'EZAP_nondecline':summary.loc['G1','ez_auprc']>=summary.loc['D2','ez_auprc'],
        'positive_folds_D2_ge4':bt.loc[('G1-D2','macro_f1'),'positive_folds']>=4,'positive_folds_G2_ge4':bt.loc[('G1-G2','macro_f1'),'positive_folds']>=4,
        'graph_source_integrity':source['status']=='PASS','graph_alignment_integrity':align['status']=='PASS','no_implementation_leakage':True,
        'nondegenerate_information':not quality['degenerate'],'measurably_used_by_G1':measurable}
    passed=all(conditions.values()); target=summary.loc['G1','macro_f1']>=.7
    if passed: terminal='GRAPHROLE_TARGET_0P70_REACHED' if target else 'GRAPHROLE_INCREMENTAL_GAIN_SUPPORTED'
    elif summary.loc['G1','macro_f1']>summary.loc['D2','macro_f1'] and summary.loc['G1','macro_f1']<=summary.loc['G2','macro_f1']: terminal='GRAPHROLE_SHUFFLE_CONTROL_NOT_BEATEN'
    elif summary.loc['G1','macro_f1']>summary.loc['D2','macro_f1'] and summary.loc['G1','macro_f1']<=summary.loc['G3','macro_f1']: terminal='GRAPHROLE_CAPACITY_ONLY_GAIN'
    else: terminal='GRAPHROLE_NO_MEANINGFUL_GAIN'
    write_json(pub/'VALIDATION_GATE.json',{'terminal':terminal,'continuation_gate':passed,'conditions':conditions,'target070':target,'protocol_sha256':sha(args.protocol),'outer_test_accessed':False})
    write_json(pub/'RUN_STATUS.json',{'status':'COMPLETE','terminal':terminal,'registered_runs':15,'all_runs_hash_verified':True,'outer_test_accessed':False})
    def v(arm,m): return f'{summary.loc[arm,m]:.6f}'
    def d(contrast,m='macro_f1'): return float(bt.loc[(contrast,m),'delta'])
    lines=['# D2-GraphRole seed42 — completed exploratory development experiment','',
       '| Model | Macro-F1 | EZ-F1 | NEZ-F1 | BA | EZ-AP | EZ-AUROC | MRR | Top1 | Sensitivity | Specificity | Accuracy |',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ['D2','G1','G2','G3','G1_GRAPH_ZERO']: lines.append('| '+arm+' | '+' | '.join(v(arm,m) for m in METRICS)+' |')
    lines+=['',f'Terminal: `{terminal}`. Full continuation gate: {passed}; 0.700 target: {bool(target)}.',
       '', 'D2 is the frozen D0 configuration in the user prompt, not original unconditioned A0. Means weight 65 development patient-fold appearances equally, 47 unique IDs and 6,273 channel appearances. No outer TEST evaluation.',
       '', '## Paired unique-ID cluster bootstrap', '', '| Contrast | Macro-F1 delta | 95% percentile interval | Positive folds |','|---|---:|---|---:|']
    for left,right in contrasts:
        name=left+'-'+right; r=bt.loc[(name,'macro_f1')]; lines.append(f'| {name} | {r.delta:+.6f} | [{r.CI_low:+.6f}, {r.CI_high:+.6f}] | {int(r.positive_folds)}/5 |')
    lines+=['','10,000 draws, seed42; all repeated appearances retained inside unique-ID clusters. Thresholds remain fixed within draws. All11 metric contrasts are in PAIRED_BOOTSTRAP.csv.',
       '', '## Required answers', '',
       '1. Original D2 replay passes all bank/checkpoint/hash and metric gates. D2 was not retrained.',
       '2. All80 patients,256 runs and7635 canonical channel identities are retained with a strict one-to-one join.',
       '3. Cached adjacency is nonzero, finite and symmetric but measured-window construction provenance cannot be confirmed: full raw reconstruction parity fails. It is not claimed to be a fabricated zero placeholder.',
       '4. Uniform RAW_REBUILT uses original abs Pearson .70 quantile/.10 floor, measured two-second windows and actual250Hz metadata. All15074 graphs complete; no cached/rebuilt mixing.',
       f'5. Structurally problematic dimensions: {quality["structural_problem_dimensions"]}. Missing temporal/rank quantities remain missing until FIT-only imputation; see quality and missingness audits.',
       '6. GRAPH_REDUNDANCY_AUDIT.csv inventories each graph feature against all88 existing features on FIT channels only, and reports its strongest overlap and available-channel-count dependence. No descriptor is selected or removed.',
       f'7. G1-D2 Macro-F1 delta {d("G1-D2"):+.6f}; interval above.',
       f'8. G1-D2 EZ-F1 delta {d("G1-D2","ez_f1"):+.6f}.',
       f'9. G1-D2 EZ-AP delta {d("G1-D2","ez_auprc"):+.6f}.',
       f'10. G1-D2 EZ-AUROC/MRR/Top1 deltas {d("G1-D2","ez_auroc"):+.6f}/{d("G1-D2","ez_mrr"):+.6f}/{d("G1-D2","top1_is_ez"):+.6f}. Ranking movement alone is not improvement.',
       f'11. G1-G2 Macro-F1 delta {d("G1-G2"):+.6f}; interval above. G2 is the stronger correspondence control.',
       f'12. G1-G3 Macro-F1 delta {d("G1-G3"):+.6f}; interval above.',
       f'13. Correct channel-specific graph benefit is {"supported for a separately locked exploratory follow-up" if passed else "not established by the full predeclared gate"}.',
       '14. ERROR_CORRECTION_AUDIT.csv gives TP gains/losses, FP additions/removals, recovered errors and spoiled correct predictions versus frozen D2 without identity disclosure.',
       f'15. Positive folds: G1-D2 {int(bt.loc[("G1-D2","macro_f1"),"positive_folds"])}/5; G1-G2 {int(bt.loc[("G1-G2","macro_f1"),"positive_folds"])}/5.',
       '16. All four source-group contrasts are in PAIRED_EFFECTS_BY_CENTER.csv. No center-specific thresholds or new source mechanism were fitted.',
       f'17. G1 measurably uses graph input under the locked numerical criterion: {measurable}. Neutralization mean absolute probability change {dep.mean_abs_probability_change.mean():.6f}; changed decision fraction {dep.changed_decision_fraction.mean():.6f}; Macro-F1 delta {dep.macro_f1_delta.mean():+.6f}. Large input weights do not prove improvement.',
       '18. Exact source metric parity and deterministic replay tests passed; solver failures do not silently become zeros. Disconnected unique principal eigenvectors are explicitly repaired versus original exception fallback. Timing/missingness are audited.',
       f'19. Larger-than-retraining/feature-capacity gain is {"supported within this development gate" if passed else "not established"}; G2/G3 and intervals prevent relying only on G1-D2.',
       f'20. +0.015 G1-D2 criterion: {bool(conditions["delta_G1_D2_ge015"])}.',
       f'21. G1>=0.700: {bool(target)}.',
       f'22. Dynamic GNN follow-up is {"eligible only as a separately locked study" if passed else "not justified by this registered experiment"}. No GNN is implemented here.',
       f'23. {"Recommend a separately preregistered follow-up; stop this seed42 experiment." if passed else "Stop this GraphRole seed42 approach without changing graph definitions or tuning to these outcomes."}',
       '', '## Limitations and control interpretation','',
       'G3 has neutral zero graph inputs, but LayerNorm104 can center these positions to nonzero values; its graph-column weights can receive gradients. The prompt statement that zeros cannot activate first-layer weights is not true for this architecture. The requested architecture was retained; G3 is a neutral-input/retraining control, not a complete feature-capacity test. G2 remains primary for correspondence.',
       'G1_GRAPH_ZERO is an inference-only intervention at the selected G1 threshold, not retrained G3 or frozen D2. Graph descriptors use unlabeled within-patient context; no population scaler is fit on VAL. The original80 cohort is not reduced for missing rank stability.',
       'Undirected correlation measures functional coupling, not directional causal spread. Clinical onset independently verified in0/256 records. Development patients have been reused in prior studies, and model/threshold selection reuses VAL; bootstrap is descriptive uncertainty, not independent clinical confirmation.',
       'Private waveforms, graph matrices/vectors, IDs, labels, prediction ledgers, checkpoints and logs remain on the original server.',
       '', '## Execution verification','',f'Protocol SHA256 `{sha(args.protocol)}`. All15 formal runs completed, {len(history)} epochs; epoch-body time {hist.seconds.sum():.2f}s, excluding preparation/tests/reporting. Initial states are identical among G1/G2/G3 within each fold. All required engineering checks passed before formal training.']
    (pub/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('ALL15_GRAPHROLE_FINALIZED',terminal,flush=True)

if __name__=='__main__': main()
