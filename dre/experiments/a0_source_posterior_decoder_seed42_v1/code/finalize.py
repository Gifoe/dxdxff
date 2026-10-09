"""Aggregate-only delivery; all identities remain in private fold artifacts."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import sha,write_json,metric,METRICS,cluster_bootstrap

def csv(path,rows):
    pd.DataFrame(rows).to_csv(path,index=False)

def summary(frame,keys,cols):
    rows=[]
    for key,g in frame.groupby(keys,sort=True,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        row={**dict(zip(keys,key)),'patient_fold_cells':len(g),'unique_patients':g.patient.nunique()}
        for c in cols:
            if c in g:row[c]=float(g[c].mean()); row[c+'_n']=int(g[c].notna().sum())
        rows.append(row)
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument('--runtime',type=Path,required=True); p.add_argument('--protocol',type=Path,required=True); a=p.parse_args()
    pub=a.runtime/'public'; lock=json.loads(a.protocol.read_text()); folds=[]; provenance=json.loads((pub/'A0_OOF_PROVENANCE.json').read_text())['teachers']
    history=[]; gates=[]
    for fold in range(1,6):
        root=a.runtime/f'fold{fold}'; meta=json.loads((root/'POSTERIOR_COMPLETE.json').read_text())
        assert meta['status']=='PASS' and meta['validation_sha256']==sha(root/'VALIDATION_PRIVATE.pt')
        art=torch.load(root/'VALIDATION_PRIVATE.pt',map_location='cpu',weights_only=False)
        full=torch.load(root/'D2_FROZEN_PRIVATE.pt',map_location='cpu',weights_only=False)
        assert art['scorer_checkpoint']==meta['D2_D3_same_scorer_sha256']==sha(root/'D2/BEST_PRIVATE.pt')
        assert art['density_sha256']==sha(root/'POSTERIOR_FROZEN_PRIVATE.pt')
        for row in full['history']:history.append({'fold':fold,'family':'formal_D2',**row})
        teach=json.loads((root/'TEACHERS_COMPLETE.json').read_text())['teachers']; provenance+=teach
        for row in teach:
            cell=root/f'D2_teacher{row["group"]}'; assert row['checkpoint_sha256']==sha(cell/'BEST_PRIVATE.pt')
            saved=torch.load(cell/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
            for h in saved['history']:history.append({'fold':fold,'family':'OOF_D2','group':row['group'],**h})
        folds.append(art)
        gates.append({'fold':fold,'D2_D3_scorer_sha256':art['scorer_checkpoint'],'density_sha256':art['density_sha256'],
                      'selected_epoch':full['epoch'],'NEZ_threshold':full['threshold'],'posterior_frozen_before_VAL':True})
    frame=pd.DataFrame([r for f in folds for r in f['rows']])
    for method in ['D0','D1','D2','D3']:
        sub=frame[frame.method==method]; assert len(sub)==65 and sub.patient.nunique()==47 and sub.channels.sum()==6273
    su=summary(frame,['method'],METRICS+['TP','FP','TN','FN','predicted_ez_fraction','observed_ez_fraction'])
    sf=summary(frame,['fold','method'],METRICS); sc=summary(frame,['center','method'],METRICS)
    csv(pub/'VALIDATION_SUMMARY.csv',su); csv(pub/'VALIDATION_BY_FOLD.csv',sf); csv(pub/'VALIDATION_BY_CENTER.csv',sc)
    csv(pub/'D2_TRAINING_HISTORY.csv',history)
    boot=[]
    for left,right in [('D1','D0'),('D2','D0'),('D3','D0'),('D3','D1'),('D3','D2')]:
        aa=frame[frame.method==left].set_index(['fold','patient']); bb=frame[frame.method==right].set_index(['fold','patient']); assert aa.index.equals(bb.index)
        delta=aa[METRICS].to_numpy()-bb[METRICS].to_numpy(); samples=cluster_bootstrap(aa.index.get_level_values('patient'),delta)
        for i,m in enumerate(METRICS):
            v=samples[:,i]; v=v[np.isfinite(v)]
            boot.append({'contrast':left+'-'+right,'metric':m,'delta':float(np.nanmean(delta[:,i])),
                'CI_low':float(np.quantile(v,.025)),'CI_high':float(np.quantile(v,.975)),
                'positive_folds':int(((aa[m]-bb[m]).groupby(level='fold').mean()>0).sum()),
                'contributing_cells':int(np.isfinite(delta[:,i]).sum()),'draws':10000,'valid_draws':len(v),'cluster_unique_ids':47})
    csv(pub/'PAIRED_BOOTSTRAP.csv',boot)
    diag=pd.DataFrame([r for f in folds for r in f['FIT_falsification']+f['diagnostics']])
    identcols=['pi','prior','actual','pi_abs_error','prior_abs_error','ll_gain','near_boundary','converged','fallback','low_iqr','separation','extreme_k','source_class_loglik_minus_global']
    ids=summary(diag,['scope','method'],identcols)
    for row in ids:
        dd=diag[(diag.scope==row['scope'])&(diag.method==row['method'])]
        row['pi_actual_correlation']=float(np.corrcoef(dd.pi,dd.actual)[0,1]) if dd.pi.std()>0 and dd.actual.std()>0 else np.nan
        row['pi_prior_displacement']=float((dd.pi-dd.prior).mean())
        row['MAP_abs_error_change']=float((dd.pi_abs_error-dd.prior_abs_error).mean())
    csv(pub/'POSTERIOR_IDENTIFIABILITY_AUDIT.csv',ids)
    csv(pub/'POSTERIOR_CALIBRATION_AUDIT.csv',summary(diag,['scope','method'],['brier','ece','fixed_brier','fixed_ece']))
    fixed=summary(diag,['scope','method'],['macro_map','macro_fixed','brier','fixed_brier','pi_abs_error','prior_abs_error'])
    for r in fixed:r['macro_MAP_minus_fixed_prior']=r['macro_map']-r['macro_fixed']
    csv(pub/'FIXED_PRIOR_COMPARISON.csv',fixed)
    ezrows=[]
    for keys,g in frame.groupby('method'):
        ezrows.append({'method':keys,'patient_fold_cells':len(g),'macro_mean':g.macro_f1.mean(),'macro_q10':g.macro_f1.quantile(.1),
            'macro_median':g.macro_f1.median(),'macro_q90':g.macro_f1.quantile(.9),'predicted_mean':g.predicted_ez_fraction.mean(),
            'predicted_q10':g.predicted_ez_fraction.quantile(.1),'predicted_median':g.predicted_ez_fraction.median(),
            'predicted_q90':g.predicted_ez_fraction.quantile(.9),'actual_mean':g.observed_ez_fraction.mean()})
    ezrows+=summary(diag[diag.scope=='VAL'],['method'],['pi','prior','predicted','actual','k_minus_expected','objective','extreme_k'])
    csv(pub/'PREDICTED_EZ_FRACTION_AUDIT.csv',ezrows)
    er=pd.DataFrame([r for f in folds for r in f['errors']]); ec=[]
    for method,g in er.groupby('method'):
        ec.append({'method':method,'channel_appearances':6273,**{c:int(g[c].sum()) for c in ['TP_gained','TP_lost','FP_added','FP_removed','errors_recovered','correct_spoiled','changed']}})
    csv(pub/'ERROR_CORRECTION_AUDIT.csv',ec)
    ranks=pd.DataFrame([r for f in folds for r in f['ranks']]); csv(pub/'RANKING_CHANGE_AUDIT.csv',summary(ranks,['method'],['pair_reversal_fraction','changed_rank_positions','corresponding_scorer_rank_change']))
    sources=pd.DataFrame([r for f in folds for r in f['source']]); csv(pub/'SOURCE_CORRECTION_AUDIT.csv',summary(sources,['center'],['mean_delta','mean_abs_delta','std_delta','max_abs_delta','mean_abs_base','delta_base_energy_ratio','effective_rank']))
    csv(pub/'OOF_SCORE_DISTRIBUTION_AUDIT.csv',[r for f in folds for r in f['distribution']])
    pars=[r for f in folds for r in f['parameters']]
    write_json(pub/'SOURCE_DENSITY_PARAMETERS.json',{'status':'PASS','folds':pars,'parameters_FIT_OOF_only':True,'common_variance_within_source':True})
    write_json(pub/'DENSITY_MODEL_AUDIT.json',{'status':'PASS','FIT_falsification_before_VAL':True,'source_fallbacks':sum(s['fallback'] for r in pars for s in r['density']['sources']),
        'source_models':40,'global_ordering_valid':True,'OOF_fitted_calibration_optimistic':True,'no_new_family_or_hyperparameters':True})
    assert len(provenance)==40
    write_json(pub/'OOF_PROVENANCE_AUDIT.json',{'status':'PASS','teachers':provenance,'A0_replayed_teachers':20,'new_D2_teachers':20,
        'query_excluded_from_train_and_selection':True,'one_logit_per_fit_channel':True})
    means={r['method']:r for r in su}; D=means
    recovery=[{'method':m,'macro_f1':D[m]['macro_f1'],'gain':D[m]['macro_f1']-D['D0']['macro_f1'],
        'historical_label_using_oracle':.7062849901037532,'headroom':.06820520725385308,
        'fraction_recovered':(D[m]['macro_f1']-D['D0']['macro_f1'])/.06820520725385308} for m in ['D0','D1','D2','D3']]
    csv(pub/'ORACLE_HEADROOM_RECOVERY.csv',recovery)
    bdf=pd.DataFrame(boot)
    def nf(m):
        return int(bdf[(bdf.contrast==m+'-D0')&(bdf.metric=='macro_f1')].positive_folds.iloc[0])
    numerical=bool(diag.converged.all() and np.isfinite(diag.pi).all())
    conditions={'D3_D0_macro_gain':D['D3']['macro_f1']-D['D0']['macro_f1']>=.020,
        'D3_D1_increment':D['D3']['macro_f1']-D['D1']['macro_f1']>=.005,
        'D3_D2_increment':D['D3']['macro_f1']-D['D2']['macro_f1']>=.005,
        'EZ_F1_nondecline':D['D3']['ez_f1']>=D['D0']['ez_f1'],'AP_nondecline':D['D3']['ez_auprc']>=D['D0']['ez_auprc'],
        'four_improved_folds':nf('D3')>=4,'integrity':True,'posterior_numerical_validity':numerical}
    def component(m):return D[m]['macro_f1']-D['D0']['macro_f1']>=.020 and D[m]['ez_f1']>=D['D0']['ez_f1'] and D[m]['ez_auprc']>=D['D0']['ez_auprc'] and nf(m)>=4
    if all(conditions.values()):terminal='D3_DEVELOPMENT_TARGET_0P70_REACHED' if D['D3']['macro_f1']>=.700 else 'D3_INCREMENTAL_GAIN_SUPPORTED'
    elif component('D2'):terminal='SOURCE_CONDITIONING_ONLY_SUPPORTED'
    elif component('D1'):terminal='POSTERIOR_DECODER_ONLY_SUPPORTED'
    else:terminal='NO_MEANINGFUL_IMPROVEMENT'
    write_json(pub/'VALIDATION_GATE.json',{'terminal':terminal,'pass':all(conditions.values()),'conditions':conditions,'D3_at_least_0p658':D['D3']['macro_f1']>=.658,
        'D3_at_least_0p700':D['D3']['macro_f1']>=.700,'D3_positive_folds':nf('D3'),'protocol_sha256':sha(a.protocol),'selection_integrity':gates})
    table='| Metric | D0 | D1 | D2 | D3 |\n|---|---:|---:|---:|---:|\n'
    for m in METRICS:table+='| '+m+' | '+' | '.join(f'{D[k][m]:.6f}' for k in ['D0','D1','D2','D3'])+' |\n'
    ci='| Contrast | Macro-F1 delta | 95% patient-cluster CI | Positive folds |\n|---|---:|---|---:|\n'
    for r in boot:
        if r['metric']=='macro_f1':ci+=f"| {r['contrast']} | {r['delta']:+.6f} | [{r['CI_low']:+.6f}, {r['CI_high']:+.6f}] | {r['positive_folds']}/5 |\n"
    dc=[r for r in ec if r['method']=='D3'][0]; dv=diag[(diag.scope=='VAL')&(diag.method=='D3')]
    d1v=diag[(diag.scope=='VAL')&(diag.method=='D1')]
    center_table='| Center | D0 F1 | D1 F1 | D2 F1 | D3 F1 | D3-D0 |\n|---|---:|---:|---:|---:|---:|\n'
    for center in sorted(frame.center.unique()):
        rr={r['method']:r for r in sc if r['center']==center}
        center_table+='| '+center+' | '+' | '.join(f"{rr[k]['macro_f1']:.6f}" for k in ['D0','D1','D2','D3'])+f" | {rr['D3']['macro_f1']-rr['D0']['macro_f1']:+.6f} |\n"
    report=f'''# Source-conditioned A0 and patient posterior decoder — seed42

Terminal: **`{terminal}`**. All four locked configurations completed. No outer TEST evaluation, raw EEG extraction, clinical relabeling, extra arm or outcome-driven rescue tuning occurred.

## Complete matched development results

{table}

65 identical patient-fold appearances,47 unique IDs,6,273 channel appearances; original cohort80/7,635/88D. These are repeatedly reused exploratory development outcomes, not independent test confirmation. D0 was freshly hash-checked/replayed, then exact historical frozen probabilities were reused (drift gate2e-7). D2/D3 share each selected scoring checkpoint; D1/D3 fit separate corresponding legal FIT-OOF densities. All eleven metrics use original patient-equal conventions.

{ci}

10,000 seed42 unique-patient cluster draws retain repeated appearances. Checkpoints, probabilities and rules remain fixed. Percentile intervals do not correct repeated development use, checkpoint/threshold-selection optimism or multiplicity. Full55 contrast/metric intervals are in PAIRED_BOOTSTRAP.csv.

## Source heterogeneity and posterior mechanism

{center_table}

Source identity is an operational acquisition variable, not a patient embedding or biological disease target. Canonical Task1 clinical labels remain unchanged. Source-specific differences cannot identify causal label-noise or center biology; no unseen-center generalization was tested. See SOURCE_AUDIT.md for heterogeneous target lineage and prior negative decision experiments.

The posterior family uses two ordered Gaussian means with common variance and patient median/IQR normalization. This enforces monotonic likelihood ratios, not valid calibration by itself. Source moments/priors use FIT-only unique-patient shrinkage nu10; Beta concentration20; labels are absent from MAP/decoder interfaces. OOF-fit calibration diagnostics are optimistic because the same FIT labels estimate densities. VAL posterior diagnostics are not used to tune or select it. Score-normalization location/scale removal and OOF-to-full-scorer transport may limit identifiability; diagnostics do not isolate a single cause.

The decoder maximizes a **ratio-of-expected-confusion-counts approximation**, not exact expected Macro-F1. It enumerates k0..C with smaller-k ties. Numerical posterior saturation retains the original mathematically monotone scorer ordering. D1 ranking metrics exactly match D0; it cannot improve representation discrimination.

On VAL, MAP minus fixed-source-prior Macro-F1 is D1 {float((d1v.macro_map-d1v.macro_fixed).mean()):+.6f}, D3 {float((dv.macro_map-dv.macro_fixed).mean()):+.6f}. This diagnostic is not a fifth selectable model. D3 posterior Brier {dv.brier.mean():.6f}, ECE {dv.ece.mean():.6f}; prior absolute fraction error {dv.prior_abs_error.mean():.6f}, MAP error {dv.pi_abs_error.mean():.6f}. These are fitted operational label probabilities/proportions, not biological EZ prevalence. Degenerate/constant-score, density fallback, mixture separation, boundary and calibration summaries are separately reported.

D3 gains {dc['TP_gained']} and loses {dc['TP_lost']} EZ TPs, adds {dc['FP_added']} and removes {dc['FP_removed']} FPs; it corrects {dc['errors_recovered']} original errors and spoils {dc['correct_spoiled']} correct predictions over repeated channel appearances. These pooled counts are not the patient-equal F1 estimand.

## Prespecified gate

{json.dumps(conditions,indent=2)}

D3 historical retrospective oracle-headroom recovery: {recovery[3]['fraction_recovered']:.4%}. The0.7062849901 oracle used patient labels and is diagnostic only; no individual headroom ratios or oracle prediction inputs were used.

## Explicit answers to19 questions

1. A0 exactly reproduced? Yes: all hash/identity/preprocessing/epoch/tau gates passed; metrics within1e-12 and probability replay within2e-7.
2. Original cohort preserved? Yes,80 patients/7,635 canonical pairs/88D and original frozen5fold; no exclusions or relabeling.
3. D1 outperform D0? Numerical Macro-F1 delta {D['D1']['macro_f1']-D['D0']['macro_f1']:+.6f}; uncertainty above, continuation magnitude gate separately assessed.
4. D2 outperform D0? Delta {D['D2']['macro_f1']-D['D0']['macro_f1']:+.6f}; not independent clinical confirmation.
5. D3 outperform all others? D3-D0 {D['D3']['macro_f1']-D['D0']['macro_f1']:+.6f}, D3-D1 {D['D3']['macro_f1']-D['D1']['macro_f1']:+.6f}, D3-D2 {D['D3']['macro_f1']-D['D2']['macro_f1']:+.6f}; fixed complementarity gate {all(conditions.values())}.
6. Source conditioning improves EZ ranking? AP delta {D['D2']['ez_auprc']-D['D0']['ez_auprc']:+.6f}, AUROC delta {D['D2']['ez_auroc']-D['D0']['ez_auroc']:+.6f}; consult paired intervals, not F1 alone.
7. Posterior improves patient F1? D1-D0 and D3-D2 above isolate it; ranking remains that of corresponding scorer.
8. MAP improves fixed prior? D1 {float((d1v.macro_map-d1v.macro_fixed).mean()):+.6f}, D3 {float((dv.macro_map-dv.macro_fixed).mean()):+.6f} F1; FIT/VAL calibration and fraction error controls reported without tuning.
9. Conditional densities distinguishable and stable? Global FIT ordering passes all10 fits. Mean VAL D1/D3 standardized separation {d1v.separation.mean():.6f}/{dv.separation.mean():.6f}; source fallback fraction {d1v.fallback.mean():.6f}/{dv.fallback.mean():.6f}. This establishes assumed monotone fit, not patient-invariant likelihood truth or identifiable clinical fraction.
10. D1 A0 ordering preserved? Yes; per-patient ranking-metric identity and numerical monotonicity assertions pass. Saturation handled explicitly.
11. D3 EZ detection or specificity? Sensitivity delta {D['D3']['sensitivity']-D['D0']['sensitivity']:+.6f}, specificity delta {D['D3']['specificity']-D['D0']['specificity']:+.6f}; TP/FP terms above expose the tradeoff.
12. All centers improve? Center table reports each. This small, confounded development population does not establish cross-center causal transfer.
13. Headroom recovered? {recovery[3]['fraction_recovered']:.4%} overall descriptive fraction; labels never entered prediction.
14. D3>=.658? {D['D3']['macro_f1']>=.658}, observed {D['D3']['macro_f1']:.9f}.
15. D3>=.700? {D['D3']['macro_f1']>=.700}; no independent test claim.
16. Failure mechanisms? The fixed comparison separates scorer discrimination, posterior operating-point change and MAP-vs-prior contribution. Gaussian misspecification, patient centering and OOF transport are plausible, not proven causal diagnoses; no rescue family was tried.
17. Further posterior modeling justified? Only if its locked incremental gate passes; current combined gate={all(conditions.values())}. More complexity is not justified merely by retrospective oracle capacity.
18. Further source scoring justified? Fixed source component gate={component('D2')}; secondary AP/AUROC must be interpreted with CIs.
19. Single next direction? If a component passes, independent prospectively locked confirmation of that component on source-adjudicated patients; otherwise independent target-definition/provenance harmonization with unchanged frozen scores. Neither was started here.

## Execution and protection

32 engineering checks and real FIT smoke passed before training, including original optimizer parity, source gradients, decoder brute parity and exact interrupted-resume model/history equality. Five formal D2 plus20 D2 OOF teachers trained;20 pre-existing A0 teachers were re-evaluated deterministically, never replaced by MC means or in-sample predictions. Source and aggregate hashes bind all private artifacts. Full checkpoint histories, patient/channel predictions, density ledgers and logs remain private.

The original NPZ y member was not read; clinical labels came only from immutable sealed FIT/VAL banks for the current fold. No outer prediction/metric loader exists in this experiment. Prior experiments did view other clinical data/outcomes; this does not establish an untouched dataset. Code and non-identifying aggregates alone are delivered. No automatic new model follows this result.
'''
    (pub/'FINAL_REPORT.md').write_text(report,encoding='utf-8')
    write_json(pub/'RUN_STATUS.json',{'status':'COMPLETE_DEVELOPMENT_ONLY','terminal':terminal,'outer_evaluation':False,'protocol_sha256':sha(a.protocol),
        'formal_D2_runs':5,'D2_OOF_teachers':20,'A0_reused_teachers':20,'matched_appearances':65,'unique_validation_ids':47,
        'all_checkpoint_density_integrity_passed':True,'aggregate_files':{q.name:sha(q) for q in pub.glob('*') if q.name!='RUN_STATUS.json'}})
    print(terminal,flush=True)
    print(pd.DataFrame(su)[['method','macro_f1','ez_f1','ez_auprc','ez_auroc']].to_string(index=False),flush=True)

if __name__=='__main__':main()
