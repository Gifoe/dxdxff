"""Reporting-only completion of a prespecified scientific density blockage.

No scorer training, model-family change, mean flip or prediction rescue is allowed.
Complete D0/D2 metrics and audit each frozen FIT density independently. D1/D3
are not reported as full-cohort methods when any fold is non-identifiable.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def main():
    p=argparse.ArgumentParser(); p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True); a=p.parse_args()
    code=Path(__file__).parents[1]/'code'; sys.path.insert(0,str(code))
    from common import sha,digest,write_json,metric,METRICS,cluster_bootstrap
    from model import SourceA0,SOURCES
    from posterior import fit_density,normalize,infer,decode,calibration
    torch.set_num_threads(2); pub=a.runtime/'public'; lock=json.loads(a.protocol.read_text())
    tests=json.loads((pub/'DECODER_UNIT_TEST_AUDIT.json').read_text()); assert tests['status']=='PASS'
    assert tests['protocol_sha256']==sha(a.protocol) and tests['code_sha256']=={q.name:sha(q) for q in code.glob('*.py')}
    history=[]; provenance=json.loads((pub/'A0_OOF_PROVENANCE.json').read_text())['teachers']
    scores=[]; densityparams=[]; densityaudits=[]; fitdiag=[]; dist=[]; errors=[]; ranks=[]; residuals=[]; selections=[]; replay=[]
    for fold in range(1,6):
        root=a.runtime/f'fold{fold}'; d=torch.load(root/'BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
        full=torch.load(root/'D2_FROZEN_PRIVATE.pt',map_location='cpu',weights_only=False)
        of=torch.load(root/'D2_OOF_PRIVATE.pt',map_location='cpu',weights_only=False)
        ts=torch.load(root/'D2/BEST_PRIVATE.pt',map_location='cpu',weights_only=False)
        assert full['binding']==of['binding'] and full['checkpoint_sha256']==sha(root/'D2/BEST_PRIVATE.pt')
        meta=json.loads((root/'D2/COMPLETE_PRIVATE.json').read_text())
        assert meta['best_sha256']==full['checkpoint_sha256'] and meta['last_sha256']==sha(root/'D2/LAST_PRIVATE.pt')
        assert ts['epoch']==full['epoch'] and ts['threshold']==full['threshold']
        model=SourceA0().cuda().eval(); model.load_state_dict(ts['model'])
        with torch.no_grad():
            fresh=model(torch.as_tensor(d['x'],device='cuda'),torch.as_tensor(d['g'],device='cuda'))
            probs=torch.sigmoid(fresh).cpu().numpy()
        drift=float(np.max(np.abs(probs-full['scores']))); assert drift<=2e-7
        replay.append({'fold':fold,'checkpoint_sha256':full['checkpoint_sha256'],'probability_drift':drift,'D2_D3_same_scorer':True})
        selections.append({'fold':fold,'epoch':full['epoch'],'threshold':full['threshold'],'completed_epochs':len(full['history']),
                           'D2_D3_checkpoint_sha256':full['checkpoint_sha256']})
        for h in full['history']:history.append({'fold':fold,'family':'formal_D2',**h})
        tr=d['train']; va=d['val']; pid=d['patient'][tr]; labels=d['y'][tr]; g=d['g'][tr]
        assert set(pid).isdisjoint(d['patient'][va]) and np.array_equal(of['patient'],pid) and np.array_equal(of['channel'],d['channel'][tr])
        assert of['audits']==json.loads((root/'TEACHERS_COMPLETE.json').read_text())['teachers']
        provenance+=of['audits']
        for teacher in of['audits']:
            k=teacher['group']; cell=root/f'D2_teacher{k}'; assert sha(cell/'BEST_PRIVATE.pt')==teacher['checkpoint_sha256']
            cm=json.loads((cell/'COMPLETE_PRIVATE.json').read_text())
            assert cm['best_sha256']==teacher['checkpoint_sha256'] and cm['last_sha256']==sha(cell/'LAST_PRIVATE.pt')
            saved=torch.load(cell/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
            for h in saved['history']:history.append({'fold':fold,'family':'OOF_D2','group':k,**h})
        for method,logits in [('D1',d['a0_oof']),('D3',of['logits'])]:
            classrows={0:[],1:[]}; low=[]; raw_source=[]
            for pat in sorted(set(pid)):
                ix=np.flatnonzero(pid==pat); z,iqr=normalize(logits[ix]); low.append(iqr<=1e-5)
                for c in [0,1]:
                    v=z[labels[ix]==c]
                    if len(v):classrows[c].append({'mean':float(v.mean()),'second':float(np.mean(v*v)),'source':int(g[ix[0]])})
            means=[float(np.mean([r['mean'] for r in classrows[c]])) for c in [0,1]]
            second=[float(np.mean([r['second'] for r in classrows[c]])) for c in [0,1]]
            try:den=fit_density(logits,labels,pid,g,set(pid)); valid=True; reason='none'
            except ValueError as exc:
                assert 'POSTERIOR_IDENTIFIABILITY_FAILED' in str(exc); den=None; valid=False; reason='global_FIT_EZ_mean_not_below_NEZ'
            da={'fold':fold,'method':method,'scope':'FIT_OOF_ONLY','identifiable':valid,'block_reason':reason,
                'global_mu_ez':means[0],'global_mu_nez':means[1],'global_mean_gap_NEZ_minus_EZ':means[1]-means[0],
                'class_bearing_patients_EZ':len(classrows[0]),'class_bearing_patients_NEZ':len(classrows[1]),
                'common_variance_raw':float(np.mean(np.asarray(second)-np.asarray(means)**2)),
                'low_iqr_fraction':float(np.mean(low)),'force_mean_separation':False}
            densityaudits.append(da); densityparams.append({'fold':fold,'method':method,'identifiable':valid,
                'global_moments_diagnostic':da,'density':den,'invalid_density_used_for_inference':False})
            if valid:
                for pat in sorted(set(pid)):
                    ix=np.flatnonzero(pid==pat); ix=ix[np.argsort(d['channel'][tr][ix].astype(str),kind='stable')]
                    q,dd=infer(logits[ix],int(g[ix[0]]),den); qf,df=infer(logits[ix],int(g[ix[0]]),den,True)
                    ez,dec=decode(q,d['channel'][tr][ix],logits[ix]); ezf,de=decode(qf,d['channel'][tr][ix],logits[ix])
                    truth=float(np.mean(labels[ix]==0)); b,e=calibration(q,labels[ix]); bf,ef=calibration(qf,labels[ix])
                    fitdiag.append({'fold':fold,'method':method,'patient':pat,'scope':'VALID_DENSITY_FIT_OOF_FITTED_DIAGNOSTIC',
                        'pi':dd['pi'],'prior':dd['prior'],'actual':truth,'predicted':float(ez.mean()),'pi_abs_error':abs(dd['pi']-truth),
                        'prior_abs_error':abs(dd['prior']-truth),'brier':b,'ece':e,'fixed_brier':bf,'fixed_ece':ef,
                        'macro_map':metric(labels[ix],1-q,.5,pred=~ez,order_score=logits[ix])['macro_f1'],
                        'macro_fixed':metric(labels[ix],1-qf,.5,pred=~ezf,order_score=logits[ix])['macro_f1'],
                        'pi_prior_displacement':dd['pi']-dd['prior'],'ll_gain':dd['ll_gain'],'fallback':dd['fallback'],
                        'near_boundary':dd['near_boundary'],'low_iqr':dd['low_iqr'],'converged':dd['converged'],
                        'extreme_k':dec['k'] in [0,len(ix)]})
        for scope,idx in [('FIT_OOF',tr),('VAL_FULL_FIT_SCORER',va)]:
            for family,logits in [('A0',d['a0_oof'] if scope=='FIT_OOF' else d['a0_logits'][va]),('D2',of['logits'] if scope=='FIT_OOF' else full['logits'][va])]:
                for label,name in [(None,'all'),(0,'EZ'),(1,'NEZ')]:
                    v=logits if label is None else logits[d['y'][idx]==label]
                    dist.append({'fold':fold,'family':family,'scope':scope,'class':name,'channels':len(v),
                        'mean':float(v.mean()),'std':float(v.std()),'q05':float(np.quantile(v,.05)),
                        'median':float(np.median(v)),'q95':float(np.quantile(v,.95))})
        for pat in sorted(set(d['patient'][va])):
            ix=va[d['patient'][va]==pat]; ix=ix[np.argsort(d['channel'][ix].astype(str),kind='stable')]
            y=d['y'][ix]; s0=d['a0_scores'][ix]; s2=full['scores'][ix]; p0=s0>=d['A0_threshold']; p2=s2>=full['threshold']
            freshm=metric(y,probs[ix],full['threshold']); storedm=metric(y,s2,full['threshold'])
            for m in METRICS:assert np.isclose(freshm[m],storedm[m],rtol=0,atol=1e-12,equal_nan=True)
            for method,s,pred in [('D0',s0,p0),('D2',s2,p2)]:
                mm=metric(y,s,.5,pred=pred); scores.append({'fold':fold,'patient':pat,'center':str(d['center'][ix[0]]),'method':method,**mm})
                errors.append({'method':method,'TP_gained':int(((y==0)&p0&~pred).sum()),'TP_lost':int(((y==0)&~p0&pred).sum()),
                    'FP_added':int(((y==1)&p0&~pred).sum()),'FP_removed':int(((y==1)&~p0&pred).sum()),
                    'errors_recovered':int(((p0!=y)&(pred==y)).sum()),'correct_spoiled':int(((p0==y)&(pred!=y)).sum())})
                aa=np.sign(s0[:,None]-s0); bb=np.sign(s[:,None]-s); upper=np.triu(np.ones_like(aa,bool),1)
                ranks.append({'fold':fold,'method':method,'patient':pat,'pair_reversal_fraction':float(np.mean((aa*bb<0)[upper])),
                    'rank_position_change_fraction':float(np.mean(np.argsort(s0,kind='stable')!=np.argsort(s,kind='stable')))})
            delta=full['delta'][ix]; base=full['base'][ix]
            residuals.append({'patient':pat,'center':str(d['center'][ix[0]]),'fold':fold,'mean_delta':float(delta.mean()),
                'mean_abs_delta':float(np.abs(delta).mean()),'std_delta':float(delta.std()),'max_abs_delta':float(np.abs(delta).max()),
                'mean_abs_base':float(np.abs(base).mean()),'delta_base_energy_ratio':float(np.mean(delta**2)/max(np.mean(base**2),1e-12)),
                'effective_rank':int(np.sum(np.asarray(full['rank_singular_values'])>1e-6))})
        print('VERIFIED_FROZEN_SCORER_AND_FIT_DENSITY',fold,flush=True)
    blocked={m:[r['fold'] for r in densityaudits if r['method']==m and not r['identifiable']] for m in ['D1','D3']}
    assert blocked['D1'] or blocked['D3'],'Use original complete finalizer when all models identifiable'
    frame=pd.DataFrame(scores); assert len(frame)==130
    d0=frame[frame.method=='D0']; assert abs(d0.macro_f1.mean()-.6380797828499001)<1e-12
    def emit(name,rows):pd.DataFrame(rows).to_csv(pub/name,index=False)
    def agg(keys,metrics=METRICS):
        out=[]
        for key,gg in frame.groupby(keys,sort=True):
            if not isinstance(key,tuple):key=(key,)
            out.append({**dict(zip(keys,key)),'status':'COMPLETE','patient_fold_cells':len(gg),'unique_patients':gg.patient.nunique(),
                        **{m:float(gg[m].mean()) for m in metrics}})
        return out
    summary=agg(['method']); byfold=agg(['fold','method']); bycenter=agg(['center','method'])
    for m in ['D1','D3']:
        summary.append({'method':m,'status':'NOT_ESTIMABLE_COMPLETE_ARM_POSTERIOR_GATE_FAILED','patient_fold_cells':0,'unique_patients':0,
            **{k:np.nan for k in METRICS},'blocked_folds':';'.join(map(str,blocked[m]))})
        for f in range(1,6):byfold.append({'method':m,'fold':f,'status':'NOT_REPORTED_PARTIAL_POSTERIOR_ARM','patient_fold_cells':0,**{k:np.nan for k in METRICS}})
        for c in sorted(frame.center.unique()):bycenter.append({'method':m,'center':c,'status':'NOT_ESTIMABLE_COMPLETE_ARM_POSTERIOR_GATE_FAILED',**{k:np.nan for k in METRICS}})
    emit('VALIDATION_SUMMARY.csv',summary); emit('VALIDATION_BY_FOLD.csv',byfold); emit('VALIDATION_BY_CENTER.csv',bycenter)
    emit('D2_TRAINING_HISTORY.csv',history); emit('OOF_SCORE_DISTRIBUTION_AUDIT.csv',dist)
    emit('POSTERIOR_IDENTIFIABILITY_AUDIT.csv',densityaudits)
    write_json(pub/'SOURCE_DENSITY_PARAMETERS.json',{'status':'POSTERIOR_IDENTIFIABILITY_FAILED','fits':densityparams})
    write_json(pub/'DENSITY_MODEL_AUDIT.json',{'status':'POSTERIOR_IDENTIFIABILITY_FAILED','blocked_folds':blocked,
        'global_density_checks':densityaudits,'new_family_or_ordering_flip':False,'all_valid_FIT_fits_before_any_new_inference':True,
        'partial_VAL_inference_already_completed_folds':[1,2],'partial_metrics_not_combined_or_used_to_rescue':True})
    fd=pd.DataFrame(fitdiag); calibr=[]; fix=[]; numerical=[]
    for m,gp in fd.groupby('method'):
        numerical.append({'method':m,'scope':'ONLY_VALID_FIT_DENSITIES_DIAGNOSTIC','valid_fold_count':gp.fold.nunique(),
            'FIT_patient_fold_appearances':len(gp),**{c:float(gp[c].mean()) for c in ['pi','prior','actual','pi_abs_error','prior_abs_error','fallback','near_boundary','low_iqr','converged','extreme_k','ll_gain','pi_prior_displacement']},
            'pi_actual_correlation':float(np.corrcoef(gp.pi,gp.actual)[0,1])})
        calibr.append({'method':m,'scope':'ONLY_VALID_FIT_DENSITIES_OPTIMISTIC_DIAGNOSTIC','cells':len(gp),**{c:float(gp[c].mean()) for c in ['brier','ece','fixed_brier','fixed_ece']}})
        fix.append({'method':m,'scope':'ONLY_VALID_FIT_DENSITIES_DIAGNOSTIC_NOT_FULL_VALIDATION','cells':len(gp),
            'macro_MAP':float(gp.macro_map.mean()),'macro_fixed_prior':float(gp.macro_fixed.mean()),
            'macro_MAP_minus_fixed_prior':float((gp.macro_map-gp.macro_fixed).mean()),'MAP_abs_error':float(gp.pi_abs_error.mean()),'prior_abs_error':float(gp.prior_abs_error.mean())})
    emit('POSTERIOR_CALIBRATION_AUDIT.csv',calibr); emit('FIXED_PRIOR_COMPARISON.csv',fix); emit('POSTERIOR_VALID_FIT_DIAGNOSTIC.csv',numerical)
    write_json(pub/'OOF_PROVENANCE_AUDIT.json',{'status':'PASS','teachers':provenance,'A0_replayed_teachers':20,'D2_trained_teachers':20,
        'once_per_fit_channel':True,'query_excluded_from_train_selection':True,'no_outer_roles':True})
    aa=frame[frame.method=='D2'].set_index(['fold','patient']); bb=frame[frame.method=='D0'].set_index(['fold','patient']); assert aa.index.equals(bb.index)
    delta=aa[METRICS].to_numpy()-bb[METRICS].to_numpy(); boot=cluster_bootstrap(aa.index.get_level_values('patient'),delta)
    bootrows=[]
    for contrast in ['D1-D0','D2-D0','D3-D0','D3-D1','D3-D2']:
        for i,m in enumerate(METRICS):
            if contrast=='D2-D0':
                v=boot[:,i]; v=v[np.isfinite(v)]
                r={'status':'ESTIMABLE','delta':float(np.nanmean(delta[:,i])),'CI_low':float(np.quantile(v,.025)),
                   'CI_high':float(np.quantile(v,.975)),'positive_folds':int(((aa[m]-bb[m]).groupby(level='fold').mean()>0).sum()),'valid_draws':len(v)}
            else:r={'status':'NOT_ESTIMABLE_POSTERIOR_GATE_FAILED','delta':np.nan,'CI_low':np.nan,'CI_high':np.nan,'valid_draws':0}
            bootrows.append({'contrast':contrast,'metric':m,'draws_prespecified':10000,'cluster_unique_ids':47,**r})
    emit('PAIRED_BOOTSTRAP.csv',bootrows)
    er=pd.DataFrame(errors); erows=[]
    for m,gp in er.groupby('method'):erows.append({'method':m,'status':'COMPLETE','channel_appearances':6273,**{c:int(gp[c].sum()) for c in ['TP_gained','TP_lost','FP_added','FP_removed','errors_recovered','correct_spoiled']}})
    for m in ['D1','D3']:erows.append({'method':m,'status':'NOT_ESTIMABLE_POSTERIOR_GATE_FAILED'})
    emit('ERROR_CORRECTION_AUDIT.csv',erows)
    ra=pd.DataFrame(ranks); emit('RANKING_CHANGE_AUDIT.csv',[{'method':m,'scope':'COMPLETE_SCORER','pair_reversal_fraction':gp.pair_reversal_fraction.mean(),
        'rank_position_change_fraction':gp.rank_position_change_fraction.mean()} for m,gp in ra.groupby('method')]+[
        {'method':m,'scope':'POSTERIOR_FULL_ARM_BLOCKED','corresponding_scorer_monotonicity':'ordered valid Gaussians proven/tested, invalid density not applied'} for m in ['D1','D3']])
    rr=pd.DataFrame(residuals); emit('SOURCE_CORRECTION_AUDIT.csv',[{'center':c,'patient_fold_cells':len(gp),'unique_patients':gp.patient.nunique(),
        **{v:float(gp[v].mean()) for v in ['mean_delta','mean_abs_delta','std_delta','max_abs_delta','mean_abs_base','delta_base_energy_ratio','effective_rank']}} for c,gp in rr.groupby('center')])
    emit('PREDICTED_EZ_FRACTION_AUDIT.csv',[{'method':m,'status':'COMPLETE','macro_mean':gp.macro_f1.mean(),'macro_q10':gp.macro_f1.quantile(.1),
        'macro_median':gp.macro_f1.median(),'macro_q90':gp.macro_f1.quantile(.9),'predicted_mean':gp.predicted_ez_fraction.mean(),
        'predicted_q10':gp.predicted_ez_fraction.quantile(.1),'predicted_median':gp.predicted_ez_fraction.median(),
        'predicted_q90':gp.predicted_ez_fraction.quantile(.9),'actual_mean':gp.observed_ez_fraction.mean()} for m,gp in frame.groupby('method')]+[
            {'method':m,'status':'NOT_ESTIMABLE_POSTERIOR_GATE_FAILED'} for m in ['D1','D3']])
    D={r['method']:r for r in summary}; recovery=[]
    for m in ['D0','D1','D2','D3']:
        gain=D[m]['macro_f1']-D['D0']['macro_f1']; recovery.append({'method':m,'status':D[m]['status'],
            'macro_f1':D[m]['macro_f1'],'gain':gain,'oracle':.7062849901037532,'headroom':.06820520725385308,'fraction_recovered':gain/.06820520725385308})
    emit('ORACLE_HEADROOM_RECOVERY.csv',recovery)
    f1b=next(r for r in bootrows if r['contrast']=='D2-D0' and r['metric']=='macro_f1')
    source_support=D['D2']['macro_f1']-D['D0']['macro_f1']>=.020 and D['D2']['ez_f1']>=D['D0']['ez_f1'] and D['D2']['ez_auprc']>=D['D0']['ez_auprc'] and f1b['positive_folds']>=4
    write_json(pub/'VALIDATION_GATE.json',{'terminal':'POSTERIOR_IDENTIFIABILITY_FAILED','pass':False,'combined_continuation':'NOT_EVALUABLE_GLOBAL_DENSITY_FAILURE',
        'posterior_blocked_folds':blocked,'D2_component_continuation_pass':source_support,'D2_D0_macro_gain':f1b['delta'],
        'D3_targets_0p658_0p700':'NOT_ESTIMABLE','no_source_rank_density_prior_solver_or_threshold_rescue':True,'protocol_sha256':sha(a.protocol),'D2_selections':selections})
    write_json(pub/'INDEPENDENT_SCORER_REPLAY.json',{'status':'PASS','folds':replay,'metric_parity_tolerance':1e-12,'all_completed_checkpoint_hashes_passed':True})
    table='| Metric | D0 frozen A0 | D1 posterior | D2 source A0 | D3 source+posterior |\n|---|---:|---|---:|---|\n'
    for m in METRICS:table+=f"| {m} | {D['D0'][m]:.6f} | not_estimable | {D['D2'][m]:.6f} | not_estimable |\n"
    dt='| Fold | Posterior family | Global EZ mean | Global NEZ mean | NEZ−EZ | Valid |\n|---|---|---:|---:|---:|---|\n'
    for r in densityaudits:dt+=f"| {r['fold']} | {r['method']} | {r['global_mu_ez']:.6f} | {r['global_mu_nez']:.6f} | {r['global_mean_gap_NEZ_minus_EZ']:+.6f} | {r['identifiable']} |\n"
    ct='| Center | D0 Macro-F1 | D2 Macro-F1 | Delta |\n|---|---:|---:|---:|\n'
    for c in sorted(frame.center.unique()):
        m0=float(frame[(frame.center==c)&(frame.method=='D0')].macro_f1.mean()); m2=float(frame[(frame.center==c)&(frame.method=='D2')].macro_f1.mean())
        ct+=f'| {c} | {m0:.6f} | {m2:.6f} | {m2-m0:+.6f} |\n'
    eb=next(r for r in erows if r['method']=='D2')
    ci='| Metric | D2-D0 | 95% cluster CI | Positive folds |\n|---|---:|---|---:|\n'
    for r in bootrows:
        if r['contrast']=='D2-D0':ci+=f"| {r['metric']} | {r['delta']:+.6f} | [{r['CI_low']:+.6f},{r['CI_high']:+.6f}] | {r['positive_folds']}/5 |\n"
    tailnote='The separate NORMALIZATION_TAIL_AUDIT.csv diagnoses contributions without modifying normalization or refitting any density.'
    if (pub/'NORMALIZATION_TAIL_AUDIT.csv').exists():
        tail=pd.read_csv(pub/'NORMALIZATION_TAIL_AUDIT.csv')
        tt=tail[(tail.method=='D1')&(tail['class']=='NEZ')&(tail.fold==3)].iloc[0]
        tailnote=f"The read-only NORMALIZATION_TAIL_AUDIT.csv identifies1–2 low-IQR FIT patients per failed fold (about1.9–3.9%). Normalized absolute logits reach {tail.patient_max_abs_z_max.max():.1f}; fold3 D1 NEZ global mean {tt.global_mean:.6f} receives {tt.low_iqr_contribution_to_global_mean:.6f} from the low-IQR group, versus {tt.other_patients_contribution_to_global_mean:.6f} from all other patients' contribution. Folds4–5 similarly show EZ mean domination by low-IQR tails. Thus floor1e-5 limits division but does not bound tail influence on Gaussian first/second moments. This directly explains the computed pooled-order reversal in this implementation; it does not establish a biological cause. No patient was excluded, score clipped, density refitted with a different rule or validation result rescued."
    report=f'''# Source-conditioned A0 and patient posterior decoder — completed seed42 audit

**Terminal: `POSTERIOR_IDENTIFIABILITY_FAILED`.** The frozen Gaussian posterior assumption fails its prespecified global FIT-OOF class-ordering requirement. No label flip, forced mean separation, alternative density, hyperparameter search or threshold rescue was performed. Five formal D2 scorers and all20 D2 OOF teachers completed;20 original A0 teachers were freshly replayed. D0/D2 full development metrics are available. D1/D3 are **not estimable as complete65-cell arms**, not assigned zero or partial means.

## All full-arm metrics

{table}

Same65 patient-fold appearances,47 unique IDs,6,273 channel appearances; original80/7,635/88D cohort/frozen5fold. A0 replay within2e-7 and all original metrics within1e-12 passed. Exact frozen A0 probabilities are reused only after that gate. D2 checkpoints/probabilities were independently replayed; D3 was bound to those exact checkpoints, never separately trained.

{ci}

10,000 seed42 paired unique-ID cluster draws preserve repeated appearances, fixed checkpoints and thresholds. All required posterior contrasts are explicitly not_estimable, because their complete arms failed FIT density validity. Intervals do not correct selection optimism, repeated development use or multiplicity. This is exploratory development, not outer test confirmation.

## The decisive FIT-only falsification

{dt}

D1 blocked folds: {blocked['D1']}; D3 blocked folds: {blocked['D3']}. These means use the frozen patient median/IQR normalized logits, equal patient weight within each observed class, and legal deterministic teacher predictions. They are not electrode-weighted estimates. Positive NEZ-minus-EZ is required. Common variance gives monotone EZ posterior only when this orientation holds. A nonpositive global gap prevents the assumed model from supplying a legal posterior: **source fallback cannot repair an invalid global density**.

Insufficient data and ordering are distinguished in POSTERIOR_IDENTIFIABILITY_AUDIT.csv; low-IQR fractions are also reported. Fixed-prior/MAP calibration diagnostics use only valid FIT densities and are explicitly optimistic/incomplete, not substitute validation results. Initial formal posterior inference completed only folds1–2 before the fold3 gate; no partial aggregate performance is used for selection or to claim D1/D3 success. Remaining folds' FIT gates were then audited separately. No invalid density generated target predictions.

{tailnote}

Patient centering/IQR normalization also removes location/scale information and can erase class-mixture information. Heterogeneous operational targets and OOF-to-full-FIT scorer transport remain possible additional limitations, not proven causes or evidence that all score posterior methods are impossible. The unchanged assumption was falsified here; it was not replaced after seeing outcomes.

## Source-conditioned scorer and errors

{ct}

D2 changes within-patient rankings and shares a jointly trained8817-parameter backbone plus404 source parameters. Exact zero initial correction and original shared initialization were verified; unknown source uses shared-head fallback. Source correction magnitudes, hidden low-rank energy ratios and effective ranks are in SOURCE_CORRECTION_AUDIT.csv. Source identity is an operational acquisition category, not biology, patient embedding or clinical-label predictor. No unseen-center transfer was evaluated.

D2 gains {eb['TP_gained']} / loses {eb['TP_lost']} EZ true positives and adds {eb['FP_added']} / removes {eb['FP_removed']} false positives. It corrects {eb['errors_recovered']} original errors and spoils {eb['correct_spoiled']} correct decisions over repeated channel appearances. Those pooled counts do not replace patient-equal F1. D2 historical oracle-headroom recovery is {recovery[2]['fraction_recovered']:.4%}; D1/D3 recovery is undefined. The historical label-using0.7062849901 oracle never enters inference.

## Explicit answers to19 questions

1. A0 exactly reproduced? Yes, all5 original checkpoint/preprocessor/identity/epoch/tau gates; metrics within1e-12, probabilities within2e-7.
2. Original cohort preserved? Yes,80 patients/7,635 canonical pairs/88 original features and frozen5fold. No clinical relabeling, alternative80-patient cohort or outer evaluation.
3. D1 outperform D0? Cannot establish: complete D1 is blocked on FIT global ordering. Do not average only successful folds.
4. D2 outperform D0? Macro-F1 delta {f1b['delta']:+.6f}, CI [{f1b['CI_low']:+.6f},{f1b['CI_high']:+.6f}]; fixed source continuation gate={source_support}.
5. D3 outperform all others? Not estimable. No fabricated full D3 result or complementarity claim.
6. Source conditioning improves ranking? AP delta {D['D2']['ez_auprc']-D['D0']['ez_auprc']:+.6f}, AUROC delta {D['D2']['ez_auroc']-D['D0']['ez_auroc']:+.6f}, with full paired intervals above.
7. Posterior improves patient Macro-F1? Not established: FIT validity blocks complete evaluation.
8. MAP improves fixed source prior? Only valid FIT-fitted diagnostics in FIXED_PRIOR_COMPARISON.csv are available; optimistic and incomplete, no full VAL benefit is claimed.
9. Conditional distributions distinguishable/stable? No under the required global monotone ordering in the failed folds. Adequate observations alone do not make this normalized Gaussian family valid.
10. D1 preserves A0 ordering? Synthetic tests and completed valid-fold assertions pass. Invalid global densities were not applied or flipped.
11. D3 improves EZ detection or specificity? Not estimable. D2 sensitivity/spec deltas {D['D2']['sensitivity']-D['D0']['sensitivity']:+.6f}/{D['D2']['specificity']-D['D0']['specificity']:+.6f}; these are not D3 metrics.
12. D3 improves all centers? Not estimable. Source-only D2 center effects are reported separately above.
13. A0 oracle headroom recovered? D2 {recovery[2]['fraction_recovered']:.4%}; D1/D3 undefined. No biological oracle claim or individual unstable recovery ratio.
14. D3 reaches.658? Not evaluable, not a numeric failure substituted for missing predictions.
15. D3 reaches.700? Not evaluable; no independent performance claim.
16. Main failure mechanism? Prespecified class-ordering failure in normalized FIT-OOF score moments. The separate arithmetic audit shows near-zero-IQR patients dominating first/second moments; the frozen epsilon floor does not prevent extreme tails. No solver, leakage or checkpoint failure caused this gate. Biological/source causes and OOF transport are not causally established.
17. Further posterior modeling justified? This particular Gaussian/common-variance/median-IQR mechanism is unsupported. Do not enlarge or tune it automatically.
18. Further source scoring justified? Fixed standalone source gate={source_support}; point gains and CIs above qualify its strength. Combined posterior success is not established.
19. Single next direction? A separately locked FIT-only score-collapse/normalization-identifiability audit on frozen A0/D2 outputs, before proposing any replacement posterior. First determine why within-patient IQR collapses while a few score tails remain. Do not tune this run or start another scoring model. This is a recommendation only; no new experiment was initiated.

## Engineering and integrity

32 named checks and real FIT smoke passed before real training, including original optimizer/selection parity,9221 parameters, source/shared gradients, exact zero initialization, legalOOF exclusions, decoder all-k/brute parity, constant/empty/unknown safety and deterministic interrupted resume. Source protocol has not changed.

Windows startup engineering deviations: initial hidden launcher produced no work; foreground startup revealed a null child ExitCode reporting problem after completed A0 replay. The PowerShell wrapper was repaired without changing Python/science, retaining all old logs. During fold3 teacher startup, Windows Application Error1000 confirmed python312.dll0xc0000005 before any teacher epoch. Exact-step resume completed remaining teachers, discarding no completed epoch. Later failures were the intended scientific ordering gate, not restarted/tuned away. Private logs/checkpoints remain preserved.

All-cohort NPZ clinical `y` member was never materialized in this run; development labels come from SHA-sealed original FIT/VAL banks. Actual OOF teacher checkpoints and final selected models are hash-verified. The final reporting-only script audits each failed density separately and independently replays completed scorers, without training, rescoring to improve outcomes, changing checkpoint/threshold, or adding a fifth model. Public files omit patient IDs, channel records, individual proportions, predictions, checkpoint tensors, caches and runtime logs.
'''
    (pub/'FINAL_REPORT.md').write_text(report,encoding='utf-8')
    write_json(pub/'RUN_STATUS.json',{'status':'COMPLETE_DEVELOPMENT_POSTERIOR_BLOCKED','terminal':'POSTERIOR_IDENTIFIABILITY_FAILED',
        'formal_D2_runs':5,'D2_OOF_teachers':20,'reused_A0_teachers':20,'outer_evaluation':False,'protocol_sha256':sha(a.protocol),
        'reporting_only_source_sha256':sha(__file__),'D0_D2_complete':True,'D1_D3_full_metrics_fabricated':False,
        'blocked_folds':blocked,'all_completed_scorer_checkpoint_replay_passed':True,
        'aggregate_files':{q.name:sha(q) for q in pub.glob('*') if q.name!='RUN_STATUS.json'}})
    print('POSTERIOR_IDENTIFIABILITY_FAILED',blocked,flush=True)
    print(pd.DataFrame(summary)[['method','macro_f1','ez_f1','ez_auprc','ez_auroc','status']].to_string(index=False),flush=True)

if __name__=='__main__':main()
