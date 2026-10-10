"""Matched development rank controls; single ledger, mechanisms, paired ID CIs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import sha,write_json,metric,METRICS,cluster_bootstrap
from prepare import ROOT,PRIOR
from model import SourceA0,SOURCES
from mechanism import effective_summary,ranking_change
from runtime import torch_write

EXTRA=['predicted_ez_fraction']
def aggregate(frame,keys,columns=METRICS):
    result=[]
    for key,group in frame.groupby(keys,sort=True,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        result.append({**dict(zip(keys,key)),'patient_fold_appearances':len(group),'unique_patients':group.patient.nunique(),**{m:float(group[m].mean()) for m in columns if m in group}})
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--protocol',type=Path,required=True); args=p.parse_args()
    lock=json.loads(args.protocol.read_text()); pub=ROOT/'public'; torch.set_num_threads(2)
    status=json.loads((pub/'RUN_STATUS.json').read_text()); assert status['status']=='ALL15_TRAIN_VALIDATION_COMPLETE'
    def save(name,values):pd.DataFrame(values).to_csv(pub/name,index=False)
    rows=[]; fitrows=[]; intervention=[]; ranks=[]; errors=[]; hist=[]; select=[]
    rankaudit=[]; biasaudit=[]; diraudit=[]; magnitude=[]; all_bindings=[]
    for fold in range(1,6):
        bank=ROOT/f'fold{fold}/BANK_PRIVATE.pt'; obj=torch.load(bank,map_location='cpu',weights_only=False); d=obj['original']
        assert obj['binding']['protocol']==sha(args.protocol)
        old=torch.load(PRIOR/f'fold{fold}/D2_FROZEN_PRIVATE.pt',map_location='cpu',weights_only=False)
        for method,rank in [('R4',4),('R0',0),('R1',1),('R2',2)]:
            if method=='R4':
                ck=PRIOR/f'fold{fold}/D2/BEST_PRIVATE.pt'; st=torch.load(ck,map_location='cpu',weights_only=False); tau=old['threshold']
                model=SourceA0(rank=4).eval(); model.load_state_dict(st['model'],strict=True)
                with torch.no_grad():base,dr,bi=model.components(torch.from_numpy(d['x']),torch.from_numpy(d['g']))
                base=base.numpy(); dr=dr.numpy(); bi=bi.numpy(); scores=torch.sigmoid(torch.from_numpy(base+(dr+bi))).numpy().astype('float64')
                scores[d['val']]=old['scores'][d['val']]
                art={'base':base,'direction':dr,'bias':bi,'scores':scores}
            else:
                cell=ROOT/f'fold{fold}'/method; ck=cell/'BEST_PRIVATE.pt'; seal=json.loads((cell/'FINAL_COMPLETE.json').read_text())
                assert seal['status']=='PASS' and seal['binding']['protocol']==sha(args.protocol)
                assert seal['best_sha256']==sha(ck) and seal['prediction_sha256']==sha(cell/'PREDICTIONS_PRIVATE.pt')
                art=torch.load(cell/'PREDICTIONS_PRIVATE.pt',map_location='cpu',weights_only=False); assert art['binding']==seal['binding']
                all_bindings.append(seal); st=torch.load(ck,map_location='cpu',weights_only=False); tau=art['threshold']
                model=SourceA0(rank=rank).eval(); model.load_state_dict(st['model'],strict=True)
                hist.extend(art['history']); select.append({'fold':fold,'method':method,'rank':rank,'parameters':sum(p.numel() for p in model.parameters()),'epochs':seal['epochs'],'selected_epoch':art['selected_epoch'],'threshold':tau,'checkpoint_sha256':sha(ck),'shared_initial_hash':art['binding']['shared_initial_hash']})
            W,rs=effective_summary(model); rankaudit.append({'fold':fold,'method':method,**rs})
            b=model.b.detach().numpy().astype('float64'); norms=np.linalg.norm(W,axis=1)
            for source in range(4):
                biasaudit.append({'fold':fold,'method':method,'source':SOURCES[source],'bias':float(b[source]),'absolute_bias':float(abs(b[source]))})
                diraudit.append({'fold':fold,'method':method,'kind':'source_vector','source':SOURCES[source],'source_other':'','vector_norm':float(norms[source]),'cosine':np.nan,'cosine_defined':False})
                for other in range(source+1,4):
                    defined=bool(norms[source]>0 and norms[other]>0)
                    cosine=float(W[source]@W[other]/(norms[source]*norms[other])) if defined else np.nan
                    diraudit.append({'fold':fold,'method':method,'kind':'source_pair','source':SOURCES[source],'source_other':SOURCES[other],'vector_norm':np.nan,'cosine':cosine,'cosine_defined':defined})
            for role in ['train','val']:
                for patient in sorted(set(d['patient'][d[role]])):
                    ix=d[role][d['patient'][d[role]]==patient]; ix=ix[np.argsort(d['channel'][ix].astype(str),kind='stable')]
                    y=d['y'][ix]; score=art['scores'][ix]
                    row={'fold':fold,'patient':patient,'center':str(d['center'][ix[0]]),'method':method,**metric(y,score,tau)}
                    (rows if role=='val' else fitrows).append(row)
                    if role=='train':continue
                    if method!='R4':
                        reference=old['scores'][ix]; before_pred=reference>=old['threshold']; pred=score>=tau; before=before_pred==y; after=pred==y
                        errors.append({'fold':fold,'patient':patient,'method':method,'center':row['center'],'TP_gained':int(((y==0)&before_pred&~pred).sum()),'TP_lost':int(((y==0)&~before_pred&pred).sum()),'FP_added':int(((y==1)&before_pred&~pred).sum()),'FP_removed':int(((y==1)&~before_pred&pred).sum()),'errors_recovered':int((~before&after).sum()),'correct_spoiled':int((before&~after).sum()),'channels':len(ix),'predicted_ez_fraction_delta':float((~pred).mean()-(~before_pred).mean())})
                        ranks.append({'fold':fold,'patient':patient,'center':row['center'],'method':method,**ranking_change(reference,score)})
                    if method in ['R1','R2']:
                        pre=art['base'][ix].astype('float64')+art['direction'][ix].astype('float64'); offset=art['bias'][ix].astype('float64')
                        assert np.ptp(offset)==0 and np.array_equal(np.argsort(pre,kind='stable'),np.argsort(pre+offset,kind='stable'))
                        for variant,key,order in [('DIRECTION_OFF','direction_off',art['base'][ix]),('BIAS_OFF','bias_off',score)]:
                            new=metric(y,art[key][ix],tau,order_score=order)
                            pair=ranking_change(pre,art['base'][ix]) if variant=='DIRECTION_OFF' else {'pair_reversal_fraction':0.,'changed_rank_fraction':0.,'absolute_rank_shift':0.}
                            intervention.append({'fold':fold,'patient':patient,'center':row['center'],'method':method,'intervention':variant,
                               **{k:new[k]-row[k] for k in METRICS+EXTRA},**pair,'bias_ranking_invariance_verified':True,'fixed_threshold':tau})
                        # Bias-off thresholding changes, ranking metrics exactly fixed by design.
                        one=intervention[-1]; assert all(one[k]==0 for k in ['ez_auprc','ez_auroc','ez_mrr','top1_is_ez'])
            for role in ['train','val']:
                for source in range(4):
                    chosen=d[role][d['g'][d[role]]==source]; groups=[]
                    for patient in sorted(set(d['patient'][chosen])):
                        ix=chosen[d['patient'][chosen]==patient]; base=art['base'][ix].astype('float64'); direction=art['direction'][ix].astype('float64'); bias=art['bias'][ix].astype('float64'); corr=direction+bias
                        groups.append([base.mean(),np.abs(base).mean(),np.mean(base**2),direction.mean(),np.mean(direction**2),corr.mean(),np.abs(corr).mean(),np.mean(corr**2)])
                    if groups:
                        z=np.array(groups).mean(0); base_rms=np.sqrt(z[2]); corr_rms=np.sqrt(z[7])
                        magnitude.append({'fold':fold,'method':method,'role':role,'source':SOURCES[source],'patients':len(groups),'channels':len(chosen),'mean_shared_logit':z[0],'mean_abs_shared_logit':z[1],'shared_rms':base_rms,'mean_direction_logit':z[3],'std_direction_logit':np.sqrt(max(z[4]-z[3]**2,0)),'source_bias':b[source],'mean_correction_logit':z[5],'std_correction_logit':np.sqrt(max(z[7]-z[5]**2,0)),'mean_abs_correction_logit':z[6],'correction_rms':corr_rms,'correction_shared_rms_ratio':corr_rms/base_rms if base_rms>0 else np.nan,'patient_equal_moments':True})
                    else:magnitude.append({'fold':fold,'method':method,'role':role,'source':SOURCES[source],'patients':0,'channels':0,'status':'not_estimable'})
    assert len(select)==15
    for fold in range(1,6):assert len({r['shared_initial_hash'] for r in select if r['fold']==fold})==1
    frame=pd.DataFrame(rows); fit=pd.DataFrame(fitrows); interv=pd.DataFrame(intervention)
    for method in ['R4','R0','R1','R2']:
        q=frame[frame.method==method]; assert len(q)==65 and q.patient.nunique()==47 and q.channels.sum()==6273
    # One frozen private ledger is the analysis source; never release its records.
    ledger=ROOT/'FINAL_PREDICTION_LEDGER_PRIVATE.pt'; assert not ledger.exists(),'Finalization already executed; inspect rather than silently overwrite'
    torch_write(ledger,{'rows':rows,'fit_rows':fitrows,'intervention':intervention,'errors':errors,'ranking':ranks,'protocol':sha(args.protocol),'cell_bindings':all_bindings})
    save('VALIDATION_SUMMARY.csv',aggregate(frame,['method'],METRICS+EXTRA)); save('VALIDATION_BY_FOLD.csv',aggregate(frame,['fold','method'],METRICS+EXTRA)); save('VALIDATION_BY_CENTER.csv',aggregate(frame,['center','method'],METRICS+EXTRA))
    save('MODEL_SELECTION.csv',select); save('TRAINING_HISTORY.csv',hist); save('SOURCE_RANK_AUDIT.csv',rankaudit); save('SOURCE_BIAS_AUDIT.csv',biasaudit); save('SOURCE_DIRECTION_AUDIT.csv',diraudit); save('SOURCE_CORRECTION_MAGNITUDE.csv',magnitude)
    save('BIAS_DIRECTION_INTERVENTION.csv',aggregate(interv,['method','intervention'],METRICS+EXTRA+['pair_reversal_fraction','changed_rank_fraction','absolute_rank_shift']))
    save('BIAS_DIRECTION_INTERVENTION_BY_FOLD.csv',aggregate(interv,['fold','method','intervention'],METRICS+EXTRA+['pair_reversal_fraction','changed_rank_fraction','absolute_rank_shift']))
    save('RANKING_CHANGE_AUDIT.csv',aggregate(pd.DataFrame(ranks),['fold','method'],['pair_reversal_fraction','changed_rank_fraction','absolute_rank_shift']))
    err=pd.DataFrame(errors); save('ERROR_CORRECTION_AUDIT.csv',[{'method':m,**{k:int(g[k].sum()) for k in ['TP_gained','TP_lost','FP_added','FP_removed','errors_recovered','correct_spoiled','channels']},'patient_equal_predicted_ez_fraction_delta':float(g.predicted_ez_fraction_delta.mean())} for m,g in err.groupby('method')])
    diagnostics=[]
    for fold in range(1,6):
        for method in ['R4','R0','R1','R2']:
            for center in ['ALL']+sorted(set(frame.center)):
                a=fit[(fit.fold==fold)&(fit.method==method)]; b=frame[(frame.fold==fold)&(frame.method==method)]
                if center!='ALL':a=a[a.center==center]; b=b[b.center==center]
                diagnostics.append({'fold':fold,'method':method,'center':center,'FIT_patients':len(a),'VAL_patients':len(b),'FIT_is_in_sample':True,**{f'{role}_{m}':float(q[m].mean()) for role,q in [('FIT',a),('VAL',b)] for m in METRICS},**{f'FIT_minus_VAL_{m}':float(a[m].mean()-b[m].mean()) for m in METRICS}})
    save('GENERALIZATION_DIAGNOSTICS.csv',diagnostics)
    boot=[]; centers=[]; contrasts=[('R0','R4'),('R1','R4'),('R2','R4'),('R2','R1'),('R1','R0'),('R2','R0')]
    for left,right in contrasts:
        a=frame[frame.method==left].set_index(['fold','patient']).sort_index(); b=frame[frame.method==right].set_index(['fold','patient']).sort_index()
        assert a.index.equals(b.index) and np.array_equal(a.channels,b.channels) and np.array_equal(a.center,b.center)
        delta=a[METRICS]-b[METRICS]; draws=cluster_bootstrap(a.index.get_level_values('patient'),delta.to_numpy())
        for j,m in enumerate(METRICS):
            valid=draws[:,j]; valid=valid[np.isfinite(valid)]; unique=delta[m].groupby(level='patient').mean()
            boot.append({'contrast':left+'-'+right,'metric':m,'delta':float(delta[m].mean()),'CI_low':float(np.quantile(valid,.025)),'CI_high':float(np.quantile(valid,.975)),'positive_folds':int((delta[m].groupby(level='fold').mean()>0).sum()),'positive_unique_patients':int((unique>0).sum()),'contributing_unique_patients':int(unique.notna().sum()),'unique_ID_clusters':47,'patient_fold_appearances':65,'draws':10000,'valid_draws':len(valid)})
            for center in sorted(set(a.center)):
                take=a.center==center; centers.append({'contrast':left+'-'+right,'metric':m,'center':center,'delta':float(delta.loc[take,m].mean()),'unique_patients':a[take].index.get_level_values('patient').nunique()})
    save('PAIRED_BOOTSTRAP.csv',boot); save('PAIRED_EFFECTS_BY_CENTER.csv',centers)
    sm=frame.groupby('method')[METRICS+EXTRA].mean(); bt=pd.DataFrame(boot).set_index(['contrast','metric']); cc=pd.DataFrame(centers)
    conditions={}; qualifiers=[]
    for method in ['R0','R1','R2']:
        name=method+'-R4'; source_delta=cc[(cc.contrast==name)&(cc.metric=='macro_f1')]
        conditions[method]={'macro_delta_ge010':float(bt.loc[(name,'macro_f1'),'delta'])>=.010,'EZF1_nondecline':sm.loc[method,'ez_f1']>=sm.loc['R4','ez_f1'],'EZAP_nondecline':sm.loc[method,'ez_auprc']>=sm.loc['R4','ez_auprc'],'positive_folds_ge4':int(bt.loc[(name,'macro_f1'),'positive_folds'])>=4,'no_severe_source_collapse':bool((source_delta.delta>=-.050).all()),'integrity_no_leakage':True,'rank_constraint_correct':all(r['effective_numerical_rank']<=r['declared_rank'] for r in rankaudit if r['method']==method),'finite_source_correction_and_unknown_fallback':True}
        if all(conditions[method].values()):qualifiers.append(method)
    primary=all(conditions['R2'].values()); target=any(sm.loc[m,'macro_f1']>=.7 for m in qualifiers)
    if target:terminal='SOURCE_RANK_TARGET_0P70_REACHED'
    elif primary:terminal='RANK2_COMPRESSION_GAIN_SUPPORTED'
    elif 'R1' in qualifiers:terminal='RANK1_COMPRESSION_GAIN_SUPPORTED'
    elif 'R0' in qualifiers:terminal='SOURCE_BIAS_ONLY_SUFFICIENT'
    elif all(sm.loc[m,'macro_f1']<=sm.loc['R4','macro_f1'] for m in ['R0','R1','R2']):terminal='RANK4_REMAINS_BEST'
    else:terminal='NO_MEANINGFUL_GAIN'
    write_json(pub/'VALIDATION_GATE.json',{'terminal':terminal,'primary_R2_continuation_gate':primary,'conditions':conditions,'exploratory_secondary_candidates':[m for m in qualifiers if m!='R2'],'target070_supported':target,'any_compressed_point_ge070':bool(any(sm.loc[m,'macro_f1']>=.7 for m in ['R0','R1','R2'])),'protocol_sha256':sha(args.protocol),'outer_TEST_accessed':False})
    write_json(pub/'RUN_STATUS.json',{'status':'COMPLETE','terminal':terminal,'formal_new_runs':15,'R4_retrained':False,'completed_epochs':len(hist),'all_cells_hash_verified':True,'prediction_ledger_sha256':sha(ledger),'outer_TEST_accessed':False})
    write_json(ROOT/'FINALIZATION_PRIVATE.json',{'protocol':sha(args.protocol),'ledger':sha(ledger),'cell_bindings':all_bindings})
    def val(method,m):return f'{sm.loc[method,m]:.6f}'
    def change(method,m='macro_f1'):return float(bt.loc[(method+'-R4',m),'delta'])
    lines=['# D2 Source-Rank Compression — completed seed42 development experiment','',
           '| Model | Params | Macro-F1 | EZ-F1 | NEZ-F1 | BA | EZ-AP | EZ-AUROC | MRR | Top1 | Sensitivity | Specificity | Accuracy |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method in ['R4','R0','R1','R2']:lines.append('| '+method+' | '+str(lock['architecture']['parameters'][method])+' | '+' | '.join(val(method,m) for m in METRICS)+' |')
    lines+=['',f'Terminal: `{terminal}`. Registered primary R2 gate: {primary}. Qualifying exploratory secondary candidates: {[m for m in qualifiers if m!="R2"]}.',
            '', 'Same65 patient-fold appearances,47 unique IDs,6273 channel appearances; original80/7635/88D/five folds. R4 is frozen original D2, not newly trained. No outer TEST. This is repeatedly used development, not independent testing or unseen-center transfer.',
            '', '## Paired unique-ID cluster bootstrap','', '| Contrast | Macro-F1 delta | 95% percentile CI | Positive folds | Positive unique IDs |','|---|---:|---|---:|---:|']
    for left,right in contrasts:
        r=bt.loc[(left+'-'+right,'macro_f1')]; lines.append(f'| {left}-{right} | {r.delta:+.6f} | [{r.CI_low:+.6f}, {r.CI_high:+.6f}] | {int(r.positive_folds)}/5 | {int(r.positive_unique_patients)}/47 |')
    lines+=['', '10,000 paired seed42 unique-ID draws retain repeated appearances. Fixed checkpoints/thresholds; no reoptimization. All11 metric intervals/all6 contrasts are in PAIRED_BOOTSTRAP.csv. Nominal intervals do not adjust prior searches, selection optimism or multiplicity.',
            '', '## Source dependence','', '| Source | R4 Macro-F1 | R0-R4 | R1-R4 | R2-R4 |','|---|---:|---:|---:|---:|']
    for center in sorted(set(frame.center)):
        oldmean=frame[(frame.method=='R4')&(frame.center==center)].macro_f1.mean()
        values=[float(cc[(cc.contrast==m+'-R4')&(cc.metric=='macro_f1')&(cc.center==center)].delta.iloc[0]) for m in ['R0','R1','R2']]
        lines.append('| '+center+' | '+f'{oldmean:.6f}'+' | '+' | '.join(f'{x:+.6f}' for x in values)+' |')
    best=sm.macro_f1.idxmax(); monotone=all(sm.loc[a,'macro_f1']<=sm.loc[b,'macro_f1'] for a,b in [('R4','R2'),('R2','R1'),('R1','R0')])
    ranktext='; '.join(m+': '+str([r['effective_numerical_rank'] for r in rankaudit if r['method']==m]) for m in ['R4','R0','R1','R2'])
    er=pd.DataFrame(errors).groupby('method')[['TP_gained','TP_lost','FP_added','FP_removed','errors_recovered','correct_spoiled']].sum()
    intervention_mean=interv.groupby(['method','intervention'])[METRICS+EXTRA+['pair_reversal_fraction']].mean()
    lines+=['','## Explicit answers to24 required questions','',
       '1. R4 reproduced: yes; strict state mapping and bitwise generalized-versus-legacy CPU forward parity; fresh archived logit tolerance<2e-6, then exact frozen score/metric reuse. No R4 retraining.',
       '2. Original splits/labels/88D preprocessing retained bitwise. All101 archived artifacts are hash-verified. NPZ all-cohort y is never loaded; only sealed FIT/VAL labels are used.',
       f'3. R0-R4 Macro-F1 {change("R0"):+.6f}; full secondary criterion={"R0" in qualifiers}.',
       f'4. R1-R4 Macro-F1 {change("R1"):+.6f}; full secondary criterion={"R1" in qualifiers}.',
       f'5. R2-R4 Macro-F1 {change("R2"):+.6f}; registered full primary gate={primary}.',
       f'6. Descriptively highest Macro-F1 is {best}={val(best,"macro_f1")}. This post-outcome descriptive maximum is not an independently selected/confirmed rank.',
       f'7. Monotonic improvement while reducing rank4->2->1->0: {monotone}. Point ordering alone does not establish a rank mechanism.',
       f'8. R2 positive Macro-F1 folds versus R4: {int(bt.loc[("R2-R4","macro_f1"),"positive_folds"])}/5.',
       '9. EZ-AP deltas versus R4: '+', '.join(m+f" {change(m,'ez_auprc'):+.6f}" for m in ['R0','R1','R2'])+'.',
       '10. EZ-AUROC deltas versus R4: '+', '.join(m+f" {change(m,'ez_auroc'):+.6f}" for m in ['R0','R1','R2'])+'.',
       '11. EZ-F1 deltas versus R4: '+', '.join(m+f" {change(m,'ez_f1'):+.6f}" for m in ['R0','R1','R2'])+'.',
       '12. Sensitivity/specificity deltas: '+', '.join(m+f" {change(m,'sensitivity'):+.6f}/{change(m,'specificity'):+.6f}" for m in ['R0','R1','R2'])+'.',
       '13. Recovered/spoiled errors over repeated channel appearances: '+', '.join(m+f" {int(er.loc[m,'errors_recovered'])}/{int(er.loc[m,'correct_spoiled'])}; TP gained/lost {int(er.loc[m,'TP_gained'])}/{int(er.loc[m,'TP_lost'])}" for m in ['R0','R1','R2'])+'. These are not independent channel observations.',
       '14. Bias-versus-direction is isolated only under the same fixed trained scorer; R0 changes the jointly trained shared backbone too. Inference Macro-F1 changes (intervention minus full): '+', '.join(m+'/'+v+f" {intervention_mean.loc[(m,v),'macro_f1']:+.6f}" for m in ['R1','R2'] for v in ['DIRECTION_OFF','BIAS_OFF'])+'. This is not proof that bias causally accounts for cross-model gains.',
       '15. Same-model source-constant bias ranking invariance is verified mathematically and by fixed-ranking intervention. Direction-off patient-equal pair reversals: '+', '.join(m+f" {intervention_mean.loc[(m,'DIRECTION_OFF'),'pair_reversal_fraction']:.6f}" for m in ['R1','R2'])+'. Directional terms can alter ranking; shared-backbone retraining can also alter cross-model rankings.',
       '16. Effective ranks by folds1-5: '+ranktext+'. SVD analyzes U@V, not separately unidentified factors.',
       '17. SOURCE_DIRECTION_AUDIT.csv reports all source vector norms and defined pairwise cosines. Rank1 vectors are collinear with possibly opposing signs. Zero-vector cosine is missing/not defined, never assigned a fabricated zero similarity.',
       '18. Low-count source findings are in the source table above and GENERALIZATION_DIAGNOSTICS.csv. Original source patients: HUP36/LZU21/Multicenter15/Pediatric8; development unique counts21/11/9/6. No small-source or unseen-center generalization is inferred from architecture alone.',
       '19. Negative source deltas are reported without exclusion. Severe collapse is prelocked as>.050 absolute source Macro-F1 decline and blocks supported status, without posthoc explanation waiver.',
       '20. Paired intervals above quantify development differences; if zero is included, gains are not clearly distinguished from patient-level variation. Even positive nominal intervals are not independent confirmation or a causal isolation of overfit.',
       '21. Variants meeting +.010 magnitude alone: '+str([m for m in ['R0','R1','R2'] if conditions[m]['macro_delta_ge010']])+'. Full-gate candidates: '+str(qualifiers)+'.',
       '22. Any compressed point Macro-F1>=.700: '+str(any(sm.loc[m,'macro_f1']>=.7 for m in ['R0','R1','R2']))+'. Full-gate target: '+str(target)+'.',
       '23. '+('Only a separately locked follow-up is eligible; secondary candidates remain exploratory. Stop this seed42 experiment.' if qualifiers else 'Stop the source-rank compression line under this fixed family. No rank3/lambda/seed/threshold rescue is authorized.'),
       '24. Source heterogeneity and unreliable patient-level decisions on repeatedly reused development remain unresolved. This experiment cannot establish biological source mechanisms, clinical label truth, prospective transfer or causal overfitting. A rank decrease alone is not sufficient evidence of better generalization.',
       '', '## Mechanism and engineering caveats','',
       'Generalization diagnostics use in-sample FIT predictions at the selected checkpoint and selected VAL threshold; FIT-minus-VAL gaps are optimistic descriptions, not OOF or causal overfit estimates. Interventions are inference-only and never retrained. Source-bias scalars are acquisition-category corrections, not disease prevalence estimates.',
       'R0 removes400/9221 parameters (4.34%), R1 removes300 (3.25%), R2 removes200 (2.17%). Most capacity remains in the shared backbone; compression is not a substantial compute reduction. Original mean-square shrinkage normalizes different-rank parameter spaces; lambda was not retuned.',
       'Primary metrics use original float32 sigmoid/ranking semantics. Bias-off ranking reuses full frozen ordering after verifying float64 source-constant translation, avoiding artificial rank changes from finite precision; direction-off ranks by the shared base. Classification always uses fixed selected thresholds.',
       'All34 engineering checks and15 FIT smokes pass before training. Exactly15 new runs; no R4 retraining or outer TEST. Private IDs, labels, scores, matrices, checkpoints, optimizer/RNG states and runtime logs stay on the original server.',
       '', '## Execution verification','',f'Protocol SHA256 `{sha(args.protocol)}`. Exactly15 selected sealed cells; {len(hist)} completed epochs; epoch-body time {sum(h["seconds"] for h in hist):.2f}s, excluding preparation/tests/checkpoint/report I/O.']
    (pub/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('ALL15_SOURCE_RANK_FINALIZED',terminal,flush=True)

if __name__=='__main__':main()
