"""One frozen head outer pass, patient-paired statistics and source diagnostics."""
import torch
from common import *
from model import Head
from head_data import bank,plan,state_hash,save_torch

def reversal(a,b):
    a=np.asarray(a);b=np.asarray(b);i,j=np.triu_indices(len(a),1);da=a[i]-a[j];db=b[i]-b[j]
    return float(((da*db)<0).mean()) if len(i) else 0.
def aggregate(frame,keys,metrics):
    rows=[]
    for keyvalue,part in frame.groupby(keys,sort=True):
        if not isinstance(keyvalue,tuple):keyvalue=(keyvalue,)
        rows.append({**dict(zip(keys,keyvalue)),'patients':len(part),**{m:float(part[m].mean()) for m in metrics}})
    return rows

def main():
    torch.set_num_threads(2);freeze_path=ROOT/'FROZEN_BEFORE_HEAD_OUTER_PRIVATE.json';frozen=json.loads(freeze_path.read_text())
    assert frozen['status']=='FROZEN_BEFORE_HEAD_OUTER_PREDICT' and frozen['protocol']==sha(PROTOCOL) and frozen['code']==code_hashes()
    selection={(r['fold'],r['method']):r for r in frozen['selections']};assert len(selection)==20
    mf=manifest();rows=[];correction=[];directions=[];rankchanges=[];errors=[];interventions=[];caches={};source_parameters=[]
    for fold in range(1,6):
        assignment=plan(mf,fold);held=assignment['held'];data=bank(fold,held,False)
        for arm in ['C0','C1','C2','C3']:
            chosen=selection[fold,arm];tau=chosen['threshold'];model=None
            if arm!='C0':
                ck=ROOT/f'fold_{fold}/{arm}/SELECTED_HEAD_PRIVATE.pt';assert sha(ck)==chosen['checkpoint_sha256'];st=torch.load(ck,map_location='cpu',weights_only=False)
                assert st['binding']['protocol']==sha(PROTOCOL) and st['threshold']==tau;model=Head(arm);model.load_state_dict(st['model']);model.eval()
            if arm=='C2':
                W=model.U.detach().numpy().astype(np.float64)@model.V.detach().numpy().astype(np.float64);sv=np.linalg.svd(W,compute_uv=False);tol=32*np.finfo(np.float64).eps*sv[0];effective=int((sv>tol).sum());assert effective<=2
                for i,source in enumerate(SOURCES):source_parameters.append({'fold':fold,'source':source,'effective_rank':effective,'singular1':sv[0],'singular2':sv[1],'singular3':sv[2],'effective_direction_norm':float(np.linalg.norm(W[i])),'source_bias':float(model.b[i].detach()),'shared_beta_norm':float(model.beta.norm().detach())})
                for i,source in enumerate(SOURCES):
                    for dim in range(32):directions.append({'fold':fold,'source':source,'embedding_coordinate':dim,'effective_UV':float(W[i,dim])})
            for patient in held:
                d=data[patient];row=mf.loc[mf.patient_name==patient].iloc[0];source=assignment['pseudo'][patient] if arm=='C3' else assignment['real'][patient]
                cache=ROOT/f'outer_predictions/{arm}/{key(patient)}.npz';seal=cache.with_suffix('.json')
                binding={'freeze':sha(freeze_path),'protocol':sha(PROTOCOL),'bank':sha(ROOT/f'embeddings/fold_{fold}/{key(patient)}.npz'),'arm':arm,'source':source,'checkpoint':chosen.get('checkpoint_sha256','frozen-CNN')}
                if cache.exists() or seal.exists():
                    assert cache.exists() and seal.exists(),'Partial outer prediction; inspect before resume'
                    verified=json.loads(seal.read_text());assert verified['binding']==binding and verified['sha256']==sha(cache)
                    with np.load(cache,allow_pickle=False) as c:logits=c['logits'];shared=c['shared'];direction=c['direction'];bias=c['bias']
                else:
                    shared=direction=bias=np.zeros(len(d['base']),np.float32)
                    if model is not None:
                        with torch.no_grad():
                            z=torch.from_numpy(d['z']);g=torch.full((len(z),),source,dtype=torch.long);ss,dd,bb=model.components(z,g)
                        shared=ss.numpy();direction=dd.numpy();bias=bb.numpy()
                    logits=d['base']+(shared+direction+bias).astype(np.float64)
                    save_npz(cache,channels=d['channels'],logits=logits,shared=shared,direction=direction,bias=bias)
                    write_json(seal,{'binding':binding,'sha256':sha(cache)})
                caches[str(cache)]=sha(cache);assert np.isfinite(logits).all()
                # Clinical labels are accessed ONLY after predictions are sealed.
                with np.load(ROOT/'labels'/f'{key(patient)}.npz',allow_pickle=False) as vault:
                    assert np.array_equal(d['channels'],vault['channels']);y=vault['labels'];valid=y>=0
                scores=sigmoid(logits);base_probability=sigmoid(d['base']);metrics=metric(y[valid],scores[valid],tau);fixed=classification(y[valid],scores[valid],.5)
                out={'patient':patient,'fold':fold,'center':str(row.center),'method':arm,'threshold':tau,'units':int(valid.sum()),**metrics,**{m+'_0_5':fixed[m] for m in CLASS}}
                rows.append(out)
                if arm=='C0':continue
                reference=metric(y[valid],base_probability[valid],selection[fold,'C0']['threshold']);pred=scores[valid]>=tau;oldpred=base_probability[valid]>=selection[fold,'C0']['threshold'];pos=y[valid]==1
                errors.append({'fold':fold,'method':arm,'center':str(row.center),'errors_recovered':int(((oldpred!=pos)&(pred==pos)).sum()),'correct_spoiled':int(((oldpred==pos)&(pred!=pos)).sum()),'TP_gained':int((pos&~oldpred&pred).sum()),'TP_lost':int((pos&oldpred&~pred).sum()),'FP_added':int((~pos&~oldpred&pred).sum()),'FP_removed':int((~pos&oldpred&~pred).sum())})
                rankchanges.append({'patient':patient,'fold':fold,'center':str(row.center),'method':arm,'rank_reversal':reversal(d['base'][valid],logits[valid]),'AUROC_change':metrics['auroc']-reference['auroc'],'AP_change':metrics['ap']-reference['ap'],'MRR_change':metrics['mrr']-reference['mrr'],'Top1_change':metrics['top1']-reference['top1']})
                if arm!='C2':continue
                corr=logits-d['base'];base_rms=float(np.sqrt(np.mean(d['base']**2)));corr_rms=float(np.sqrt(np.mean(corr**2)))
                correction.append({'patient':patient,'fold':fold,'center':str(row.center),'shared_rms':float(np.sqrt(np.mean(shared**2))),'direction_rms':float(np.sqrt(np.mean(direction**2))),'direction_variance':float(np.var(direction)),'bias':float(bias[0]),'correction_rms':corr_rms,'correction_to_original_RMS':corr_rms/max(base_rms,1e-12),'mean_absolute_prediction_change':float(abs(scores-base_probability).mean()),'fraction_changed_decisions_FIT_thresholds':float((pred!=oldpred).mean()),'fraction_changed_decisions_same_C2_numeric_threshold':float((pred!=(base_probability[valid]>=tau)).mean()),'rank_reversal':reversal(d['base'][valid],logits[valid]),'AUROC_change':metrics['auroc']-reference['auroc'],'AP_change':metrics['ap']-reference['ap']})
                pre=d['base']+shared.astype(np.float64)+direction.astype(np.float64);constant=bias.astype(np.float64);assert np.ptp(constant)==0
                # Exact-arithmetic translation property; avoid invented rank
                # effects from float32 correction sum/saturated sigmoid ties.
                for variant,ll,order in [('DIRECTION_OFF',d['base']+(shared+bias).astype(np.float64),None),('BIAS_OFF',pre,scores[valid]),('ALL_OFF',d['base'],None)]:
                    probability=sigmoid(ll);new=metric(y[valid],probability[valid],tau,order_score=order)
                    if variant=='ALL_OFF':assert np.array_equal(probability,base_probability)
                    if variant=='BIAS_OFF':assert all(new[m]==metrics[m] for m in RANK)
                    interventions.append({'patient':patient,'fold':fold,'center':str(row.center),'intervention':variant,'threshold':tau,**new,**{m+'_delta':new[m]-metrics[m] for m in METRICS},'rank_reversal_vs_full':0. if variant=='BIAS_OFF' else reversal(logits[valid],ll[valid]),'mean_absolute_probability_change':float(abs(probability-scores).mean())})
            print('OUTER_FROZEN_ARM_COMPLETE',fold,arm,flush=True)
    frame=pd.DataFrame(rows);assert len(frame)==496
    for arm in ['C0','C1','C2','C3']:
        part=frame[frame.method==arm];assert part.patient.nunique()==124 and part.units.sum()==10202
    allmetrics=METRICS+[m+'_0_5' for m in CLASS]+['predicted_pathological_fraction']
    csv('OOF_PATIENT_SUMMARY.csv',aggregate(frame,['method'],allmetrics));csv('OOF_BY_FOLD.csv',aggregate(frame,['fold','method'],allmetrics));csv('OOF_BY_CENTER.csv',aggregate(frame,['center','method'],allmetrics))
    # Number of improved patients remains aggregate per source/method.
    improved=[];baseline=frame[frame.method=='C0'].set_index('patient')
    for arm in ['C1','C2','C3']:
        x=frame[frame.method==arm].set_index('patient').reindex(baseline.index)
        for center in SOURCES:
            take=x.center==center;improved.append({'center':center,'method':arm,'patients':int(take.sum()),'AUROC_improved_patients':int((x.loc[take,'auroc']>baseline.loc[take,'auroc']).sum()),'AP_improved_patients':int((x.loc[take,'ap']>baseline.loc[take,'ap']).sum())})
    csv('CENTER_IMPROVEMENT_COUNTS.csv',improved)
    csv('SOURCE_DIRECTION_MATRIX.csv',directions);csv('SOURCE_PARAMETER_AUDIT.csv',source_parameters)
    corr=pd.DataFrame(correction);corrcolumns=[c for c in corr if c not in ['patient','fold','center']]
    csv('SOURCE_CORRECTION_AUDIT.csv',aggregate(corr,['fold','center'],corrcolumns))
    rank=pd.DataFrame(rankchanges);csv('RANKING_CHANGE_AUDIT.csv',aggregate(rank,['fold','center','method'],['rank_reversal','AUROC_change','AP_change','MRR_change','Top1_change']))
    error=pd.DataFrame(errors);csv('ERROR_CORRECTION_AUDIT.csv',[{'fold':f,'method':a,'center':c,**{m:int(g[m].sum()) for m in ['errors_recovered','correct_spoiled','TP_gained','TP_lost','FP_added','FP_removed']}} for (f,a,c),g in error.groupby(['fold','method','center'])])
    intr=pd.DataFrame(interventions);intrcols=METRICS+[m+'_delta' for m in METRICS]+['rank_reversal_vs_full','mean_absolute_probability_change']
    csv('SOURCE_DIRECTION_INTERVENTIONS.csv',aggregate(intr,['intervention'],intrcols));csv('SOURCE_INTERVENTIONS_BY_CENTER.csv',aggregate(intr,['center','intervention'],intrcols))
    boot=[];source_effect=[];bootstrap_metrics=METRICS+[m+'_0_5' for m in CLASS];contrasts=[('C1','C0'),('C2','C0'),('C2','C1'),('C2','C3')]
    for left,right in contrasts:
        a=frame[frame.method==left].set_index('patient').sort_index();b=frame[frame.method==right].set_index('patient').sort_index();assert a.index.equals(b.index)
        delta=a[bootstrap_metrics]-b[bootstrap_metrics];idx=np.random.default_rng(42).integers(0,124,size=(10000,124));draws=delta.to_numpy()[idx].mean(1)
        for j,m in enumerate(bootstrap_metrics):
            effects=delta[m].to_numpy();boot.append({'contrast':left+'-'+right,'metric':m,'policy':'fixed0.5' if m.endswith('_0_5') else 'ranking' if m in RANK else 'FIT_selected','delta':float(effects.mean()),'CI_low':float(np.quantile(draws[:,j],.025)),'CI_high':float(np.quantile(draws[:,j],.975)),'positive_folds':int((delta[m].groupby(a.fold).mean()>0).sum()),'positive_patients':int((effects>0).sum()),'patients':124,'draws':10000,'seed':42})
            for center in SOURCES:
                take=a.center==center;source_effect.append({'contrast':left+'-'+right,'metric':m,'center':center,'delta':float(delta.loc[take,m].mean()),'patients':int(take.sum())})
    csv('PATIENT_BOOTSTRAP.csv',boot);csv('CENTER_PAIRED_EFFECTS.csv',source_effect)
    sm=frame.groupby('method')[allmetrics].mean();effects=pd.DataFrame(boot).set_index(['contrast','metric']);center_effect=pd.DataFrame(source_effect);direction_delta=float(intr.loc[intr.intervention=='DIRECTION_OFF','auroc_delta'].mean())
    conditions={
        'C2_C0_AUROC_ge010':sm.loc['C2','auroc']-sm.loc['C0','auroc']>=.010,
        'C2_C1_AUROC_ge005':sm.loc['C2','auroc']-sm.loc['C1','auroc']>=.005,
        'C2_C3_AUROC_ge005':sm.loc['C2','auroc']-sm.loc['C3','auroc']>=.005,
        'AP_nondecline':sm.loc['C2','ap']>=sm.loc['C0','ap'],
        'matched_FIT_threshold_MacroF1_nondecline':sm.loc['C2','macro_f1']>=sm.loc['C0','macro_f1'],
        'pathological_F1_decline_le010':sm.loc['C2','pathological_f1']>=sm.loc['C0','pathological_f1']-.010,
        'AUROC_positive_folds_ge4':int(effects.loc[('C2-C0','auroc'),'positive_folds'])>=4,
        'small_source_AUROC_decline_le050':bool((center_effect[(center_effect.contrast=='C2-C0')&(center_effect.metric=='auroc')&center_effect.center.isin(['HUP','SourceSink'])].delta>=-.050).all()),
        'meaningful_direction':-direction_delta>=.001 and float(intr.loc[intr.intervention=='DIRECTION_OFF','rank_reversal_vs_full'].mean())>0,
        'MRR_Top1_nondecline':bool(sm.loc['C2','mrr']>=sm.loc['C0','mrr'] and sm.loc['C2','top1']>=sm.loc['C0','top1']),
        'cohort_checkpoint_alignment_integrity':True}
    collapsed=float(corr.direction_rms.mean())<=1e-8
    shared=bool(sm.loc['C1','auroc']-sm.loc['C0','auroc']>=.010 and all(sm.loc['C1',m]>=sm.loc['C0',m] for m in ['ap','macro_f1','mrr','top1']) and effects.loc[('C1-C0','auroc'),'positive_folds']>=4)
    if collapsed:terminal='SOURCE_CORRECTION_COLLAPSED'
    elif all(conditions.values()):terminal='OMNI_D2_PATIENT_GAIN_SUPPORTED'
    elif not conditions['C2_C3_AUROC_ge005']:terminal='SOURCE_SHUFFLE_CONTROL_NOT_BEATEN'
    elif not conditions['C2_C1_AUROC_ge005'] and shared:terminal='SHARED_PATIENT_RELATIVE_ONLY_SUPPORTED'
    elif not conditions['C2_C1_AUROC_ge005'] or not conditions['meaningful_direction']:terminal='SOURCE_DIRECTION_NOT_NEEDED'
    else:terminal='NO_MEANINGFUL_GAIN'
    write_json(ROOT/'public/VALIDATION_GATE.json',{'terminal':terminal,'primary_conditions':{k:bool(v) for k,v in conditions.items()},'primary_supported':bool(all(conditions.values())),'AUROC_target_reached':bool(sm.loc['C2','auroc']>=.7722),'protocol':sha(PROTOCOL),'reused_cohort_exploratory':True,'pooled8061_not_used':True})
    save_torch(ROOT/'FINAL_OUTER_LEDGER_PRIVATE.pt',{'patient_rows':rows,'interventions':interventions,'source_correction':correction,'freeze':sha(freeze_path),'prediction_hashes':caches})
    write_json(ROOT/'public/RUN_STATUS.json',{'status':'COMPLETE','terminal':terminal,'patients':124,'folds':5,'labeled_unique_patient_channels':10202,'inner_training_runs':45,'refits':15,'new_CNN_training':False,'outer_labels_used_for_selection':False,'prediction_ledger_sha256':sha(ROOT/'FINAL_OUTER_LEDGER_PRIVATE.pt')})
    # Explicit26 answers, all values come from the single frozen private ledger.
    def value(arm,m):return f'{sm.loc[arm,m]:.6f}'
    def diff(left,right,m='auroc'):return float(sm.loc[left,m]-sm.loc[right,m])
    lines=['# Omni-D2 seed42: frozen patient-level outer evaluation','',f'Terminal: `{terminal}`. Registered primary continuation gate: {all(conditions.values())}.','',
           '| Model | Patient AUROC | AP | MRR | Top1 | NDCG | MacroF1 FIT tau | PathF1 FIT tau | MacroF1@0.5 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ['C0','C1','C2','C3']:lines.append('| '+arm+' | '+' | '.join(value(arm,m) for m in ['auroc','ap','mrr','top1','ndcg','macro_f1','pathological_f1','macro_f1_0_5'])+' |')
    lines+=['','All124 unique patients and10202 labeled patient-channels are included exactly once per arm. Original137 records/five folds retained. Ranking is within-patient, summaries patient-equal; published pooled0.8061 is not a criterion. Classification columns distinguish fixed0.5 and own FIT-selected numeric thresholds. All CLASS/RANK metrics and center/fold tables are in CSVs.','',
      'C0@0.5 historical AUROC0.762163/AP0.477255/MacroF10.545188; old A1Context AUROC0.763012/AP0.472740/MacroF10.509492 is historical only. No old Ictal numbers are numerically compared to Omni.','',
      '## Paired bootstrap','', '| Contrast | Patient AUROC delta |95% percentile CI | Positive folds | Positive patients |','|---|---:|---|---:|---:|']
    for left,right in contrasts:
        r=effects.loc[(left+'-'+right,'auroc')];lines.append(f'| {left}-{right} | {r.delta:+.6f} | [{r.CI_low:+.6f}, {r.CI_high:+.6f}] | {int(r.positive_folds)}/5 | {int(r.positive_patients)}/124 |')
    lines+=['','10,000 paired seed42 unique-patient draws. Fixed checkpoints and numeric thresholds, no bootstrap reoptimization. Nominal intervals are descriptive and do not adjust repeated historical cohort viewing or research selection. This is not an untouched external test.','',
      '## Source effects','', '| Source | n | C0 AUROC | C2 AUROC | C2-C0 | C2-C3 |','|---|---:|---:|---:|---:|---:|']
    for source in SOURCES:
        a=frame[(frame.center==source)&(frame.method=='C2')].auroc.mean();b=frame[(frame.center==source)&(frame.method=='C0')].auroc.mean();c=frame[(frame.center==source)&(frame.method=='C3')].auroc.mean()
        lines.append(f'| {source} | {int((mf.center==source).sum())} | {b:.6f} | {a:.6f} | {a-b:+.6f} | {a-c:+.6f} |')
    intrmean=intr.groupby('intervention')[METRICS+[m+'_delta' for m in METRICS]].mean()
    lines+=['','## Answers to26 required questions','',
      '1. Original TimeConv reproduced: fresh bank hard gate passed every historical patient metric; scores within1e-6, no archived-score substitution. BASELINE_REPRODUCTION.json reports exact maximum error.',
      '2. All124 patients,137 records and original folds preserved; label orientation pathological1/normal0 and unknown masks unchanged.',
      '3. Only frozen CNN32D embeddings plus original frozen logits are used;35/105 parameter correction heads, no CNN optimization or raw feature redefinition.',
      f'4. C1 patient AUROC={value("C1","auroc")}, delta C0={diff("C1","C0"):+.6f}.',
      f'5. C2 patient AUROC={value("C2","auroc")}, delta C0={diff("C2","C0"):+.6f}.',
      f'6. C2 AP={value("C2","ap")}, delta C0={diff("C2","C0","ap"):+.6f}.',
      f'7. C2 matched FIT-tau MacroF1={value("C2","macro_f1")}, delta C0={diff("C2","C0","macro_f1"):+.6f}; fixed0.5={value("C2","macro_f1_0_5")}.',
      f'8. C2 pathologicalF1={value("C2","pathological_f1")}, delta matched C0={diff("C2","C0","pathological_f1"):+.6f}.',
      f'9. C2 MRR/Top1={value("C2","mrr")}/{value("C2","top1")}; deltas={diff("C2","C0","mrr"):+.6f}/{diff("C2","C0","top1"):+.6f}.',
      f'10. C2-C1 AUROC={diff("C2","C1"):+.6f}; registered requirement +0.005={conditions["C2_C1_AUROC_ge005"]}.',
      f'11. C2-C3 AUROC={diff("C2","C3"):+.6f}; matched-capacity source placebo beaten by +0.005={conditions["C2_C3_AUROC_ge005"]}.',
      '12. All five folds,45 inner heads,15 selected-epoch full FIT refits and four OOF arms completed.',
      f'13. Positive C2-C0 AUROC folds={int(effects.loc[("C2-C0","auroc"),"positive_folds"])}/5.',
      '14. Source effect table above reports the Open-iEEG majority and both smaller sources without exclusion; do not infer general source transfer from pooled or majority performance.',
      '15. HUP10 and SourceSink11 AUROC changes are reported above. Small-source gate passed='+str(conditions['small_source_AUROC_decline_le050'])+'. Their uncertainty is substantial.',
      f'16. Mean patient C2 source-direction RMS={float(corr.direction_rms.mean()):.6f}; meaningful-direction gate={conditions["meaningful_direction"]}. Source_PARAMETER/SOURCE_DIRECTION_MATRIX and source correction CSVs give all singular values, ranks, norms, biases, variances and RMS ratios.',
      f'17. Mean direction-off pair reversal={float(intr.loc[intr.intervention=="DIRECTION_OFF","rank_reversal_vs_full"].mean()):.6f}; full C2-vs-C0 pair reversal={float(corr.rank_reversal.mean()):.6f}.',
      f'18. Removing source bias changes matched-threshold MacroF1 by {float(intrmean.loc["BIAS_OFF","macro_f1_delta"]):+.6f}; within-patient ranking deltas are exactly0 by constant-translation audit. Bias can affect classification, not patient ranking.',
      f'19. Direction-off AUROC/AP/MacroF1 deltas={float(intrmean.loc["DIRECTION_OFF","auroc_delta"]):+.6f}/{float(intrmean.loc["DIRECTION_OFF","ap_delta"]):+.6f}/{float(intrmean.loc["DIRECTION_OFF","macro_f1_delta"]):+.6f}. This is same-model inference diagnosis, not retrained ablation.',
      '20. All added corrections off recovers C0 probabilities exactly; classification intervention uses C2 numeric threshold, unlike separately FIT-selected C0 threshold. No false claim of classification equality across different thresholds.',
      '21. Bootstrap intervals above quantify patient sampling variation. Intervals including0 do not clearly distinguish gains. Even positive nominal intervals do not constitute independent validation or causal source evidence.',
      '22. Yes: inner head-validation patients were previously used to train the corresponding frozen outer CNN. Inner head-OOF is NOT full-pipeline OOF. Only outer held patients are excluded by both CNN and head.',
      '23. No outer labels enter head initialization, loss, inner epoch/threshold selection or source shuffling; predictions are atomically sealed before final label access.',
      f'24. Patient AUROC>=0.7722: {sm.loc["C2","auroc"]>=.7722}. This target is exploratory, not external validation.',
      '25. '+('The full gate supports only a separately predeclared joint fine-tuning follow-up; it does not itself prove clinical/source causality.' if all(conditions.values()) else 'The registered gate fails; this evidence does not justify joint fine-tuning as a rescue of this mechanism.'),
      '26. '+('Continue only as separately registered exploratory follow-up; stop this seed42 task without extra tuning.' if all(conditions.values()) else 'Stop this source-direction family under the current prespecified experiment; no post-outcome rescue.'),
      '', '## Gate and limitations','',json.dumps({k:bool(v) for k,v in conditions.items()},indent=2),
      '', 'Unknown good channels enter label-blind per-patient normalization, never loss/metrics. Head CV uses only FIT labels and one common three-way split. C3 has identical initial tensors/parameter count/budget but source-category assignment only is shuffled; imbalanced Open-iEEG composition makes many assignments unchanged. Changed-source fractions are disclosed, not forced to impossible100% derangement.',
      'This transfers the operational source-correction mechanism, not Ictal weights or biological meaning. Head-only in-sample-CNN inner validation can be optimistic; thresholds may not transfer to outer held scores. Inference interventions do not identify effects of retraining or prove source causality. Unknown-source fallback retains shared beta only.',
      'Bias-off ranking reuses full frozen ordering to avoid finite-precision/sigmoid saturation ties while verifying constant source offset. Classification is recomputed at the frozen C2 threshold. Effective U@V numerical rank uses float64 SVD; zero vectors have no assigned cosine.',
      'Private waveforms, labels, patient/channel identities, embeddings, scores, head/CNN tensor checkpoints, optimizers and logs remain on server. Public output is aggregate-only. No raw EEG extraction, no CNN training, no architecture/seed sweep, no test-derived threshold.',
      '', '## Provenance','',f'Protocol SHA256 `{sha(PROTOCOL)}`; pre-outer head freeze `{sha(freeze_path)}`. All prediction and model hashes retained privately; compact integrity verification delivered separately.']
    (ROOT/'public/FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('OMNI_D2_FROZEN_OUTER_COMPLETE',terminal,flush=True)

if __name__=='__main__':main()
