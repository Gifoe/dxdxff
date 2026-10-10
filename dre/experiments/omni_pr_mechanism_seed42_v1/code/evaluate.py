"""One immutable outer pass for the two new controls; original references reused."""
from source import *
from transforms import *
from report import final_report
def reversal(a,b):
    i,j=np.triu_indices(len(a),1);return float(((a[i]-a[j])*(b[i]-b[j])<0).mean()) if len(i) else 0.
def seal_prediction(path,binding,channels,logits):
    seal=path.with_suffix('.json')
    if path.exists() or seal.exists():
        assert path.exists() and seal.exists(),'Incomplete prediction; inspect before resume'
        previous=json.loads(seal.read_text());assert previous['binding']==binding and sha(path)==previous['sha256']
        with np.load(path,allow_pickle=False) as v:assert np.array_equal(v['channels'],channels) and np.array_equal(v['logits'],logits)
    else:
        save_npz(path,channels=channels,logits=logits);write_json(seal,{'binding':binding,'sha256':sha(path)})
    return sha(path)
def main():
    torch.set_num_threads(2);fpath=ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json';freeze=json.loads(fpath.read_text())
    assert freeze['status']=='ALL_CONTROLS_FROZEN_BEFORE_OUTER' and freeze['protocol']==sha(PROTOCOL) and freeze['code']==code_hashes()
    assert len(freeze['choices'])==10 and freeze['historical_replay_sha256']==sha(ROOT/'HISTORICAL_ROWS_PRIVATE.pt')
    for p,h in freeze['transform_hashes'].items():assert sha(p)==h
    mf=manifest();rows=torch.load(ROOT/'HISTORICAL_ROWS_PRIVATE.pt',map_location='cpu',weights_only=False)['rows'];rows=list(rows)
    choices={(r['fold'],r['method']):r for r in freeze['choices']};hashes={};diagnostics=[];scalechecks=[];ranks=[];errors=[]
    for fold in range(1,6):
        assignment=plan(mf,fold);held=assignment['held'];raw=bank(fold,held)
        for arm in ['C1-ABS','C1-SHUFFLE']:
            chosen=choices[fold,arm];path=ROOT/f'fold_{fold}/{arm}/SELECTED_HEAD_PRIVATE.pt';assert sha(path)==chosen['checkpoint_sha256']
            st=torch.load(path,map_location='cpu',weights_only=False);head=Head('C1');head.load_state_dict(st['model']);head.eval();assert st['threshold']==chosen['threshold']
            scaler=None
            if arm=='C1-ABS':
                path=ROOT/f'transforms/fold_{fold}/ABS_full_FIT.pt';assert sha(path)==chosen['scaler_sha256'];scaler=torch.load(path,map_location='cpu',weights_only=False)
                assert scaler['fit_patient_digest']==digest(assignment['fit'])
            data=transform(raw,held,arm,fold,'HELD',scaler);pred=predict(head,data,held,assignment['real'])
            for patient in held:
                d=raw[patient];ll=pred[patient];p=ROOT/f'outer_predictions/{arm}/{key(patient)}.npz'
                binding={'protocol':sha(PROTOCOL),'freeze':sha(fpath),'bank':d['bank_sha'],'head':chosen['checkpoint_sha256'],'scaler':chosen['scaler_sha256'],'transform_manifest':digest(freeze['transform_hashes'])}
                hashes[str(p)]=seal_prediction(p,binding,d['channels'],ll)
                # Held labels are loaded only after the new arm's prediction is sealed.
                with np.load(HIST/'labels'/f'{key(patient)}.npz',allow_pickle=False) as v:
                    assert np.array_equal(v['channels'],d['channels']);y=v['labels'];valid=y>=0
                probability=sigmoid(ll);tau=chosen['threshold'];q=metric(y[valid],probability[valid],tau);fixed=classification(y[valid],probability[valid],.5)
                center=str(mf.loc[mf.patient_name==patient,'center'].iloc[0]);row={'patient':patient,'fold':fold,'center':center,'method':arm,'threshold':tau,'units':int(valid.sum()),**q,**{m+'_0_5':fixed[m] for m in CLASS}};rows.append(row)
                base=next(r for r in rows if r['patient']==patient and r['method']=='C0');oldpred=sigmoid(d['base'][valid])>=base['threshold'];newpred=probability[valid]>=tau;positive=y[valid]==1
                errors.append({'fold':fold,'center':center,'method':arm,'recovered':int(((oldpred!=positive)&(newpred==positive)).sum()),'spoiled':int(((oldpred==positive)&(newpred!=positive)).sum()),'TP_gained':int((positive&~oldpred&newpred).sum()),'TP_lost':int((positive&oldpred&~newpred).sum()),'FP_added':int((~positive&~oldpred&newpred).sum()),'FP_removed':int((~positive&oldpred&~newpred).sum())})
                ranks.append({'patient':patient,'fold':fold,'center':center,'method':arm,'rank_reversal_vs_C0':reversal(d['base'][valid],ll[valid]),**{m+'_delta_C0':q[m]-base[m] for m in RANK}})
            print('NEW_CONTROL_OUTER_COMPLETE',fold,arm,flush=True)
        head=pr_head(fold);prdata=transform(raw,held,'C1-PR',fold,'HELD');original=predict(head,prdata,held,assignment['real'])
        shuffled=predict(head,transform(raw,held,'C1-SHUFFLE',fold,'HELD'),held,assignment['real'])
        for patient in held:
            d=raw[patient];ref,scaled,offset,err=scale_diagnostic(d['h'],d['base'],head,assignment['real'][patient])
            center=str(mf.loc[mf.patient_name==patient,'center'].iloc[0]);tau=selected(fold,'C1-PR')['threshold']
            for name,ll in [('SCALE_ONLY',scaled),('INFERENCE_SHUFFLE',shuffled[patient])]:
                path=ROOT/f'diagnostics/{name}/{key(patient)}.npz';binding={'freeze':sha(fpath),'bank':d['bank_sha'],'historical_PR_head':selected(fold,'C1-PR')['checkpoint_sha256'],'threshold':tau,'diagnostic':name}
                hashes[str(path)]=seal_prediction(path,binding,d['channels'],ll)
            with np.load(HIST/'labels'/f'{key(patient)}.npz',allow_pickle=False) as v:y=v['labels'];valid=y>=0
            full=next(r for r in rows if r['patient']==patient and r['method']=='C1-PR')
            rank_ref=metric(y[valid],sigmoid(ref[valid]),order_score=ref[valid]);rank_scaled=metric(y[valid],sigmoid(scaled[valid]),order_score=scaled[valid])
            assert all(abs(rank_ref[m]-rank_scaled[m])<=1e-12 for m in RANK)
            operational_rank=metric(y[valid],sigmoid(original[patient][valid]),order_score=original[patient][valid])
            scalechecks.append({'constant_identity_error':err,'math_rank_error':max(abs(rank_ref[m]-rank_scaled[m]) for m in RANK),'operational_PR_logit_ranking_error':max(abs(operational_rank[m]-rank_scaled[m]) for m in RANK),'historical_probability_metric_error':max(abs(full[m]-rank_scaled[m]) for m in RANK)})
            for name,ll,order in [('SCALE_ONLY',scaled,scaled[valid]),('INFERENCE_SHUFFLE',shuffled[patient],None)]:
                probability=sigmoid(ll);q=metric(y[valid],probability[valid],tau,order_score=order);fixed=classification(y[valid],probability[valid],.5)
                diagnostics.append({'patient':patient,'fold':fold,'center':center,'diagnostic':name,'threshold':tau,**q,**{m+'_delta_PR':q[m]-full[m] for m in METRICS},**{m+'_0_5':fixed[m] for m in CLASS}})
    frame=pd.DataFrame(rows);assert len(frame)==496
    for arm,g in frame.groupby('method'):assert g.patient.nunique()==124 and g.units.sum()==10202
    columns=METRICS+[m+'_0_5' for m in CLASS];csv('OOF_PATIENT_SUMMARY.csv',aggregate(frame,['method'],columns));csv('OOF_BY_FOLD.csv',aggregate(frame,['fold','method'],columns));csv('OOF_BY_CENTER.csv',aggregate(frame,['center','method'],columns))
    rank=pd.DataFrame(ranks);csv('RANKING_CHANGE_AUDIT.csv',aggregate(rank,['fold','center','method'],['rank_reversal_vs_C0']+[m+'_delta_C0' for m in RANK]))
    csv('ERROR_CORRECTION_AUDIT.csv',[{'fold':f,'center':c,'method':a,**{m:int(g[m].sum()) for m in ['recovered','spoiled','TP_gained','TP_lost','FP_added','FP_removed']}} for (f,c,a),g in pd.DataFrame(errors).groupby(['fold','center','method'])])
    diagnostic=pd.DataFrame(diagnostics);diagnostic_columns=columns+[m+'_delta_PR' for m in METRICS]
    csv('INFERENCE_SHUFFLE_INTERVENTION.csv',aggregate(diagnostic[diagnostic.diagnostic=='INFERENCE_SHUFFLE'],['diagnostic','center'],diagnostic_columns)+aggregate(diagnostic[diagnostic.diagnostic=='INFERENCE_SHUFFLE'],['diagnostic'],diagnostic_columns))
    csv('SCALE_ONLY_CLASSIFICATION.csv',aggregate(diagnostic[diagnostic.diagnostic=='SCALE_ONLY'],['diagnostic','center'],diagnostic_columns)+aggregate(diagnostic[diagnostic.diagnostic=='SCALE_ONLY'],['diagnostic'],diagnostic_columns))
    audit=json.loads((ROOT/'public/SCALE_ONLY_INVARIANCE_AUDIT.json').read_text());audit.update({'formal_max_operational_PR_logit_ranking_error':max(q['operational_PR_logit_ranking_error'] for q in scalechecks),'formal_max_historical_probability_metric_error':max(q['historical_probability_metric_error'] for q in scalechecks),'formal_max_math_rank_error':max(q['math_rank_error'] for q in scalechecks),'classification_outputs':'SCALE_ONLY_CLASSIFICATION.csv'});write_json(ROOT/'public/SCALE_ONLY_INVARIANCE_AUDIT.json',audit)
    contrasts=[('C1-PR','C1-ABS'),('C1-PR','C1-SHUFFLE'),('C1-ABS','C0'),('C1-SHUFFLE','C0'),('C1-PR','C0')];boots=[];effects=[];centers=[];counts=[]
    idx=np.random.default_rng(42).integers(0,124,(10000,124))
    for left,right in contrasts:
        a=frame[frame.method==left].set_index('patient').sort_index();b=frame[frame.method==right].set_index('patient').sort_index();assert a.index.equals(b.index)
        delta=a[columns]-b[columns];draws=delta.to_numpy()[idx].mean(1)
        for j,m in enumerate(columns):
            d=delta[m];boots.append({'contrast':left+'-'+right,'metric':m,'policy':'0.5' if m.endswith('_0_5') else 'ranking' if m in RANK else 'frozen_FIT_threshold','delta':float(d.mean()),'CI_low':float(np.quantile(draws[:,j],.025)),'CI_high':float(np.quantile(draws[:,j],.975)),'positive_patients':int((d>0).sum()),'positive_folds':int((d.groupby(a.fold).mean()>0).sum()),'patients':124,'draws':10000,'seed':42})
            effects.append({'contrast':left+'-'+right,'metric':m,'patients':124,'mean':float(d.mean()),'median':float(d.median()),'q25':float(d.quantile(.25)),'q75':float(d.quantile(.75))})
            for center in SOURCES:
                mask=a.center==center;dd=d[mask];centers.append({'contrast':left+'-'+right,'metric':m,'center':center,'patients':int(mask.sum()),'delta':float(dd.mean()),'positive_patients':int((dd>0).sum())})
        for center in SOURCES:
            mask=a.center==center;counts.append({'contrast':left+'-'+right,'center':center,'patients':int(mask.sum()),'AUROC_improved':int((delta.loc[mask,'auroc']>0).sum()),'AP_improved':int((delta.loc[mask,'ap']>0).sum())})
    csv('PAIRED_PATIENT_BOOTSTRAP.csv',boots);csv('PATIENT_EFFECT_DISTRIBUTION.csv',effects);csv('CENTER_PAIRED_EFFECTS.csv',centers);csv('CENTER_IMPROVEMENT_COUNTS.csv',counts)
    means=frame.groupby('method')[columns].mean();boot=pd.DataFrame(boots).set_index(['contrast','metric']);prabs=boot.loc[('C1-PR-C1-ABS','auroc')];prsh=boot.loc[('C1-PR-C1-SHUFFLE','auroc')]
    conditions={'AUROC_delta_ge010':bool(prabs.delta>=.010),'AUROC_CI_lower_positive':bool(prabs.CI_low>0),'AP_nondecline':bool(means.loc['C1-PR','ap']>=means.loc['C1-ABS','ap']),'positive_folds_ge4':bool(prabs.positive_folds>=4),'MacroF1_loss_le005':bool(means.loc['C1-PR','macro_f1']>=means.loc['C1-ABS','macro_f1']-.005),'integrity':True}
    alignment={'AUROC_delta_ge010':bool(prsh.delta>=.010),'AUROC_CI_lower_positive':bool(prsh.CI_low>0),'AP_strict_gain':bool(means.loc['C1-PR','ap']>means.loc['C1-SHUFFLE','ap']),'logits_labels_sources_unpermuted':True}
    prpass=all(conditions.values());alignpass=all(alignment.values())
    abs_better=bool(prabs.delta<0 and prabs.CI_high<0 and means.loc['C1-ABS','ap']>=means.loc['C1-PR','ap'])
    comparable=bool(prabs.CI_low>=-.010 and prabs.CI_high<=.010 and boot.loc[('C1-ABS-C0','auroc'),'CI_low']>0 and all(means.loc['C1-ABS',m]>=means.loc['C0',m] for m in ['ap','macro_f1']) and means.loc['C1-ABS','auroc']-means.loc['C1-SHUFFLE','auroc']>=.010)
    terminal='PATIENT_RELATIVE_INCREMENT_SUPPORTED' if prpass and alignpass else 'EMBEDDING_ALIGNMENT_NOT_SUPPORTED' if not alignpass else 'ABSOLUTE_EMBEDDING_BETTER' if abs_better else 'LINEAR_RECALIBRATION_SUFFICIENT' if comparable else 'MECHANISM_INCONCLUSIVE'
    gate={'terminal':terminal,'PR_mechanism_gate':conditions,'embedding_alignment_gate':alignment,'PR_gate_pass':prpass,'alignment_gate_pass':alignpass,'ABS_better_with_CI':abs_better,'descriptive_comparability_rule_pass':comparable,'exploratory_historical_reuse':True,'protocol_sha256':sha(PROTOCOL)};write_json(ROOT/'public/VALIDATION_GATE.json',gate)
    save_torch(ROOT/'FINAL_LEDGER_PRIVATE.pt',{'rows':rows,'diagnostics':diagnostics,'prediction_hashes':hashes,'freeze':sha(fpath)})
    write_json(ROOT/'public/RUN_STATUS.json',{'status':'COMPLETE','terminal':terminal,'patients':124,'labeled_units':10202,'outer_new_prediction_cells':248,'diagnostic_cells':248,'inner_runs':30,'refits':10,'new_CNN_training':False,'PR_retrained':False,'private_ledger_sha256':sha(ROOT/'FINAL_LEDGER_PRIVATE.pt')})
    final_report(frame,diagnostic,boot,gate)
    print('OMNI_PR_SINGLE_FROZEN_OUTER_COMPLETE',terminal,flush=True)
if __name__=='__main__':main()
