"""CPU-only complete A0 audit. Hash gates -> FIT smoke/selection -> VAL diagnostics."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from core import METRICS,sha,write_json,seed,metric,oracle,episode,fit_pca,cluster_bootstrap
from clinical import provenance

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save_csv(path,rows):
    frame=pd.DataFrame(rows); tmp=Path(path).with_suffix('.partial'); frame.to_csv(tmp,index=False); tmp.replace(path)

def summarize(frame,groupby):
    rows=[]
    for keys,g in frame.groupby(groupby,dropna=False,sort=True):
        if not isinstance(keys,tuple):keys=(keys,)
        # Suppress sparse clinical groups, including fold-by-center intersections.
        if 'center' in groupby and g.patient.nunique()<5:continue
        row={**dict(zip(groupby,keys)),'patient_fold_cells':len(g),'unique_patients':g.patient.nunique()}
        for m in METRICS+['TP','FP','TN','FN','predicted_ez_fraction','observed_ez_fraction']:
            if m in g:
                row[m]=float(g[m].mean()); row[m+'_n']=int(g[m].notna().sum())
        rows.append(row)
    return rows

def contrasts(frame,policy):
    g=frame[frame.policy==policy]
    rows=[]
    for left,right in [('P1','P0'),('P2','P0'),('P2','P1'),('P2','P2_permuted')]:
        a=g[g.method==left].set_index(['fold','patient']); b=g[g.method==right].set_index(['fold','patient'])
        assert a.index.equals(b.index)
        for m in METRICS:
            delta=(a[m]-b[m]).to_numpy(); boot=cluster_bootstrap(a.index.get_level_values('patient'),delta)[:,0]
            valid=np.isfinite(delta); bv=boot[np.isfinite(boot)]
            foldgain=(a[m]-b[m]).groupby(level='fold').mean()
            rows.append({'policy':policy,'contrast':left+'-'+right,'metric':m,'delta':float(np.nanmean(delta)),
                'CI_low':float(np.quantile(bv,.025)) if len(bv) else np.nan,'CI_high':float(np.quantile(bv,.975)) if len(bv) else np.nan,
                'contributing_cells':int(valid.sum()),'unique_patients':len(set(a.index.get_level_values('patient')[valid])),
                'positive_folds':int((foldgain>0).sum()),'draws':10000,'valid_draws':len(bv)})
    return rows

def main():
    p=argparse.ArgumentParser()
    for n in ['source','runtime','protocol','bank-runtime','oof-runtime']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args(); root=a.runtime; pub=root/'public'; pub.mkdir(parents=True,exist_ok=True)
    lock=read(a.protocol); code=Path(__file__).parent
    binding={'protocol':sha(a.protocol),'code':{q.name:sha(q) for q in sorted(code.glob('*.py'))}}
    status=lambda s,**kw:write_json(pub/'RUN_STATUS.json',{'status':s,'binding':binding,'outer_test_accessed':False,**kw})
    status('VERIFYING_FROZEN_INPUTS')
    core=a.source/'pr_uncertainty_aware_supervision_seed42_v1/code/uas_core.py'
    assert sha(core)==lock['a0_core_sha256']; sys.path.insert(0,str(core.parent))
    from uas_core import PRMLP,prepare,oof_plan,validate_exclusion
    torch.set_num_threads(2)
    export=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert sha(export)==lock['private_feature_sha256']
    split=Path(r'D:\nips-temp\task1_aaai_completion_training\audit\fixed_partition_manifest.csv')
    assert sha(split)==lock['split_sha256']
    with np.load(export,allow_pickle=True) as f:raw={k:f[k] for k in f.files}
    assert raw['x'].shape==(7635,88) and len(set(raw['patient']))==80 and len(set(zip(raw['patient'],raw['channel'])))==7635
    assert hashlib.sha256(json.dumps(raw['features'].tolist(),separators=(',',':')).encode()).hexdigest()==lock['feature_order_sha256']
    assert sha(a.oof_runtime/'gate/FEATURES_PRIVATE.npz')==sha(export)
    teacher_audit=read(a.oof_runtime/'OOF_TEACHER_AUDIT.json'); assert teacher_audit['status']=='PASS'
    banks=[]; replay=[]; checks=[]; pcas=[]; oofs=[]; oofchecks=[]; block=None
    for fold in range(1,6):
        bankpath=a.bank_runtime/f'fold{fold}/DEVELOPMENT_PRIVATE.pt'
        assert sha(bankpath)==lock['bank_sha256'][fold-1]
        bank=torch.load(bankpath,map_location='cpu',weights_only=False)
        # Verify identities and exact prepared data anew, never read outer predictions.
        s=raw['split_fold']==fold
        ids={r:set(raw['split_patient'][s&(raw['split_role']==r)]) for r in ['fit','validation','test']}
        assert all(not ids[r]&ids[t] for r,t in [('fit','validation'),('fit','test'),('validation','test')])
        di=np.flatnonzero(np.isin(raw['patient'],list(ids['fit']|ids['validation'])))
        for k in ['y','patient','channel','center']:assert np.array_equal(raw[k][di],bank[k])
        for k,r in [('train','fit'),('val','validation')]:assert set(bank['patient'][bank[k]])==ids[r]
        xx,pre=prepare(raw['x'][di],bank['patient'],bank['train'])
        assert np.array_equal(xx,bank['x']) and np.array_equal(bank['features'],raw['features'])
        assert all(np.array_equal(pre[k],bank['pre'][k]) for k in ['mean','scale','var','imputer_statistics'])
        assert set(pre['fit_patient_ids'])==ids['fit']
        ck=Path(bank['A0_checkpoint']); assert sha(ck)==lock['checkpoint_sha256'][fold-1]
        saved=torch.load(ck,map_location='cpu',weights_only=False)
        assert saved['epoch']==lock['epochs'][fold-1] and saved['threshold']==lock['thresholds'][fold-1]==bank['A0_threshold']
        model=PRMLP().eval().requires_grad_(False); model.load_state_dict(saved['model']); assert sum(v.numel() for v in model.parameters())==8817
        va=bank['val']
        with torch.no_grad():score=torch.sigmoid(model(torch.from_numpy(xx[va]))).numpy()
        ref=pd.read_csv(a.bank_runtime/f'fold{fold}/A0_CHANNEL_PRIVATE.csv')
        new=pd.DataFrame({'patient':bank['patient'][va],'channel':bank['channel'][va],'y_nez':bank['y'][va],'score':score})
        merged=new.merge(ref,on=['patient','channel','y_nez'],validate='one_to_one')
        assert len(merged)==len(va)==len(ref); drift=float(np.abs(merged.score-merged.score_nez).max()); assert drift<=lock['probability_tolerance']
        frozen_map={(str(r.patient),str(r.channel)):float(r.score_nez) for r in ref.itertuples()}
        bank['score']=np.full(len(xx),np.nan)
        bank['score'][va]=[frozen_map[(str(bank['patient'][i]),str(bank['channel'][i]))] for i in va]
        # Canonical order is locked for stable ranking tie resolution.
        oldpm=pd.read_csv(a.bank_runtime/f'fold{fold}/A0_PATIENT_PRIVATE.csv').set_index('patient')
        for patient in sorted(ids['validation']):
            ix=va[bank['patient'][va]==patient]; ix=ix[np.argsort(bank['channel'][ix].astype(str),kind='stable')]
            m=metric(bank['y'][ix],bank['score'][ix],saved['threshold'])
            for k in METRICS:assert np.isclose(m[k],oldpm.loc[patient,k],rtol=0,atol=1e-12,equal_nan=True)
            replay.append({'fold':fold,'patient':str(patient),'center':str(bank['center'][ix[0]]),'method':'A0',**m})
        pca,z=fit_pca(xx,bank['train']); bank['z']=z; banks.append(bank)
        pcas.append({'fold':fold,'fit_patients':len(ids['fit']),'fit_channels':len(bank['train']),
            'dimension':4,'label_used':False,'outer_test_rows':0,'explained_variance_ratio':pca.explained_variance_ratio_.tolist()})
        checks.append({'fold':fold,'bank_sha256':sha(bankpath),'checkpoint_sha256':sha(ck),'epoch':saved['epoch'],
            'threshold':saved['threshold'],'probability_drift':drift,'identities_features_preprocessor_exact':True})
        # B provenance failure must not prevent independent A/C audits.
        try:
            fi=np.flatnonzero(np.isin(raw['patient'],list(ids['fit']))); fr=a.oof_runtime/f'fold{fold}'
            frozen=torch.load(fr/'OOF_FROZEN_PRIVATE.pt',map_location='cpu',weights_only=False)
            bind=json.loads(frozen['binding']); assert bind['data']==sha(export) and bind['uas_core.py']==sha(core)
            assert np.array_equal(frozen['patient'],raw['patient'][fi]) and np.array_equal(frozen['channel'],raw['channel'][fi]) and np.array_equal(frozen['y_nez'],raw['y'][fi])
            q=np.full(len(fi),np.nan); seen=np.zeros(len(fi),int)
            for plan in oof_plan(ids['fit'],fold):
                k=plan['group']; cell=fr/f'teacher{k}'; record=torch.load(cell/'OOF_PRIVATE.pt',map_location='cpu',weights_only=False)
                assert record['source']['plan']==plan and record['source']['binding']==frozen['binding']
                validate_exclusion(plan['train'],plan['validation'],plan['query'],ids['fit'])
                assert not set(plan['train']+plan['validation']+plan['query'])&(ids['validation']|ids['test'])
                check=next(r for r in teacher_audit['folds'] if r['fold']==fold and r['teacher_group']==k)
                assert sha(cell/'OOF_PRIVATE.pt')==check['OOF_file_sha256']==frozen['teacher_hashes'][k]
                assert sha(cell/'BEST_PRIVATE.pt')==check['checkpoint_sha256']==record['source']['checkpoint']
                ix=np.flatnonzero(np.isin(raw['patient'][fi],plan['query'])); tr=np.flatnonzero(np.isin(raw['patient'][fi],plan['train']))
                assert np.array_equal(record['patient'],raw['patient'][fi][ix]) and np.array_equal(record['channel'],raw['channel'][fi][ix])
                _,tp=prepare(raw['x'][fi],raw['patient'][fi],tr)
                assert all(np.array_equal(tp[t],record['preprocessor'][t]) for t in ['mean','scale','var','imputer_statistics'])
                assert tp['fit_patient_ids']==record['preprocessor']['fit_patient_ids']==plan['train']
                q[ix]=record['mc']['q']; seen[ix]+=1
                oofchecks.append({'fold':fold,'group':k,'output_sha256':sha(cell/'OOF_PRIVATE.pt'),'checkpoint_sha256':sha(cell/'BEST_PRIVATE.pt'),
                    'train_query_overlap':0,'selection_query_overlap':0,'teacher_preprocessor_exact':True})
            assert (seen==1).all() and np.array_equal(q,frozen['q']) and np.isfinite(q).all()
            mapping={(str(p),str(c)):float(v) for p,c,v in zip(frozen['patient'],frozen['channel'],q)}
            tr=bank['train']; oq=np.asarray([mapping[(str(bank['patient'][i]),str(bank['channel'][i]))] for i in tr])
            oofs.append((tr,oq))
        except (AssertionError,FileNotFoundError,KeyError,ValueError) as e:
            block={'fold':fold,'type':type(e).__name__,'reason':'FIT OOF artifact/exclusion/exact-source gate failed; no in-sample substitute'}; oofs.append(None)
        print('FROZEN_A0_REPLAY_PASS',fold,'OOF_GATE',oofs[-1] is not None,flush=True)
    baseline=pd.DataFrame(replay); assert len(baseline)==65 and baseline.patient.nunique()==47
    expected={'macro_f1':.6380797828499001,'ez_f1':.431198131752259,'ez_auprc':.518235100110259,'ez_auroc':.7106669304311658}
    assert all(abs(baseline[k].mean()-v)<1e-12 for k,v in expected.items())
    write_json(pub/'A0_REPRODUCTION.json',{'status':'PASS','folds':checks,'metrics':{k:float(baseline[k].mean()) for k in METRICS},'patient_fold_cells':65,'unique_patients':47,'retrained':False,'binding':binding})
    write_json(pub/'PCA_REPRESENTATION_AUDIT.json',{'status':'PASS','folds':pcas,'predictive_input_fields':['original_A0_prepared_88D'],'center_outcome_resection_used_as_predictors':False})
    # FIT-only smoke and lambda choice are completed before ANY new VAL probes.
    choices=[]; gridrows=[]; smoke=[]
    if block is None:
        for fold,(bank,(tr,oq)) in enumerate(zip(banks,oofs),1):
            foldgrid=[]
            for lam in lock['lambda_grid']:
                scores={'P1':[],'P2':[]}
                for patient in sorted(set(bank['patient'][tr])):
                    local=np.flatnonzero(bank['patient'][tr]==patient); local=local[np.argsort(bank['channel'][tr][local].astype(str),kind='stable')]; ix=tr[local]
                    if len(ix)<=8:continue
                    result,audit=episode(bank['y'][ix],oq[local],bank['z'][ix],bank['channel'][ix],bank['A0_threshold'],8,'uncertainty',seed(fold,patient,'fit',0),(lam,lam))
                    assert all(r['converged'] for r in audit['checks'])
                    for method in scores:scores[method].append(result[method]['macro_f1'])
                for method,values in scores.items():
                    row={'fold':fold,'method':method,'lambda':lam,'FIT_OOF_query_macro_f1':float(np.mean(values)),'FIT_patients':len(values),'VAL_labels_used':False}; gridrows.append(row); foldgrid.append(row)
            choice={m:max([r for r in foldgrid if r['method']==m],key=lambda r:(r['FIT_OOF_query_macro_f1'],r['lambda']))['lambda'] for m in ['P1','P2']}
            choices.append({'fold':fold,**choice}); smoke.append({'fold':fold,'solver_finite_converged':True,'FIT_episodes_checked':len(scores['P1']),'VAL_query_read':False})
        write_json(pub/'REGULARIZATION_SELECTION_AUDIT.json',{'status':'PASS','choices':choices,'grid':gridrows,'OOF_provenance':oofchecks,
            'score_definition':'frozen 10-pass MC dropout mean; not deterministic final-A0 score','selection_before_new_VAL_probes':True,'cross_outer_fold_selection_pooling':False,'binding':binding})
    else:write_json(pub/'REGULARIZATION_SELECTION_AUDIT.json',{'status':'BLOCKED',**block})
    write_json(pub/'FIT_ONLY_SMOKE.json',{'status':'PASS' if block is None else 'B_BLOCKED','folds':smoke,'no_outer_test':True})
    frozenchoices=sha(pub/'REGULARIZATION_SELECTION_AUDIT.json')
    write_json(pub/'PRE_VAL_FREEZE.json',{'binding':binding,'regularization_sha256':frozenchoices,'all_A0_hashes_verified':True})
    status('RUNNING_FROZEN_VAL_DIAGNOSTICS',regularization_sha256=frozenchoices)
    # Exhaustive oracle on all validation channels; private per-cell ledger only.
    oracle_rows=[]; summaries=[]
    for fold,bank in enumerate(banks,1):
        va=bank['val']
        for patient in sorted(set(bank['patient'][va])):
            ix=va[bank['patient'][va]==patient]; ix=ix[np.argsort(bank['channel'][ix].astype(str),kind='stable')]
            value=oracle(bank['y'][ix],bank['score'][ix],bank['A0_threshold']); c=str(bank['center'][ix[0]])
            for name in ['baseline','oracle','grid_oracle','true_count_topk']:oracle_rows.append({'fold':fold,'patient':str(patient),'center':c,'method':name,**value[name]})
            summaries.append({'fold':fold,'patient':str(patient),'center':c,**{k:v for k,v in value.items() if k not in ['baseline','oracle','grid_oracle','true_count_topk']},
                'baseline_macro':value['baseline']['macro_f1'],'oracle_macro':value['oracle']['macro_f1'],
                **{m:value['baseline'][m] for m in ['ez_auprc','ez_auroc','ez_mrr','top1_is_ez']}})
        print('ORACLE_FOLD_COMPLETE',fold,flush=True)
    of=pd.DataFrame(oracle_rows); hs=pd.DataFrame(summaries)
    save_csv(pub/'PATIENT_ORACLE_SUMMARY.csv',summarize(of,['method'])); save_csv(pub/'ORACLE_BY_FOLD.csv',summarize(of,['fold','method'])); save_csv(pub/'ORACLE_BY_CENTER.csv',summarize(of,['center','method']))
    hs['headroom_group']=np.where(hs.headroom<.01,'little_lt_0.01',np.where(hs.headroom<.05,'moderate_0.01_to_0.05','large_ge_0.05'))
    hrows=[]
    for label,g in hs.groupby('headroom_group'):
        hrows.append({'group':label,'cells':len(g),'unique_patients':g.patient.nunique(),**{k:float(g[k].mean()) for k in ['headroom','baseline_macro','oracle_macro','distance','cardinality_change','ez_auprc','ez_auroc','ez_mrr','top1_is_ez']}})
    hrows.append({'group':'ALL','cells':65,'unique_patients':47,'headroom':float(hs.headroom.mean()),'q10':float(hs.headroom.quantile(.1)),'median':float(hs.headroom.median()),'q90':float(hs.headroom.quantile(.9))})
    save_csv(pub/'ORACLE_HEADROOM_DISTRIBUTION.csv',hrows)
    write_json(pub/'THRESHOLD_INTERVAL_AUDIT.json',{'status':'PASS','comparison':'NEZ score >= tau','ties':'equal scores move together; canonical ties only in separate TopK',
        'multi_optimal_decision_set_fraction':float((hs.optimal_decision_sets>1).mean()),'disconnected_region_fraction':float(hs.disconnected.mean()),
        'mean_distance_to_optimal_region':float(hs.distance.mean()),'mean_grid_resolution_loss':float(hs.grid_resolution_loss.mean()),'max_grid_resolution_loss':float(hs.grid_resolution_loss.max()),
        'true_count_topk_cuts_tie_cells':int(hs.topk_cuts_tie.sum()),'optimal_thresholds_private':True,'unbounded_extreme_intervals_supported':True})
    associations=[]
    for m in ['ez_auprc','ez_auroc','ez_mrr','top1_is_ez']:
        g=hs[['headroom',m]].dropna(); r=spearmanr(g.headroom,g[m]); associations.append({'metric':m,'spearman_headroom_association':float(r.statistic),'cells':len(g),'interpretation':'descriptive noncausal; not a unique calibration-ranking error partition'})
    save_csv(pub/'RANKING_DECISION_DECOMPOSITION.csv',associations)
    # Private resume cache binds code, protocol, choices and exact patient inputs.
    prows=[]; episodes=[]; failures=0
    if block is None:
        cache=root/'episodes_private'; cache.mkdir(exist_ok=True)
        for fold,bank in enumerate(banks,1):
            va=bank['val']; lam=choices[fold-1]; lambdas=(lam['P1'],lam['P2'])
            for patient in sorted(set(bank['patient'][va])):
                ix=va[bank['patient'][va]==patient]; ix=ix[np.argsort(bank['channel'][ix].astype(str),kind='stable')]
                for policy,reps,budget,kind in [('B8_uncertainty',1,8,'uncertainty'),('B8_random',20,8,'random'),('half_random',10,len(ix)//2,'random')]:
                    if budget>=len(ix):continue
                    repetitions=[]
                    for rep in range(reps):
                        key=hashlib.sha256(json.dumps([binding,frozenchoices,fold,str(patient),policy,rep],sort_keys=True).encode()).hexdigest()
                        path=cache/(key+'.json')
                        if path.exists():
                            cell=read(path); assert cell['key']==key
                        else:
                            result,check=episode(bank['y'][ix],bank['score'][ix],bank['z'][ix],bank['channel'][ix],bank['A0_threshold'],budget,kind,seed(fold,str(patient),policy,rep),lambdas)
                            cell={'key':key,'metrics':result,'check':check}; write_json(path,cell)
                        repetitions.append(cell['metrics']); episodes.append({'fold':fold,'patient':str(patient),'policy':policy,'one_class':cell['check']['one_class'],
                            'ranking_change_fraction':cell['check']['ranking_change_fraction'],'budget':budget,'query':len(ix)-budget})
                        failures+=sum(not r['converged'] for r in cell['check']['checks'])
                    for method in ['P0','P1','P2','P2_permuted']:
                        row={'fold':fold,'patient':str(patient),'center':str(bank['center'][ix[0]]),'policy':policy,'method':method,'repetitions':reps}
                        for m in METRICS:
                            values=[c[method][m] for c in repetitions]; valid=[v for v in values if v is not None and np.isfinite(v)]
                            row[m]=float(np.mean(valid)) if valid else np.nan; row[m+'_valid_reps']=len(valid)
                        prows.append(row)
            print('PROBE_FOLD_COMPLETE',fold,flush=True)
    pf=pd.DataFrame(prows); bootstrap=[]
    oracle_delta=of[of.method=='oracle'].macro_f1.to_numpy()-of[of.method=='baseline'].macro_f1.to_numpy()
    ob=cluster_bootstrap(hs.patient,oracle_delta)[:,0]; bootstrap.append({'policy':'full_channel_oracle','contrast':'oracle-baseline','metric':'macro_f1','delta':float(np.mean(oracle_delta)),
        'CI_low':float(np.quantile(ob,.025)),'CI_high':float(np.quantile(ob,.975)),'contributing_cells':65,'unique_patients':47,'draws':10000,'valid_draws':10000})
    if block is None:
        for policy in ['B8_uncertainty','B8_random','half_random']:bootstrap.extend(contrasts(pf,policy))
        primary=pf[pf.policy=='B8_uncertainty']
        save_csv(pub/'B8_PROBE_SUMMARY.csv',summarize(primary,['policy','method']))
        save_csv(pub/'B8_PROBE_BY_FOLD.csv',summarize(primary,['fold','method']))
        save_csv(pub/'B8_PROBE_BY_CENTER.csv',summarize(primary,['center','method']))
        save_csv(pub/'B8_RANDOM_CONTROL.csv',summarize(pf[pf.policy=='B8_random'],['policy','method']))
        save_csv(pub/'HALF_SUPPORT_CAPACITY.csv',summarize(pf[pf.policy=='half_random'],['policy','method']))
        save_csv(pub/'DIRECTION_VS_BIAS.csv',[r for r in bootstrap if r['contrast']=='P2-P1'])
        save_csv(pub/'SUPPORT_LABEL_PERMUTATION.csv',[r for r in bootstrap if r['contrast']=='P2-P2_permuted'])
        ep=pd.DataFrame(episodes); g=ep.groupby(['fold','patient','policy'],as_index=False).agg(one_class_fraction=('one_class','mean'),ranking_change_fraction=('ranking_change_fraction','mean'),budget=('budget','mean'),query=('query','mean'))
        diagnostics=[]
        for policy,h in g.groupby('policy'):
            diagnostics.append({'policy':policy,'patient_fold_cells':len(h),'unique_patients':h.patient.nunique(),'mean_one_class_support_fraction':h.one_class_fraction.mean(),
                'mean_ranking_change_fraction':h.ranking_change_fraction.mean(),'mean_support_channels':h.budget.mean(),'mean_query_channels':h['query'].mean()})
        save_csv(pub/'PATIENT_GEOMETRY_DIAGNOSTICS.csv',diagnostics)
        write_json(pub/'FEWSHOT_SUPPORT_AUDIT.json',{'status':'PASS','patient_fold_cells':65,'unique_patients':47,'episodes':len(ep),'policies':diagnostics,
            'support_query_disjoint':True,'support_selection_label_blind':True,'same_query_all_methods':True,'query_labels_not_fitted':True,
            'repetitions_averaged_before_cluster_bootstrap':True,'one_class_regularized_no_fabrication':True,'solver_fallbacks':failures,
            'P1_ranking_identity_all_episodes':True,'permutation_degenerate_one_class_count':int(ep.one_class.sum()),'coefficients_predictions_private':True})
    else:
        for name in ['B8_PROBE_SUMMARY','B8_PROBE_BY_FOLD','B8_PROBE_BY_CENTER','B8_RANDOM_CONTROL','HALF_SUPPORT_CAPACITY','DIRECTION_VS_BIAS','SUPPORT_LABEL_PERMUTATION','PATIENT_GEOMETRY_DIAGNOSTICS']:
            save_csv(pub/(name+'.csv'),[{'status':'BLOCKED','reason':block['reason']}])
        write_json(pub/'FEWSHOT_SUPPORT_AUDIT.json',{'status':'BLOCKED',**block})
    save_csv(pub/'PATIENT_CLUSTER_BOOTSTRAP.csv',bootstrap)
    status('CLINICAL_METADATA_AND_FINALIZATION')
    clinical,ca=provenance(a.source,export,sha)
    write_json(pub/'CLINICAL_SOURCE_INVENTORY.json',ca); write_json(pub/'LABEL_ALIGNMENT_AUDIT.json',ca)
    save_csv(pub/'LABEL_SOURCE_COUNTS.csv',ca['label_source_counts']); save_csv(pub/'TARGET_DEFINITION_COMPARISON.csv',ca['target_comparison'])
    assoc=[]
    for row in ca['label_source_counts']:
        ids={p for p,m in clinical.items() if m['center']==row['center'] and (m['source']==row['label_source'] or row['label_source'].startswith('CENTER_AGGREGATE'))}
        h=baseline[baseline.patient.isin(ids)]
        if h.patient.nunique()<5:continue
        assoc.append({**row,'development_unique_patients':h.patient.nunique(),'development_patient_fold_cells':len(h),
            **{m:float(h[m].mean()) for m in ['macro_f1','ez_f1','ez_auprc','observed_ez_fraction']},'causal_claim':False})
    save_csv(pub/'LABEL_SOURCE_ERROR_ASSOCIATION.csv',assoc)
    # Gate hierarchy is explicitly reported with all component findings.
    def value(policy,contrast,m):return next(r for r in bootstrap if r['policy']==policy and r['contrast']==contrast and r['metric']==m)
    def gates(policy):
        b=value(policy,'P1-P0','macro_f1'); d=value(policy,'P2-P1','macro_f1'); perm=value(policy,'P2-P2_permuted','macro_f1')
        bias=b['delta']>=.015 and b['CI_low']>0 and b['positive_folds']>=4 and value(policy,'P1-P0','ez_f1')['delta']>=0
        direction=d['delta']>=.01 and d['CI_low']>0 and d['positive_folds']>=4 and value(policy,'P2-P1','ez_auprc')['delta']>=.005 and value(policy,'P2-P1','ez_f1')['delta']>=0 and perm['delta']>0 and perm['CI_low']>0
        return {'bias_pass':bias,'direction_pass':direction,'bias_macro':b,'direction_macro':d,'direction_AP':value(policy,'P2-P1','ez_auprc'),'permutation_macro':perm}
    bg=gates('B8_uncertainty') if block is None else {'status':'BLOCKED'}; hg=gates('half_random') if block is None else {'status':'BLOCKED'}
    headroom=float(hs.headroom.mean()); astat='DECISION_HEADROOM_PRESENT' if headroom>=.03 else 'LIMITED_THRESHOLD_HEADROOM'
    unresolved=sum(r.get('run_union_disagree',0) for r in ca['target_comparison'])
    cstat='TARGET_PROVENANCE_INCOMPLETE' if unresolved else 'HETEROGENEOUS_SOURCE_DEFINITIONS_CONFIRMED'
    terminal=('EXPERIMENT_BLOCKED' if block is not None else 'PATIENT_DIRECTION_ADAPTATION_SUPPORTED' if bg.get('direction_pass') else
        'PATIENT_BIAS_ADAPTATION_SUPPORTED' if bg.get('bias_pass') else 'HIGH_LABEL_BUDGET_REQUIRED' if hg.get('bias_pass') or hg.get('direction_pass') else
        'LIMITED_DECISION_HEADROOM' if headroom<.03 else 'PATIENT_ORACLE_HEADROOM_NOT_RECOVERABLE')
    if cstat=='TARGET_PROVENANCE_INCOMPLETE':terminal='LABEL_PROVENANCE_REQUIRES_RESOLUTION'
    decision={'terminal':terminal,'A':{'status':astat,'headroom':headroom,'oracle_macro_f1':float(of[of.method=='oracle'].macro_f1.mean())},
        'B8':bg,'half_capacity':hg,'C':{'status':cstat,'run_union_disagreements':unresolved,'source_execution_version_not_embedded':True},
        'zero_shot_established':False,'all_results_exploratory':True,'outer_test_accessed':False,
        'next_experiment':'Independent clinical target-definition harmonization/provenance validation with frozen score analysis; do not build another zero-shot head',
        'next_experiment_started':False}
    write_json(pub/'FINAL_DECISION_MATRIX.json',decision)
    private=root/'AUDIT_PRIVATE.pt'
    torch.save({'binding':binding,'baseline':baseline,'oracle':of,'headroom':hs,'probes':pf,'clinical':clinical},private)
    status('COMPLETE',terminal=terminal,patient_fold_cells=65,unique_patients=47,regularization_sha256=frozenchoices)
    print(json.dumps(decision),flush=True)

if __name__=='__main__':main()
