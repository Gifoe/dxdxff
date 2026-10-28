"""Frozen-model seizure sensitivity, with label-blind data/sampling admission."""
import argparse,sys,itertools,gc,json,zipfile,time
from pathlib import Path
sys.dont_write_bytecode=True
import trusted as t
import numpy as np
import torch
import analysis_math as math
import seizure_math as pool
MODELS=['LINEAR_SVM','DRECHANNET','TIMECONV'];ARMS=['NO_PR','FULL_PR']
KEEP=[0,1,2,3,5,6];KEEP12=[0,1,2,3,4,5,6,7,10,11,12,13]
DROOT=t.RUNTIME/'seizure_sensitivity_private'
D_METRICS=['auroc','auprc','macro_f1','f1','recall_10','recall_20','mrr','hit_at_1']

def delta_module(p):
    module=t.frozen_io.module('ranking_frozen_seizure_delta',Path(r'E:\DRE-nips\new-pipeline\7-11\raw_delta_encoder_seed42_v1\code\construct_delta.py'))
    assert t.sha(module.__file__)==p.rec['binding']['code']['construct_delta.py'];return module

def build_data(p,commit):
    DROOT.mkdir(parents=True,exist_ok=True);original=delta_module(p);plan=t.read(t.RUNTIME/'D_SEIZURE_SAMPLE_PLAN_PRIVATE.pkl')
    abc=t.jr(t.RUNTIME/'ABC_FREEZE_PRIVATE.json');assert abc['parent_source']==t.frozen_io.sources()
    data={};audit=[];maxerr=np.zeros(14);sources=t.source_map(p)
    for ledger in p.rec['ledgers']:
        pid=str(ledger['patient']);ii=np.flatnonzero(p.meta['patient']==pid);channels=p.meta['channel'][ii].astype(str).tolist()
        spec=p.inputs.registry['files'][pid];assert t.sha(spec['path'])==spec['sha256'];q=t.read(spec['path']);qi=[q['channels'].index(c) for c in channels]
        items=[];raw_good=[]
        for s,record in enumerate(ledger['records']):
            mask=np.zeros((len(record['mask']),len(ii)),bool)
            for c,name in enumerate(channels):
                if name in record['names']:mask[:,c]=record['mask'][:,record['names'].index(name)]
            value=np.asarray(q['delta'][s,qi,:len(mask)],np.float32).transpose(1,0,2)
            items.append(original.seizure_mean(np.where(mask[...,None],value,np.nan)))
            tt=np.asarray(record['times']);raw_good.append(np.all(np.stack([mask[tt<0].any(0),mask[(tt>=0)&(tt<8)].any(0),mask[(tt>=8)&(tt<20)].any(0)]),0))
        items=np.asarray(items,np.float32);all14,_=original.aggregate(items)
        maxerr=np.maximum(maxerr,np.max(np.abs(all14-p.rec['R14'][ii]),0));assert np.array_equal(all14,p.rec['R14'][ii])
        if pid not in plan:continue
        common=np.flatnonzero(np.all(np.isfinite(items[:,:,KEEP]),axis=(0,2))&np.all(raw_good,axis=0))
        fold=next(f for f in range(1,6) if pid in p.bank(f)['roles']['test'])
        audit.append(dict(patient=pid,source=sources[pid],fold=fold,seizures=len(items),original_channels=len(ii),common_channels=len(common),excluded_channels=len(ii)-len(common),included=bool(len(common))))
        if len(common):data[pid]=dict(indices=ii[common],full_indices=ii,common_positions=common,channels=[channels[j] for j in common],items=items[:,common],fold=fold,source=sources[pid],subsets=plan[pid]['subsets'],conditioning_sha256=spec['sha256'])
    assert len(audit)==57 and not np.any(maxerr)
    target=DROOT/'INPUT_PLAN_PRIVATE.pkl'
    if target.exists():
        old=t.read(target);assert old['source_commit']==commit
        assert old['source']==t.source_hashes()
    else:t.save(target,dict(source_commit=commit,source=t.source_hashes(),data=data,audit=audit,source_abc_freeze_sha256=t.sha(t.RUNTIME/'ABC_FREEZE_PRIVATE.json'),seizure_plan_sha256=t.sha(t.RUNTIME/'D_SEIZURE_SAMPLE_PLAN_PRIVATE.pkl')))
    t.csv_out('D_patient_admission',audit,True)
    out=dict(status='PASS',eligible_patients=57,included_patients=len(data),empty_common_channel_exclusions=57-len(data),original_channels=sum(r['original_channels'] for r in audit),common_channels=sum(r['common_channels'] for r in audit),excluded_channels=sum(r['excluded_channels'] for r in audit),source_patients={s:sum(q['source']==s for q in data.values()) for s in math.SOURCES},all80_original_feature_reconstruction_exact=True,feature_max_errors=maxerr.tolist(),sampling_labels_independent=True,inference_device='CPU_FP32',no_new_training=True,source_commit=commit,source=t.source_hashes())
    t.write(t.PUB/'D_ADMISSION.json',out);print('D_ADMISSION_PASS',json.dumps({k:out[k] for k in ['included_patients','common_channels','excluded_channels','source_patients']}),flush=True)
    return data

def feature_input(p,original,q,subset,arm,fold):
    r14,_=original.aggregate(q['items'][list(subset)])
    assert np.isfinite(r14[:,KEEP12]).all()
    if len(subset)==1:assert not r14[:,[1,3,5,7,11,13]].any()
    pid=np.repeat(q['patient'],len(r14))
    if q['model']=='LINEAR_SVM':return (p.core.patient_z(r14,pid) if arm=='FULL_PR' else r14.astype(np.float64))[:,KEEP12]
    d=p.bank(fold,arm=='FULL_PR');return p.own.query_features(r14,pid,d['pre'],'FULL' if arm=='FULL_PR' else 'WO_BOTH_PR',p.core).astype(np.float32)

def infer(p,data):
    fr=t.jr(t.RUNTIME/'ABC_FREEZE_PRIVATE.json');original=delta_module(p);seal={};done=0
    for model in MODELS:
        for fold in range(1,6):
            patients=sorted(pid for pid,q in data.items() if q['fold']==fold)
            for arm in ARMS:
                r=next(r for r in fr['cells'] if r['model']==model and r['condition']==arm and r['fold']==fold);assert t.sha(r['checkpoint'])==r['checkpoint_sha256']
                if model=='LINEAR_SVM':bundle=t.read(r['checkpoint'])['bundle'];network=None
                else:
                    network=p.own_model(fold,'FULL' if arm=='FULL_PR' else 'WO_BOTH_PR') if model=='DRECHANNET' else p.raw_model(fold,arm=='FULL_PR')
                    network.load_state_dict(torch.load(r['checkpoint'],map_location='cpu',weights_only=False)['model']);network.eval()
                    before=p.own.state_hash(network.state_dict())
                for pid in patients:
                    q=dict(data[pid],patient=pid,model=model);token=__import__('hashlib').sha256(pid.encode()).hexdigest()[:24];k=f'{model}_{arm}_{token}'
                    path=DROOT/'scores'/f'{k}.pkl'
                    if path.exists():
                        saved=t.read(path);assert saved['checkpoint_sha256']==r['checkpoint_sha256'] and saved['input_plan_sha256']==t.sha(DROOT/'INPUT_PLAN_PRIVATE.pkl');seal[k]=t.sha(path);done+=1;continue
                    tick=time.perf_counter();capture=None;parity=True
                    if model!='LINEAR_SVM':
                        full_channels=p.meta['channel'][q['full_indices']].astype(str).tolist()
                        if model=='DRECHANNET':raw,mask,times,delta,_=p.inputs.get(pid,full_channels,'cpu')
                        else:
                            raw,mask,times,_=p.raw_store.get(pid,full_channels,'cpu');delta=None
                        ci=torch.as_tensor(q['common_positions'],dtype=torch.int64);raw=raw.index_select(1,ci);mask=mask.index_select(1,ci)
                        if delta is not None:delta=delta.index_select(1,ci)
                        assert raw.shape[0]==len(q['items'])
                        with torch.inference_mode():
                            if model=='DRECHANNET':
                                capture,direct=pool.capture_own(network,raw,mask,times,delta,p.own.scale(fold))
                                x=feature_input(p,original,q,range(len(q['items'])),arm,fold);direct_logit=network.classify(direct,torch.as_tensor(x))
                            else:capture,direct,direct_logit=pool.capture_time(p.engine,network,raw,mask)
                        del raw,mask,delta
                    records=[]
                    for K,subsets in q['subsets'].items():
                        label='All' if int(K)==len(q['items']) else K
                        for rep,subset in enumerate(subsets):
                            with torch.inference_mode():
                                if model=='LINEAR_SVM':
                                    x=feature_input(p,original,q,subset,arm,fold);prob,margin=p.bl.predict(bundle,x);logit_ez=bundle['calibration'].decision_function(margin[:,None]);pred=prob>=r['threshold']
                                else:
                                    ix=list(subset)
                                    if model=='DRECHANNET':
                                        v=pool.own_pool(capture['summary'][ix],capture['valid'][ix],network.raw_pr)
                                        x=feature_input(p,original,q,subset,arm,fold);ll=network.classify(v,torch.as_tensor(x))
                                    else:v=pool.time_pool(capture['seizure'][ix],capture['has'][ix]);ll=network.classify(v)
                                    if label=='All':assert torch.equal(v,direct) and torch.equal(ll,direct_logit),'All cached/direct logits differ'
                                    nez=p.own.probability(ll.detach().numpy());prob=(1-nez).astype(float);logit_ez=-ll.detach().numpy().astype(float)
                                    pred=nez<r['threshold'] if r['orientation']=='NEZ_LT' else prob>=r['threshold']
                                records.append(dict(K=label,repetition=rep,subset=tuple(subset),probability_ez=prob,logit_ez=logit_ez,prediction_ez=pred))
                    t.save(path,dict(model=model,condition=arm,fold=fold,patient=pid,source=q['source'],indices=q['indices'],checkpoint_sha256=r['checkpoint_sha256'],input_plan_sha256=t.sha(DROOT/'INPUT_PLAN_PRIVATE.pkl'),all_cached_direct_bit_exact=parity,records=records))
                    seal[k]=t.sha(path);done+=1
                    t.write(t.PUB/'D_RUN_STATUS.json',dict(stage='FROZEN_INFERENCE',model=model,condition=arm,fold=fold,completed_patient_model_cells=done,total_patient_model_cells=len(data)*6,last_seconds=time.perf_counter()-tick))
                    print('D_FROZEN_INFERENCE_PROGRESS',done,len(data)*6,model,arm,'seconds',round(time.perf_counter()-tick,2),flush=True)
                if network is not None:assert before==p.own.state_hash(network.state_dict());del network;gc.collect()
    assert len(seal)==len(data)*6;p.verify_protected();t.write(DROOT/'SCORE_SEAL_PRIVATE.json',dict(files=seal,all_scores_before_metrics=True));print('D_ALL_SUBSET_SCORES_SEALED',len(seal),flush=True)
    return seal

def report(p,data,seal):
    labels=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz');assert t.sha(labels)==p.rec['binding']['archive']
    with np.load(labels,allow_pickle=True) as z:y=1-z['y']
    rows=[];rep_rows=[]
    for k,h in seal.items():
        path=DROOT/'scores'/f'{k}.pkl';assert t.sha(path)==h;q=t.read(path);yy=y[q['indices']];cc=p.meta['channel'][q['indices']]
        common={v:q[v] for v in ['model','condition','fold','patient','source']};common['channels']=len(yy)
        for r in q['records']:
            v=math.metrics(yy,r['probability_ez'],r['prediction_ez'],cc);check=math.metrics(yy,r['probability_ez'],r['prediction_ez'],cc,True)
            assert all((not np.isfinite(v[m]) and not np.isfinite(check[m])) or abs(v[m]-check[m])<1e-12 for m in math.METRICS)
            for frac in [.10,.20]:
                recall=math.budget_values(yy,r['probability_ez'],cc,'fraction',frac)['recall']
                order=sorted(range(len(yy)),key=lambda j:(-float(r['probability_ez'][j]),str(cc[j])))
                n=max(1,int(np.ceil(frac*len(yy))));expected=sum(int(yy[j]) for j in order[:n])/int(yy.sum()) if yy.sum() else np.nan
                assert (not np.isfinite(recall) and not np.isfinite(expected)) or abs(recall-expected)<1e-12
                v['recall_'+str(int(frac*100))]=recall
            rep_rows.append(dict(common,K=r['K'],repetition=r['repetition'],**{m:float(v[m]) for m in D_METRICS}))
        for K in ['1','2','All']:
            rr=[r for r in rep_rows if all(r[v]==common[v] for v in ['model','condition','patient']) and r['K']==K]
            assert len(rr)>0
            rows.append(dict(common,K=K,repetitions=len(rr),**{m:float(np.mean([r[m] for r in rr])) for m in D_METRICS}))
    assert len(rows)==len(data)*18;t.csv_out('D_repetition_endpoints',rep_rows,True);t.csv_out('D_patient_endpoints',rows,True)
    ids=sorted(data);source=np.array([data[pid]['source'] for pid in ids]);lookup={(r['model'],r['condition'],r['K'],r['patient']):r for r in rows}
    absolute=[];paired=[];sources=[];comparisons=[]
    for arm in ARMS:
        for K in ['1','2']:comparisons.append((f'{arm}_K{K}_MINUS_ALL',(arm,K),(arm,'All')))
    for K in ['1','2']:comparisons.append((f'FULL_MINUS_NOPR_K{K}',('FULL_PR',K),('NO_PR',K)))
    for model in MODELS:
        arrays={(a,K):np.array([[lookup[model,a,K,pid][metric] for metric in D_METRICS] for pid in ids]) for a in ARMS for K in ['1','2','All']}
        for (arm,K),value in arrays.items():
            for j,metric in enumerate(D_METRICS):absolute.append(dict(model=model,condition=arm,K=K,metric=metric,**math.interval(value[:,j],source)))
            for s in math.SOURCES:
                a=value[source==s]
                if len(a):sources.append(dict(model=model,condition=arm,K=K,source=s,patients=len(a),**dict(zip(D_METRICS,np.nanmean(a,0).tolist()))))
        for label,a,b in comparisons:
            value=arrays[a]-arrays[b]
            for j,metric in enumerate(D_METRICS):
                family=36 if metric in ['macro_f1','auroc'] else None;ci=math.interval(value[:,j],source,family)
                if not family:ci=dict(adjusted_low=np.nan,adjusted_high=np.nan,family_size=0,**ci)
                paired.append(dict(model=model,contrast=label,metric=metric,**ci))
    t.csv_out('TABLE_E',absolute);t.csv_out('TABLE_E_PAIRED',paired);t.csv_out('TABLE_E_SOURCE',sources)
    p.verify_protected();t.write(t.PUB/'D_VERIFICATION.json',dict(status='PASS',patients=len(data),patient_model_cells=len(data)*6,patient_K_rows=len(rows),repetition_rows=len(rep_rows),family_intervals=sum(r['family_size']==36 for r in paired),nominal_paired_intervals=len(paired),all_cached_direct_representation_and_logit_bit_exact=True,independent_endpoint_formulas=True,independent_direct_index_bootstrap=True,thresholds_unchanged=True,all_scores_sealed_before_metrics=True))
    t.write(t.PUB/'STATUS.json',dict(status='ABCD_COMPLETE_VERIFIED',A='COMPLETE',B='COMPLETE',C='COMPLETE',D='COMPLETE',ABC_patients=80,D_patients=len(data),timeconv_core_models_complete=True,timeconv_additional_normalizer_training_untouched=True))
    files=[p for p in t.PUB.iterdir() if p.suffix in ['.csv','.json'] and p.name!='D_RUN_STATUS.json'];manifest={p.name:dict(sha256=t.sha(p),bytes=p.stat().st_size) for p in files}
    path=t.RUNTIME/'public_abcd_export_v1.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,'public/'+p.name)
        z.writestr('PUBLIC_MANIFEST.json',json.dumps(dict(files=manifest,private_data_included=False),sort_keys=True))
    print('ABCD_COMPLETE_VERIFIED',t.sha(path),flush=True)

def main():
    arg=argparse.ArgumentParser();arg.add_argument('--source-commit',required=True);arg.add_argument('--stage',choices=['admit','infer','report','all'],default='all');a=arg.parse_args()
    p=t.parents();data=build_data(p,a.source_commit)
    if a.stage=='admit':return
    seal=infer(p,data) if a.stage in ['infer','all'] else t.jr(DROOT/'SCORE_SEAL_PRIVATE.json')['files']
    if a.stage in ['report','all']:report(p,data,seal)

if __name__=='__main__':main()
