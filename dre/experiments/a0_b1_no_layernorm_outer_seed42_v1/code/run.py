"""Explicit B1-only outer authorization, separate from failed PMA gate.

Unchanged parent runner/dependencies. Freeze before signal-feature test access;
seal every prediction before full y access. No training or selection calls.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch


def context():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','evaluate','verify'])
    p.add_argument('--runtime',type=Path,default=Path(r'C:\a0_b1_no_layernorm_outer_seed42_v1_runtime'))
    p.add_argument('--pma-runtime',type=Path,default=Path(r'C:\a0_pma_seed42_v1_runtime'))
    p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'))
    a=p.parse_args();sys.path.insert(0,str(a.source/'a0_pma_seed42_v1/code'))
    import runtime
    pa=SimpleNamespace(runtime=a.pma_runtime,source=a.source,previous=Path(r'C:\d2_ictal_ablation_seed42_v1_runtime'),
        original=Path(r'C:\a0_source_posterior_decoder_seed42_runtime'),
        features=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz'),
        protocol=a.source/'a0_pma_seed42_v1/PROTOCOL_LOCK.json')
    o=runtime.setup(pa);lock,ref,c,core,original,trainer,models,banks,binding=o
    pre=json.loads((pa.runtime/'public/PRETRAIN_AUDIT.json').read_text())
    assert pre['status']=='PASS' and pre['binding']==binding
    eng=json.loads((pa.runtime/'public/ENGINEERING_TESTS.json').read_text());assert eng['status']=='PASS' and eng['binding']==binding
    ver=json.loads((pa.runtime/'public/INDEPENDENT_VERIFICATION.json').read_text());assert ver['status']=='PASS'
    amendment_path=Path(__file__).parents[1]/'PROTOCOL_AMENDMENT.json';amend=json.loads(amendment_path.read_text())
    dp=pa.runtime/'DEVELOPMENT_FREEZE.json';assert c.sha(dp)==amend['development_freeze_sha256']==ver['development_freeze_sha256']
    frozen=json.loads(dp.read_text());assert frozen['binding']==binding
    oldseal=pa.previous/'OUTER_PREDICTIONS_SEAL.json';assert c.sha(oldseal)==amend['B0_outer_seal_sha256']
    oldfiles=json.loads(oldseal.read_text())['files']
    paths={'amendment':amendment_path,'development_freeze':dp,'parent_pretrain':pa.runtime/'public/PRETRAIN_AUDIT.json',
        'parent_tests':pa.runtime/'public/ENGINEERING_TESTS.json','parent_verification':pa.runtime/'public/INDEPENDENT_VERIFICATION.json',
        'parent_gate':pa.runtime/'public/VALIDATION_GATE.json','B0_outer_seal':oldseal,
        'transform':a.source/'d2_ictal_outer_seed42_v1/code/evaluate_outer.py'}
    assert not json.loads(paths['parent_gate'].read_text())['pass'],'Do not rewrite parent PMA gate'
    for fold in range(1,6):
        rec=next(r for r in frozen['selections'] if r['arm']=='B1' and r['fold']==fold)
        cp=runtime.checkpoint(pa,'B1',fold);bp=pa.runtime/f'fold{fold}/BANK_PRIVATE.pt';vp=bp.parent/'B1_VAL_PRIVATE.pt'
        assert c.sha(cp)==rec['checkpoint_sha256']==amend['checkpoint_sha256'][fold-1]
        assert c.sha(bp)==rec['bank_sha256']==pre['folds'][fold-1]['bank_sha256']
        assert rec['epoch']==amend['epochs'][fold-1] and rec['threshold']==amend['nez_thresholds'][fold-1]
        old=pa.previous/f'no_source/fold{fold}/OUTER_PREDICTIONS_PRIVATE.pt'
        assert c.sha(old)==oldfiles[f'no_source_{fold}']
        paths.update({f'B1_ck{fold}':cp,f'bank{fold}':bp,f'B1_val{fold}':vp,f'B0_outer{fold}':old})
    for path in sorted(Path(__file__).parent.glob('*.py')):paths['code_'+path.name]=path
    paths['contract_tests']=Path(__file__).parents[1]/'tests/contract.py'
    sys.path.insert(0,str(a.source/'d2_ictal_outer_seed42_v1/code'))
    from evaluate_outer import frozen_transform
    a.runtime.mkdir(parents=True,exist_ok=True);(a.runtime/'public').mkdir(exist_ok=True)
    return a,pa,o,amend,paths,frozen,frozen_transform,runtime


def hashes(c,paths):return {k:c.sha(v) for k,v in paths.items()}


def prepare(ctx):
    a,pa,o,amend,paths,frozen,transform,rt=ctx;lock,ref,c,core,original,trainer,models,banks,binding=o
    target=a.runtime/'FROZEN_BEFORE_OUTER.json';assert not target.exists()
    before=hashes(c,paths);rows=[];seen=[];channels=[]
    with np.load(pa.features,allow_pickle=True) as f:
        meta={k:f[k] for k in ['patient','channel','center','split_fold','split_patient','split_role']}
    for fold in range(1,6):
        d=torch.load(paths[f'bank{fold}'],map_location='cpu',weights_only=False)
        ck=torch.load(paths[f'B1_ck{fold}'],map_location='cpu',weights_only=False)
        completion=json.loads(paths[f'B1_ck{fold}'].parent.joinpath('COMPLETE_PRIVATE.json').read_text())
        assert completion['best_sha256']==c.sha(paths[f'B1_ck{fold}']) and ck['binding']['seed']==42+1009*fold
        assert ck['epoch']==amend['epochs'][fold-1] and ck['threshold']==amend['nez_thresholds'][fold-1]
        assert ck['binding']['max_epochs']==30 and ck['binding']['penalty']==0
        x,pre=core.prepare(banks[fold-1]['raw_x'],d['patient'],d['train']);assert np.array_equal(x,d['x'])
        for key in ['mean','scale','var','imputer_statistics']:assert np.array_equal(pre[key],d['pre'][key])
        val=torch.load(paths[f'B1_val{fold}'],map_location='cpu',weights_only=False)
        assert val['binding']==amend['development_freeze_sha256'] and val['checkpoint_sha256']==c.sha(paths[f'B1_ck{fold}'])
        assert np.array_equal(val['indices'],d['val']) and val['threshold']==ck['threshold']
        m=models.constructor(original.SourceA0,'B1',d['stats_normalizer'])().eval();m.load_state_dict(ck['model'],strict=True)
        assert isinstance(m.network[0],torch.nn.Identity) and sum(p.numel() for p in m.parameters())==8641
        with torch.inference_mode():ll=m(torch.from_numpy(x[d['val']]));ss=torch.sigmoid(ll).numpy()
        ld=float(np.max(np.abs(ll.numpy()-val['logits'])));sd=float(np.max(np.abs(ss-val['scores'])))
        assert ld<=amend['logit_replay_absolute_tolerance'] and sd<=amend['probability_replay_absolute_tolerance']
        assert np.array_equal(ss>=ck['threshold'],val['scores']>=ck['threshold'])
        mask=meta['split_fold']==fold
        ids={role:set(meta['split_patient'][mask&(meta['split_role']==role)]) for role in ['fit','validation','test']}
        assert all(ids[u].isdisjoint(ids[v]) for u,v in [('fit','validation'),('fit','test'),('validation','test')])
        assert set(d['patient'][d['train']])==ids['fit'] and set(d['patient'][d['val']])==ids['validation']
        assert set(d['pre']['fit_patient_ids'])==ids['fit'] and set(d['patient']).isdisjoint(ids['test'])
        ix=np.flatnonzero(np.isin(meta['patient'],list(ids['test'])))
        assert len(ids['test'])==amend['test_patients_by_fold'][fold-1] and len(ix)==amend['test_channels_by_fold'][fold-1]
        seen.extend(ids['test']);channels.extend(ix.tolist())
        rows.append(dict(fold=fold,epoch=ck['epoch'],threshold=ck['threshold'],parameters=8641,
            test_patients=len(ids['test']),test_channels=len(ix),FIT_only_preprocessor_bitwise=True,
            validation_logit_max_drift=ld,validation_probability_max_drift=sd,validation_decision_parity=True,
            checkpoint_sha256=c.sha(paths[f'B1_ck{fold}']),bank_sha256=c.sha(paths[f'bank{fold}']),
            FIT_TEST_overlap=0,full_feature_x_y_loaded=False))
    assert len(seen)==len(set(seen))==80 and sorted(channels)==list(range(7635))
    assert before==hashes(c,paths)
    f=dict(status='PASS',hashes=before,parent_binding=binding,rows=rows,new_inference_models=['B1'],
        reused_B0_only=True,scientific_status=amend['scientific_status'],device='cpu',torch=torch.__version__,
        numpy=np.__version__,full_TEST_x_y_loaded=False)
    c.write_json(target,f);c.write_json(a.runtime/'public/FROZEN_DEPENDENCY_AUDIT.json',f)
    print('B1_ALL5_CHECKPOINT_SCALER_THRESHOLDS_FROZEN_BEFORE_OUTER',flush=True)


def evaluate(ctx):
    a,pa,o,amend,paths,frozen,transform,rt=ctx;lock,ref,c,core,original,trainer,models,banks,binding=o
    pub=a.runtime/'public';assert not (pub/'OUTER_STATUS.json').exists(),'Completed outer cannot rerun'
    fp=a.runtime/'FROZEN_BEFORE_OUTER.json';fr=json.loads(fp.read_text());assert fr['hashes']==hashes(c,paths)
    fs=c.sha(fp)
    with np.load(pa.features,allow_pickle=True) as f:raw={k:f[k] for k in f.files if k!='y'}
    assert raw['x'].shape==(7635,88) and len(set(zip(raw['patient'],raw['channel'])))==7635
    files={};cells=[];seen=[]
    for fold in range(1,6):
        ids=raw['split_patient'][(raw['split_fold']==fold)&(raw['split_role']=='test')]
        ix=np.flatnonzero(np.isin(raw['patient'],ids));path=a.runtime/f'fold{fold}_B1_OUTER_PRIVATE.pt'
        ck=torch.load(paths[f'B1_ck{fold}'],map_location='cpu',weights_only=False)
        d=torch.load(paths[f'bank{fold}'],map_location='cpu',weights_only=False)
        if path.exists():
            sh=json.loads(path.with_suffix('.sha256.json').read_text());assert sh==dict(binding=fs,sha256=c.sha(path))
            cell=torch.load(path,map_location='cpu',weights_only=False)
            assert cell['binding']==fs and np.array_equal(ix,cell['indices']) and cell['threshold']==ck['threshold']
            assert cell['checkpoint_sha256']==c.sha(paths[f'B1_ck{fold}']) and 'y' not in cell
        else:
            x=transform(raw['x'][ix],raw['patient'][ix],d['pre'],core)
            m=models.constructor(original.SourceA0,'B1',d['stats_normalizer'])().eval();m.load_state_dict(ck['model'],strict=True)
            with torch.inference_mode():ll=m(torch.from_numpy(x));ss=torch.sigmoid(ll).numpy()
            assert np.isfinite(ss).all()
            cell=dict(binding=fs,fold=fold,indices=ix,logits=ll.numpy(),scores=ss,
                threshold=ck['threshold'],checkpoint_sha256=c.sha(paths[f'B1_ck{fold}']))
            core.torch_write(path,cell);c.write_json(path.with_suffix('.sha256.json'),dict(binding=fs,sha256=c.sha(path)))
        seen.extend(ix.tolist());files[str(fold)]=c.sha(path);cells.append(cell)
        print('B1_OUTER_PREDICTIONS_SEALED',fold,len(ix),flush=True)
    assert sorted(seen)==list(range(7635))
    seal=dict(binding=fs,files=files,all_B1_scores_before_full_y=True)
    sp=a.runtime/'OUTER_PREDICTION_SEAL.json'
    if sp.exists():assert json.loads(sp.read_text())==seal
    else:c.write_json(sp,seal)
    # No full-cohort label materialization until all B1 scores are sealed.
    with np.load(pa.features,allow_pickle=True) as f:y=f['y']
    assert len(y)==7635 and int((y==0).sum())==1743 and int((y==1).sum())==5892
    d=dict(patient=raw['patient'],channel=raw['channel'],center=raw['center'],y=y);rows=[];pooled=[]
    all_scores={arm:np.empty(7635) for arm in ['B0','B1']};all_decisions={arm:np.empty(7635,dtype=bool) for arm in ['B0','B1']}
    for fold,cell in enumerate(cells,1):
        ix=cell['indices'];old=torch.load(paths[f'B0_outer{fold}'],map_location='cpu',weights_only=False)
        assert np.array_equal(ix,old['indices'])
        oldsel=next(r for r in frozen['selections'] if r['arm']=='B0' and r['fold']==fold)
        assert old['threshold']==oldsel['threshold']
        bank=torch.load(paths[f'bank{fold}'],map_location='cpu',weights_only=False)
        di=np.flatnonzero(np.isin(raw['patient'],bank['patient']));assert np.array_equal(y[di],bank['y'])
        for arm,pred in [('B0',old),('B1',cell)]:
            rows.extend(rt.patient_rows(c,d,ix,pred['scores'],pred['threshold'],arm,fold))
            all_scores[arm][ix]=pred['scores'];all_decisions[arm][ix]=pred['scores']>=pred['threshold']
    df=pd.DataFrame(rows);assert len(df)==160 and df.patient.nunique()==80
    assert all(len(g)==80 and g.patient.nunique()==80 and g.channels.sum()==7635 for _,g in df.groupby('arm'))
    core.torch_write(a.runtime/'PATIENT_METRICS_PRIVATE.pt',rows)
    summary=df.groupby('arm')[c.METRICS].mean().reset_index();summary['unit']='patient_equal_outer80';summary['parameters']=summary.arm.map({'B0':8817,'B1':8641})
    summary.to_csv(pub/'OUTER_SUMMARY.csv',index=False)
    absolute=[]
    for arm,g in df.groupby('arm'):
        boot=c.cluster_bootstrap(g.patient,g[c.METRICS].to_numpy(),10000)
        for j,key in enumerate(c.METRICS):absolute.append(dict(arm=arm,metric=key,estimate=g[key].mean(),
            ci_low=np.nanpercentile(boot[:,j],2.5),ci_high=np.nanpercentile(boot[:,j],97.5),estimable_patients=int(g[key].notna().sum()),patients=80,draws=10000,seed=42))
    pd.DataFrame(absolute).to_csv(pub/'OUTER_PATIENT_EQUAL_METRICS.csv',index=False)
    indexed={arm:g.set_index('patient').sort_index() for arm,g in df.groupby('arm')};assert indexed['B0'].index.equals(indexed['B1'].index)
    delta=indexed['B1'][c.METRICS].to_numpy()-indexed['B0'][c.METRICS].to_numpy();boot=c.cluster_bootstrap(indexed['B1'].index,delta,10000)
    pd.DataFrame([dict(comparison='B1 minus B0',metric=k,delta=np.nanmean(delta[:,j]),ci_low=np.nanpercentile(boot[:,j],2.5),
        ci_high=np.nanpercentile(boot[:,j],97.5),draws=10000,seed=42,cluster='80_unique_outer_patients',thresholds_reoptimized=False) for j,k in enumerate(c.METRICS)]).to_csv(pub/'OUTER_PAIRED_BOOTSTRAP.csv',index=False)
    for key,name in [('fold','OUTER_BY_FOLD.csv'),('center','OUTER_BY_CENTER.csv')]:
        pd.DataFrame([dict(arm=arm,**{key:v},patients=len(g),channels=int(g.channels.sum()),**g[c.METRICS].mean().to_dict()) for (arm,v),g in df.groupby(['arm',key])]).to_csv(pub/name,index=False)
    canonical=np.lexsort((raw['channel'].astype(str),raw['patient'].astype(str)))
    for arm in ['B0','B1']:
        metrics=c.metric(y[canonical],all_scores[arm][canonical],0.,pred=all_decisions[arm][canonical]);metrics.pop('ez_mrr');metrics.pop('top1_is_ez')
        pooled.append(dict(arm=arm,unit='secondary_pooled7635_channels_fold_specific_thresholds',**metrics))
    pd.DataFrame(pooled).to_csv(pub/'OUTER_POOLED_CHANNEL_METRICS.csv',index=False)
    assert hashes(c,paths)==fr['hashes']
    c.write_json(pub/'IMMUTABILITY_AND_COVERAGE_AUDIT.json',dict(status='PASS',all_dependencies_unchanged=True,
        full_B1_predictions_sealed_before_y=True,freeze_sha256=fs,prediction_seal_sha256=c.sha(sp),
        patients_unique_once=80,channels_unique_once=7635,EZ=1743,NEZ=5892,B0_scores_reused=True,
        B1_parameters=8641,frozen_B1_thresholds=amend['nez_thresholds'],training=False,threshold_reselection=False,
        B2_B3_outer_accessed=False,PMA_gate_unchanged=True,FIT_VAL_label_parity=True))
    c.write_json(pub/'OUTER_STATUS.json',dict(status='ONE_USER_AUTHORIZED_FROZEN_B1_OUTER_COMPLETE',
        scientific_status=amend['scientific_status'],patients=80,channels=7635,new_outer_model='B1',
        B0_reused=True,independent_verification_pending=True,freeze_sha256=fs))
    print('ONE_FROZEN_B1_OUTER_COMPLETE',flush=True)


def verify(ctx):
    a,pa,o,amend,paths,frozen,transform,rt=ctx;lock,ref,c,core,original,trainer,models,banks,binding=o
    from verify import independent
    pub=a.runtime/'public';fr=json.loads((a.runtime/'FROZEN_BEFORE_OUTER.json').read_text());assert hashes(c,paths)==fr['hashes']
    fs=c.sha(a.runtime/'FROZEN_BEFORE_OUTER.json');seal=json.loads((a.runtime/'OUTER_PREDICTION_SEAL.json').read_text());assert seal['binding']==fs and seal['all_B1_scores_before_full_y']
    with np.load(pa.features,allow_pickle=True) as f:raw={k:f[k] for k in ['patient','channel','center','y']}
    saved=pd.DataFrame(torch.load(a.runtime/'PATIENT_METRICS_PRIVATE.pt',map_location='cpu',weights_only=False)).set_index(['arm','patient']).sort_index()
    rows=[];maxdrift=0.
    for fold in range(1,6):
        path=a.runtime/f'fold{fold}_B1_OUTER_PRIVATE.pt';assert c.sha(path)==seal['files'][str(fold)]
        cell=torch.load(path,map_location='cpu',weights_only=False);assert cell['binding']==fs
        ck=torch.load(paths[f'B1_ck{fold}'],map_location='cpu',weights_only=False)
        assert cell['threshold']==ck['threshold']==amend['nez_thresholds'][fold-1] and cell['checkpoint_sha256']==c.sha(paths[f'B1_ck{fold}'])
        old=torch.load(paths[f'B0_outer{fold}'],map_location='cpu',weights_only=False);assert np.array_equal(cell['indices'],old['indices'])
        for arm,pred in [('B1',cell),('B0',old)]:
            ix=pred['indices']
            for pat in sorted(set(raw['patient'][ix])):
                k=np.flatnonzero(raw['patient'][ix]==pat);k=k[np.argsort(raw['channel'][ix][k].astype(str),kind='stable')];si=ix[k]
                m=independent(raw['y'][si],pred['scores'][k],pred['threshold']);r=saved.loc[(arm,pat)]
                for key,v in m.items():
                    assert np.isclose(v,r[key],atol=1e-12,rtol=0,equal_nan=True),(arm,key)
                    if np.isfinite(v):maxdrift=max(maxdrift,abs(v-r[key]))
                rows.append(dict(arm=arm,patient=pat,fold=fold,center=str(raw['center'][si[0]]),**m))
    df=pd.DataFrame(rows);assert len(df)==160 and df.patient.nunique()==80
    for filename,keys in [('OUTER_SUMMARY.csv',['arm']),('OUTER_BY_FOLD.csv',['arm','fold']),('OUTER_BY_CENTER.csv',['arm','center'])]:
        actual=pd.read_csv(pub/filename).set_index(keys).sort_index();expected=df.groupby(keys)[c.METRICS].mean().sort_index()
        assert actual.index.equals(expected.index) and np.allclose(actual[c.METRICS],expected,atol=1e-12,rtol=0,equal_nan=True)
    draws=np.random.default_rng(42).integers(0,80,size=(10000,80));indexed={arm:g.set_index('patient').sort_index() for arm,g in df.groupby('arm')}
    absolute=pd.read_csv(pub/'OUTER_PATIENT_EQUAL_METRICS.csv')
    for arm,g in indexed.items():
        v=g[c.METRICS].to_numpy();boot=np.nanmean(v[draws],axis=1)
        for j,key in enumerate(c.METRICS):
            r=absolute[(absolute.arm==arm)&(absolute.metric==key)].iloc[0]
            assert np.allclose([np.nanmean(v[:,j]),*np.nanpercentile(boot[:,j],[2.5,97.5])],[r.estimate,r.ci_low,r.ci_high],atol=1e-12,rtol=0,equal_nan=True)
    delta=indexed['B1'][c.METRICS].to_numpy()-indexed['B0'][c.METRICS].to_numpy();boot=np.nanmean(delta[draws],axis=1)
    paired=pd.read_csv(pub/'OUTER_PAIRED_BOOTSTRAP.csv')
    for j,key in enumerate(c.METRICS):
        r=paired[paired.metric==key].iloc[0]
        assert np.allclose([np.nanmean(delta[:,j]),*np.nanpercentile(boot[:,j],[2.5,97.5])],[r.delta,r.ci_low,r.ci_high],atol=1e-12,rtol=0,equal_nan=True)
    assert hashes(c,paths)==fr['hashes']
    c.write_json(pub/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',patient_cells=160,max_absolute_metric_drift=maxdrift,
        absolute_intervals_checked=22,paired_intervals_checked=11,draws=10000,seed=42,backend='independent sklearn',
        no_new_inference=True,no_threshold_refitting=True,freeze_sha256=fs,numpy=np.__version__,sklearn=__import__('sklearn').__version__))
    print('B1_OUTER_INDEPENDENT_VERIFICATION_PASS',maxdrift,flush=True)


if __name__=='__main__':
    ctx=context();{'prepare':prepare,'evaluate':evaluate,'verify':verify}[ctx[0].phase](ctx)
