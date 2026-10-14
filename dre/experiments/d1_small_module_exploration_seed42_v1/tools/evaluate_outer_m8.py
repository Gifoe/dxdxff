"""Explicitly authorized frozen M8/D1 Outer; no training or new extraction."""
import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/'code'))
from explore import context,make_base,original_modules,Inputs,EV,sha,read,save,write
from best_recipe import build

def transform(x,patient,pre,core):
    z=core.patient_z(x,patient)
    assert np.isfinite(pre['imputer_statistics']).all() and (pre['scale']>0).all()
    q=((np.where(np.isnan(z),pre['imputer_statistics'],z)-pre['mean'])/pre['scale']).astype(np.float32)
    assert np.isfinite(q).all()
    return q

def roles(meta,fold):
    q={r:set(meta['split_patient'][(meta['split_fold']==fold)&(meta['split_role']==r)]) for r in ['fit','validation','test']}
    assert all(q[a].isdisjoint(q[b]) for a,b in [('fit','validation'),('fit','test'),('validation','test')])
    return q

def prepare(a):
    lock=json.loads((ROOT/'OUTER_PROTOCOL_AMENDMENT.json').read_text())
    c,core,fac,trainer,records,rec,banks,hist,deps=context(a.source,a.baseline)
    old=json.loads((a.runtime/'MODEL_FREEZE_PRIVATE.json').read_text())
    source=json.loads((a.runtime/'INPUT_FREEZE_PRIVATE.json').read_text())
    assert old['binding']==source
    assert source['code']=={p.name:sha(p) for p in (ROOT/'phase3/code').glob('*.py')}
    assert source['protocol']==sha(ROOT/'phase3/PROTOCOL_LOCK.json')
    assert source['prior_code']=={p.name:sha(p) for p in (ROOT/'phase2/code').glob('*.py')}
    p1=json.loads((a.phase1/'INPUT_FREEZE_PRIVATE.json').read_text())
    assert p1['code']=={p.name:sha(p) for p in (ROOT/'code').glob('*.py')}
    dfr=json.loads((a.baseline/'STAGE_B_MODEL_FREEZE_PRIVATE.json').read_text())
    sels=[];visited=[];meta=rec['meta']
    assert len(meta['patient'])==7635 and len(set(meta['patient']))==80
    assert len(set(zip(meta['patient'],meta['channel'])))==7635
    assert rec['binding']['archive']=='1bdcc479a2bd148bdc94614c8253bea67e4008bd98bc6dad9e8030df827db9bd'
    for fold in range(1,6):
        rr=roles(meta,fold);ix=np.flatnonzero(np.isin(meta['patient'],list(rr['test'])))
        assert len(rr['test'])==lock['patients_by_fold'][fold-1] and len(ix)==lock['channels_by_fold'][fold-1]
        visited.extend(ix.tolist());bp=a.baseline/f'D1/fold{fold}/BANK_PRIVATE.pt'
        d=torch.load(bp,map_location='cpu',weights_only=False)
        assert set(d['patient'][d['train']])==rr['fit'] and set(d['patient'][d['val']])==rr['validation']
        assert set(d['patient']).isdisjoint(rr['test']) and set(d['pre']['fit_patient_ids'])==rr['fit']
        assert np.array_equal(transform(rec['R14'][banks[fold-1]['archive_indices']],d['patient'],d['pre'],core),d['x'])
        sp=a.phase1/f'scale_fold{fold}_PRIVATE.pkl';sq=read(sp)
        assert set(sq['fit_patients'])==rr['fit'] and sha(sp)==source['phase1_scales'][str(fold)]
        assert np.isfinite(sq['scale']).all() and (sq['scale']>0).all()
        for arm in lock['models']:
            ref=next(q for q in (dfr if arm=='D1' else old)['selections'] if q['arm']==arm and q['fold']==fold)
            cp=(a.baseline if arm=='D1' else a.runtime)/f'{arm}/fold{fold}/TRAIN/BEST_PRIVATE.pt'
            assert sha(cp)==ref['checkpoint_sha256']
            if arm=='D1':assert sha(bp)==ref['bank_sha256']
            else:assert ref['D1_checkpoint_sha256']==sha(a.baseline/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt')
            ck=torch.load(cp,map_location='cpu',weights_only=False)
            assert ck['epoch']==ref['epoch'] and ck['threshold']==ref['threshold']
            done=json.loads((cp.parent/'COMPLETE_PRIVATE.json').read_text())
            assert done['best_sha256']==sha(cp) and done['last_sha256']==sha(cp.parent/'LAST_PRIVATE.pt')
            sels.append(dict(arm=arm,fold=fold,checkpoint=str(cp),checkpoint_sha256=sha(cp),bank=str(bp),bank_sha256=sha(bp),scale_sha256=sha(sp),epoch=ck['epoch'],threshold=ck['threshold']))
    assert sorted(visited)==list(range(7635))
    reg=read(a.data/'CONDITIONING_REGISTRY_PRIVATE.pkl');assert set(reg['files'])==set(meta['patient'])
    assert json.loads((a.data/'results/WINDOW_DELTA_PARITY.json').read_text())['maximum_actual_7D_reaggregation_error']==0
    code={str(p.relative_to(ROOT)):sha(p) for folder in ['code','phase2/code','phase3/code','tools'] for p in (ROOT/folder).glob('*.py')}
    binding=dict(amendment=sha(ROOT/'OUTER_PROTOCOL_AMENDMENT.json'),code=code,dependencies=deps,reconstruction=sha(a.baseline/'RECONSTRUCTION_PRIVATE.pkl'),raw_registry=sha(a.baseline/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl'),conditioning_registry=sha(a.data/'CONDITIONING_REGISTRY_PRIVATE.pkl'),window_gate=sha(a.data/'results/WINDOW_DELTA_PARITY.json'),phase3_input=sha(a.runtime/'INPUT_FREEZE_PRIVATE.json'),phase3_models=sha(a.runtime/'MODEL_FREEZE_PRIVATE.json'),original_input_hashes=p1['input_hashes'],torch=torch.__version__,numpy=np.__version__)
    a.out.mkdir(parents=True,exist_ok=True);pub=a.out/'public';pub.mkdir(exist_ok=True)
    return lock,c,core,fac,records,rec,banks,sels,binding,pub

def freeze(a):
    lock,c,core,fac,records,rec,banks,sels,binding,pub=prepare(a)
    q=dict(status='PASS',binding=binding,selections=sels,test_labels_not_loaded=True)
    fp=a.out/'FROZEN_BEFORE_OUTER_PRIVATE.json'
    if fp.exists():assert json.loads(fp.read_text())==q
    else:write(fp,q)
    write(pub/'OUTER_FREEZE_AUDIT.json',dict(status='PASS',freeze_sha256=sha(fp),binding=binding,selections=[{k:v for k,v in r.items() if k not in ['checkpoint','bank']} for r in sels],patients=80,channels=7635,own_fold_test_excluded_from_fit_val=True,labels_not_loaded=True,no_training=True,no_refit=True))
    print('M8_D1_TEN_CHECKPOINTS_FROZEN_BEFORE_OUTER',flush=True)

def cached(path,freeze_hash,cp_hash,ix):
    q=read(path)
    assert q['freeze_sha256']==freeze_hash and q['checkpoint_sha256']==cp_hash and np.array_equal(q['indices'],ix) and 'y' not in q
    assert np.isfinite(q['logit']).all()
    return q

def evaluate(a):
    lock,c,core,fac,records,rec,banks,sels,binding,pub=prepare(a)
    fp=a.out/'FROZEN_BEFORE_OUTER_PRIVATE.json';fr=json.loads(fp.read_text());fh=sha(fp)
    assert fr['binding']==binding and fr['selections']==sels
    if (pub/'OUTER_STATUS.json').exists():
        assert json.loads((pub/'OUTER_STATUS.json').read_text())['freeze_sha256']==fh
        print('OUTER_ALREADY_COMPLETE_NO_NEW_INFERENCE',flush=True);return
    store=Inputs(a.baseline,a.data,a.runtime);store.arm='M1_SCALED_DGM';cells={};files={};identity=[];meta=rec['meta']
    for r in sels:
        arm,fold=r['arm'],r['fold'];ix=np.flatnonzero(np.isin(meta['patient'],list(roles(meta,fold)['test'])))
        pp=a.out/f'scores_private/{arm}_fold{fold}.pkl'
        if pp.exists():q=cached(pp,fh,r['checkpoint_sha256'],ix)
        else:
            bd=torch.load(r['bank'],map_location='cpu',weights_only=False)
            bank=dict(patient=meta['patient'][ix],channel=meta['channel'][ix],x=transform(rec['R14'][ix],meta['patient'][ix],bd['pre'],core))
            base=make_base(core,fac,records,a.baseline/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt',42+1009*fold)
            model=(base if arm=='D1' else build(base)).cuda().eval()
            ck=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True)
            scale=torch.tensor(read(a.phase1/f'scale_fold{fold}_PRIVATE.pkl')['scale'],device='cuda')
            outputs={k:np.empty(len(ix),np.float32) for k in (['logit'] if arm=='D1' else ['logit','base_logit','residual'])}
            for n,pid in enumerate(sorted(set(bank['patient'])),1):
                jj=np.flatnonzero(bank['patient']==pid);gix=ix[jj]
                token=hashlib.sha256(str(pid).encode()).hexdigest()[:24]
                pf=a.out/f'patients_private/{arm}_fold{fold}/{token}.pkl'
                if pf.exists():v=cached(pf,fh,r['checkpoint_sha256'],gix)
                else:
                    if arm=='D1':
                        with torch.inference_mode():ll=model(torch.as_tensor(bank['x'][jj],device='cuda')).cpu().numpy()
                        v=dict(logit=ll)
                    else:
                        v=EV.predict(model,bank,jj,store,scale)
                        v={k:v[k] for k in outputs}
                    v.update(indices=gix,freeze_sha256=fh,checkpoint_sha256=r['checkpoint_sha256']);save(pf,v)
                for k in outputs:outputs[k][jj]=v[k]
                store.cache.clear();store.raw.cache.clear();gc.collect()
                print('OUTER_PATIENT_SEALED',arm,fold,n,'of',len(set(bank['patient'])),flush=True)
            assert core.state_hash(model.state_dict())==core.state_hash(ck['model'])
            q=dict(**outputs,indices=ix,freeze_sha256=fh,checkpoint_sha256=r['checkpoint_sha256']);save(pp,q);del model,bank,bd,ck;torch.cuda.empty_cache()
        cells[arm,fold]=q;files[f'{arm}_{fold}']=sha(pp)
    for fold in range(1,6):
        b=cells['D1',fold]['logit'];m=cells['M8_MACRO_AWARE',fold]
        drift=float(np.max(np.abs(b-m['base_logit'])));assert drift<=1e-6
        assert np.max(np.abs(m['logit']-m['base_logit']-m['residual']))<=1e-6
        assert np.max(np.abs(m['residual']))<=1+1e-6
        identity.append(dict(fold=fold,embedded_D1_logit_max_difference=drift,residual_cap_verified=True))
    seal=a.out/'OUTER_PREDICTIONS_SEAL_PRIVATE.json';obj=dict(freeze_sha256=fh,files=files,all10_cells_sealed_before_test_labels=True)
    if seal.exists():assert json.loads(seal.read_text())==obj
    else:write(seal,obj)
    # First full TEST label read in this new evaluation, after all scores are sealed.
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz');assert sha(archive)==rec['binding']['archive']
    with np.load(archive,allow_pickle=True) as z:y=z['y']
    assert len(y)==7635 and (y==0).sum()==1743 and (y==1).sum()==5892
    metrics,boot=original_modules();rows=[];pooled=[]
    for r in sels:
        arm,fold=r['arm'],r['fold'];q=cells[arm,fold];ix=q['indices'];s=torch.sigmoid(torch.from_numpy(q['logit'])).numpy()
        d=dict(patient=meta['patient'][ix],channel=meta['channel'][ix],center=meta['center'][ix],y=y[ix],val=np.arange(len(ix)))
        rows.extend(metrics.patient_rows(c,d,s,r['threshold'],arm,fold))
    df=pd.DataFrame(rows)
    for arm,g in df.groupby('arm'):assert len(g)==g.patient.nunique()==80 and g.channels.sum()==7635
    save(a.out/'OUTER_METRICS_PRIVATE.pkl',rows)
    df.groupby('arm')[metrics.METRICS].mean().reset_index().to_csv(pub/'OUTER_SUMMARY.csv',index=False)
    for key in ['fold','center']:
        pd.DataFrame([dict(arm=arm,**{key:value},patients=len(g),channels=int(g.channels.sum()),**g[metrics.METRICS].mean().to_dict()) for (arm,value),g in df.groupby(['arm',key])]).to_csv(pub/f'OUTER_BY_{key.upper()}.csv',index=False)
    paired=boot.compare(c,df[df.arm=='M8_MACRO_AWARE'],df[df.arm=='D1'],metrics.METRICS)
    pd.DataFrame(paired).to_csv(pub/'OUTER_PAIRED_BOOTSTRAP.csv',index=False)
    intervals=[]
    for arm,g in df.groupby('arm'):
        b=c.cluster_bootstrap(g.patient.to_numpy(),g[metrics.METRICS].to_numpy(),10000)
        for j,m in enumerate(metrics.METRICS):intervals.append(dict(arm=arm,metric=m,estimate=float(g[m].mean()),ci_low=float(np.nanpercentile(b[:,j],2.5)),ci_high=float(np.nanpercentile(b[:,j],97.5)),draws=10000,seed=42))
        score=np.empty(7635);pred=np.zeros(7635,bool)
        for r in [v for v in sels if v['arm']==arm]:
            q=cells[arm,r['fold']];ix=q['indices'];s=torch.sigmoid(torch.from_numpy(q['logit'])).numpy();score[ix]=s;pred[ix]=s>=r['threshold']
        order=np.lexsort((meta['channel'],meta['patient']));v=c.metric(y[order],score[order],0,pred=pred[order]);v.pop('ez_mrr');v.pop('top1_is_ez');pooled.append(dict(arm=arm,unit='secondary_pooled_channels',**v))
    pd.DataFrame(intervals).to_csv(pub/'OUTER_ABSOLUTE_INTERVALS.csv',index=False)
    pd.DataFrame(pooled).to_csv(pub/'OUTER_POOLED_CHANNELS.csv',index=False)
    write(pub/'OUTER_D1_IDENTITY.json',dict(status='PASS',folds=identity))
    write(pub/'OUTER_STATUS.json',dict(status='ONE_FROZEN_OUTER_PASS_COMPLETE',scientific_status=lock['scientific_status'],patients_per_arm=80,channels_per_arm=7635,freeze_sha256=fh,seal_sha256=sha(seal),all10_cells_sealed_before_test_labels=True,no_training=True,no_threshold_or_scaler_refit=True,no_new_extraction=True))
    print('M8_D1_OUTER_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','evaluate']);p.add_argument('--runtime',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_phase3_runtime'));p.add_argument('--phase1',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_runtime'));p.add_argument('--data',type=Path,default=Path(r'D:\dgm_net_seed42_v1_runtime'));p.add_argument('--baseline',type=Path,default=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime'));p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));p.add_argument('--out',type=Path,default=Path(r'D:\d1_small_module_exploration_seed42_v1_outer_runtime'));a=p.parse_args();(freeze if a.phase=='freeze' else evaluate)(a)
