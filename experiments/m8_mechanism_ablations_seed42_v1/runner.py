"""Four independently trained M8 single-switch ablations and frozen late fusion."""
import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn  # Load scientific runtime before torch; never allow duplicate OpenMP.
import torch

TASK=Path(__file__).resolve().parent
SOURCE=Path(r'E:\DRE-nips\new-pipeline\7-11')
BASELINE=Path(r'D:\raw_delta_encoder_seed42_v1_exact_resume_runtime')
NOPR=Path(r'D:\d1_m8_progressive_ablation_seed42_v1_runtime')
OLD=Path(r'D:\d1_small_module_exploration_seed42_v1_outer_runtime')
M8=Path(r'D:\d1_small_module_exploration_seed42_v1_phase3_runtime')
P1=Path(r'D:\d1_small_module_exploration_seed42_v1_runtime')
DATA=Path(r'D:\dgm_net_seed42_v1_runtime')
RAW=Path(r'D:\m8_raw_only_seed42_v1_runtime')
OUT=Path(r'D:\m8_mechanism_ablations_seed42_v1_runtime')
sys.path.insert(0,str(SOURCE/'raw_delta_encoder_seed42_v1/code'))
from io_utils import sha,read,save,write,load
UP=load('mechanism_trusted_no_pr_support',Path(r'D:\m8_d1_no_pr_outer_seed42_v1\code\run.py'))
P3=UP.P3
M8Mechanism=load('m8_single_mechanism_model',TASK/'model.py').M8Mechanism
EV=load('m8_mechanism_expanded_metrics',Path(r'D:\m8_raw_backbones_pr_seed42_v1\evaluate.py'))
PRE=UP.PRE


def jr(p):return json.loads(Path(p).read_text())
def digest(o):return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def public(r):return {k:v for k,v in r.items() if k not in ['checkpoint','bank']}


def transform_pr(x,patient,pre,core):
    z=core.patient_z(x,patient)
    out=((np.where(np.isnan(z),pre['imputer_statistics'],z)-pre['mean'])/pre['scale']).astype(np.float32)
    assert np.isfinite(out).all()
    return out


def setup():
    lock=jr(TASK/'PROTOCOL_LOCK.json')
    c,core,fac,records,rec,banks,refs,upstream=UP.setup(OUT/'dependency_gate')
    for p,key in [(OLD/'FROZEN_BEFORE_OUTER_PRIVATE.json','old_reference_freeze_sha256'),
                  (OLD/'OUTER_PREDICTIONS_SEAL_PRIVATE.json','old_reference_seal_sha256'),
                  (RAW/'FROZEN_BEFORE_OUTER_PRIVATE.json','raw_only_freeze_sha256'),
                  (RAW/'OUTER_PREDICTIONS_SEAL_PRIVATE.json','raw_only_seal_sha256')]:assert sha(p)==lock[key]
    assert jr(RAW/'public/INDEPENDENT_VERIFICATION.json')['status']=='PASS'
    assert jr(RAW/'public/FINAL_STATUS.json')['status']=='M8_RAW_ONLY_FIVE_FOLD_OUTER_COMPLETE'
    assert len(rec['meta']['patient'])==7635 and len(set(rec['meta']['patient']))==80
    assert sum(len(q['records']) for q in rec['ledgers'])==256
    align=jr(BASELINE/'results/RAW_DYNAMIC_ALIGNMENT.json')
    assert align['status']=='PASS' and align['valid_channel_windows']==1471965
    assert align['registry_sha256']==sha(BASELINE/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl')
    original=jr(M8/'INPUT_FREEZE_PRIVATE.json')
    assert original['code']=={p.name:sha(p) for p in Path(r'D:\d1_small_module_exploration_seed42_v1\phase3\code').glob('*.py')}
    protected={}
    paths=[OLD/'FROZEN_BEFORE_OUTER_PRIVATE.json',OLD/'OUTER_PREDICTIONS_SEAL_PRIVATE.json',OLD/'OUTER_METRICS_PRIVATE.pkl',
        RAW/'FROZEN_BEFORE_OUTER_PRIVATE.json',RAW/'OUTER_PREDICTIONS_SEAL_PRIVATE.json',RAW/'PATIENT_METRICS_PRIVATE.pkl',
        M8/'INPUT_FREEZE_PRIVATE.json',M8/'MODEL_FREEZE_PRIVATE.json',BASELINE/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl',
        NOPR/'MODEL_FREEZE_PRIVATE.json',NOPR/'OUTER_PREDICTIONS_SEAL_PRIVATE.json',
        Path(r'D:\m8_d1_no_pr_outer_seed42_v1\code\run.py'),Path(r'D:\m8_d1_no_pr_outer_seed42_v1\code\model.py'),
        Path(r'D:\m8_raw_backbones_pr_seed42_v1\evaluate.py')]
    for r in refs:
        paths += [Path(r['checkpoint'])]
        if 'bank' in r:paths.append(Path(r['bank']))
    for r in jr(RAW/'FROZEN_BEFORE_OUTER_PRIVATE.json')['selections']:
        assert sha(r['checkpoint'])==r['checkpoint_sha256'];paths.append(Path(r['checkpoint']))
    for f in range(1,6):
        paths += [BASELINE/f'D1/fold{f}/BANK_PRIVATE.pt',P1/f'scale_fold{f}_PRIVATE.pkl',
            BASELINE/f'scores/D1_fold{f}_PRIVATE.pkl',M8/f'predictions_private/M8_MACRO_AWARE_fold{f}.pkl']
        assert sha(P1/f'scale_fold{f}_PRIVATE.pkl')==original['phase1_scales'][str(f)]
    for p in paths:protected[str(p)]=sha(p)
    binding=dict(source={p.name:sha(p) for p in TASK.iterdir() if p.suffix in ['.py','.json']},
        upstream=upstream,protected=protected,raw_alignment_sha256=sha(BASELINE/'results/RAW_DYNAMIC_ALIGNMENT.json'),
        torch=torch.__version__,numpy=np.__version__)
    OUT.mkdir(exist_ok=True);(OUT/'public').mkdir(exist_ok=True)
    ip=OUT/'INPUT_FREEZE_PRIVATE.json'
    if ip.exists():assert jr(ip)==binding,'Source/input/protocol changed: refuse silent resume'
    else:write(ip,binding)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
    write(OUT/'public/INPUT_AUDIT.json',dict(status='PASS',patients=80,channels=7635,seizures=256,
        valid_raw_channel_windows=1471965,own_fold_patient_exclusion=True,no_extraction=True,
        exact_no_PR_base_and_bank_reused=True,original_gate_scales_verified=True,torch=torch.__version__,
        source=binding['source'],raw_alignment_sha256=binding['raw_alignment_sha256']))
    return lock,c,core,fac,records,rec,banks,refs,binding


def bank(fold,arm,banks):
    original=banks[fold-1]
    bp=(NOPR/f'A1/fold{fold}/BANK_PRIVATE.pt' if arm=='A1_WO_FEATURE_PR' else BASELINE/f'D1/fold{fold}/BANK_PRIVATE.pt')
    d=torch.load(bp,map_location='cpu',weights_only=False)
    for key in ['patient','channel','y','train','val']:assert np.array_equal(d[key],original[key])
    d=dict(d,archive_indices=original['archive_indices'],roles=original['roles'])
    assert bool(d['pre'].get('patient_z',True))==(arm!='A1_WO_FEATURE_PR')
    assert set(d['patient'][d['train']])==d['roles']['fit'] and set(d['patient'][d['val']])==d['roles']['validation']
    assert set(d['patient']).isdisjoint(d['roles']['test'])
    if arm!='A1_WO_FEATURE_PR':
        # Exact original PR+FIT imputer/scaler, not a refitted transform.
        assert d['x'].shape[1]==14
    return d,bp


def make(core,fac,records,arm,fold):
    seed=42+1009*fold;oldcp=BASELINE/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt'
    original=P3.make(core,fac,records,oldcp,seed,'M8_MACRO_AWARE')
    ck=torch.load(M8/f'M8_MACRO_AWARE/fold{fold}/TRAIN/BEST_PRIVATE.pt',map_location='cpu',weights_only=False)
    assert core.state_hash(original.state_dict())==ck['binding']['initial']
    cp=NOPR/f'A1/fold{fold}/TRAIN/BEST_PRIVATE.pt' if arm=='A1_WO_FEATURE_PR' else oldcp
    model=M8Mechanism(P3.make_base(core,fac,records,cp,seed),raw_pr=arm!='A2_WO_RAW_PR',
        pre_reference=arm!='A4_WO_PRE_REFERENCE',identity_gate=arm=='A3_IDENTITY_GATE').cuda()
    keep=lambda k:not k.startswith('base.') and not (arm=='A3_IDENTITY_GATE' and k.startswith('gate.'))
    assert core.state_hash({k:v for k,v in model.state_dict().items() if keep(k)})==core.state_hash({k:v for k,v in original.state_dict().items() if keep(k)})
    return model,original


def admission(lock,core,fac,records,banks,store):
    path=OUT/'ADMISSION_PRIVATE.json'
    if path.exists():
        q=jr(path);assert q['source']=={p.name:sha(p) for p in TASK.iterdir() if p.suffix in ['.py','.json']}
        return
    audits=[]
    for arm in lock['train_arms']:
        d,_=bank(1,arm,banks);model,original=make(core,fac,records,arm,1)
        scale=torch.tensor(read(P1/'scale_fold1_PRIVATE.pkl')['scale'],device='cuda')
        tr=d['train'];ii=tr[d['patient'][tr]==sorted(d['roles']['fit'])[0]]
        raw,mask,times,delta,perms=store.get(str(d['patient'][ii[0]]),d['channel'][ii].astype(str).tolist(),'cuda')
        x=torch.as_tensor(d['x'][ii],device='cuda');y=torch.as_tensor(d['y'][ii],device='cuda',dtype=torch.float32)
        model.eval();original.eval()
        # With both math switches enabled, compare actual full-patient features to immutable original.
        enabled=M8Mechanism(P3.make_base(core,fac,records,BASELINE/'D1/fold1/TRAIN/BEST_PRIVATE.pt',1051)).cuda().eval()
        with torch.inference_mode():
            a=original.features(raw,mask,times,delta,scale)[0];b=enabled.features(raw,mask,times,delta,scale)[0]
            error=float((a-b).abs().max());assert error==0
            q=model(x,raw,mask,times,delta,scale);assert torch.equal(q[0],q[2])
            if arm=='A3_IDENTITY_GATE':
                assert torch.equal(q[4],torch.ones_like(q[4]))
                assert torch.equal(model.features(raw,mask,times,delta,scale)[0],model.features(raw,mask,times,delta*0,scale)[0])
            if arm=='A2_WO_RAW_PR':
                full=model.features(raw,mask,times,delta,scale)[0]
                sub=model.features(raw[:,:-1],mask[:,:-1],times,delta[:,:-1],scale)[0]
                assert torch.allclose(full[:-1],sub,atol=1e-5,rtol=0)
        frozen=core.state_hash(model.base.state_dict())
        fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((1-d['y'][tr]).sum()/d['y'][tr].sum()),device='cuda'))
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.001)
        norms=[]
        for step in range(3):
            model.train();opt.zero_grad(set_to_none=True);q=model(x,raw,mask,times,delta,scale)
            loss=P3.objective(q[0],y,fn,'M8_MACRO_AWARE');loss.backward()
            assert torch.isfinite(loss) and all(p.grad is None for p in model.base.parameters())
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            ns={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
            assert ns['head.2.weight']>0
            if step>=1:assert ns['head.0.weight']>0 and ns['cnn.0.weight']>0
            if arm!='A3_IDENTITY_GATE' and step>=2:assert ns['gate.network.0.weight']>0
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step();norms.append(ns)
        assert core.state_hash(model.base.state_dict())==frozen
        count=sum(p.numel() for p in model.parameters() if p.requires_grad);assert count==lock['parameter_counts'][arm]
        audits.append(dict(arm=arm,trainable=count,full_feature_parity_max_error=error,zero_initial_residual=True,
            frozen_base_unchanged=True,three_FIT_steps_discarded=True,raw_PR=model.raw_pr,raw_pre_reference=model.pre_reference,gradient_norms=norms))
        del model,original,enabled,opt,raw,mask,times,delta,perms,x,y,q,loss;gc.collect();torch.cuda.empty_cache()
    obj=dict(status='PASS',source={p.name:sha(p) for p in TASK.iterdir() if p.suffix in ['.py','.json']},arms=audits)
    write(path,obj);write(OUT/'public/ADMISSION_AUDIT.json',obj)
    print('ALL4_REAL_FIT_PARITY_AND_GRADIENT_ADMISSIONS_PASS',flush=True)


def train(lock,core,fac,records,banks,binding,store):
    selections=[];history=[]
    for arm in lock['train_arms']:
        runtime=OUT/arm;(runtime/'results').mkdir(parents=True,exist_ok=True)
        for fold in range(1,6):
            d,bp=bank(fold,arm,banks);model,original=make(core,fac,records,arm,fold);del original
            scale=torch.tensor(read(P1/f'scale_fold{fold}_PRIVATE.pkl')['scale'],device='cuda')
            initial=core.state_hash(model.state_dict())
            best,h=P3.fit(model,d,store,scale,runtime/f'fold{fold}/TRAIN',42+1009*fold,binding,core,runtime,'M8_MACRO_AWARE',fold)
            cp=runtime/f'fold{fold}/TRAIN/BEST_PRIVATE.pt'
            pp=OUT/f'validation_private/{arm}_fold{fold}.pkl'
            if pp.exists():q=read(pp);assert q['checkpoint_sha256']==sha(cp)
            else:
                pred=P3.EV.predict(model,d,d['val'],store,scale)
                q=dict(logit=pred['logit'],checkpoint_sha256=sha(cp),validation_indices=d['archive_indices'][d['val']]);save(pp,q)
            sel=core.select_threshold(d['y'][d['val']],torch.sigmoid(torch.from_numpy(q['logit'])).numpy(),d['patient'][d['val']])
            assert sel['threshold']==best['threshold']
            r=dict(arm=arm,fold=fold,checkpoint=str(cp),checkpoint_sha256=sha(cp),bank=str(bp),bank_sha256=sha(bp),
                threshold=best['threshold'],epoch=best['epoch'],epochs_run=len(h),initial_state_sha256=initial,
                validation_sha256=sha(pp),trainable=lock['parameter_counts'][arm])
            selections.append(r);history += [dict(row,source_arm=row['arm'],arm=arm) for row in h]
            pd.DataFrame([public(s) for s in selections]).to_csv(OUT/'public/MODEL_SELECTION.csv',index=False)
            pd.DataFrame(history).to_csv(OUT/'public/TRAINING_HISTORY.csv',index=False)
            write(OUT/'public/RUN_STATUS.json',dict(status='TRAINING',completed_model_folds=len(selections),total_model_folds=20,arm=arm,fold=fold))
            print('ABLATION_MODEL_FOLD_COMPLETE',arm,fold,'epochs',len(h),'best',best['epoch'],flush=True)
            del model,d;gc.collect();torch.cuda.empty_cache()
    return selections


def fusion(core,banks):
    results=[];search=[];rawfr=jr(RAW/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    for fold,d in enumerate(banks,1):
        d1=read(BASELINE/f'scores/D1_fold{fold}_PRIVATE.pkl');p1=d1['score']
        r=next(r for r in rawfr['selections'] if r['fold']==fold)
        assert sha(r['checkpoint'])==r['checkpoint_sha256']
        ck=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
        p2=torch.sigmoid(torch.from_numpy(ck['validation_logit'])).numpy()
        assert p1.shape==p2.shape==(len(d['val']),)
        assert core.select_threshold(d['y'][d['val']],p2,d['patient'][d['val']])['threshold']==r['threshold_nez']
        candidates=[]
        for weight in np.round(np.arange(0,1.00001,.05),2):
            # Float64 convex probability fusion, explicitly frozen before test.
            score=(1-float(weight))*p1.astype(float)+float(weight)*p2.astype(float)
            sel=core.select_threshold(d['y'][d['val']],score,d['patient'][d['val']])
            candidates.append((sel['macro_f1'],sel['ez_f1'],-float(weight),dict(arm='A6_LATE_FUSION',fold=fold,raw_weight=float(weight),**sel)))
            search.append(dict(fold=fold,raw_weight=float(weight),**sel))
        chosen=max(candidates,key=lambda x:x[:3])[3]
        results.append(dict(chosen,D1_checkpoint_sha256=d1['checkpoint_sha256'],raw_checkpoint_sha256=r['checkpoint_sha256'],
            validation_D1_sha256=sha(BASELINE/f'scores/D1_fold{fold}_PRIVATE.pkl'),validation_raw_checkpoint_sha256=sha(r['checkpoint'])))
    pd.DataFrame(search).to_csv(OUT/'public/FUSION_VALIDATION_SEARCH.csv',index=False)
    pd.DataFrame(results).to_csv(OUT/'public/FUSION_SELECTION.csv',index=False)
    return results


def freeze(binding,selections,fusions,refs):
    for r in selections:assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['bank'])==r['bank_sha256']
    obj=dict(binding_sha256=digest(binding),selections=selections,fusions=fusions,
        references=refs,raw_references=jr(RAW/'FROZEN_BEFORE_OUTER_PRIVATE.json')['selections'],all25_new_selections_before_outer=True)
    p=OUT/'FROZEN_BEFORE_OUTER_PRIVATE.json'
    if p.exists():assert jr(p)==obj
    else:write(p,obj)
    write(OUT/'public/OUTER_FREEZE_AUDIT.json',dict(status='PASS',models=20,fusions=5,
        historical_reference_models=15,freeze_sha256=sha(p),all_selected_before_new_outer=True,own_fold_test_excluded=True))
    return obj,sha(p)


def reference_score(arm,fold,meta):
    ix=np.flatnonzero(np.isin(meta['patient'],list(UP.SUP.roles(meta,fold)['test'])))
    if arm in ['D1_PR','M8_FULL']:
        oldarm='D1' if arm=='D1_PR' else 'M8_MACRO_AWARE'
        path=OLD/f'scores_private/{oldarm}_fold{fold}.pkl';seal=jr(OLD/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')
        assert sha(path)==seal['files'][f'{oldarm}_{fold}'];q=read(path)
        assert q['freeze_sha256']==seal['freeze_sha256'] and np.array_equal(ix,q['indices'])
        return ix,torch.sigmoid(torch.from_numpy(q['logit'])).numpy()
    seal=jr(RAW/'OUTER_PREDICTIONS_SEAL_PRIVATE.json');logit=np.full(len(meta['patient']),np.nan,np.float32);seen=[]
    for key,h in seal['files'].items():
        if not key.startswith(f'{fold}_'):continue
        token=key.split('_',1)[1];p=RAW/f'fold{fold}/outer_private/{token}.pkl';assert sha(p)==h;q=read(p)
        assert q['freeze_sha256']==seal['freeze_sha256']
        logit[q['indices']]=q['logit_nez'];seen+=q['indices'].tolist()
    assert sorted(seen)==ix.tolist() and np.isfinite(logit[ix]).all()
    return ix,torch.sigmoid(torch.from_numpy(logit[ix])).numpy()


def outer(lock,core,fac,records,rec,banks,fr,fh,store):
    meta=rec['meta'];cells={};files={};audits=[]
    for r in fr['selections']:
        arm,fold=r['arm'],r['fold'];ix=np.flatnonzero(np.isin(meta['patient'],list(UP.SUP.roles(meta,fold)['test'])))
        path=OUT/f'outer_scores_private/{arm}_fold{fold}.pkl'
        if path.exists():q=read(path)
        else:
            model,original=make(core,fac,records,arm,fold);del original
            ck=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);model.eval()
            bd=torch.load(r['bank'],map_location='cpu',weights_only=False)
            if arm=='A1_WO_FEATURE_PR':z=PRE.transform(rec['R14'][ix],bd['pre'])
            else:z=transform_pr(rec['R14'][ix],meta['patient'][ix],bd['pre'],core)
            score=np.empty(len(ix),np.float32);base=np.empty(len(ix),np.float32)
            scale=torch.tensor(read(P1/f'scale_fold{fold}_PRIVATE.pkl')['scale'],device='cuda')
            before=core.state_hash(model.state_dict())
            for pid in sorted(set(meta['patient'][ix])):
                jj=np.flatnonzero(meta['patient'][ix]==pid);ii=ix[jj];token=hashlib.sha256(str(pid).encode()).hexdigest()[:24]
                pp=OUT/f'patient_scores_private/{arm}/fold{fold}/{token}.pkl'
                if pp.exists():pq=read(pp)
                else:
                    raw,mask,times,delta,perms=store.get(str(pid),meta['channel'][ii].astype(str).tolist(),'cuda')
                    with torch.inference_mode():out=model(torch.as_tensor(z[jj],device='cuda'),raw,mask,times,delta,scale)
                    pq=dict(logit=out[0].cpu().numpy(),base_logit=out[2].cpu().numpy(),indices=ii,
                        freeze_sha256=fh,checkpoint_sha256=r['checkpoint_sha256']);save(pp,pq)
                    del raw,mask,times,delta,perms,out
                assert pq['freeze_sha256']==fh and pq['checkpoint_sha256']==r['checkpoint_sha256'] and np.array_equal(pq['indices'],ii)
                score[jj]=torch.sigmoid(torch.from_numpy(pq['logit'])).numpy();base[jj]=pq['base_logit']
            if arm=='A1_WO_FEATURE_PR':
                p=NOPR/f'outer_scores_private/A1_fold{fold}.pkl';oldseal=jr(NOPR/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')
                assert sha(p)==oldseal['files'][f'A1_{fold}'];old=read(p)
                assert np.array_equal(ix,old['indices']);err=float(np.max(np.abs(base-old['logit'])))
            else:
                _,p=reference_score('D1_PR',fold,meta);err=float(np.max(np.abs(torch.sigmoid(torch.from_numpy(base)).numpy()-p)))
            assert err<=1e-6 and core.state_hash(model.state_dict())==before
            audits.append(dict(arm=arm,fold=fold,frozen_base_replay_max_error=err))
            q=dict(indices=ix,score_nez=score,threshold=r['threshold'],freeze_sha256=fh,checkpoint_sha256=r['checkpoint_sha256'])
            save(path,q);del model,ck,bd;gc.collect();torch.cuda.empty_cache()
        assert q['freeze_sha256']==fh and q['checkpoint_sha256']==r['checkpoint_sha256'] and np.array_equal(q['indices'],ix) and 'y' not in q
        cells[arm,fold]=q;files[f'{arm}_{fold}']=sha(path);print('OUTER_CELL_SEALED',arm,fold,flush=True)
    rawfr=jr(RAW/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    for fold in range(1,6):
        for arm in lock['reference_arms']+['A6_LATE_FUSION']:
            if arm=='A6_LATE_FUSION':
                r=next(r for r in fr['fusions'] if r['fold']==fold);ix,p1=reference_score('D1_PR',fold,meta);ix2,p2=reference_score('M8_RAW_ONLY',fold,meta)
                assert np.array_equal(ix,ix2);score=(1-r['raw_weight'])*p1.astype(float)+r['raw_weight']*p2.astype(float);threshold=r['threshold']
            else:
                ix,score=reference_score(arm,fold,meta)
                if arm=='M8_RAW_ONLY':threshold=next(r for r in rawfr['selections'] if r['fold']==fold)['threshold_nez']
                else:threshold=next(r for r in fr['references'] if r['arm']==('D1_PR' if arm=='D1_PR' else 'M8_PR') and r['fold']==fold)['threshold']
            path=OUT/f'outer_scores_private/{arm}_fold{fold}.pkl'
            q=dict(indices=ix,score_nez=score,threshold=threshold,freeze_sha256=fh)
            if path.exists():old=read(path);assert np.array_equal(old['score_nez'],score) and old['freeze_sha256']==fh
            else:save(path,q)
            cells[arm,fold]=q;files[f'{arm}_{fold}']=sha(path)
    assert len(cells)==40
    p=OUT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json';obj=dict(freeze_sha256=fh,files=files,all40_scores_sealed_before_full_labels=True)
    if p.exists():assert jr(p)==obj
    else:write(p,obj)
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz');assert sha(archive)==rec['binding']['archive']
    with np.load(archive,allow_pickle=True) as z:y=1-z['y']
    assert len(y)==7635 and y.sum()==1743
    frames=[]
    for (arm,fold),q in cells.items():
        ix=q['indices'];frames.append(pd.DataFrame(dict(model=arm,version='seed42',fold=fold,patient=meta['patient'][ix],
            channel=meta['channel'][ix],center=meta['center'][ix],y=y[ix],probability=1-q['score_nez'].astype(float),prediction=q['score_nez']<q['threshold'])))
    frame=pd.concat(frames,ignore_index=True);save(OUT/'PREDICTIONS_PRIVATE.pkl',frame)
    write(OUT/'public/BASE_REPLAY_AUDIT.json',dict(status='PASS',folds=audits,tolerance=1e-6))
    return frame


def main():
    lock,c,core,fac,records,rec,banks,refs,binding=setup()
    if (OUT/'public/FINAL_STATUS.json').exists():
        assert jr(OUT/'public/FINAL_STATUS.json')['status']=='M8_MECHANISM_5FOLD_OUTER_COMPLETE'
        print('ALREADY_COMPLETE_NO_NEW_TRAINING_OR_INFERENCE',flush=True);return
    store=P3.Inputs(BASELINE,DATA,OUT);store.arm='M1_SCALED_DGM'
    admission(lock,core,fac,records,banks,store)
    selections=train(lock,core,fac,records,banks,binding,store);fusions=fusion(core,banks)
    fr,fh=freeze(binding,selections,fusions,refs)
    frame=outer(lock,core,fac,records,rec,banks,fr,fh,store)
    report=load('mechanism_report',TASK/'reporting.py');report.finish(sys.modules[__name__],frame,fr,banks,binding,core)
    print('M8_MECHANISM_5FOLD_OUTER_COMPLETE',flush=True)


if __name__=='__main__':main()
