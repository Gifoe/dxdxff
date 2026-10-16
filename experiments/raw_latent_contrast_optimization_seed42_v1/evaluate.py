"""Validation replay,pre-test freeze,one resumed score seal,then full labels/statistics."""
import argparse
import gc
import pandas as pd
from scipy.special import expit
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,average_precision_score
from common import *
from train import stage2_setup,build,predict,select,ARMS
from engine import cached
from metrics import patient_rows


def bootstrap(frame):
    models=sorted(frame.model.unique());base=ARMS[0]
    order=sorted(frame.patient.unique());assert len(order)==80
    matrices={}
    for model in models:
        g=frame[frame.model==model].set_index('patient').loc[order]
        assert len(g)==80 and np.isfinite(g[METRICS].to_numpy()).all()
        matrices[model]=g[METRICS].to_numpy(float)
    draws=np.random.default_rng(42).integers(0,80,size=(10000,80));absolute=[];paired=[]
    for model in models:
        v=matrices[model];boot=v[draws].mean(1)
        for j,m in enumerate(METRICS):
            lo,hi=np.quantile(boot[:,j],[.025,.975])
            absolute.append(dict(model=model,metric=m,mean=float(v[:,j].mean()),low=float(lo),high=float(hi),patients=80,draws=10000))
    contrasts=[(m,base) for m in ARMS[1:]+['SELECTED_POLICY']]
    contrasts += [(ARMS[2],ARMS[1]),(ARMS[3],ARMS[2])]
    for left,right in contrasts:
        d=matrices[left]-matrices[right];boot=d[draws].mean(1)
        for j,m in enumerate(METRICS):
            lo,hi=np.quantile(boot[:,j],[.025,.975])
            paired.append(dict(left=left,right=right,metric=m,delta=float(d[:,j].mean()),low=float(lo),high=float(hi),patients=80,draws=10000,interval='unadjusted_95percent_paired_patient_percentile'))
    return pd.DataFrame(absolute),pd.DataFrame(paired)


def verify_rows(prediction,patients):
    maximum=0.
    for (model,fold,pid),q in prediction.groupby(['model','fold','patient']):
        row=patients[(patients.model==model)&(patients.fold==fold)&(patients.patient==pid)].iloc[0]
        y=q.y.to_numpy(int);p=q.probability.to_numpy(float);pred=q.prediction.to_numpy(bool)
        independent=dict(accuracy=accuracy_score(y,pred),f1=f1_score(y,pred,pos_label=1,zero_division=0),
            nez_f1=f1_score(y,pred,pos_label=0,zero_division=0),macro_f1=f1_score(y,pred,average='macro',labels=[0,1],zero_division=0),
            auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p))
        # Independent Python sort instead of the production NumPy ranking path.
        channels=q.channel.astype(str).tolist();sort=sorted(range(len(y)),key=lambda j:(-p[j],channels[j]));ranks=[k+1 for k,j in enumerate(sort) if y[j]]
        independent['mrr']=1/ranks[0];independent['hit_at_1']=float(y[sort[0]])
        independent['ez_fraction_bias']=float(pred.mean()-y.mean())
        for name,value in independent.items():
            error=abs(float(row[name])-float(value));assert error<=1e-12,(model,name,error);maximum=max(maximum,error)
    return maximum


def results(frozen,meta):
    seal=jr(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json');assert seal['freeze_sha256']==sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    for rel,h in seal['files'].items():assert sha(ROOT/rel)==h
    rec=read(BASE/'RECONSTRUCTION_PRIVATE.pkl');archive_hash=rec['binding']['archive'];del rec
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert sha(archive)==archive_hash
    with np.load(archive,allow_pickle=True) as z:y=1-z['y']
    assert y.shape==(7635,) and int(y.sum())==1743
    _,banks=identity_and_banks(labels=True)
    for d in banks:assert np.array_equal(y[d['archive_indices']],1-d['y'])
    patient=[];prediction=[]
    for row in frozen['selections']:
        f=row['fold'];arm=row['arm'];path=ROOT/'outer_cells_private'/f'{arm}_fold{f}.pkl';q=read(path)
        assert q['freeze_sha256']==seal['freeze_sha256'] and q['checkpoint_sha256']==row['checkpoint_sha256']
        ix=q['archive_indices'];p=expit(-q['logit_nez'].astype(float));pid=meta['patient'][ix];ch=meta['channel'][ix];center=meta['center'][ix]
        alternatives=[(arm,row['threshold']),
            (arm+'__ALT_THRESHOLD',row['ez_threshold'] if arm.startswith('B0') else row['macro_threshold'])]
        if frozen['deployment'][str(f)]==arm:alternatives.append(('SELECTED_POLICY',row['threshold']))
        for name,theta in alternatives:
            patient.extend([dict(model=name,fold=f,**r) for r in patient_rows(y[ix],p,pid,ch,center,theta)])
            prediction.append(pd.DataFrame(dict(model=name,fold=f,patient=pid,channel=ch,center=center,
                y=y[ix],probability=p,prediction=p>=theta,threshold=theta,archive_index=ix)))
    frame=pd.DataFrame(patient);pred=pd.concat(prediction,ignore_index=True)
    assert len(frame)==720 and len(frame.model.unique())==9
    for model,g in frame.groupby('model'):
        assert g.patient.nunique()==len(g)==80 and int(g.n.sum())==7635
        assert pred[pred.model==model].archive_index.nunique()==7635
    error=verify_rows(pred,frame);save(ROOT/'PATIENT_METRICS_PRIVATE.pkl',frame);save(ROOT/'PREDICTIONS_PRIVATE.pkl',pred)
    main=frame.groupby('model')[METRICS].mean().reset_index();main.to_csv(PUB/'MAIN_RESULTS.csv',index=False)
    frame.groupby(['model','fold'])[METRICS].mean().reset_index().to_csv(PUB/'FOLD_RESULTS.csv',index=False)
    frame.groupby('model')[METRICS].agg(['mean','std']).to_csv(PUB/'PATIENT_DISPERSION.csv')
    fold=frame.groupby(['model','fold'])[METRICS].mean()
    fold.groupby('model').agg(['mean','std','min','max']).to_csv(PUB/'FOLD_STABILITY.csv')
    source=[];coverage=[];pooled=[]
    for (model,center),g in frame.groupby(['model','center']):
        n=g.patient.nunique();source.append(dict(model=model,center=center,patients=n,suppressed=n<5,
            **{m:float(g[m].mean()) if n>=5 else np.nan for m in METRICS}))
    for model,g in frame.groupby('model'):
        for metric in METRICS:coverage.append(dict(model=model,metric=metric,finite_patients=int(g[metric].notna().sum()),patients=80))
        q=pred[pred.model==model];yy=q.y.to_numpy();pp=q.probability.to_numpy();hh=q.prediction.to_numpy()
        pooled.append(dict(model=model,channels=7635,accuracy=float(accuracy_score(yy,hh)),f1=float(f1_score(yy,hh,pos_label=1)),
            nez_f1=float(f1_score(yy,hh,pos_label=0)),macro_f1=float(f1_score(yy,hh,average='macro')),
            auroc=float(roc_auc_score(yy,pp)),auprc=float(average_precision_score(yy,pp)),ez_fraction_bias=float(hh.mean()-yy.mean())))
    pd.DataFrame(source).to_csv(PUB/'SOURCE_RESULTS.csv',index=False)
    pd.DataFrame(coverage).to_csv(PUB/'METRIC_COVERAGE.csv',index=False)
    pd.DataFrame(pooled).to_csv(PUB/'POOLED_RESULTS.csv',index=False)
    absolute,paired=bootstrap(frame);absolute.to_csv(PUB/'ABSOLUTE_INTERVALS.csv',index=False);paired.to_csv(PUB/'PAIRED_INTERVALS.csv',index=False)
    primary=main[main.model=='SELECTED_POLICY'].iloc[0];ci=absolute[(absolute.model=='SELECTED_POLICY')&(absolute.metric=='f1')].iloc[0]
    write(PUB/'TARGET_STATUS.json',dict(primary_model='SELECTED_POLICY',metric='patient_average_EZ_positive_F1',
        target_greater_than=.65,observed=float(primary.f1),patient_bootstrap_95percent=[float(ci.low),float(ci.high)],
        point_target_met=bool(primary.f1>.65),interval_lower_above_target=bool(ci.low>.65),
        selected_using_Development_not_Outer=True,Outer_is_exploratory_repeated_test=True,
        no_independent_clinical_confirmation=True))
    write(PUB/'OUTER_STATUS.json',dict(status='ONE_FROZEN_FIVEFOLD_OUTER_COMPLETE_VERIFIED',
        freeze_sha256=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json'),score_seal_sha256=sha(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json'),
        actual_trained_cells=20,scored_patient_cells=320,reported_models=9,primary_models=5,metric_patient_rows=720,
        patients_per_model=80,channels_per_model=7635,bootstrap_draws=10000,independent_metric_max_error=error,
        scientific_status='EXPLORATORY_REPEATED_TEST',no_post_Outer_tuning=True))
    print('RAW_OUTER_COMPLETE_VERIFIED',float(primary.f1),flush=True)


def run():
    binding,recipes,banks=stage2_setup();meta,_=identity_and_banks(labels=False)
    if (PUB/'OUTER_STATUS.json').exists() and jr(PUB/'OUTER_STATUS.json')['status']=='ONE_FROZEN_FIVEFOLD_OUTER_COMPLETE_VERIFIED':
        status=jr(PUB/'OUTER_STATUS.json')
        assert status['freeze_sha256']==sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
        assert status['score_seal_sha256']==sha(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')
        for rel,h in jr(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')['files'].items():assert sha(ROOT/rel)==h
        print('RAW_OUTER_ALREADY_COMPLETE_VERIFIED_NO_INFERENCE',flush=True);return
    status=jr(PUB/'TRAIN_STATUS.json');assert status['status']=='ALL20_RAW_MODELS_AND_DEPLOYMENT_FROZEN'
    assert status['model_freeze_sha256']==sha(ROOT/'MODEL_FREEZE_PRIVATE.json')
    models=jr(ROOT/'MODEL_FREEZE_PRIVATE.json');assert models['binding']==binding and len(models['selections'])==20
    verification=jr(PUB/'STAGE1_INDEPENDENT_VERIFICATION.json');assert verification['status']=='PASS'
    assert verification['verified_recipe_freeze_sha256']==sha(ROOT/'POOL_RECIPE_FREEZE_PRIVATE.json')
    store=RawStore();checks=[]
    for row in models['selections']:
        assert sha(row['checkpoint'])==row['checkpoint_sha256'] and sha(row['validation'])==row['validation_sha256']
        cp=torch.load(row['checkpoint'],map_location='cpu',weights_only=False);d=banks[row['fold']-1]
        m,c=build(row['arm'],row['fold'],recipes);m.load_state_dict(cp['model'])
        logit=predict(m,d,d['val'],store);saved=read(row['validation'])
        assert np.array_equal(saved['archive_indices'],d['archive_indices'][d['val']])
        error=float(np.max(np.abs(logit-saved['logit_nez'])));assert error<=1e-6
        selected=select(d,logit,c['threshold_kind']);assert selected==row['selection']
        assert select(d,logit,'ez')['threshold']==row['ez_threshold'] and select(d,logit,'macro')['threshold']==row['macro_threshold']
        checks.append(dict(arm=row['arm'],fold=row['fold'],maximum_logit_error=error))
        del m,cp;gc.collect();torch.cuda.empty_cache()
    write(PUB/'VALIDATION_REPLAY_AUDIT.json',dict(status='PASS',cells=20,maximum_logit_error=max(r['maximum_logit_error'] for r in checks),checks=checks))
    frozen=dict(**models,evaluation_source={n:sha(HERE/n) for n in ['evaluate.py','metrics.py','model.py','common.py','engine.py','train.py']},
        stage1_verification_sha256=sha(PUB/'STAGE1_INDEPENDENT_VERIFICATION.json'),
        new_full_Outer_labels_not_read=True)
    path=ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json';freeze(path,frozen);fh=sha(path)
    files={}
    for row in frozen['selections']:
        fold=row['fold'];d=banks[fold-1];ix=np.flatnonzero(np.isin(meta['patient'],sorted(d['roles']['test'])))
        assert len(set(meta['patient'][ix]))==[16,16,17,15,16][fold-1]
        assert len(ix)==[1631,1498,1746,1425,1335][fold-1]
        cp=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
        m,c=build(row['arm'],fold,recipes);m.load_state_dict(cp['model']);m.eval();before=state_hash(m.state_dict());score=np.empty(len(ix),np.float32)
        for pid in sorted(d['roles']['test']):
            local=np.flatnonzero(meta['patient'][ix]==pid);ai=ix[local]
            target=ROOT/'outer_patients_private'/f"{row['arm']}_fold{fold}"/(private_key(pid)+'.pkl')
            pb=dict(freeze_sha256=fh,checkpoint_sha256=row['checkpoint_sha256'],archive_indices=list(map(int,ai)),waveform_sha256=store.plans[str(pid)]['sha256'])
            if target.exists():q=read(target);assert q['binding']==pb
            else:
                with torch.inference_mode():
                    raw,mask,times=store.get(pid,meta['channel'][ai]);ll,z=cached(m,raw,mask,times)
                q=dict(binding=pb,logit_nez=ll.cpu().numpy(),channel_embedding=z.cpu().numpy(),labels_not_read=True);save(target,q)
            assert np.isfinite(q['logit_nez']).all() and np.isfinite(q['channel_embedding']).all()
            score[local]=q['logit_nez'];files[str(target.relative_to(ROOT))]=sha(target)
        assert state_hash(m.state_dict())==before==state_hash(cp['model'])
        cell=ROOT/'outer_cells_private'/f"{row['arm']}_fold{fold}.pkl"
        if cell.exists():
            old=read(cell)
            assert old['freeze_sha256']==fh and old['checkpoint_sha256']==row['checkpoint_sha256']
            assert np.array_equal(old['archive_indices'],ix) and np.array_equal(old['logit_nez'],score)
        else:save(cell,dict(freeze_sha256=fh,checkpoint_sha256=row['checkpoint_sha256'],archive_indices=ix,logit_nez=score))
        files[str(cell.relative_to(ROOT))]=sha(cell)
        write(PUB/'OUTER_STATUS.json',dict(status='SEALING_SCORES_BEFORE_FULL_LABELS',completed_model_cells=len([p for p in files if p.startswith('outer_cells_private')]),total_model_cells=20,freeze_sha256=fh))
        print('RAW_OUTER_FROZEN_CELL',row['arm'],fold,flush=True)
        del m,cp;gc.collect();torch.cuda.empty_cache()
    assert len([p for p in files if p.startswith('outer_patients_private')])==320
    freeze(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json',dict(freeze_sha256=fh,files=files,all320_patient_scores_before_full_labels=True))
    results(frozen,meta)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['all']);p.parse_args();run()
