"""Registered frozen fullFIT encoder/readout stage; all three policies reported.

Actions are separate, hash-resumable steps. No full Outer labels are loaded until
all15 score cells are sealed. Waveforms/identities/scores stay on the server.
"""
import argparse
import gc
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score
import common as c
import train_internal as trainer
import execute_cpu_bridge_v2 as bridge
from head import Head, RECIPES

FINAL=c.ROOT/'cpu_bridge_execution_v2'/'final_stage_v1'
PUBLIC=FINAL/'public'
PROCEDURES=['NATIVE','REFIT','SELECTED']
pure=c.load('macro_final_pure_raw',c.REFERENCE/'pure_model.py')
poolpath=c.io.HERE/'model.py'
assert c.sha(poolpath)=='2a837032d9f2f2bc7a7b42c6d71c87863920a37ec5e35313c7ed1a6b54b20605'
pool=c.load('macro_final_immutable_raw_pool',poolpath)


def reference_rows():
    fr=c.read(c.HISTORY/'MODEL_FREEZE_PRIVATE.pkl')
    status=c.jr(c.HISTORY/'public/NEURAL_STATUS.json')
    assert status['status']=='ALL80_LOGICAL_MODELS_FROZEN'
    assert status['model_freeze_sha256']==c.sha(c.HISTORY/'MODEL_FREEZE_PRIVATE.pkl')
    rows=sorted([r for r in fr['selections'] if r['family']=='Concat' and r['recipe']=='RAW_ONLY'],key=lambda r:r['fold'])
    assert [r['fold'] for r in rows]==list(range(1,6))
    for r in rows:
        assert r['groups']==[]
        assert c.sha(r['checkpoint'])==r['checkpoint_sha256'] and c.sha(r['validation'])==r['validation_sha256']
    return rows


def setup():
    c.ROOT=bridge.ROOT;c.PUB=bridge.PUB
    stage1=bridge.setup()
    terminal=c.jr(c.PUB/'INTERNAL_TERMINAL.json');verify=c.jr(c.PUB/'INTERNAL_VERIFICATION.json')
    assert terminal['status']=='COMPLETE_VERIFIED_MACRO_HEAD_INTERNAL' and verify['status']=='PASS'
    choice_path=c.ROOT/'OWN_FIT_HEAD_SELECTION_PRIVATE.json'
    assert verify['selection_sha256']==c.sha(choice_path)
    choice=c.jr(choice_path);assert choice['input']==stage1
    assert choice['seal']==c.sha(c.ROOT/'INTERNAL_SCORE_SEAL_PRIVATE.json')
    choices={int(r['fold']):r['chosen'] for r in choice['choices']};assert len(choices)==5
    FINAL.mkdir(exist_ok=True);PUBLIC.mkdir(exist_ok=True)
    reference=reference_rows()
    binding=dict(internal_binding_sha256=c.digest(stage1),ownFIT_selection_sha256=c.sha(choice_path),
                 source={str(p.name):c.sha(p) for p in [c.HERE/'final_stage.py',c.REFERENCE/'pure_model.py',poolpath]},
                 reference_freeze_sha256=c.sha(c.HISTORY/'MODEL_FREEZE_PRIVATE.pkl'),
                 reference_rows=[{k:r[k] for k in ['fold','checkpoint_sha256','validation_sha256','threshold','epoch']} for r in reference],
                 raw_registry=c.io.EXPECTED_REG,reconstruction=c.io.EXPECTED_REC,
                 original_Outer_seal=c.sha(c.HISTORY/'OUTER_PREDICTIONS_SEAL_PRIVATE.json'))
    c.freeze(FINAL/'INPUT_FREEZE_PRIVATE.json',binding)
    trainer.ROOT=FINAL;trainer.PUB=PUBLIC;trainer.predict=bridge.predict
    return binding,choices,reference


def part_path(fold,pid):
    return FINAL/'features_private'/f'fold{fold}'/(c.io.private_key(pid)+'.pkl')


def embeddings(model,raw,mask,times,recipes):
    with torch.inference_mode():
        x=model.normalized_windows(raw,mask,times)
        dense=raw.new_zeros((*mask.shape,24))
        dense[mask]=torch.cat([model.cnn(q[:,None]).flatten(1) for q in x.split(4096)])
        z={'1111':model.aggregate(dense,mask,times)}
        for recipe in recipes:
            if recipe!='1111':z[recipe]=pool.pool_embeddings(dense,mask,times,tuple(map(int,recipe)))
        native=model.classify(z['1111']).cpu().numpy()
        return {k:v.cpu().numpy() for k,v in z.items()},native


def prepare():
    binding,choices,reference=setup();_,banks=c.io.identity_and_banks(labels=False);store=c.io.RawStore()
    parts={};reports=[];started=time.perf_counter()
    for r in reference:
        fold=r['fold'];d=banks[fold-1]
        assert set(d['patient'])==d['roles']['fit']|d['roles']['validation']
        assert set(d['patient']).isdisjoint(d['roles']['test'])
        cp=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
        assert cp['epoch']==r['epoch'] and cp['threshold']==r['threshold']
        model=pure.PureRaw().cuda().eval();model.load_reference(cp['model']);model.requires_grad_(False)
        recipes=sorted({'1111',RECIPES[choices[fold]]})
        logit=np.empty(len(d['patient']),np.float32);windows=0
        for pid in sorted(set(d['patient'])):
            ix=np.flatnonzero(d['patient']==pid);p=part_path(fold,pid)
            pb=dict(input=binding,fold=fold,archive_indices=d['archive_indices'][ix].tolist(),
                    checkpoint_sha256=r['checkpoint_sha256'],waveform_sha256=store.plans[str(pid)]['sha256'],recipes=recipes)
            if p.exists():q=c.read(p);assert q['binding']==pb
            else:
                raw,mask,times=store.get(pid,d['channel'][ix]);z,native=embeddings(model,raw,mask,times,recipes)
                assert all(v.shape==(len(ix),96) and np.isfinite(v).all() for v in z.values())
                q=dict(binding=pb,features=z,logit_nez=native,valid_windows=int(mask.sum()),
                       complete_patient_channels=True,no_manual_features_or_labels_model_inputs=True)
                c.save(p,q)
            parts[str(p.relative_to(FINAL))]=c.sha(p);logit[ix]=q['logit_nez'];windows+=q['valid_windows']
            c.write(PUBLIC/'PREPARE_STATUS.json',dict(status='PREPARING_FIT_DEVELOPMENT_ONLY',complete_patient_cells=len(parts),
                                                    elapsed_seconds=time.perf_counter()-started,new_Outer_passes=0))
            print('MACRO_FINAL_RAW_PATIENT',len(parts),'fold',fold,flush=True)
        saved=c.read(r['validation']);va=d['val']
        assert np.array_equal(saved['archive_indices'],d['archive_indices'][va])
        error=float(np.max(np.abs(logit[va]-saved['logit'])));assert error<=1e-6
        pn=c.probability(logit[va]);old=c.probability(saved['logit'])
        assert np.array_equal(pn>=r['threshold'],old>=r['threshold'])
        reports.append(dict(fold=fold,FIT_patients=len(d['roles']['fit']),Development_patients=len(d['roles']['validation']),
                            patient_cells=len(set(d['patient'])),channels=len(d['patient']),valid_windows=windows,
                            Native_validation_logit_error=error,Native_threshold_decisions_unchanged=True,
                            frozen_encoder_sha256=r['checkpoint_sha256'],selected_arm=choices[fold]))
        del model,cp;gc.collect();torch.cuda.empty_cache()
    c.freeze(FINAL/'FEATURES_SEAL_PRIVATE.json',dict(input=binding,parts=parts))
    c.write(PUBLIC/'FINAL_PREPARATION.json',dict(status='PASS',reports=reports,patient_cells=len(parts),
          feature_seal_sha256=c.sha(FINAL/'FEATURES_SEAL_PRIVATE.json'),fullOuter_labels_read=False,new_Outer_passes=0))


def final_context(d,r,choices,binding):
    fold=r['fold'];seal=c.jr(FINAL/'FEATURES_SEAL_PRIVATE.json');assert seal['input']==binding
    z={k:np.empty((len(d['patient']),96),np.float32) for k in sorted({'1111',RECIPES[choices[fold]]})}
    logit=np.empty(len(d['patient']),np.float32)
    for pid in sorted(set(d['patient'])):
        ix=np.flatnonzero(d['patient']==pid);p=part_path(fold,pid)
        assert c.sha(p)==seal['parts'][str(p.relative_to(FINAL))]
        q=c.read(p);assert q['binding']['input']==binding and q['binding']['archive_indices']==d['archive_indices'][ix].tolist()
        assert q['binding']['checkpoint_sha256']==r['checkpoint_sha256'] and q['complete_patient_channels']
        for key in z:z[key][ix]=q['features'][key]
        logit[ix]=q['logit_nez']
    ctx=dict(fold=fold,screen='final',data=d,roles=dict(train=d['train'],choice=np.array([],int),cal=d['val']),
             stored=dict(raw_logit_nez=logit,checkpoint_sha256=r['checkpoint_sha256']),
             representation_sha256=c.sha(FINAL/'FEATURES_SEAL_PRIVATE.json'))
    return ctx,z


def train():
    binding,choices,reference=setup();assert c.jr(PUBLIC/'FINAL_PREPARATION.json')['status']=='PASS'
    _,banks=c.io.identity_and_banks(labels=True);rows=[];public=[];actual_fits=0
    for r in reference:
        fold=r['fold'];d=banks[fold-1];ctx,z=final_context(d,r,choices,binding)
        cp=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
        results={}
        for arm in dict.fromkeys(['NATIVE','REFIT',choices[fold]]):
            results[arm]=trainer.fit(ctx,arm,z[RECIPES[arm]],cp,binding);actual_fits+=int(arm!='NATIVE')
        assert results['NATIVE']['threshold']==r['threshold']
        for procedure,arm in [('NATIVE','NATIVE'),('REFIT','REFIT'),('SELECTED',choices[fold])]:
            q=results[arm];path=FINAL/'heads_private'/f'fold{fold}_screenfinal'/arm/'BEST_PRIVATE.pt'
            rows.append(dict(fold=fold,procedure=procedure,arm=arm,recipe=RECIPES[arm],model=str(path),
                             model_sha256=c.sha(path),threshold=q['threshold'],epoch=q['epoch'],
                             reference_checkpoint=r['checkpoint'],reference_sha256=r['checkpoint_sha256']))
            public.append(dict(fold=fold,procedure=procedure,arm=arm,selected_epoch=q['epoch'],threshold_NEZ=q['threshold'],
                               fitted_epochs=q['epochs'],trainable_parameters=0 if arm=='NATIVE' else q['parameters'],
                               reference_encoder_selected_epoch=r['epoch']))
    assert len(rows)==15 and actual_fits<=10
    c.save(FINAL/'MODEL_FREEZE_PRIVATE.pkl',dict(input=binding,features=c.sha(FINAL/'FEATURES_SEAL_PRIVATE.json'),rows=rows))
    pd.DataFrame(public).to_csv(PUBLIC/'FINAL_SELECTION.csv',index=False)
    c.write(PUBLIC/'FINAL_TRAINING_STATUS.json',dict(status='ALL15_FINAL_LOGICAL_MODELS_FROZEN',logical_models=15,
          actual_trainable_head_fits=actual_fits,encoder_updates=0,new_Outer_passes=0,
          model_freeze_sha256=c.sha(FINAL/'MODEL_FREEZE_PRIVATE.pkl'),selection_ownFIT_only=True))


def frozen_test():
    binding,choices,reference=setup();status=c.jr(PUBLIC/'FINAL_TRAINING_STATUS.json')
    assert status['status']=='ALL15_FINAL_LOGICAL_MODELS_FROZEN'
    assert status['model_freeze_sha256']==c.sha(FINAL/'MODEL_FREEZE_PRIVATE.pkl')
    fr=c.read(FINAL/'MODEL_FREEZE_PRIVATE.pkl');assert fr['input']==binding
    for row in fr['rows']:assert c.sha(row['model'])==row['model_sha256']
    freeze=dict(input=binding,model_freeze=c.sha(FINAL/'MODEL_FREEZE_PRIVATE.pkl'),procedures=PROCEDURES,
                evaluation_source=c.sha(c.HERE/'final_stage.py'),original_score_seal=binding['original_Outer_seal'])
    c.freeze(FINAL/'FROZEN_BEFORE_OUTER_PRIVATE.json',freeze);fh=c.sha(FINAL/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    meta,banks=c.io.identity_and_banks(labels=False);store=c.io.RawStore();cells={};parity=[]
    oldseal=c.jr(c.HISTORY/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')
    assert oldseal['freeze_sha256']==c.sha(c.HISTORY/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    for r in reference:
        fold=r['fold'];d=banks[fold-1];ix=np.flatnonzero(np.isin(meta['patient'],sorted(d['roles']['test'])))
        assert set(meta['patient'][ix]).isdisjoint(set(d['patient']))
        oldpath=c.HISTORY/'outer_scores_private'/f'Concat__RAW_ONLY_{fold}.pkl'
        assert c.sha(oldpath)==oldseal['files'][f'Concat__RAW_ONLY_{fold}']
        old=c.read(oldpath);assert old['freeze_sha256']==oldseal['freeze_sha256']
        assert old['checkpoint_sha256']==r['checkpoint_sha256'] and old['threshold']==r['threshold']
        assert np.array_equal(old['indices'],ix)
        cp=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
        rawmodel=pure.PureRaw().cuda().eval();rawmodel.load_reference(cp['model']);rawmodel.requires_grad_(False)
        foldrows=[q for q in fr['rows'] if q['fold']==fold];models={};pred={p:np.empty(len(ix),np.float32) for p in PROCEDURES}
        for row in foldrows:
            state=torch.load(row['model'],map_location='cpu',weights_only=False)
            md=Head(row['arm'],cp['model'],fold);md.load_state_dict(state['model']);md.eval();models[row['procedure']]=md
        recipes=sorted({row['recipe'] for row in foldrows});maximum=0.
        for pid in sorted(set(meta['patient'][ix])):
            jj=np.flatnonzero(meta['patient'][ix]==pid);ai=ix[jj]
            cache=FINAL/'outer_patient_scores_private'/f'fold{fold}'/(c.io.private_key(pid)+'.pkl')
            pb=dict(freeze=fh,fold=fold,archive_indices=ai.tolist(),waveform_sha256=store.plans[str(pid)]['sha256'])
            if cache.exists():q=c.read(cache);assert q['binding']==pb
            else:
                raw,mask,times=store.get(pid,meta['channel'][ai]);z,logit=embeddings(rawmodel,raw,mask,times,recipes)
                pn_native=c.probability(logit);scores={'NATIVE':pn_native}
                for row in foldrows:
                    p=row['procedure']
                    if p=='NATIVE':continue
                    scores[p]=pn_native.copy() if row['arm']=='NATIVE' else c.probability(trainer.predict(models[p],z[row['recipe']],meta['patient'][ai]))
                error=float(np.max(np.abs(pn_native-old['score_nez'][jj])));assert error<=1e-6
                assert np.array_equal(pn_native<r['threshold'],old['score_nez'][jj]<r['threshold'])
                # Native is the exact prior sealed reference after verified replay.
                scores['NATIVE']=old['score_nez'][jj].copy()
                if choices[fold]=='NATIVE':scores['SELECTED']=scores['NATIVE'].copy()
                q=dict(binding=pb,scores=scores,Native_probability_error=error,valid_windows=int(mask.sum()))
                c.save(cache,q)
            maximum=max(maximum,q['Native_probability_error'])
            for p in PROCEDURES:pred[p][jj]=q['scores'][p]
            print('MACRO_FINAL_FROZEN_OUTER_PATIENT','fold',fold,flush=True)
        for row in foldrows:
            p=row['procedure'];key=f'{p}_{fold}';path=FINAL/'outer_scores_private'/(key+'.pkl')
            result=dict(freeze=fh,model_sha256=row['model_sha256'],indices=ix,score_nez=pred[p],threshold=row['threshold'])
            if path.exists():previous=c.read(path);assert previous['freeze']==fh and np.array_equal(previous['score_nez'],pred[p])
            else:c.save(path,result)
            cells[key]=c.sha(path)
        parity.append(dict(fold=fold,patients=len(set(meta['patient'][ix])),channels=len(ix),
                           native_probability_max_error=maximum,Native_decisions_unchanged=True))
        c.write(PUBLIC/'OUTER_STATUS.json',dict(status='SEALING_SCORES_FULL_LABELS_NOT_READ',score_cells=len(cells),total=15))
        del rawmodel,models;gc.collect();torch.cuda.empty_cache()
    assert len(cells)==15
    c.freeze(FINAL/'OUTER_SCORE_SEAL_PRIVATE.json',dict(freeze=fh,cells=cells,all_scores_sealed_before_fullOuter_labels=True))
    c.write(PUBLIC/'OUTER_STATUS.json',dict(status='ALL15_FROZEN_SCORE_CELLS_SEALED',score_cells=15,
          score_seal_sha256=c.sha(FINAL/'OUTER_SCORE_SEAL_PRIVATE.json'),parity=parity,
          scientific_status='EXPLORATORY_REPEATED_TEST',fullOuter_labels_read=False))


def summarize():
    binding,_,_=setup();status=c.jr(PUBLIC/'OUTER_STATUS.json');assert status['status']=='ALL15_FROZEN_SCORE_CELLS_SEALED'
    assert status['score_seal_sha256']==c.sha(FINAL/'OUTER_SCORE_SEAL_PRIVATE.json')
    seal=c.jr(FINAL/'OUTER_SCORE_SEAL_PRIVATE.json');assert seal['freeze']==c.sha(FINAL/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    meta,_=c.io.identity_and_banks(labels=False)
    archive=c.Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert c.sha(archive)==c.read(c.io.BASE/'RECONSTRUCTION_PRIVATE.pkl')['binding']['archive']
    with np.load(archive,allow_pickle=True) as zz:y=1-np.asarray(zz['y'],int)
    assert len(y)==7635 and y.sum()==1743
    rows=[];pooled=[]
    for procedure in PROCEDURES:
        pp=[];yy=[];predicted=[];coverage=[]
        for fold in range(1,6):
            path=FINAL/'outer_scores_private'/f'{procedure}_{fold}.pkl'
            assert c.sha(path)==seal['cells'][f'{procedure}_{fold}']
            q=c.read(path);assert q['freeze']==seal['freeze'];ix=q['indices'];pn=q['score_nez'];prediction=pn<q['threshold'];prob=1-pn.astype(float)
            pp.extend(prob);yy.extend(y[ix]);predicted.extend(prediction);coverage.extend(ix)
            for pid in sorted(set(meta['patient'][ix])):
                jj=np.flatnonzero(meta['patient'][ix]==pid);ai=ix[jj]
                metrics=c.ev.classification(y[ai],prob[jj],prediction[jj]);rank=c.ev.ranking(y[ai],prob[jj],meta['channel'][ai])
                independent=dict(macro_f1=f1_score(y[ai],prediction[jj],average='macro',zero_division=0),
                                 f1=f1_score(y[ai],prediction[jj],pos_label=1,zero_division=0),
                                 nez_f1=f1_score(y[ai],prediction[jj],pos_label=0,zero_division=0),
                                 accuracy=accuracy_score(y[ai],prediction[jj]),auroc=roc_auc_score(y[ai],prob[jj]),
                                 auprc=average_precision_score(y[ai],prob[jj]))
                for m,v in independent.items():assert abs(v-metrics[m])<1e-12
                rows.append(dict(procedure=procedure,fold=fold,patient=str(pid),center=str(meta['center'][ai][0]),**metrics,**rank))
        assert len(coverage)==len(set(coverage))==7635 and set(coverage)==set(range(7635))
        pooled.append(dict(procedure=procedure,**c.ev.classification(yy,pp,predicted)))
    frame=pd.DataFrame(rows);assert len(frame)==240 and all(frame[frame.procedure==p].patient.nunique()==80 for p in PROCEDURES)
    c.save(FINAL/'OUTER_PATIENT_METRICS_PRIVATE.pkl',rows)
    frame.groupby('procedure')[c.METRICS].mean().reset_index().to_csv(PUBLIC/'OUTER_PATIENT_MEAN.csv',index=False)
    pd.DataFrame(pooled).to_csv(PUBLIC/'OUTER_POOLED.csv',index=False)
    folds=frame.groupby(['procedure','fold'])[c.METRICS].mean().reset_index();folds.to_csv(PUBLIC/'OUTER_FOLDS.csv',index=False)
    folds.groupby('procedure')[c.METRICS].std(ddof=1).reset_index().to_csv(PUBLIC/'OUTER_FOLD_SD.csv',index=False)
    source=[];absolute=[];paired=[]
    for (p,center),g in frame.groupby(['procedure','center']):
        n=g.patient.nunique();source.append(dict(procedure=p,center=center,patients=n,suppressed=n<5,
                            **{m:float(g[m].mean()) if n>=5 else np.nan for m in c.METRICS}))
    pd.DataFrame(source).to_csv(PUBLIC/'OUTER_SOURCE.csv',index=False)
    for procedure,g in frame.groupby('procedure'):
        absolute.extend([dict(procedure=procedure,**r) for r in c.cluster_interval(g.to_dict('records'))])
    for left,right in [('REFIT','NATIVE'),('SELECTED','NATIVE'),('SELECTED','REFIT')]:
        merged=frame[frame.procedure==left].merge(frame[frame.procedure==right],on=['patient','center','fold'],suffixes=('_l','_r'),validate='one_to_one')
        diff=pd.DataFrame(dict(patient=merged.patient,**{m:merged[m+'_l']-merged[m+'_r'] for m in c.METRICS}))
        paired.extend([dict(left=left,right=right,**r) for r in c.cluster_interval(diff.to_dict('records'))])
    pd.DataFrame(absolute).to_csv(PUBLIC/'OUTER_ABSOLUTE_CI.csv',index=False)
    pd.DataFrame(paired).to_csv(PUBLIC/'OUTER_PAIRED_CI.csv',index=False)
    base=frame[frame.procedure=='NATIVE'].macro_f1.mean();assert abs(base-.6304930529317453)<1e-12
    verified={p:float(frame[frame.procedure==p].macro_f1.mean()) for p in PROCEDURES}
    c.write(PUBLIC/'FINAL_TERMINAL.json',dict(status='COMPLETE_VERIFIED_THREE_RAW_MACRO_PROCEDURES',patients_per_procedure=80,
          channels_per_procedure=7635,patient_rows=240,all_three_predeclared_procedures_reported=True,
          patient_equal_macro_f1=verified,point_target_over_065={p:v>.65 for p,v in verified.items()},
          goal_point_established=any(verified[p]>.65 for p in ['REFIT','SELECTED']),independent_confirmation=False,
          scientific_status='EXPLORATORY_REPEATED_TEST',score_seal_sha256=c.sha(FINAL/'OUTER_SCORE_SEAL_PRIVATE.json'),
          independent_sklearn_patient_metrics_pass=True,no_postOuter_tuning=True))
    print('MACRO_FINAL_VERIFIED',verified,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','train','test','summarize']);args=parser.parse_args()
    {'prepare':prepare,'train':train,'test':frozen_test,'summarize':summarize}[args.action]()
