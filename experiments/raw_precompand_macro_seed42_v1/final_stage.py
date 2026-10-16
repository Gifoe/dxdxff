"""Fixed four-procedure final stage; no test-driven model/threshold selection."""
import argparse
import gc
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,accuracy_score
import common as c
import train as trainer
from model import Model,ARMS,TRAINED

FINAL=c.ROOT/'final_stage_v1';PUBLIC=FINAL/'public'
PROCEDURES=['NATIVE','WINDOW_RESIDUAL','PRECOMPAND_RESIDUAL','SELECTED']
HISTORY=c.parent.HISTORY


def references():
    baseline=c.jr(c.parent.REFERENCE_RUNTIME/'BASELINE_INPUT_FREEZE_PRIVATE.json')
    assert c.sha(HISTORY/'MODEL_FREEZE_PRIVATE.pkl')==baseline['historical_freeze_sha256']
    fr=c.read(HISTORY/'MODEL_FREEZE_PRIVATE.pkl')
    rows=sorted([r for r in fr['selections'] if r['family']=='Concat' and r['recipe']=='RAW_ONLY'],key=lambda r:r['fold'])
    assert [r['fold'] for r in rows]==list(range(1,6))
    for r in rows:
        assert r['groups']==[] and c.sha(r['checkpoint'])==r['checkpoint_sha256'] and c.sha(r['validation'])==r['validation_sha256']
    return rows


def setup():
    initial=c.setup();status=c.jr(c.PUB/'INDEPENDENT_INTERNAL_VERIFICATION.json')
    assert status['status']=='PASS' and c.jr(c.PUB/'INTERNAL_TERMINAL.json')['status']=='COMPLETE_VERIFIED_PRECOMPAND_INTERNAL'
    selection=c.ROOT/'OWN_FIT_SELECTION_PRIVATE.json';assert c.sha(selection)==status['ownFIT_selection_sha256']
    selected=c.jr(selection);assert selected['input']==initial and selected['internal_seal']==c.sha(c.ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json')
    choices={r['fold']:r['selected_arm'] for r in selected['choices']};assert len(choices)==5
    FINAL.mkdir(exist_ok=True);PUBLIC.mkdir(exist_ok=True)
    rows=references();binding=dict(internal_binding_sha256=c.digest(initial),ownFIT_selection_sha256=c.sha(selection),
             source=c.sha(c.HERE/'final_stage.py'),body_freeze=c.sha(HISTORY/'MODEL_FREEZE_PRIVATE.pkl'),
             body_rows=[{k:r[k] for k in ['fold','checkpoint_sha256','validation_sha256','threshold','epoch']} for r in rows],
             raw_registry=c.io.EXPECTED_REG,reconstruction=c.io.EXPECTED_REC,
             original_Outer_seal=c.sha(HISTORY/'OUTER_PREDICTIONS_SEAL_PRIVATE.json'))
    registry=c.read(c.parent.ENC/'ENCODER_REGISTRY_PRIVATE.pkl')
    assert binding['original_Outer_seal']==registry['binding']['parent_Outer_seal_protection']
    c.freeze(FINAL/'INPUT_FREEZE_PRIVATE.json',binding)
    trainer.ROOT=FINAL;trainer.PUB=PUBLIC
    return binding,choices,rows


def phase_path(fold,pid):return FINAL/'phase_views_private'/f'fold{fold}'/(c.io.private_key(pid)+'.pkl')


def prepare():
    binding,choices,rows=setup();_,banks=c.io.identity_and_banks(labels=False);store=c.RawStore();parts={};reports=[]
    for r in rows:
        fold=r['fold'];d=banks[fold-1];cp=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
        assert cp['epoch']==r['epoch'] and cp['threshold']==r['threshold']
        assert set(d['patient'])==d['roles']['fit']|d['roles']['validation'] and set(d['patient']).isdisjoint(d['roles']['test'])
        model=Model('NATIVE',fold,cp['model']).cuda().eval();logit=np.empty(len(d['patient']),np.float32)
        for pid in sorted(set(d['patient'])):
            ix=np.flatnonzero(d['patient']==pid);p=phase_path(fold,pid)
            pb=dict(input=binding,fold=fold,archive_indices=d['archive_indices'][ix].tolist(),body_sha256=r['checkpoint_sha256'],
                    waveform_sha256=store.plans[str(pid)]['sha256'])
            if p.exists():q=c.read(p);assert q['binding']==pb
            else:
                raw,mask,times=store.get(pid,d['channel'][ix]);cache,audit=model.encode(raw,mask,times)
                with torch.inference_mode():score=model.forward_stats(cache).cpu().numpy()
                q=dict(binding=pb,**{k:v.cpu().numpy() for k,v in cache.items()},native_logit_nez=score,audit=audit,
                       valid_windows=int(mask.sum()),complete_patient_channels=True,no_labels_or_manual_features_used=True)
                c.save(p,q)
            logit[ix]=q['native_logit_nez'];parts[str(p.relative_to(FINAL))]=c.sha(p)
            c.write(PUBLIC/'PREPARATION_STATUS.json',dict(status='ENCODING_ORIGINAL_FIT_DEVELOPMENT_ONLY',patient_cells=len(parts),new_Outer_passes=0))
        saved=c.read(r['validation']);va=d['val'];assert np.array_equal(saved['archive_indices'],d['archive_indices'][va])
        error=float(np.max(np.abs(logit[va]-saved['logit'])));assert error<=1e-6
        grid=np.round(np.arange(0,1.0001,.005),3);a=c.probability(logit[va]);b=c.probability(saved['logit'])
        assert np.array_equal(a[:,None]>=grid,b[:,None]>=grid)
        reports.append(dict(fold=fold,FIT_patients=len(d['roles']['fit']),Development_patients=len(d['roles']['validation']),
                            channels=len(d['patient']),Native_validation_logit_error=error,all201_CAL_grid_decisions_unchanged=True,
                            chosen_arm=choices[fold],body_sha256=r['checkpoint_sha256']))
        print('PRECOMPAND_FINAL_VIEWS_FOLD_COMPLETE',fold,len(parts),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    assert len(parts)==320
    c.freeze(FINAL/'PHASE_VIEWS_SEAL_PRIVATE.json',dict(input=binding,parts=parts))
    c.write(PUBLIC/'FINAL_PREPARATION.json',dict(status='PASS',patient_cells=320,reports=reports,
          phase_seal_sha256=c.sha(FINAL/'PHASE_VIEWS_SEAL_PRIVATE.json'),new_Outer_passes=0))


def context(d,r,binding):
    seal=c.jr(FINAL/'PHASE_VIEWS_SEAL_PRIVATE.json');assert seal['input']==binding
    caches={}
    for pid in sorted(set(d['patient'])):
        ix=np.flatnonzero(d['patient']==pid);p=phase_path(r['fold'],pid)
        assert c.sha(p)==seal['parts'][str(p.relative_to(FINAL))];q=c.read(p)
        assert q['binding']['input']==binding and q['binding']['archive_indices']==d['archive_indices'][ix].tolist()
        assert q['binding']['body_sha256']==r['checkpoint_sha256'] and q['complete_patient_channels']
        caches[str(pid)]=c.payload(q)
    cp=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
    ctx=dict(fold=r['fold'],screen='final',data=d,reference_state=cp['model'],reference_threshold=r['threshold'],
             roles=dict(train=d['train'],choice=np.array([],int),cal=d['val'],held=d['val']),
             stored=dict(checkpoint_sha256=r['checkpoint_sha256']),
             representation_sha256=c.digest(dict(fold=r['fold'],archive_indices=d['archive_indices'].tolist(),phase_seal=c.sha(FINAL/'PHASE_VIEWS_SEAL_PRIVATE.json'))))
    # The reused generic trainer's HELD output is only Development here; never
    # report it as a new independent held set or use it to change chosen arms.
    return ctx,caches


def fit():
    binding,choices,refs=setup();assert c.jr(PUBLIC/'FINAL_PREPARATION.json')['status']=='PASS'
    _,banks=c.io.identity_and_banks(labels=True);rows=[];public=[]
    for r in refs:
        fold=r['fold'];d=banks[fold-1];ctx,caches=context(d,r,binding);byarm={};epoch_counts={'NATIVE':0}
        native=dict(fold=fold,procedure='NATIVE',arm='NATIVE',body=r['checkpoint'],body_sha256=r['checkpoint_sha256'],
                    matrix=None,matrix_sha256=None,threshold=r['threshold'],epoch=0)
        rows.append(native);byarm['NATIVE']=native
        for arm in TRAINED:
            summary,_,_=trainer.fit(ctx,arm,binding,caches)
            p=FINAL/'matrices_private'/f'fold{fold}_screenfinal'/arm/'BEST_PRIVATE.pt'
            row=dict(fold=fold,procedure=arm,arm=arm,body=r['checkpoint'],body_sha256=r['checkpoint_sha256'],
                     matrix=str(p),matrix_sha256=c.sha(p),threshold=summary['threshold'],epoch=summary['selected_epoch'])
            rows.append(row);byarm[arm]=row
            epoch_counts[arm]=summary['epochs']
            public.append(dict(fold=fold,procedure=arm,arm=arm,epochs=summary['epochs'],selected_epoch=summary['selected_epoch'],
                               threshold_NEZ=summary['threshold'],trainable_parameters=576,body_selected_epoch=r['epoch']))
        rows.append(dict(byarm[choices[fold]],procedure='SELECTED'))
        public.extend([dict(fold=fold,procedure='NATIVE',arm='NATIVE',epochs=0,selected_epoch=0,threshold_NEZ=r['threshold'],trainable_parameters=0,body_selected_epoch=r['epoch']),
                       dict(fold=fold,procedure='SELECTED',arm=choices[fold],epochs=epoch_counts[choices[fold]],selected_epoch=byarm[choices[fold]]['epoch'],
                            threshold_NEZ=byarm[choices[fold]]['threshold'],trainable_parameters=0 if choices[fold]=='NATIVE' else 576,body_selected_epoch=r['epoch'])])
        del caches;gc.collect();torch.cuda.empty_cache()
    assert len(rows)==20
    c.save(FINAL/'MODEL_FREEZE_PRIVATE.pkl',dict(input=binding,phase_seal=c.sha(FINAL/'PHASE_VIEWS_SEAL_PRIVATE.json'),rows=rows))
    pd.DataFrame(public).to_csv(PUBLIC/'FINAL_SELECTION.csv',index=False)
    c.write(PUBLIC/'FINAL_TRAINING_STATUS.json',dict(status='ALL20_LOGICAL_MODELS_FROZEN',actual_matrix_fits=10,
          body_head_updates=0,model_freeze_sha256=c.sha(FINAL/'MODEL_FREEZE_PRIVATE.pkl'),new_Outer_passes=0,
          selected_rows_are_aliases_not_additional_fits=True,development_not_independent=True))


def test():
    binding,choices,refs=setup();status=c.jr(PUBLIC/'FINAL_TRAINING_STATUS.json');assert status['status']=='ALL20_LOGICAL_MODELS_FROZEN'
    assert status['model_freeze_sha256']==c.sha(FINAL/'MODEL_FREEZE_PRIVATE.pkl');fr=c.read(FINAL/'MODEL_FREEZE_PRIVATE.pkl');assert fr['input']==binding
    for r in fr['rows']:
        assert c.sha(r['body'])==r['body_sha256']
        if r['matrix']:assert c.sha(r['matrix'])==r['matrix_sha256']
    c.freeze(FINAL/'FROZEN_BEFORE_OUTER_PRIVATE.json',dict(input=binding,model_freeze=c.sha(FINAL/'MODEL_FREEZE_PRIVATE.pkl'),
             evaluation_source=c.sha(c.HERE/'final_stage.py'),procedures=PROCEDURES))
    fh=c.sha(FINAL/'FROZEN_BEFORE_OUTER_PRIVATE.json');meta,banks=c.io.identity_and_banks(labels=False);store=c.RawStore();cells={};parity=[]
    oldseal=c.jr(HISTORY/'OUTER_PREDICTIONS_SEAL_PRIVATE.json');assert oldseal['freeze_sha256']==c.sha(HISTORY/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    for r in refs:
        fold=r['fold'];d=banks[fold-1];ix=np.flatnonzero(np.isin(meta['patient'],sorted(d['roles']['test'])))
        assert set(meta['patient'][ix]).isdisjoint(d['roles']['fit']|d['roles']['validation'])
        oldpath=HISTORY/'outer_scores_private'/f'Concat__RAW_ONLY_{fold}.pkl'
        assert c.sha(oldpath)==oldseal['files'][f'Concat__RAW_ONLY_{fold}'];old=c.read(oldpath)
        assert np.array_equal(old['indices'],ix) and old['checkpoint_sha256']==r['checkpoint_sha256'] and old['threshold']==r['threshold']
        cp=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);models={}
        foldrows=[q for q in fr['rows'] if q['fold']==fold]
        for row in foldrows:
            if row['procedure']=='SELECTED':continue
            model=Model(row['arm'],fold,cp['model']).cuda().eval()
            if row['matrix']:
                state=torch.load(row['matrix'],map_location='cpu',weights_only=False);assert state['binding']['input']==binding
                with torch.no_grad():model.matrix.copy_(state['matrix'])
            models[row['procedure']]=model
        scores={p:np.empty(len(ix),np.float32) for p in PROCEDURES};maximum=0.
        for pid in sorted(set(meta['patient'][ix])):
            jj=np.flatnonzero(meta['patient'][ix]==pid);ai=ix[jj]
            cache=FINAL/'outer_patient_scores_private'/f'fold{fold}'/(c.io.private_key(pid)+'.pkl')
            pb=dict(freeze=fh,fold=fold,archive_indices=ai.tolist(),waveform_sha256=store.plans[str(pid)]['sha256'])
            if cache.exists():q=c.read(cache);assert q['binding']==pb
            else:
                raw,mask,times=store.get(pid,meta['channel'][ai]);features,audit=models['NATIVE'].encode(raw,mask,times)
                with torch.inference_mode():pn={arm:c.probability(model.forward_stats(features).cpu().numpy()) for arm,model in models.items()}
                error=float(np.max(np.abs(pn['NATIVE']-old['score_nez'][jj])));assert error<=1e-6
                assert np.array_equal(pn['NATIVE']<r['threshold'],old['score_nez'][jj]<r['threshold'])
                pn['NATIVE']=old['score_nez'][jj].copy();pn['SELECTED']=pn[choices[fold]].copy()
                q=dict(binding=pb,scores=pn,Native_probability_error=error,valid_windows=int(mask.sum()),normalization_audit=audit)
                c.save(cache,q)
            maximum=max(maximum,q['Native_probability_error'])
            for p in PROCEDURES:scores[p][jj]=q['scores'][p]
        for row in foldrows:
            key=f"{row['procedure']}_{fold}";path=FINAL/'outer_scores_private'/(key+'.pkl')
            result=dict(freeze=fh,indices=ix,score_nez=scores[row['procedure']],threshold=row['threshold'],arm=row['arm'],
                        body_sha256=row['body_sha256'],matrix_sha256=row['matrix_sha256'])
            if path.exists():prior=c.read(path);assert prior['freeze']==fh and np.array_equal(prior['score_nez'],result['score_nez'])
            else:c.save(path,result)
            cells[key]=c.sha(path)
        parity.append(dict(fold=fold,patients=len(set(meta['patient'][ix])),channels=len(ix),Native_probability_max_error=maximum,Native_decisions_unchanged=True))
        c.write(PUBLIC/'OUTER_STATUS.json',dict(status='SEALING_FROZEN_SCORES_FULL_LABELS_NOT_READ',score_cells=len(cells),total=20))
        print('PRECOMPAND_FROZEN_OUTER_FOLD_COMPLETE',fold,len(cells),flush=True)
        del models;gc.collect();torch.cuda.empty_cache()
    assert len(cells)==20
    c.freeze(FINAL/'OUTER_SCORE_SEAL_PRIVATE.json',dict(freeze=fh,cells=cells,all_scores_sealed_before_fullOuter_truth=True))
    c.write(PUBLIC/'OUTER_STATUS.json',dict(status='ALL20_FROZEN_SCORE_CELLS_SEALED',score_cells=20,parity=parity,
          score_seal_sha256=c.sha(FINAL/'OUTER_SCORE_SEAL_PRIVATE.json'),fullOuter_truth_read=False,scientific_status='EXPLORATORY_REPEATED_TEST'))


def report():
    binding,_,_=setup();status=c.jr(PUBLIC/'OUTER_STATUS.json');assert status['status']=='ALL20_FROZEN_SCORE_CELLS_SEALED'
    assert status['score_seal_sha256']==c.sha(FINAL/'OUTER_SCORE_SEAL_PRIVATE.json')
    seal=c.jr(FINAL/'OUTER_SCORE_SEAL_PRIVATE.json');assert seal['freeze']==c.sha(FINAL/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    meta,banks=c.io.identity_and_banks(labels=False)
    archive=c.Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert c.sha(archive)==c.read(c.io.BASE/'RECONSTRUCTION_PRIVATE.pkl')['binding']['archive']
    with np.load(archive,allow_pickle=True) as z:
        assert np.array_equal(z['patient'],meta['patient']) and np.array_equal(z['channel'],meta['channel']);y=1-z['y']
    assert len(y)==7635 and y.sum()==1743
    rows=[];pooled=[];maximum=0.
    for procedure in PROCEDURES:
        allix=[];truth=[];probs=[];predictions=[]
        for fold in range(1,6):
            path=FINAL/'outer_scores_private'/f'{procedure}_{fold}.pkl';assert c.sha(path)==seal['cells'][f'{procedure}_{fold}']
            q=c.read(path);assert q['freeze']==seal['freeze'];ix=q['indices'];pn=q['score_nez'];pp=1-pn.astype(float);pred=pn<q['threshold']
            assert set(meta['patient'][ix])==banks[fold-1]['roles']['test']
            allix.extend(ix);truth.extend(y[ix]);probs.extend(pp);predictions.extend(pred)
            for pid in sorted(set(meta['patient'][ix])):
                loc=meta['patient'][ix]==pid;ai=ix[loc];yy=y[ai];p=pp[loc];pr=pred[loc];ch=meta['channel'][ai]
                metric=dict(**c.parent.ev.classification(yy,p,pr),**c.parent.ev.ranking(yy,p,ch))
                order=np.lexsort((ch,-p));first=int(np.flatnonzero(yy[order])[0])
                independent=dict(macro_f1=f1_score(yy,pr,average='macro',zero_division=0),f1=f1_score(yy,pr,pos_label=1,zero_division=0),
                                 nez_f1=f1_score(yy,pr,pos_label=0,zero_division=0),accuracy=accuracy_score(yy,pr),
                                 auroc=roc_auc_score(yy,p),auprc=average_precision_score(yy,p),mrr=1/(first+1),hit_at_1=float(first==0),
                                 ez_fraction_bias=float(pr.mean()-yy.mean()))
                for m,v in independent.items():maximum=max(maximum,abs(v-metric[m]));assert abs(v-metric[m])<1e-12
                rows.append(dict(procedure=procedure,fold=fold,patient=str(pid),center=str(meta['center'][ai][0]),**metric))
        assert len(allix)==len(set(allix))==7635 and set(allix)==set(range(7635))
        pooled.append(dict(procedure=procedure,**c.parent.ev.classification(truth,probs,predictions)))
    frame=pd.DataFrame(rows);assert len(frame)==320 and all(frame[frame.procedure==p].patient.nunique()==80 for p in PROCEDURES)
    c.save(FINAL/'OUTER_PATIENT_METRICS_PRIVATE.pkl',rows)
    frame.groupby('procedure')[c.METRICS].mean().reset_index().to_csv(PUBLIC/'OUTER_PATIENT_MEAN.csv',index=False)
    pd.DataFrame(pooled).to_csv(PUBLIC/'OUTER_POOLED.csv',index=False)
    foldmeans=frame.groupby(['procedure','fold'])[c.METRICS].mean().reset_index();foldmeans.to_csv(PUBLIC/'OUTER_FOLDS.csv',index=False)
    foldmeans.groupby('procedure')[c.METRICS].std(ddof=1).reset_index().to_csv(PUBLIC/'OUTER_FOLD_SD.csv',index=False)
    sources=[];coverage=[];absolute=[];paired=[]
    for (procedure,center),g in frame.groupby(['procedure','center']):
        n=g.patient.nunique();sources.append(dict(procedure=procedure,center=center,patients=n,suppressed=n<5,
                           **{m:float(g[m].mean()) if n>=5 else np.nan for m in c.METRICS}))
    for procedure,g in frame.groupby('procedure'):
        absolute.extend([dict(procedure=procedure,**r) for r in c.parent.cluster_interval(g.to_dict('records'))])
        coverage.extend([dict(procedure=procedure,metric=m,finite_patients=int(np.isfinite(g[m]).sum()),total_patients=80) for m in c.METRICS])
    for left,right in [('WINDOW_RESIDUAL','NATIVE'),('PRECOMPAND_RESIDUAL','NATIVE'),('PRECOMPAND_RESIDUAL','WINDOW_RESIDUAL'),('SELECTED','NATIVE')]:
        merged=frame[frame.procedure==left].merge(frame[frame.procedure==right],on=['patient','center','fold'],suffixes=('_l','_r'),validate='one_to_one')
        diff=pd.DataFrame(dict(patient=merged.patient,**{m:merged[m+'_l']-merged[m+'_r'] for m in c.METRICS}))
        paired.extend([dict(left=left,right=right,**r) for r in c.parent.cluster_interval(diff.to_dict('records'))])
    pd.DataFrame(sources).to_csv(PUBLIC/'OUTER_SOURCE.csv',index=False);pd.DataFrame(coverage).to_csv(PUBLIC/'OUTER_METRIC_COVERAGE.csv',index=False)
    pd.DataFrame(absolute).to_csv(PUBLIC/'OUTER_ABSOLUTE_CI.csv',index=False);pd.DataFrame(paired).to_csv(PUBLIC/'OUTER_PAIRED_CI.csv',index=False)
    point={p:float(frame[frame.procedure==p].macro_f1.mean()) for p in PROCEDURES};assert abs(point['NATIVE']-.6304930529317453)<1e-12
    c.write(PUBLIC/'FINAL_VERIFICATION.json',dict(status='PASS',patients_per_procedure=80,channels_per_procedure=7635,patient_rows=320,
          archive_identity_exact=True,all9_metrics_independently_recomputed=True,maximum_metric_error=maximum,
          all20_scores_sealed_before_full_labels=True,score_seal_sha256=c.sha(FINAL/'OUTER_SCORE_SEAL_PRIVATE.json')))
    c.write(PUBLIC/'FINAL_TERMINAL.json',dict(status='COMPLETE_VERIFIED_FOUR_PRECOMPAND_PROCEDURES',patient_equal_macro_f1=point,
          point_over_065={p:v>.65 for p,v in point.items()},goal_point_established=any(point[p]>.65 for p in PROCEDURES[1:]),
          all_predeclared_procedures_reported=True,postOuter_tuning=False,scientific_status='EXPLORATORY_REPEATED_TEST',independent_confirmation=False))
    print('PRECOMPAND_FINAL_VERIFIED',point,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','fit','test','report']);args=parser.parse_args()
    {'prepare':prepare,'fit':fit,'test':test,'report':report}[args.action]()
