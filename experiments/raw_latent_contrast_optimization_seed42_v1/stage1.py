"""Frozen own-FIT encoders: numerical bridge,16-way screen,fold-local recipe lock."""
import argparse
import gc
import time
import warnings
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from common import *
from model import RawModel,FACTORS,recipe_key,normalize_windows,pool_embeddings
from metrics import patient_rows,threshold


def contexts(banks):
    registry=read(OLD/'ENCODER_REGISTRY_PRIVATE.pkl');out=[]
    assert len(registry['entries'])==20
    for entry in registry['entries']:
        fold=int(entry['fold']);screen=int(entry['screen']);d=banks[fold-1]
        assert sha(entry['path'])==entry['sha256'];stored=read(entry['path'])
        fit=d['train'];assert np.array_equal(stored['archive_indices'],d['archive_indices'][fit])
        rr=stored['roles'];sets=[set(d['patient'][fit][rr[k]]) for k in rr]
        assert all(sets[i].isdisjoint(sets[j]) for i in range(len(sets)) for j in range(i))
        assert set.union(*sets)==d['roles']['fit']
        cp=OLD/'encoders_private'/f'fold{fold}_screen{screen}'/'BEST_PRIVATE.pt'
        assert sha(cp)==stored['checkpoint_sha256']
        out.append(dict(fold=fold,screen=screen,bank=d,fit=fit,stored=stored,checkpoint=cp,
            binding=dict(representation_sha256=entry['sha256'],checkpoint_sha256=stored['checkpoint_sha256'])))
    return out


def part_path(ctx,pid):
    return ROOT/'screen_features_private'/f"fold{ctx['fold']}_screen{ctx['screen']}"/(private_key(pid)+'.pkl')


def part_binding(ctx,global_binding,pid,indices,store):
    return dict(input=global_binding,encoder=ctx['binding'],patient_key=private_key(pid),
                archive_indices=list(map(int,indices)),waveform_bank_sha256=store.plans[str(pid)]['sha256'])


def extract():
    binding=setup();_,banks=identity_and_banks(labels=False);cc=contexts(banks);store=RawStore()
    models={};pids=set();totals={};parity=[]
    for ctx in cc:
        key=(ctx['fold'],ctx['screen']);state=torch.load(ctx['checkpoint'],map_location='cpu',weights_only=False)['model']
        m=RawModel(ctx['fold']).cuda().eval()
        cnn={n.removeprefix('cnn.'):v for n,v in state.items() if n.startswith('cnn.')}
        m.cnn.load_state_dict(cnn,strict=True)
        # Frozen historical160-input head has64 exactly-zero physiology slots.
        head={n.removeprefix('head.'):v for n,v in state.items() if n.startswith('head.')}
        assert head['0.weight'].shape==(64,160)
        head['0.weight']=head['0.weight'][:,:96].clone();m.head.load_state_dict(head,strict=True)
        models[key]=m;pids.update(ctx['bank']['patient'][ctx['fit']]);totals[key]=0
    seen=0;target=sum(len(set(ctx['bank']['patient'][ctx['fit']])) for ctx in cc)
    started=time.perf_counter()
    for pid in sorted(pids):
        raw=mask=times=None
        for ctx in cc:
            d=ctx['bank'];fit=ctx['fit'];loc=np.flatnonzero(d['patient'][fit]==pid)
            if not len(loc):continue
            ix=fit[loc];path=part_path(ctx,pid);pb=part_binding(ctx,binding,pid,d['archive_indices'][ix],store)
            if path.exists():
                result=read(path);assert result['binding']==pb
            else:
                tick=time.perf_counter()
                channels=d['channel'][ix].astype(str).tolist()
                # Patient channel order can differ between legal banks: reload by exact join.
                raw,mask,times=store.get(str(pid),channels)
                m=models[(ctx['fold'],ctx['screen'])]
                with torch.inference_mode():
                    flat=normalize_windows(raw[mask]);emb=raw.new_zeros((*mask.shape,24))
                    emb[mask]=torch.cat([m.cnn(q[:,None]).flatten(1) for q in flat.split(4096)])
                    features=np.stack([pool_embeddings(emb,mask,times,r).cpu().numpy() for r in FACTORS])
                    old=ctx['stored']['raw'][loc,:96]
                    error=float(np.max(np.abs(features[-1]-old)))
                    assert error<=1e-6,('Legacy Raw summary parity failed',ctx['fold'],ctx['screen'],error)
                    logit=m.classify(torch.from_numpy(features[-1]).cuda()).cpu().numpy()
                    logit_error=float(np.max(np.abs(logit-ctx['stored']['raw_logit_nez'][loc])))
                    assert logit_error<=1e-6,('Mapped legacy head parity failed',logit_error)
                assert features.shape==(16,len(ix),96) and np.isfinite(features).all()
                result=dict(binding=pb,features=features,embedding_error=error,logit_error=logit_error,
                    valid_windows=int(mask.sum().cpu()),seconds=time.perf_counter()-tick,
                    all_complete_patient_channels=True,no_labels_or_manual_features_used=True)
                save(path,result);del flat,emb,features,logit
            seen+=1;totals[(ctx['fold'],ctx['screen'])]+=1
            parity.append(dict(fold=ctx['fold'],screen=ctx['screen'],embedding_error=result['embedding_error'],
                               logit_error=result['logit_error'],valid_windows=result['valid_windows']))
            write(PUB/'STAGE1_STATUS.json',dict(status='EXTRACTING_FACTORIAL_SUMMARIES',
                complete_patient_cells=seen,total_patient_cells=target,elapsed_seconds=time.perf_counter()-started,
                no_new_Outer_access=True))
            print('RAW_FACTORIAL_PATIENT_CELL',seen,'of',target,'fold',ctx['fold'],'screen',ctx['screen'],flush=True)
        del raw,mask,times;gc.collect()
    assert seen==target
    for ctx in cc:
        key=(ctx['fold'],ctx['screen']);assert totals[key]==len(set(ctx['bank']['patient'][ctx['fit']]))
    report=pd.DataFrame(parity).groupby(['fold','screen']).agg(patient_cells=('embedding_error','size'),
        max_embedding_error=('embedding_error','max'),max_mapped_head_error=('logit_error','max'),
        valid_windows=('valid_windows','sum')).reset_index()
    report.to_csv(PUB/'FROZEN_ENCODER_BRIDGE.csv',index=False)
    parts={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'screen_features_private').rglob('*.pkl'))}
    freeze(ROOT/'SCREEN_FEATURES_SEAL_PRIVATE.json',dict(binding=binding,parts=parts))
    write(PUB/'EXTRACTION_ADMISSION.json',dict(status='PASS',complete_cells=20,patient_cells=seen,
        recipes=16,dimensions=96,maximum_embedding_error=float(report.max_embedding_error.max()),
        maximum_mapped_head_error=float(report.max_mapped_head_error.max()),all_genuine_windows_used=True,
        all_complete_patient_Z=True,no_new_Outer_access=True,seal_sha256=sha(ROOT/'SCREEN_FEATURES_SEAL_PRIVATE.json')))
    print('RAW_FACTORIAL_EXTRACTION_PASS',seen,flush=True)


def weights(y,pid):
    w=np.zeros(len(y),float)
    for who in np.unique(pid):
        ix=pid==who
        for label in np.unique(y[ix]):
            q=ix&(y==label);w[q]=1/(len(np.unique(y[ix]))*q.sum())
    return w*len(w)/w.sum()


def standardize(x,tr):
    a=np.asarray(x,dtype=np.float64);mean=a[tr].mean(0);sd=a[tr].std(0)
    assert np.isfinite(a).all()
    sd=np.where(sd>1e-12,sd,1.)
    return (a-mean)/sd,dict(mean=mean,scale=sd)


def fit_probe(x,y,pid,tr):
    z,pre=standardize(x,tr)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model=LogisticRegression(C=1,solver='lbfgs',max_iter=2000,tol=1e-7,random_state=42)
        model.fit(z[tr],y[tr],sample_weight=weights(y[tr],pid[tr]))
    return model.predict_proba(z)[:,list(model.classes_).index(1)],dict(pre=pre,coef=model.coef_,intercept=model.intercept_,iterations=model.n_iter_)


def feature_matrix(ctx,binding,store):
    d=ctx['bank'];fit=ctx['fit'];x=np.empty((16,len(fit),96),np.float32)
    for pid in sorted(set(d['patient'][fit])):
        loc=np.flatnonzero(d['patient'][fit]==pid);ix=fit[loc]
        q=read(part_path(ctx,pid))
        assert q['binding']==part_binding(ctx,binding,pid,d['archive_indices'][ix],store)
        x[:,loc]=q['features']
    assert np.isfinite(x).all();return x


def summarized(rows):
    return {m:float(np.nanmean([r[m] for r in rows])) for m in METRICS}


def source_penalty(rows):
    sources=sorted(set(r['center'] for r in rows))
    values=[np.mean([r['f1'] for r in rows if r['center']==s]) for s in sources]
    return float(np.std(values,ddof=1)) if len(values)>1 else 0.


def screen():
    binding=setup();admission=jr(PUB/'EXTRACTION_ADMISSION.json');assert admission['status']=='PASS'
    assert admission['seal_sha256']==sha(ROOT/'SCREEN_FEATURES_SEAL_PRIVATE.json')
    seal=jr(ROOT/'SCREEN_FEATURES_SEAL_PRIVATE.json');assert seal['binding']==binding
    for rel,h in seal['parts'].items():assert sha(ROOT/rel)==h
    _,banks=identity_and_banks(labels=True);cc=contexts(banks);store=RawStore()
    all_assessment=[];search=[];fits=0;final={}
    for ctx in sorted(cc,key=lambda c:(c['fold'],c['screen'])):
        fold=ctx['fold'];s=ctx['screen'];d=ctx['bank'];fit=ctx['fit'];stored=ctx['stored'];rr=stored['roles']
        pid=d['patient'][fit];channel=d['channel'][fit];center=d['center'][fit];y=(d['y'][fit]==0).astype(int)
        x=feature_matrix(ctx,binding,store);path=ROOT/'screen_results_private'/f'fold{fold}_screen{s}.pkl'
        bind=dict(input=binding,features_seal=sha(ROOT/'SCREEN_FEATURES_SEAL_PRIVATE.json'),fold=fold,screen=s)
        if path.exists():
            result=read(path);assert result['binding']==bind
        else:
            records=[];assessment=[];models=[]
            for j,recipe in enumerate(FACTORS):
                prob,model=fit_probe(x[j],y,pid,rr['train']);theta=threshold(y[rr['cal']],prob[rr['cal']],pid[rr['cal']])
                ix=rr['choice'];rows=patient_rows(y[ix],prob[ix],pid[ix],channel[ix],center[ix],theta['threshold'])
                records.append(dict(fold=fold,screen=s,recipe=recipe_key(recipe),stage='choice',**summarized(rows),threshold=theta['threshold']))
                models.append(dict(recipe=recipe_key(recipe),kind='choice',model=model,threshold=theta['threshold']))
                if s:
                    refit=np.r_[rr['train'],rr['choice']]
                    prob,model=fit_probe(x[j],y,pid,refit);theta=threshold(y[rr['cal']],prob[rr['cal']],pid[rr['cal']])
                    ix=rr['held'];rows=patient_rows(y[ix],prob[ix],pid[ix],channel[ix],center[ix],theta['threshold'])
                    assessment.extend([dict(fold=fold,screen=s,recipe=recipe_key(recipe),**r) for r in rows])
                    models.append(dict(recipe=recipe_key(recipe),kind='assessment',model=model,threshold=theta['threshold']))
            result=dict(binding=bind,search=records,assessment=assessment,models=models,fit_count=len(models))
            save(path,result)
        all_assessment.extend(result['assessment']);search.extend(result['search']);fits+=result['fit_count']
        pd.DataFrame(search).to_csv(PUB/'SCREEN_CHOICE_RESULTS.csv',index=False)
        write(PUB/'STAGE1_STATUS.json',dict(status='SCREENING',complete_encoder_paths=sum(1 for p in (ROOT/'screen_results_private').glob('*.pkl')),
            logistic_fits_counted=fits,no_Development_or_Outer_used=True))
        print('RAW_FACTORIAL_SCREEN_PATH_COMPLETE',fold,s,'fits',fits,flush=True)
    assert fits==560
    frame=pd.DataFrame(all_assessment);save(ROOT/'ASSESSMENT_ROWS_PRIVATE.pkl',all_assessment)
    aggregate=frame.groupby(['fold','recipe'])[METRICS].mean().reset_index()
    choices=[]
    for fold in range(1,6):
        candidates=[]
        for recipe in map(recipe_key,FACTORS):
            rows=[r for r in all_assessment if r['fold']==fold and r['recipe']==recipe]
            score=np.mean([r['f1'] for r in rows])-.1*source_penalty(rows)
            candidates.append(dict(recipe=recipe,score=float(score),active_components=recipe.count('1')))
        maximum=max(r['score'] for r in candidates)
        chosen=min([r for r in candidates if r['score']>=maximum-.005],
                   key=lambda r:(r['active_components'],-r['score'],r['recipe']))
        final[str(fold)]=chosen['recipe'];choices.append(dict(fold=fold,**chosen,maximum_score=maximum,own_FIT_only=True))
    # Source and factor effects are descriptive on own-FIT selection data, not confirmation.
    source=frame.groupby(['fold','recipe','center'])[METRICS].mean().reset_index()
    contrasts=[]
    for factor in range(4):
        pairs=[]
        for recipe in map(recipe_key,FACTORS):
            if recipe[factor]=='1':continue
            other=recipe[:factor]+'1'+recipe[factor+1:]
            a=frame[frame.recipe==recipe];b=frame[frame.recipe==other]
            merged=a.merge(b,on=['fold','screen','patient','center'],suffixes=('_off','_on'),validate='one_to_one')
            for m in METRICS:
                pairs.append(dict(factor=factor,off_recipe=recipe,on_recipe=other,metric=m,
                    paired_delta=float((merged[m+'_on']-merged[m+'_off']).mean()),logical_patients=len(merged)))
        contrasts.extend(pairs)
    aggregate.to_csv(PUB/'FACTORIAL_ASSESSMENT_RESULTS.csv',index=False)
    source.to_csv(PUB/'FACTORIAL_SOURCE_RESULTS.csv',index=False)
    pd.DataFrame(contrasts).to_csv(PUB/'FACTOR_CONDITIONAL_EFFECTS.csv',index=False)
    pd.DataFrame(choices).to_csv(PUB/'OWN_FIT_POOL_SELECTION.csv',index=False)
    freeze(ROOT/'POOL_RECIPE_FREEZE_PRIVATE.json',dict(binding=binding,feature_seal=sha(ROOT/'SCREEN_FEATURES_SEAL_PRIVATE.json'),
        recipes=final,own_FIT_only=True,selection=jr(HERE/'PROTOCOL_LOCK.json')['stage1']['recipe_choice']))
    write(PUB/'STAGE1_STATUS.json',dict(status='ALL16_FACTORIAL_SCREEN_AND_FOLD_LOCAL_RECIPES_COMPLETE',
        logistic_fits=560,encoder_paths=20,encoder_updates_in_this_stage=0,recipes=final,
        total_assessment_patient_rows=len(frame),independent_patients_not_equal_to_logical_rows=True,
        new_Outer_access=False,original_Development_used=False,
        recipe_freeze_sha256=sha(ROOT/'POOL_RECIPE_FREEZE_PRIVATE.json'),
        stage2_not_yet_run=True,target_F1_not_established=True))
    print('RAW_FACTORIAL_SCREEN_COMPLETE',fits,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['extract','screen']);args=p.parse_args()
    extract() if args.action=='extract' else screen()
