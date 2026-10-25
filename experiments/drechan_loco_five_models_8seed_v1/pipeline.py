"""Sequential fit -> independent selection freeze -> score seal -> metrics."""
import argparse
import re
import time
import zipfile
import pandas as pd
from common import *
import svm

def cellpaths(arm,k):
    r=ROOT/'models_private'/arm/f'held{k}';r.mkdir(parents=True,exist_ok=True)
    return r,r/'BEST_PRIVATE.pt',r/'VALIDATION_PRIVATE.pkl',r/'COMPLETE_PRIVATE.json'

def fit(arm,k,binding,d,store):
    root,bp,vp,done=cellpaths(arm,k)
    if arm=='LINEAR_SVM_FEATURE_PR':bp=root/'MODEL_PRIVATE.pkl'
    m=make(arm,k) if arm!='LINEAR_SVM_FEATURE_PR' else None
    initial=state_hash(m.state_dict()) if m is not None else 'NOT_APPLICABLE'
    bind=dict(input=binding,arm=arm,held_index=k,seed=TRAIN_SEED+1009*k,initial=initial,
              preprocessor_sha256=digest(io.clean(d['pre'])),roles_sha256=sha(ROOT/'ROLES_PRIVATE.json'))
    if done.exists():
        q=jr(done);assert q['binding']==digest(bind) and sha(bp)==q['checkpoint_sha256'] and sha(vp)==q['validation_sha256']
        v=read(vp);assert v['binding']==bind
        del m;gc.collect();torch.cuda.empty_cache();return v['summary']
    if m is None:
        tick=time.perf_counter();bundle=svm.fit(d,k);vscore=svm.predict(bundle,d['raw'][d['val']],d['patient'][d['val']])
        selected=select(arm,d,vscore);history=[];epoch=0
        save(bp,dict(binding=bind,bundle=bundle,selection=selected))
        seconds=time.perf_counter()-tick
    else:
        seed_all(TRAIN_SEED+1009*k);tc=arm=='TIMECONVCNN_RAW'
        opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.0001 if tc else .001,**({'fused':True} if tc else {}))
        tr=d['train'];pos=torch.tensor(float((1-d['y'][tr]).sum()/d['y'][tr].sum()),device='cuda')
        groups=[tr[d['patient'][tr]==pid] for pid in sorted(set(d['patient'][tr]))]
        last=root/'LAST_PRIVATE.pt';best=key=None;history=[];stale=0;start=1
        if last.exists():
            q=torch.load(last,weights_only=False,map_location='cpu');assert q['binding']==bind
            m.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);restore_rng(q['rng'])
            best,key,history,stale,start=q['best'],q['key'],q['history'],q['stale'],q['epoch']+1
            if q['epoch']>=6 and stale>=6:start=31
        for epoch in range(start,31):
            tick=time.perf_counter();m.train();losses=[]
            for n,j in enumerate(np.random.default_rng(TRAIN_SEED+1009*k+epoch).permutation(len(groups)),1):
                ii=groups[int(j)];opt.zero_grad(set_to_none=True)
                if tc:loss,_=forward(m,arm,d,ii,store,True,pos)
                else:
                    logit=forward(m,arm,d,ii,store);target=torch.as_tensor(d['y'][ii],device='cuda',dtype=torch.float32)
                    loss=LEG.MODEL.objective(logit,target,pos);assert torch.isfinite(loss);loss.backward();loss=float(loss.detach())
                    del logit,target
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
                torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
                if tc:m.constraints()
                losses.append(loss)
                if n%10==0:write(PUB/'TRAIN_STATUS.json',dict(status='TRAINING',arm=arm,held_source=SOURCES[k-1],epoch=epoch,patients_complete=n,patients_total=len(groups)))
            score=predict(m,arm,d,d['val'],store);sel=select(arm,d,score);candidate=(sel['macro_f1'],sel['ez_f1'],-epoch)
            if key is None or candidate>key:
                key=candidate;stale=0;best=dict(model={n:v.detach().cpu().clone() for n,v in m.state_dict().items()},epoch=epoch,selection=sel)
            else:stale+=1
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),seconds=time.perf_counter()-tick,**sel))
            atomic_torch(last,dict(binding=bind,model=m.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),best=best,key=key,history=history,stale=stale,epoch=epoch))
            print('LOCO_EPOCH',arm,k,epoch,'best',best['epoch'],'seconds',round(history[-1]['seconds'],3),flush=True)
            if epoch>=6 and stale>=6:break
        assert best is not None;m.load_state_dict(best['model']);atomic_torch(bp,dict(binding=bind,**best))
        vscore=predict(m,arm,d,d['val'],store);selected=select(arm,d,vscore)
        assert selected==best['selection'];epoch=best['epoch'];seconds=sum(r['seconds'] for r in history)
    summary=dict(train_seed=TRAIN_SEED,arm=arm,held_index=k,held_source=SOURCES[k-1],epoch=epoch,epochs=len(history),
                 threshold=selected['threshold'],rule=selected['rule'],parameters=sum(p.numel() for p in m.parameters()) if m is not None else None,
                 initial_sha256=initial,checkpoint=str(bp),checkpoint_sha256=sha(bp),validation=str(vp),
                 preprocessor_sha256=bind['preprocessor_sha256'],seconds=seconds,reused=False)
    save(vp,dict(binding=bind,summary=summary,score=vscore,selection=selected,history=history,indices=d['archive_indices'][d['val']]))
    summary['validation_sha256']=sha(vp)
    write(done,dict(binding=digest(bind),checkpoint_sha256=sha(bp),validation_sha256=sha(vp)))
    del m;gc.collect();torch.cuda.empty_cache();return summary

def train():
    assert TRAIN_SEED in [45,46,47,48,49]
    binding,rec,banks=setup();a=jr(PUB/'ADMISSION.json');assert a['status']=='PASS' and a['binding']==binding
    commit=os.environ.get('LOCO_SOURCE_COMMIT','');assert re.fullmatch('[a-f0-9]{40}',commit)
    freeze(PUB/'SOURCE_PUSH_BEFORE_TRAIN.json',dict(commit=commit,source=sources(),admission_sha256=sha(PUB/'ADMISSION.json'),pushed_before_fit=True))
    store=io.RawStore();rows=[]
    for k,d in enumerate(banks,1):
        for arm in ARMS:
            r=fit(arm,k,binding,d,store)
            if 'validation_sha256' not in r:r=dict(r,validation_sha256=sha(r['validation']))
            rows.append(r);csv_out('TRAIN_CELLS',[{n:v for n,v in q.items() if n not in ['checkpoint','validation']} for q in rows])
            histories=[]
            for q in rows:
                for h in read(q['validation'])['history']:histories.append(dict(train_seed=TRAIN_SEED,arm=q['arm'],held_source=q['held_source'],**h))
            if histories:csv_out('TRAIN_HISTORY',histories)
            print('LOCO_CELL_COMPLETE',arm,k,len(rows),20,flush=True)
    freeze(ROOT/'MODEL_SELECTION_PRIVATE.json',dict(binding=binding,cells=rows))
    write(PUB/'TRAIN_STATUS.json',dict(status='ALL20_SELECTIONS_COMPLETE',cells=20,reused=0))

def verify_internal():
    binding,rec,banks=setup();q=jr(ROOT/'MODEL_SELECTION_PRIVATE.json');assert q['binding']==binding and len(q['cells'])==20
    checks=[];store=io.RawStore()
    for r in q['cells']:
        arm,k=r['arm'],r['held_index'];d=banks[k-1];v=read(r['validation']);assert sha(r['validation'])==r['validation_sha256']
        assert sha(r['checkpoint'])==r['checkpoint_sha256'] and v['binding']['input']==binding
        assert np.array_equal(v['indices'],d['archive_indices'][d['val']])
        assert v['binding']['preprocessor_sha256']==digest(io.clean(d['pre']))==r['preprocessor_sha256']
        if arm=='LINEAR_SVM_FEATURE_PR':
            cp=read(r['checkpoint']);score=svm.predict(cp['bundle'],d['raw'][d['val']],d['patient'][d['val']])
            assert len(cp['bundle']['calibration_audit'])==3 and all(a['patient_disjoint'] and a['all_preprocessing_subFIT_only'] for a in cp['bundle']['calibration_audit'])
        else:
            m=make(arm,k);assert state_hash(m.state_dict())==r['initial_sha256']
            cp=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);m.load_state_dict(cp['model'])
            score=predict(m,arm,d,d['val'],store);best=max(v['history'],key=lambda h:(h['macro_f1'],h['ez_f1'],-h['epoch']))
            assert best['epoch']==r['epoch'] and best['threshold']==r['threshold']
            del m;gc.collect();torch.cuda.empty_cache()
        assert np.array_equal(score,v['score']) and select(arm,d,score)==v['selection']
        checks.append(dict(arm=arm,held_source=SOURCES[k-1],validation_replay_max_error=0.,preprocessing_FIT_only=True,threshold_exact=True,old_fitted_artifacts_reused=False))
    csv_out('VALIDATION_REPLAY',checks)
    write(PUB/'INTERNAL_VERIFICATION.json',dict(status='PASS',cells=20,all_validation_scores_bit_exact=True,held_sources_excluded=True))
    freeze(ROOT/'FROZEN_BEFORE_TEST_PRIVATE.json',dict(**q,internal_verification=sha(PUB/'INTERNAL_VERIFICATION.json')))
    write(PUB/'FREEZE_STATUS.json',dict(status='ALL20_FROZEN_BEFORE_TEST',freeze_sha256=sha(ROOT/'FROZEN_BEFORE_TEST_PRIVATE.json')))

def score_test():
    binding,rec,banks=setup();fr=frozen();assert TRAIN_SEED in [45,46,47,48,49];store=io.RawStore();meta=rec['meta'];sealed={};coverage=[]
    for r in fr['cells']:
        arm,k=r['arm'],r['held_index'];train=banks[k-1];ix=np.flatnonzero(np.isin(meta['patient'],sorted(train['roles']['test'])))
        assert set(meta['center'][ix])=={SOURCES[k-1]}
        key=f'{arm}_held{k}';p=ROOT/'scores_private'/(key+'.pkl')
        if p.exists():q=read(p)
        else:
            x=feature_query(rec['R14'][ix],meta['patient'][ix],train['pre'])
            d=dict(x=x,patient=meta['patient'][ix],channel=meta['channel'][ix])
            if arm=='LINEAR_SVM_FEATURE_PR':
                cp=read(r['checkpoint']);prob=svm.predict(cp['bundle'],rec['R14'][ix],meta['patient'][ix]);logit=None
            else:
                m=make(arm,k);cp=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);m.load_state_dict(cp['model'])
                logit=predict(m,arm,d,np.arange(len(ix)),store);prob=1-pn(logit).astype(float)
                del m;gc.collect();torch.cuda.empty_cache()
            pred=(pn(logit)<r['threshold']) if r['rule']=='NEZ_LT' else prob>=r['threshold']
            q=dict(freeze=sha(ROOT/'FROZEN_BEFORE_TEST_PRIVATE.json'),checkpoint=r['checkpoint_sha256'],indices=ix,probability_ez=prob,prediction_ez=pred,logit_nez=logit,threshold=r['threshold'],rule=r['rule'])
            save(p,q)
        assert q['freeze']==sha(ROOT/'FROZEN_BEFORE_TEST_PRIVATE.json') and q['checkpoint']==r['checkpoint_sha256'] and np.array_equal(q['indices'],ix)
        assert 'y' not in q and q['threshold']==r['threshold'];sealed[key]=sha(p)
        nw=0
        if arm in ['OURS_RAW_FEATURE','OURS_RAW','TIMECONVCNN_RAW']:
            for pid in sorted(set(meta['patient'][ix])):
                jj=ix[meta['patient'][ix]==pid];raw,mask,times=store.get(pid,meta['channel'][jj].astype(str).tolist());nw+=int(mask.sum())
        coverage.append(dict(arm=arm,held_source=SOURCES[k-1],patients=len(set(meta['patient'][ix])),channels=len(ix),valid_raw_windows=nw,raw_consumed=arm in ['OURS_RAW_FEATURE','OURS_RAW','TIMECONVCNN_RAW']))
        print('LOCO_SCORE_SEALED',arm,k,flush=True)
    assert len(sealed)==20
    for arm in ARMS:
        rows=[r for r in coverage if r['arm']==arm];assert sum(r['patients'] for r in rows)==80 and sum(r['channels'] for r in rows)==7635
        if rows[0]['raw_consumed']:assert sum(r['valid_raw_windows'] for r in rows)==1471965
    freeze(ROOT/'TEST_SCORE_SEAL_PRIVATE.json',dict(binding=binding,freeze=sha(ROOT/'FROZEN_BEFORE_TEST_PRIVATE.json'),files=sealed))
    csv_out('COVERAGE',coverage);write(PUB/'PREDICTION_STATUS.json',dict(status='ALL20_TEST_SCORES_SEALED',cells=20,seal_sha256=sha(ROOT/'TEST_SCORE_SEAL_PRIVATE.json')))

def all_scores():
    fr=frozen();seal=jr(ROOT/'TEST_SCORE_SEAL_PRIVATE.json')
    assert seal['binding']==CTX['binding'] and seal['freeze']==sha(ROOT/'FROZEN_BEFORE_TEST_PRIVATE.json') and len(seal['files'])==20
    meta=CTX['rec']['meta'];out=[]
    for r in fr['cells']:
        key=f"{r['arm']}_held{r['held_index']}";p=ROOT/'scores_private'/(key+'.pkl');assert sha(p)==seal['files'][key]
        q=read(p);ix=q['indices'];assert q['checkpoint']==r['checkpoint_sha256']
        out.append(pd.DataFrame(dict(arm=r['arm'],held_source=r['held_source'],patient=meta['patient'][ix],channel=meta['channel'][ix],
                                    y=1-labels(ix),probability=q['probability_ez'],prediction=q['prediction_ez'])))
    return pd.concat(out,ignore_index=True)

def patient_metrics(frame):
    rows=[]
    for (arm,held,pid),g in frame.groupby(['arm','held_source','patient'],sort=True):
        v=dict(**EV.classification(g.y,g.probability,g.prediction),**EV.ranking(g.y,g.probability,g.channel))
        rows.append(dict(arm=arm,held_source=held,patient=pid,**{m:v[m] for m in METRICS}))
    assert len(rows)==400 and all(np.isfinite(r[m]) for r in rows for m in METRICS)
    return pd.DataFrame(rows)

def bootstrap_draws(sources):
    sources=np.asarray(sources);draw=np.empty((10000,len(sources)),int);rng=np.random.default_rng(42)
    for s in SOURCES:
        ix=np.flatnonzero(sources==s)
        if len(ix):draw[:,ix]=ix[rng.integers(0,len(ix),size=(10000,len(ix)))]
    return draw

def boot(values,draw):
    return np.concatenate([values[draw[i:i+250]].mean(1) for i in range(0,len(draw),250)])

def tables(frame,patients):
    tables={};main=[];centers=[];pooled=[];sourceequal=[];sourcesd=[]
    for arm in ARMS:
        g=patients[patients.arm==arm]
        main.append(dict(arm=arm,patients=80,**{m:float(g[m].mean()) for m in METRICS}))
        cm=[]
        for s in SOURCES:
            h=g[g.held_source==s];v={m:float(h[m].mean()) for m in METRICS};cm.append(v)
            centers.append(dict(arm=arm,held_source=s,patients=len(h),**v))
        sourceequal.append(dict(arm=arm,sources=4,**{m:float(np.mean([r[m] for r in cm])) for m in METRICS}))
        sourcesd.append(dict(arm=arm,sources=4,**{m:float(np.std([r[m] for r in cm],ddof=1)) for m in METRICS}))
        for s in ['ALL']+SOURCES:
            h=frame[(frame.arm==arm)&((frame.held_source==s) if s!='ALL' else True)]
            v=EV.classification(h.y,h.probability,h.prediction)
            # Ranking is patient-local; there is no meaningful cross-patient pooled MRR.
            pooled.append(dict(arm=arm,held_source=s,channels=len(h),**{m:v[m] for m in METRICS if m not in ['mrr','hit_at_1']}))
    tables.update(MAIN_RESULTS=main,SOURCE_RESULTS=centers,SOURCE_EQUAL_RESULTS=sourceequal,SOURCE_SD=sourcesd,POOLED_RESULTS=pooled)
    intervals=[];family=[]
    for scope in ['ALL']+SOURCES:
        sub=patients if scope=='ALL' else patients[patients.held_source==scope]
        first=sub[sub.arm==ARMS[0]].sort_values('patient');draw=bootstrap_draws(first.held_source.to_numpy());ids=first.patient.tolist()
        values={a:sub[sub.arm==a].set_index('patient').loc[ids,METRICS].to_numpy() for a in ARMS}
        comparisons={a:v for a,v in values.items()}
        comparisons.update({'FULL_MINUS_'+a:values[ARMS[0]]-values[a] for a in ARMS[1:]})
        for name,value in comparisons.items():
            dist=boot(value,draw);ci=np.quantile(dist,[.025,.975],axis=0)
            for j,m in enumerate(METRICS):
                intervals.append(dict(scope=scope,comparison=name,metric=m,estimate=float(value[:,j].mean()),ci_low=float(ci[0,j]),ci_high=float(ci[1,j]),patients=len(ids),draws=10000,seed=42))
            if scope=='ALL' and name.startswith('FULL_MINUS_'):
                ci=np.quantile(dist,[.05/16,1-.05/16],axis=0)
                for m in ['macro_f1','auroc']:
                    j=METRICS.index(m);family.append(dict(comparison=name,metric=m,estimate=float(value[:,j].mean()),ci_low=float(ci[0,j]),ci_high=float(ci[1,j]),family_tests=8,draws=10000,seed=42))
    assert len(intervals)==405 and len(family)==8
    tables.update(INTERVALS=intervals,PRIMARY_FAMILY_INTERVALS=family)
    return tables

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','verify_internal','score_test']);args=parser.parse_args()
    globals()[args.stage]()
