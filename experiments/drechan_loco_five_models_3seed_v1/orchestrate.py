"""Phase barriers, strict historical reuse, and seed-averaged endpoint reporting."""
import argparse
import subprocess
import zipfile
import pandas as pd
from common import *
import common as C
import pipeline as P

OLD_CODE=Path(r'D:\drechan_loco_five_models_seed42_v1')
OLD_ROOT=Path(r'D:\drechan_loco_five_models_seed42_v1_runtime')
GLOBAL_PUB=BASE_ROOT/'public'
SEEDS=[42,43,44]
STATUS='COMPLETE_VERIFIED_LOCO_5_MODELS_4_SOURCES_3SEED'
BASIC_ALLOW=['REUSE_AUDIT.json','SEED_RESULTS.csv','SOURCE_SEED_RESULTS.csv','SEED_MEAN_SD.csv',
             'SOURCE_SEED_SD.csv','MAIN_RESULTS.csv','SOURCE_RESULTS.csv','SOURCE_EQUAL_RESULTS.csv',
             'SOURCE_SD.csv','POOLED_SEED_RESULTS.csv','INTERVALS.csv','PRIMARY_FAMILY_INTERVALS.csv',
             'TRAIN_CELLS.csv','VALIDATION_REPLAY.csv','COVERAGE.csv','SOURCE_PUSH_BEFORE_TRAIN.json',
             'FREEZE_STATUS.json','PREDICTION_STATUS.json','REPORT.md','REPORT_STATUS.json',
             'INDEPENDENT_VERIFICATION.json','OUTER_STATUS.json']
ADMISSION_ALLOW=['ADMISSION.json','ROLE_AUDIT.csv','INPUT_AUDIT.json','INITIALIZATION_AUDIT.csv']

def init():
    setup();GLOBAL_PUB.mkdir(parents=True,exist_ok=True)

def seed_root(seed):return OLD_ROOT if seed==42 else BASE_ROOT/f'seed{seed}'

def reuse_gate():
    lock=jr(HERE/'PARENT_LOCK.json')
    for n,h in lock['source'].items():assert sha(OLD_CODE/n)==h,('PARENT_SOURCE_CHANGED',n)
    for n,h in {**lock['dependencies'],**lock['extra_dependencies']}.items():assert sha(Path(n))==h,('PARENT_DEPENDENCY_CHANGED',n)
    assert sha(OLD_ROOT/'FROZEN_BEFORE_TEST_PRIVATE.json')==lock['freeze_sha256']
    assert sha(OLD_ROOT/'TEST_SCORE_SEAL_PRIVATE.json')==lock['seal_sha256']
    assert sha(OLD_ROOT/'public_aggregate_export_v1.zip')==lock['archive_sha256']
    old=jr(OLD_ROOT/'public/OUTER_STATUS.json');assert old['status']=='COMPLETE_VERIFIED_LOCO_5_MODELS_4_SOURCES_SEED42'
    fr=jr(OLD_ROOT/'FROZEN_BEFORE_TEST_PRIVATE.json');seal=jr(OLD_ROOT/'TEST_SCORE_SEAL_PRIVATE.json')
    assert len(fr['cells'])==len(seal['files'])==20 and seal['binding']==fr['binding']
    assert fr['binding']['archive']==CTX['binding']['archive'] and fr['binding']['reconstruction']==CTX['binding']['reconstruction']
    oldroles=jr(OLD_ROOT/'ROLES_PRIVATE.json')['roles']
    assert oldroles==[{n:sorted(v) for n,v in r.items()} for r in CTX['roles']]
    cells=[]
    for r in fr['cells']:
        arm,k=r['arm'],r['held_index'];d=CTX['banks'][k-1]
        assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['validation'])==r['validation_sha256']
        v=read(r['validation']);assert v['binding']['input']==fr['binding']
        assert v['binding']['seed']==42+1009*k and v['binding']['arm']==arm
        assert r['preprocessor_sha256']==digest(io.clean(d['pre']))
        assert np.array_equal(v['indices'],d['archive_indices'][d['val']]) and np.isfinite(v['score']).all()
        key=f'{arm}_held{k}';scorepath=OLD_ROOT/'scores_private'/(key+'.pkl')
        assert sha(scorepath)==seal['files'][key]
        q=read(scorepath);ix=np.flatnonzero(np.isin(CTX['rec']['meta']['patient'],sorted(d['roles']['test'])))
        assert np.array_equal(q['indices'],ix) and q['freeze']==lock['freeze_sha256'] and q['checkpoint']==r['checkpoint_sha256']
        assert 'y' not in q and q['threshold']==r['threshold'] and q['rule']==r['rule']
        if TRAIN_SEED==42 and arm!='LINEAR_SVM_FEATURE_PR':
            m=make(arm,k);assert state_hash(m.state_dict())==r['initial_sha256']
            del m;gc.collect();torch.cuda.empty_cache()
        cells.append(dict(arm=arm,held_source=SOURCES[k-1],seed=42,initial_or_SVM_recipe_exact=True,
                          roles_exact=True,FIT_preprocessor_exact=True,selection_hash_exact=True,sealed_score_hash_exact=True))
    return dict(status='PASS',cells=20,parent_commit=lock['parent_commit'],old_predictions_reused_not_recomputed=True,
                historical_results_already_viewed=True,old_source_and_dependencies_immutable=True,
                old_freeze_sha256=lock['freeze_sha256'],old_seal_sha256=lock['seal_sha256'],
                old_public_archive_sha256=lock['archive_sha256'],checks=cells)

def admission_package():
    init();reuse=reuse_gate();write(GLOBAL_PUB/'REUSE_AUDIT.json',reuse)
    files={'REUSE_AUDIT.json':GLOBAL_PUB/'REUSE_AUDIT.json'}
    for seed in SEEDS:
        root=BASE_ROOT/f'seed{seed}'
        a=jr(root/'public/ADMISSION.json');assert a['status']=='PASS' and a['source']==sources()
        assert a['binding']['train_seed']==seed and a['binding']['split_seed']==42 and a['initialization_cells']==16
        for n in ADMISSION_ALLOW:files[f'seed{seed}/{n}']=root/'public'/n
    manifest=dict(source=sources(),files={n:sha(p) for n,p in files.items()},schema='LOCO5_3SEED_ADMISSION_V1',private_artifacts_exported=False)
    write(BASE_ROOT/'ADMISSION_PACKAGE_MANIFEST.json',manifest)
    p=BASE_ROOT/'admission_public_v1.zip'
    with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
        for n,fp in files.items():z.write(fp,n)
        z.write(BASE_ROOT/'ADMISSION_PACKAGE_MANIFEST.json','PACKAGE_MANIFEST.json')
    print('MULTISEED_ADMISSION_PACKAGE',sha(p),p.stat().st_size,flush=True)

def freeze_all():
    init();reuse=reuse_gate();write(GLOBAL_PUB/'REUSE_AUDIT.json',reuse);records={}
    for seed in SEEDS:
        r=seed_root(seed);fr=jr(r/'FROZEN_BEFORE_TEST_PRIVATE.json')
        assert len(fr['cells'])==20
        if seed!=42:
            assert fr['binding']['source']==sources() and fr['binding']['train_seed']==seed
            assert jr(r/'public/INTERNAL_VERIFICATION.json')['status']=='PASS'
        for q in fr['cells']:
            assert sha(q['checkpoint'])==q['checkpoint_sha256'] and sha(q['validation'])==q['validation_sha256']
        records[str(seed)]=sha(r/'FROZEN_BEFORE_TEST_PRIVATE.json')
    commit=os.environ.get('LOCO_SOURCE_COMMIT','');assert __import__('re').fullmatch('[0-9a-f]{40}',commit)
    for seed in [43,44]:
        pushed=jr(seed_root(seed)/'public/SOURCE_PUSH_BEFORE_TRAIN.json')
        assert pushed['commit']==commit and pushed['source']==sources() and pushed['pushed_before_fit']
    write(GLOBAL_PUB/'SOURCE_PUSH_BEFORE_TRAIN.json',dict(source_commit=commit,source=sources(),all_new_fits_require_same_source_commit=True))
    freeze(BASE_ROOT/'ALL60_SELECTIONS_FROZEN_PRIVATE.json',dict(status='ALL60_FROZEN_BEFORE_NEW_TEST',source=sources(),records=records,cells=60,reused=20,fresh=40))
    write(GLOBAL_PUB/'FREEZE_STATUS.json',dict(status='ALL60_FROZEN_BEFORE_NEW_TEST',cells=60,sha256=sha(BASE_ROOT/'ALL60_SELECTIONS_FROZEN_PRIVATE.json')))

def seal_all():
    init();fr=jr(BASE_ROOT/'ALL60_SELECTIONS_FROZEN_PRIVATE.json');assert fr['source']==sources();records={}
    for seed in SEEDS:
        root=seed_root(seed);assert sha(root/'FROZEN_BEFORE_TEST_PRIVATE.json')==fr['records'][str(seed)]
        seal=jr(root/'TEST_SCORE_SEAL_PRIVATE.json');assert len(seal['files'])==20 and seal['freeze']==fr['records'][str(seed)]
        for n,h in seal['files'].items():assert sha(root/'scores_private'/(n+'.pkl'))==h
        records[str(seed)]=sha(root/'TEST_SCORE_SEAL_PRIVATE.json')
    freeze(BASE_ROOT/'ALL60_TEST_SCORES_SEALED_PRIVATE.json',dict(status='ALL60_SEALED_BEFORE_METRICS',source=sources(),freeze=sha(BASE_ROOT/'ALL60_SELECTIONS_FROZEN_PRIVATE.json'),records=records))
    write(GLOBAL_PUB/'PREDICTION_STATUS.json',dict(status='ALL60_SEALED_BEFORE_METRICS',cells=60,sha256=sha(BASE_ROOT/'ALL60_TEST_SCORES_SEALED_PRIVATE.json')))

def frames():
    seal=jr(BASE_ROOT/'ALL60_TEST_SCORES_SEALED_PRIVATE.json');fr=jr(BASE_ROOT/'ALL60_SELECTIONS_FROZEN_PRIVATE.json')
    assert seal['source']==sources() and seal['freeze']==sha(BASE_ROOT/'ALL60_SELECTIONS_FROZEN_PRIVATE.json')
    meta=CTX['rec']['meta'];rows=[]
    for seed in SEEDS:
        root=seed_root(seed);assert sha(root/'TEST_SCORE_SEAL_PRIVATE.json')==seal['records'][str(seed)]
        q=jr(root/'TEST_SCORE_SEAL_PRIVATE.json');sel=jr(root/'FROZEN_BEFORE_TEST_PRIVATE.json')
        assert sha(root/'FROZEN_BEFORE_TEST_PRIVATE.json')==fr['records'][str(seed)]
        for r in sel['cells']:
            key=f"{r['arm']}_held{r['held_index']}";p=root/'scores_private'/(key+'.pkl');assert sha(p)==q['files'][key]
            v=read(p);ix=v['indices'];assert v['checkpoint']==r['checkpoint_sha256'] and v['freeze']==fr['records'][str(seed)]
            rows.append(pd.DataFrame(dict(train_seed=seed,arm=r['arm'],held_source=r['held_source'],patient=meta['patient'][ix],
                                         channel=meta['channel'][ix],y=1-labels(ix),probability=v['probability_ez'],prediction=v['prediction_ez'])))
    return pd.concat(rows,ignore_index=True)

def endpoint_rows(frame,independent=False):
    rows=[]
    if independent:
        from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,accuracy_score
    for (seed,arm,held,pid),g in frame.groupby(['train_seed','arm','held_source','patient'],sort=True):
        if not independent:
            v=dict(**EV.classification(g.y,g.probability,g.prediction),**EV.ranking(g.y,g.probability,g.channel))
        else:
            y=g.y.to_numpy().astype(int);pr=g.probability.to_numpy();pred=g.prediction.to_numpy();assert set(y)=={0,1}
            f=f1_score(y,pred,labels=[0,1],average=None,zero_division=0)
            order=sorted(range(len(g)),key=lambda i:(-float(pr[i]),str(g.channel.iloc[i])))
            rank=next(j for j,i in enumerate(order,1) if y[i])
            v=dict(macro_f1=float(f.mean()),auroc=roc_auc_score(y,pr),auprc=average_precision_score(y,pr),f1=float(f[1]),nez_f1=float(f[0]),
                   accuracy=accuracy_score(y,pred),mrr=1/rank,hit_at_1=float(rank==1),ez_fraction_bias=float(np.mean(pred)-np.mean(y)))
        rows.append(dict(train_seed=int(seed),arm=arm,held_source=held,patient=pid,**{m:float(v[m]) for m in METRICS}))
    assert len(rows)==1200 and all(np.isfinite(r[m]) for r in rows for m in METRICS)
    return pd.DataFrame(rows)

def combined_tables(frame,patients):
    assert set(patients.train_seed)==set(SEEDS)
    # Equal seed averaging of each patient's endpoints, NOT probabilities.
    count=patients.groupby(['arm','held_source','patient']).size();assert len(count)==400 and (count==3).all()
    average=patients.groupby(['arm','held_source','patient'],as_index=False)[METRICS].mean()
    tables=P.tables(frame[frame.train_seed==42],average);tables.pop('POOLED_RESULTS')
    seedrows=[];sourcerows=[];pooled=[]
    for seed in SEEDS:
        t=P.tables(frame[frame.train_seed==seed],patients[patients.train_seed==seed])
        seedrows.extend([dict(train_seed=seed,**r) for r in t['MAIN_RESULTS']])
        sourcerows.extend([dict(train_seed=seed,**r) for r in t['SOURCE_RESULTS']])
        pooled.extend([dict(train_seed=seed,**r) for r in t['POOLED_RESULTS']])
    def seed_sd(rows,by):
        f=pd.DataFrame(rows);out=[]
        for key,g in f.groupby(by,sort=True):
            keys=key if isinstance(key,tuple) else (key,)
            assert len(g)==3
            out.append(dict(zip(by,keys),seeds=3,**{m+'_mean':float(g[m].mean()) for m in METRICS},
                            **{m+'_sample_sd':float(g[m].std(ddof=1)) for m in METRICS}))
        return out
    tables.update(SEED_RESULTS=seedrows,SOURCE_SEED_RESULTS=sourcerows,POOLED_SEED_RESULTS=pooled,
                  SEED_MEAN_SD=seed_sd(seedrows,['arm']),SOURCE_SEED_SD=seed_sd(sourcerows,['arm','held_source']))
    return tables

def export_csv(name,rows):
    original=P.PUB
    try:P.PUB=GLOBAL_PUB;P.csv_out(name,rows)
    finally:P.PUB=original

def ancillary():
    for name in ['TRAIN_CELLS','VALIDATION_REPLAY','COVERAGE']:
        rows=[]
        for seed in SEEDS:
            t=pd.read_csv(seed_root(seed)/'public'/(name+'.csv'))
            for r in t.to_dict('records'):
                r['train_seed']=seed
                if name=='TRAIN_CELLS':r['reused']=(seed==42)
                rows.append(r)
        assert len(rows)==60;export_csv(name,rows)

def report():
    init();frame=frames();p=endpoint_rows(frame);save(BASE_ROOT/'ALL_PATIENT_METRICS_PRIVATE.pkl',p)
    out=combined_tables(frame,p)
    for n,rs in out.items():export_csv(n,rs)
    # Seed42 historical public means must reconstruct exactly within serialization tolerance.
    old=pd.read_csv(OLD_ROOT/'public/MAIN_RESULTS.csv').set_index('arm')
    for r in out['SEED_RESULTS']:
        if r['train_seed']==42:assert all(abs(r[m]-old.loc[r['arm'],m])<1e-12 for m in METRICS)
    ancillary()
    text=['# Five-model four-source LOCO: seeds42/43/44','',
          'All60 logical selections frozen before any new test; all60 sealed score cells before metrics.20 strict historical seed42 reuses,40 fresh fits.',
          'Original seed42 patient roles held FIXED. Neural initialization/order/dropout and SVM OOF calibration vary with training seed.',
          'Primary: mean3 endpoints for each patient, then mean80patients; NOT bestseed or a probability ensemble.',
          'Exploratory repeatedly viewed historical cohort; three seeds do not confer external confirmation.','',
          '| Model | Macro-F1 | AUROC | AP | EZ-F1 | NEZ-F1 | ACC | MRR | Top-1 | EZ-Frac. Bias |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in out['MAIN_RESULTS']:text.append('| '+r['arm']+' | '+' | '.join(f'{r[m]:.6f}' for m in METRICS)+' |')
    (GLOBAL_PUB/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    write(GLOBAL_PUB/'REPORT_STATUS.json',dict(status='ALL5_MODELS_3SEED_REPORTED',patient_rows=1200,metric_values=10800))

def verify():
    init();frame=frames();expected=endpoint_rows(frame,True);actual=read(BASE_ROOT/'ALL_PATIENT_METRICS_PRIVATE.pkl')
    keys=['train_seed','arm','held_source','patient']
    a=actual.set_index(keys).sort_index();e=expected.set_index(keys).sort_index()
    assert a.index.equals(e.index) and np.max(np.abs(a[METRICS].to_numpy()-e[METRICS].to_numpy()))<1e-12
    tables=combined_tables(frame,expected);maximum=0.
    for name,data in tables.items():
        observed=pd.read_csv(GLOBAL_PUB/(name+'.csv'));assert len(observed)==len(data)
        for i,r in enumerate(data):
            for n,v in r.items():
                if isinstance(v,(int,float,np.number)):
                    err=abs(float(observed.iloc[i][n])-float(v));maximum=max(maximum,err);assert err<1e-12,(name,i,n)
                else:assert observed.iloc[i][n]==v
    # Independent counts/matmul bootstrap on independently recomputed seed-averaged endpoints.
    avg=expected.groupby(['arm','held_source','patient'],as_index=False)[METRICS].mean()
    nom=pd.read_csv(GLOBAL_PUB/'INTERVALS.csv');fam=pd.read_csv(GLOBAL_PUB/'PRIMARY_FAMILY_INTERVALS.csv')
    nominal=family=0
    for scope in ['ALL']+SOURCES:
        sub=avg if scope=='ALL' else avg[avg.held_source==scope]
        ref=sub[sub.arm==ARMS[0]].sort_values('patient');ids=ref.patient.tolist();centers=ref.held_source.to_numpy();n=len(ids)
        rng=np.random.default_rng(42);counts=np.zeros((10000,n),np.float64)
        for source in SOURCES:
            ix=np.flatnonzero(centers==source)
            if len(ix):
                sampled=ix[rng.integers(0,len(ix),size=(10000,len(ix)))]
                np.add.at(counts,(np.arange(10000)[:,None],sampled),1.)
        assert np.all(counts.sum(1)==n)
        values={arm:sub[sub.arm==arm].set_index('patient').loc[ids,METRICS].to_numpy() for arm in ARMS}
        for name in ARMS+['FULL_MINUS_'+arm for arm in ARMS[1:]]:
            v=values[name] if name in values else values[ARMS[0]]-values[name[len('FULL_MINUS_'):]]
            dist=counts@v/n;ci=np.percentile(dist,[2.5,97.5],axis=0)
            for j,m in enumerate(METRICS):
                r=nom[(nom.scope==scope)&(nom.comparison==name)&(nom.metric==m)];assert len(r)==1
                assert np.max(np.abs(r[['ci_low','ci_high']].to_numpy()[0]-ci[:,j]))<1e-12;nominal+=1
            if scope=='ALL' and name.startswith('FULL_MINUS_'):
                ci=np.quantile(dist,[.05/16,1-.05/16],axis=0)
                for m in ['macro_f1','auroc']:
                    j=METRICS.index(m);r=fam[(fam.comparison==name)&(fam.metric==m)];assert len(r)==1
                    assert np.max(np.abs(r[['ci_low','ci_high']].to_numpy()[0]-ci[:,j]))<1e-12;family+=1
    assert nominal==405 and family==8
    ancillary()
    cells=pd.read_csv(GLOBAL_PUB/'TRAIN_CELLS.csv')
    assert len(cells)==60 and cells.reused.sum()==20 and set(zip(cells.train_seed,cells.arm,cells.held_index))=={(s,a,k) for s in SEEDS for a in ARMS for k in range(1,5)}
    check=dict(status='PASS',patient_rows=1200,metric_values=10800,nominal_intervals=405,family_intervals=8,cells=60,fresh=40,reused=20,
               maximum_numeric_error=maximum,patient_roles_fixed=True,seed_metric_mean_not_prediction_ensemble=True)
    write(GLOBAL_PUB/'INDEPENDENT_VERIFICATION.json',check)
    write(GLOBAL_PUB/'OUTER_STATUS.json',{**check,'status':STATUS,
          'freeze_sha256':sha(BASE_ROOT/'ALL60_SELECTIONS_FROZEN_PRIVATE.json'),'seal_sha256':sha(BASE_ROOT/'ALL60_TEST_SCORES_SEALED_PRIVATE.json')})

def package():
    init();assert jr(GLOBAL_PUB/'OUTER_STATUS.json')['status']==STATUS
    files={n:GLOBAL_PUB/n for n in BASIC_ALLOW}
    for seed in SEEDS:
        for n in ADMISSION_ALLOW:files[f'admission_seed{seed}/{n}']=BASE_ROOT/f'seed{seed}'/'public'/n
    manifest=dict(files={n:sha(p) for n,p in files.items()},source=sources(),schema='LOCO5_3SEED_V1',private_artifacts_exported=False)
    write(GLOBAL_PUB/'PACKAGE_MANIFEST.json',manifest);p=BASE_ROOT/'public_aggregate_export_v1.zip'
    with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
        for n,fp in files.items():z.write(fp,n)
        z.write(GLOBAL_PUB/'PACKAGE_MANIFEST.json','PACKAGE_MANIFEST.json')
    print('LOCO3_PUBLIC_PACKAGE',sha(p),p.stat().st_size,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['admission_package','freeze_all','seal_all','report','verify','package']);args=parser.parse_args()
    globals()[args.stage]()
