"""Independent sklearn/rank/bootstrap reconstruction, no retraining or selection."""
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,accuracy_score
from common import *
def table(name):
    with (PUB/(name+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def equal_metrics(row,values):
    assert all(abs(float(row[m])-values[j])<1e-12 for j,m in enumerate(METRICS))
def main():
    fr=frozen();fh=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json');seal=jr(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')
    assert len(fr['cells'])==len(seal['files'])==10 and seal['freeze']==fh and seal['all10_cells_sealed_before_metric_reporting']
    ctx=read(ROOT/'METADATA_PRIVATE.pkl');meta=ctx['meta']
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert sha(archive)==ctx['archive_sha256']
    with np.load(archive,allow_pickle=True) as q:y=1-q['y']
    stored=read(ROOT/'PATIENT_METRICS_PRIVATE.pkl')
    lookup={(r['seed'],r['fold'],r['patient']):r for r in stored};assert len(lookup)==len(stored)==160
    rows=[];pooled={s:dict(y=[],p=[],v=[]) for s in SEEDS};numeric=0
    for r in fr['cells']:
        seed,fold=r['seed'],r['fold'];key=key_for(r['arm'],seed,fold);path=ROOT/'scores_private'/(key+'.pkl')
        assert sha(path)==seal['files'][key];q=read(path);ii=q['indices'];p=1-q['score_nez'].astype(float);v=q['score_nez']<r['threshold']
        assert q['checkpoint']==r['checkpoint_sha256'] and q['freeze']==fh and q['threshold']==r['threshold']
        state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)['model']
        assert state['keep7'][4]==0 and not any('conditioner' in n or n.startswith('gate.') or 'interaction' in n for n in state)
        for pid in sorted(set(meta['patient'][ii])):
            j=meta['patient'][ii]==pid;yy,pp,vv=y[ii][j],p[j],v[j];ch=meta['channel'][ii][j].astype(str)
            order=sorted(range(len(yy)),key=lambda k:(-float(pp[k]),ch[k]));first=next(k for k,i in enumerate(order) if yy[i])
            values=dict(macro_f1=f1_score(yy,vv,labels=[0,1],average='macro',zero_division=0),
                f1=f1_score(yy,vv,zero_division=0),nez_f1=f1_score(1-yy,1-vv.astype(int),zero_division=0),
                auroc=roc_auc_score(yy,pp),auprc=average_precision_score(yy,pp),accuracy=accuracy_score(yy,vv),
                mrr=1/(first+1),hit_at_1=float(first==0),ez_fraction_bias=float(vv.mean()-yy.mean()))
            row=lookup[(seed,fold,str(pid))]
            for m,val in values.items():assert abs(row[m]-val)<1e-12;numeric+=1
            assert row['arm']==ARMS[0] and row['center']==str(meta['center'][ii][j][0]) and row['channels']==int(j.sum())
            rows.append(dict(arm=ARMS[0],seed=seed,fold=fold,patient=str(pid),center=str(meta['center'][ii][j][0]),channels=int(j.sum()),**values))
        for k,a in [('y',y[ii]),('p',p),('v',v)]:pooled[seed][k].append(a)
    assert len(rows)==160 and numeric==1440
    ids=sorted({r['patient'] for r in rows});assert len(ids)==80
    mat={}
    for seed in SEEDS:
        rr=sorted([r for r in rows if r['seed']==seed],key=lambda r:r['patient'])
        assert [r['patient'] for r in rr]==ids and sum(r['channels'] for r in rr)==7635
        mat[seed]=vector(rr)
    mean=sum(mat.values())/2
    mainrows,seedrows,seedsd=table('MAIN_RESULTS'),table('SEED_RESULTS'),table('SEED_SD')
    folds,foldsd=table('FOLD_RESULTS'),table('FOLD_SD')
    assert len(mainrows)==len(seedsd)==1 and len(seedrows)==len(foldsd)==2 and len(folds)==10
    equal_metrics(mainrows[0],mean.mean(0));equal_metrics(seedsd[0],np.std([mat[s].mean(0) for s in SEEDS],axis=0,ddof=1))
    for seed in SEEDS:
        equal_metrics(next(r for r in seedrows if int(r['seed'])==seed),mat[seed].mean(0));ff=[]
        for fold in range(1,6):
            rr=[r for r in rows if r['seed']==seed and r['fold']==fold]
            assert len(rr)==[16,16,17,15,16][fold-1];val=vector(rr).mean(0);ff.append(val)
            r=next(r for r in folds if int(r['seed'])==seed and int(r['fold'])==fold)
            assert int(r['patients'])==len(rr);equal_metrics(r,val)
        equal_metrics(next(r for r in foldsd if int(r['seed'])==seed),np.std(ff,axis=0,ddof=1))
    for name,perseed in [('SOURCE_RESULTS',False),('SOURCE_SEED_RESULTS',True)]:
        for r in table(name):
            rr=[z for z in rows if z['center']==r['center'] and (not perseed or z['seed']==int(r['seed']))];n=len(rr) if perseed else len(rr)//2
            assert n==int(r['patients']) and (r['suppressed']=='True')==(n<5)
            if n>=5:equal_metrics(r,vector(rr).mean(0))
            else:assert all(np.isnan(float(r[m])) for m in METRICS)
    cov=table('METRIC_COVERAGE');assert len(cov)==18
    assert {(int(r['seed']),r['metric']) for r in cov}=={(s,m) for s in SEEDS for m in METRICS}
    assert all(int(r['finite_patients'])==int(r['patients'])==80 for r in cov)
    ci=table('INTERVALS');assert len(ci)==27;checks=0;maximum=0.
    draws=np.random.default_rng(42).integers(0,80,(10000,80))
    for label,values in [(f'SEED{s}',mat[s]) for s in SEEDS]+[('TWO_SEED_METRIC_MEAN',mean)]:
        for j,m in enumerate(METRICS):
            samples=values[:,j][draws].mean(1);lo,hi=np.quantile(samples,[.025,.975]);point=values[:,j].mean()
            r=next(r for r in ci if r['estimate']==label and r['metric']==m)
            error=max(abs(float(r['mean'])-point),abs(float(r['low'])-lo),abs(float(r['high'])-hi))
            assert error<1e-12 and int(r['patients'])==80 and int(r['draws'])==10000
            maximum=max(maximum,error);checks+=1
    assert checks==27
    poolrows=table('POOLED_RESULTS');assert len(poolrows)==2
    for r in poolrows:
        z=pooled[int(r['seed'])];yy,pp,vv=[np.concatenate(z[k]) for k in ['y','p','v']]
        vals=dict(macro_f1=f1_score(yy,vv,average='macro'),f1=f1_score(yy,vv),nez_f1=f1_score(1-yy,1-vv.astype(int)),
            auroc=roc_auc_score(yy,pp),auprc=average_precision_score(yy,pp),accuracy=accuracy_score(yy,vv),
            ez_fraction_bias=float(vv.mean()-yy.mean()),tp=int((vv&(yy==1)).sum()),tn=int((~vv&(yy==0)).sum()),
            fp=int((vv&(yy==0)).sum()),fn=int((~vv&(yy==1)).sum()))
        assert len(yy)==7635 and int(r['channels'])==7635 and all(abs(float(r[k])-val)<1e-12 for k,val in vals.items())
    pm=table('POOLED_SEED_MEAN');assert len(pm)==1
    for m in METRICS:
        if m not in ['mrr','hit_at_1']:assert abs(float(pm[0][m])-np.mean([float(r[m]) for r in poolrows]))<1e-12
    raw=table('RAW_COVERAGE');assert len(raw)==10
    assert {(int(r['seed']),int(r['fold'])) for r in raw}=={(s,f) for s in SEEDS for f in range(1,6)}
    for seed in SEEDS:
        z=[r for r in raw if int(r['seed'])==seed]
        assert sum(int(r['patients']) for r in z)==80 and sum(int(r['channels']) for r in z)==7635 and sum(int(r['valid_windows']) for r in z)==1471965
        for r in z:
            assert r['raw_consumed']=='True' and r['reused_scientific_cell']=='False'
            for k in ['relative_active_coordinates','relative_used_coordinates','global_fallback_coordinates',
                'relative_singleton_valid_windows','mismatch_changed_valid_windows','mismatch_eligible_valid_windows','mismatch_singleton_valid_windows']:assert int(r[k])==0
    assert jr(PUB/'REPORT_STATUS.json')['selected_seed'] is None
    write(PUB/'INDEPENDENT_OUTER_VERIFICATION.json',dict(status='PASS',patient_rows=160,patient_metric_checks=numeric,
        nominal_interval_checks=27,maximum_interval_difference=maximum,all10_model_selections_frozen=True,
        unique_patient_resampling=True,metric_seed_averaging_not_ensemble=True,no_post_Outer_tuning=True))
    write(PUB/'OUTER_STATUS.json',dict(status='COMPLETE_VERIFIED_NO_EARLY_SEEDS52_62_FIVE_FOLD_OUTER',arms=1,seeds=[52,62],
        patients_each=80,channels_each=7635,logical_model_cells=10,patient_rows=160,metric_checks=numeric,selected_seed=None,
        scientific_status='EXPLORATORY_REPEATED_TEST',independent_confirmation=False,freeze_sha256=fh,
        score_seal_sha256=sha(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')))
    print('NEL2_INDEPENDENT_VERIFICATION_PASS',flush=True)
if __name__=='__main__':main()
