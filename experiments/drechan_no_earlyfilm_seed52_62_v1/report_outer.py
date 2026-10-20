"""Two requested seeds: equal-patient metrics, never best-seed selection."""
from common import *
def table(name):
    with (PUB/(name+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def load_rows():
    fr=frozen();fh=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json');seal=jr(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')
    assert seal['freeze']==fh and len(seal['files'])==10 and seal['all10_cells_sealed_before_metric_reporting']
    ctx=read(ROOT/'METADATA_PRIVATE.pkl');meta=ctx['meta']
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert sha(archive)==ctx['archive_sha256']
    with np.load(archive,allow_pickle=True) as q:y=1-q['y']
    assert len(y)==7635 and int(y.sum())==1743
    rows=[];pool={s:dict(y=[],p=[],v=[]) for s in SEEDS}
    for r in fr['cells']:
        seed,fold=r['seed'],r['fold'];key=key_for(r['arm'],seed,fold);p=ROOT/'scores_private'/(key+'.pkl')
        assert sha(p)==seal['files'][key];q=read(p);ii=q['indices'];pp=1-q['score_nez'].astype(float);vv=q['score_nez']<r['threshold']
        assert q['freeze']==fh and q['checkpoint']==r['checkpoint_sha256'] and q['threshold']==r['threshold']
        for pid in sorted(set(meta['patient'][ii])):
            jj=meta['patient'][ii]==pid
            vals=dict(**EV.classification(y[ii][jj],pp[jj],vv[jj]),**EV.ranking(y[ii][jj],pp[jj],meta['channel'][ii][jj]))
            assert all(np.isfinite(vals[m]) for m in METRICS)
            rows.append(dict(arm=r['arm'],seed=seed,fold=fold,patient=str(pid),center=str(meta['center'][ii][jj][0]),
                             channels=int(jj.sum()),**{m:float(vals[m]) for m in METRICS}))
        for k,v in [('y',y[ii]),('p',pp),('v',vv)]:pool[seed][k].append(v)
    assert len(rows)==160
    return rows,pool
def matrices(rows):
    ids=sorted({r['patient'] for r in rows});assert len(ids)==80;out={}
    for s in SEEDS:
        z=sorted([r for r in rows if r['seed']==s],key=lambda r:r['patient'])
        assert [r['patient'] for r in z]==ids and sum(r['channels'] for r in z)==7635
        out[s]=vector(z)
    return out
def main():
    rows,pool=load_rows();mat=matrices(rows);mean=np.mean([mat[s] for s in SEEDS],axis=0)
    seedrows=[];folds=[];foldsd=[];sources=[];source_seed=[];coverage=[]
    for s in SEEDS:
        rr=[r for r in rows if r['seed']==s];seedrows.append(dict(arm=ARMS[0],seed=s,patients=80,**dict(zip(METRICS,mat[s].mean(0).tolist()))))
        ff=[]
        for f in range(1,6):
            z=[r for r in rr if r['fold']==f];point=vector(z).mean(0);ff.append(point)
            folds.append(dict(arm=ARMS[0],seed=s,fold=f,patients=len(z),**dict(zip(METRICS,point.tolist()))))
        foldsd.append(dict(arm=ARMS[0],seed=s,**dict(zip(METRICS,np.std(ff,axis=0,ddof=1).tolist()))))
        for c in sorted({r['center'] for r in rr}):
            z=[r for r in rr if r['center']==c];n=len(z)
            source_seed.append(dict(arm=ARMS[0],seed=s,center=c,patients=n,suppressed=n<5,
                **dict(zip(METRICS,vector(z).mean(0).tolist() if n>=5 else [float('nan')]*9))))
        coverage.extend(dict(arm=ARMS[0],seed=s,metric=m,finite_patients=int(np.isfinite(mat[s][:,j]).sum()),patients=80) for j,m in enumerate(METRICS))
    for c in sorted({r['center'] for r in rows}):
        z=[r for r in rows if r['center']==c];n=len(z)//2
        sources.append(dict(arm=ARMS[0],center=c,seeds=2,patients=n,suppressed=n<5,
            **dict(zip(METRICS,vector(z).mean(0).tolist() if n>=5 else [float('nan')]*9))))
    csv_out('MAIN_RESULTS',[dict(arm=ARMS[0],seeds=2,patients=80,**dict(zip(METRICS,mean.mean(0).tolist())))])
    csv_out('SEED_SD',[dict(arm=ARMS[0],seeds=2,**dict(zip(METRICS,np.std([mat[s].mean(0) for s in SEEDS],axis=0,ddof=1).tolist())))])
    for name,data in [('SEED_RESULTS',seedrows),('FOLD_RESULTS',folds),('FOLD_SD',foldsd),('SOURCE_RESULTS',sources),
                       ('SOURCE_SEED_RESULTS',source_seed),('METRIC_COVERAGE',coverage)]:csv_out(name,data)
    draws=np.random.default_rng(42).integers(0,80,(10000,80));ci=[]
    for label,values in [(f'SEED{s}',mat[s]) for s in SEEDS]+[('TWO_SEED_METRIC_MEAN',mean)]:
        point,lim=interval(values,draws)
        ci.extend(dict(estimate=label,metric=m,mean=float(point[j]),low=float(lim[0,j]),high=float(lim[1,j]),patients=80,draws=10000) for j,m in enumerate(METRICS))
    assert len(ci)==27;csv_out('INTERVALS',ci)
    pooled=[];poolmetrics=[m for m in METRICS if m not in ['mrr','hit_at_1']]
    for s,z in pool.items():
        yy,pp,vv=[np.concatenate(z[k]) for k in ['y','p','v']];v=EV.classification(yy,pp,vv)
        pooled.append(dict(arm=ARMS[0],seed=s,channels=len(yy),aggregation='POOLED_CHANNELS_PER_SEED_NOT_PRIMARY',
            **{m:float(v[m]) for m in poolmetrics},tp=int((vv&(yy==1)).sum()),tn=int((~vv&(yy==0)).sum()),
            fp=int((vv&(yy==0)).sum()),fn=int((~vv&(yy==1)).sum())))
    csv_out('POOLED_RESULTS',pooled)
    csv_out('POOLED_SEED_MEAN',[dict(arm=ARMS[0],seeds=2,channels_each=7635,aggregation='MEAN_OF_SEED_METRICS_NOT_ENSEMBLE',
        **{m:float(np.mean([r[m] for r in pooled])) for m in poolmetrics})])
    save(ROOT/'PATIENT_METRICS_PRIVATE.pkl',rows)
    write(PUB/'REPORT_STATUS.json',dict(status='NO_EARLY_SEEDS52_62_REPORTED',patient_rows=160,nominal_intervals=27,
        selected_seed=None,metric_seed_averaging_not_ensemble=True,no_matched_Full_52_62_run=True,
        scientific_status='EXPLORATORY_REPEATED_TEST',independent_confirmation=False,no_post_Outer_tuning=True))
    print('NEL2_REPORT_COMPLETE',flush=True)
if __name__=='__main__':main()
