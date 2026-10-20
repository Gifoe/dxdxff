"""Three-seed means of patient endpoints, never a score ensemble or seed search."""
from common import *


def table(name):
    with (PUB/(name+'.csv')).open(encoding='utf-8', newline='') as f: return list(csv.DictReader(f))


def load_rows():
    fr = frozen(); fh = sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json'); seal = jr(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')
    assert seal['freeze'] == fh and len(seal['files']) == 285 and seal['all285_cells_sealed_before_metric_reporting']
    ctx = read(ROOT/'METADATA_PRIVATE.pkl'); meta = ctx['meta']
    archive = Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert sha(archive) == ctx['archive_sha256']
    with np.load(archive, allow_pickle=True) as q: y = 1-q['y']
    assert len(y) == 7635 and int(y.sum()) == 1743
    rows, pool = [], {(a,s):dict(y=[],p=[],v=[]) for a in ARMS for s in SEEDS}
    for r in fr['cells']:
        arm, seed, fold = r['arm'], r['seed'], r['fold']; key = key_for(arm,seed,fold)
        path = ROOT/'scores_private'/(key+'.pkl'); assert sha(path) == seal['files'][key]
        q = read(path); ii = q['indices']; p = 1-q['score_nez'].astype(float); v = q['score_nez'] < r['threshold']
        assert q['freeze'] == fh and q['checkpoint'] == r['checkpoint_sha256'] and q['threshold'] == r['threshold']
        for pid in sorted(set(meta['patient'][ii])):
            j = meta['patient'][ii] == pid
            values = dict(**EV.classification(y[ii][j],p[j],v[j]), **EV.ranking(y[ii][j],p[j],meta['channel'][ii][j]))
            assert all(np.isfinite(values[m]) for m in METRICS)
            rows.append(dict(arm=arm,seed=seed,fold=fold,patient=str(pid),center=str(meta['center'][ii][j][0]),
                             channels=int(j.sum()), **{m:float(values[m]) for m in METRICS}))
        for k,a in [('y',y[ii]),('p',p),('v',v)]: pool[(arm,seed)][k].append(a)
    assert len(rows) == 4560
    return rows,pool


def seed_matrices(rows):
    ids = sorted({r['patient'] for r in rows}); assert len(ids) == 80
    out = {}
    for arm in ARMS:
        for seed in SEEDS:
            z = sorted([r for r in rows if r['arm']==arm and r['seed']==seed],key=lambda r:r['patient'])
            assert [r['patient'] for r in z] == ids and sum(r['channels'] for r in z) == 7635
            out[(arm,seed)] = vector(z)
    return out


def collapse_seeds(matrices):
    # Unit of resampling is the same unique patient, not 240 independent rows.
    return {a:np.mean([matrices[(a,s)] for s in SEEDS],axis=0) for a in ARMS}


def contrast_definitions():
    return [('absolute',a,{a:1}) for a in ARMS]+[('paired',n,w) for n,w in PAIRS.items()]+[('interaction',n,w) for n,w in INTERACTIONS.items()]


def direction_evidence(rows, matrices, collapsed, family):
    summary, details = [], []
    for name,weights in PAIRS.items():
        mean = sum(collapsed[a]*w for a,w in weights.items()).mean(0)
        spos=fpos=0
        for seed in SEEDS:
            point = sum(matrices[(a,seed)]*w for a,w in weights.items()).mean(0)
            spos += int(point[0]>0 and point[1]>0)
            for fold in range(1,6):
                v = sum(vector([r for r in rows if r['arm']==a and r['seed']==seed and r['fold']==fold]).mean(0)*w for a,w in weights.items())
                fpos += int(v[0]>0 and v[1]>0)
                details.append(dict(contrast=name,seed=seed,fold=fold,**dict(zip(METRICS,v.tolist()))))
        lows={m:next(float(r['low']) for r in family if r['contrast']==name and r['metric']==m) for m in ['macro_f1','auroc']}
        summary.append(dict(contrast=name,macro_f1_difference=float(mean[0]),auroc_difference=float(mean[1]),
            both_point_estimates_positive=bool(mean[0]>0 and mean[1]>0),seeds_both_positive=spos,seeds=3,
            fold_seed_cells_both_positive=fpos,fold_seed_cells=15,macro_family_low=lows['macro_f1'],auroc_family_low=lows['auroc'],
            both_family_lower_bounds_positive=bool(min(lows.values())>0)))
    return summary,details


def main():
    rows,pool=load_rows(); matrices=seed_matrices(rows); collapsed=collapse_seeds(matrices)
    mainrows,seedrows,seedsd,folds,foldsd,sources,source_seed,cov = [],[],[],[],[],[],[],[]
    for arm in ARMS:
        mainrows.append(dict(arm=arm,seeds=3,patients=80,**dict(zip(METRICS,collapsed[arm].mean(0).tolist()))))
        means=[matrices[(arm,s)].mean(0) for s in SEEDS]
        seedsd.append(dict(arm=arm,seeds=3,**dict(zip(METRICS,np.std(means,axis=0,ddof=1).tolist()))))
        for seed in SEEDS:
            rr=[r for r in rows if r['arm']==arm and r['seed']==seed]
            seedrows.append(dict(arm=arm,seed=seed,patients=80,**dict(zip(METRICS,matrices[(arm,seed)].mean(0).tolist()))))
            ff=[]
            for fold in range(1,6):
                z=[r for r in rr if r['fold']==fold]; point=vector(z).mean(0);ff.append(point)
                folds.append(dict(arm=arm,seed=seed,fold=fold,patients=len(z),**dict(zip(METRICS,point.tolist()))))
            foldsd.append(dict(arm=arm,seed=seed,**dict(zip(METRICS,np.std(ff,axis=0,ddof=1).tolist()))))
            for center in sorted({r['center'] for r in rr}):
                z=[r for r in rr if r['center']==center]
                source_seed.append(dict(arm=arm,seed=seed,center=center,patients=len(z),suppressed=len(z)<5,
                    **dict(zip(METRICS,vector(z).mean(0).tolist() if len(z)>=5 else [float('nan')]*9))))
            cov.extend(dict(arm=arm,seed=seed,metric=m,finite_patients=int(np.isfinite(matrices[(arm,seed)][:,j]).sum()),patients=80) for j,m in enumerate(METRICS))
        rr=[r for r in rows if r['arm']==arm and r['seed']==42]
        for center in sorted({r['center'] for r in rr}):
            z=[r for r in rows if r['arm']==arm and r['center']==center]; n=len(z)//3
            sources.append(dict(arm=arm,center=center,seeds=3,patients=n,suppressed=n<5,
                **dict(zip(METRICS,vector(z).mean(0).tolist() if n>=5 else [float('nan')]*9))))
    for name,data in [('MAIN_RESULTS',mainrows),('SEED_RESULTS',seedrows),('SEED_SD',seedsd),('FOLD_RESULTS',folds),
        ('FOLD_SD',foldsd),('SOURCE_RESULTS',sources),('SOURCE_SEED_RESULTS',source_seed),('METRIC_COVERAGE',cov)]: csv_out(name,data)
    draw=np.random.default_rng(42).integers(0,80,(10000,80));ci=[];family=[]
    for kind,name,w in contrast_definitions():
        value=sum(collapsed[a]*weight for a,weight in w.items());point,lim=interval(value,draw)
        ci.extend(dict(kind=kind,contrast=name,metric=m,mean=float(point[j]),low=float(lim[0,j]),high=float(lim[1,j]),patients=80,draws=10000) for j,m in enumerate(METRICS))
        if kind=='paired':
            tail=.05/72;_,limits=interval(value,draw,(tail,1-tail))
            for m in ['macro_f1','auroc']:
                j=METRICS.index(m);family.append(dict(kind=kind,contrast=name,metric=m,mean=float(point[j]),
                    low=float(limits[0,j]),high=float(limits[1,j]),comparisons=36,draws=10000,tail=tail))
    assert len(ci)==351 and len(family)==36
    csv_out('INTERVALS',ci);csv_out('PRIMARY_FAMILY_INTERVALS',family)
    evidence,directions=direction_evidence(rows,matrices,collapsed,family)
    csv_out('COMPONENT_EVIDENCE',evidence);csv_out('FOLD_SEED_DIRECTIONS',directions)
    pooled=[];poolmetrics=[m for m in METRICS if m not in ['mrr','hit_at_1']]
    for (arm,seed),z in pool.items():
        yy,pp,vv=[np.concatenate(z[k]) for k in ['y','p','v']];q=EV.classification(yy,pp,vv)
        pooled.append(dict(arm=arm,seed=seed,channels=len(yy),aggregation='POOLED_CHANNELS_PER_SEED_NOT_PRIMARY',
            **{m:float(q[m]) for m in poolmetrics},tp=int((vv&(yy==1)).sum()),tn=int((~vv&(yy==0)).sum()),
            fp=int((vv&(yy==0)).sum()),fn=int((~vv&(yy==1)).sum())))
    csv_out('POOLED_RESULTS',pooled)
    csv_out('POOLED_SEED_MEAN',[dict(arm=a,seeds=3,channels_each=7635,aggregation='MEAN_OF_SEED_METRICS_NOT_ENSEMBLE',
        **{m:float(np.mean([r[m] for r in pooled if r['arm']==a])) for m in poolmetrics}) for a in ARMS])
    save(ROOT/'PATIENT_METRICS_PRIVATE.pkl',rows)
    with (current.PUB/'MAIN_RESULTS.csv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            assert all(abs(float(r[m])-matrices[(r['arm'],42)][:,j].mean())<1e-12 for j,m in enumerate(METRICS))
    full,b0=collapsed['FULL'].mean(0),collapsed['B0_ORIGINAL'].mean(0)
    write(PUB/'GOAL_AUDIT.json',dict(status='DESCRIPTIVE_NOT_SELECTION_GATE',selected_arm=None,
        full_macro_f1=float(full[0]),full_auroc=float(full[1]),original_macro_f1=float(b0[0]),original_auroc=float(b0[1]),
        main_macro_f1_above_0p64=bool(full[0]>.64),macro_within1pp_B0=bool(full[0]-b0[0]>=-.01),
        main_auroc_above_B0=bool(full[1]>b0[1]),no_ablation_direction_assumed=True,no_model_promotion=True,
        independent_confirmation=False,bootstrap_unit='80_UNIQUE_PATIENTS_AFTER_METRIC_SEED_AVERAGING'))
    write(PUB/'REPORT_STATUS.json',dict(status='ALL19_FIXED_GROUPS_3SEEDS_REPORTED',patient_rows=4560,
        nominal_intervals=351,primary_family_intervals=36,selected_arm=None,scientific_status='EXPLORATORY_REPEATED_TEST',
        independent_confirmation=False,no_post_Outer_tuning=True))
    print('CFA3_ALL19_THREE_SEED_REPORT_COMPLETE',flush=True)


if __name__=='__main__': main()
