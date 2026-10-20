"""Fixed-no-Late seed42 patient endpoints; all scores sealed before reporting."""
from common import *


def table(name):
    with (PUB/(name+'.csv')).open(encoding='utf-8', newline='') as f: return list(csv.DictReader(f))


def load_rows():
    fr = frozen(); fh = sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json'); seal = jr(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')
    assert seal['freeze'] == fh and len(seal['files']) == 25 and seal['all25_cells_sealed_before_metric_reporting']
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
    assert len(rows) == 400
    return rows,pool



def contrast_definitions():
    return [('absolute',a,{a:1}) for a in ARMS]+[('paired',n,w) for n,w in PAIRS.items()]+[('interaction',n,w) for n,w in INTERACTIONS.items()]

def direction_evidence(rows,matrix,family):
    result=[];details=[]
    for name,w in PAIRS.items():
        mean=sum(matrix[a]*x for a,x in w.items()).mean(0);positive=0
        for fold in range(1,6):
            v=sum(vector([r for r in rows if r['arm']==a and r['fold']==fold]).mean(0)*x for a,x in w.items())
            positive+=int(v[0]>0 and v[1]>0)
            details.append(dict(contrast=name,fold=fold,**dict(zip(METRICS,v.tolist()))))
        lows={m:next(float(r['low']) for r in family if r['contrast']==name and r['metric']==m) for m in ['macro_f1','auroc']}
        result.append(dict(contrast=name,macro_f1_difference=float(mean[0]),auroc_difference=float(mean[1]),
            both_point_estimates_positive=bool(mean[0]>0 and mean[1]>0),folds_both_positive=positive,folds=5,
            macro_family_low=lows['macro_f1'],auroc_family_low=lows['auroc'],both_family_lower_bounds_positive=bool(min(lows.values())>0)))
    return result,details

def main():
    rows,pool=load_rows();ids=sorted({r['patient'] for r in rows});assert len(ids)==80
    matrix={};mainrows=[];foldrows=[];sd=[];source=[];cov=[]
    for arm in ARMS:
        rr=sorted([r for r in rows if r['arm']==arm],key=lambda r:r['patient'])
        assert [r['patient'] for r in rr]==ids and sum(r['channels'] for r in rr)==7635
        v=vector(rr);matrix[arm]=v
        mainrows.append(dict(arm=arm,seed=42,patients=80,**dict(zip(METRICS,v.mean(0).tolist()))))
        ff=[]
        for fold in range(1,6):
            z=[r for r in rr if r['fold']==fold];values=vector(z).mean(0);ff.append(values)
            foldrows.append(dict(arm=arm,seed=42,fold=fold,patients=len(z),**dict(zip(METRICS,values.tolist()))))
        sd.append(dict(arm=arm,**dict(zip(METRICS,np.std(ff,axis=0,ddof=1).tolist()))))
        for center in sorted({r['center'] for r in rr}):
            z=[r for r in rr if r['center']==center]
            source.append(dict(arm=arm,center=center,patients=len(z),suppressed=len(z)<5,
                **dict(zip(METRICS,vector(z).mean(0).tolist() if len(z)>=5 else [float('nan')]*9))))
        cov.extend(dict(arm=arm,metric=m,finite_patients=int(np.isfinite(v[:,j]).sum()),patients=80) for j,m in enumerate(METRICS))
    for name,data in [('MAIN_RESULTS',mainrows),('FOLD_RESULTS',foldrows),('FOLD_SD',sd),('SOURCE_RESULTS',source),('METRIC_COVERAGE',cov)]:csv_out(name,data)
    draw=np.random.default_rng(42).integers(0,80,(10000,80));ci=[];family=[]
    for kind,label,w in contrast_definitions():
        values=sum(matrix[a]*weight for a,weight in w.items());mean,lim=interval(values,draw)
        ci.extend(dict(kind=kind,contrast=label,metric=m,mean=float(mean[j]),low=float(lim[0,j]),high=float(lim[1,j]),patients=80,draws=10000) for j,m in enumerate(METRICS))
        if kind=='paired':
            tail=.05/20;_,limits=interval(values,draw,(tail,1-tail))
            for m in ['macro_f1','auroc']:
                j=METRICS.index(m);family.append(dict(kind=kind,contrast=label,metric=m,mean=float(mean[j]),low=float(limits[0,j]),
                    high=float(limits[1,j]),comparisons=10,draws=10000,tail=tail))
    assert len(ci)==90 and len(family)==10
    csv_out('INTERVALS',ci);csv_out('PRIMARY_FAMILY_INTERVALS',family)
    evidence,detail=direction_evidence(rows,matrix,family);csv_out('COMPONENT_EVIDENCE',evidence);csv_out('FOLD_DIRECTIONS',detail)
    pooled=[]
    for (arm,seed),z in pool.items():
        yy,pp,vv=[np.concatenate(z[k]) for k in ['y','p','v']];q=EV.classification(yy,pp,vv)
        pooled.append(dict(arm=arm,seed=seed,channels=len(yy),aggregation='POOLED_CHANNELS_NOT_PRIMARY',
            **{m:float(q[m]) for m in METRICS if m not in ['mrr','hit_at_1']},tp=int((vv&(yy==1)).sum()),tn=int((~vv&(yy==0)).sum()),
            fp=int((vv&(yy==0)).sum()),fn=int((~vv&(yy==1)).sum())))
    csv_out('POOLED_RESULTS',pooled);save(ROOT/'PATIENT_METRICS_PRIVATE.pkl',rows)
    with (parent.PUB/'MAIN_RESULTS.csv').open(encoding='utf-8',newline='') as f:
        old=[r for r in csv.DictReader(f) if int(r['seed'])==42]
    for arm,original in MODEL.REUSE.items():
        r=next(r for r in old if r['arm']==original)
        assert all(abs(float(r[m])-matrix[arm][:,j].mean())<1e-12 for j,m in enumerate(METRICS))
    assert abs(matrix['FULL'][:,0].mean()-.6534651290238689)<1e-12 and abs(matrix['FULL'][:,1].mean()-.7652863490156572)<1e-12
    write(PUB/'GOAL_AUDIT.json',dict(status='DESCRIPTIVE_NOT_SELECTION_GATE',selected_arm=None,late_gate_absent_in_all_groups=True,
        historical_reference_groups=0,seed=42,independent_confirmation=False,no_ablation_direction_assumed=True,
        full_macro_f1=float(matrix['FULL'][:,0].mean()),full_auroc=float(matrix['FULL'][:,1].mean())))
    write(PUB/'REPORT_STATUS.json',dict(status='ALL5_FIXED_GROUPS_SEED42_REPORTED',patient_rows=400,nominal_intervals=90,
        primary_family_intervals=10,no_extra_aliases=True,selected_arm=None,
        scientific_status='EXPLORATORY_REPEATED_TEST',independent_confirmation=False,no_post_Outer_tuning=True))
    print('PCA_ALL5_SEED42_REPORT_COMPLETE',flush=True)

if __name__=='__main__':main()
