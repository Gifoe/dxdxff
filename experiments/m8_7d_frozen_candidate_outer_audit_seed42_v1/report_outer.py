"""All eleven arms reported without retrospective candidate selection."""
from common import *

def load_rows():
    fr=read_frozen();seal=jr(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json');fh=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    assert seal['freeze']==fh and seal['all55_cells_sealed_before_full_truth'] and len(seal['files'])==55
    for key,h in seal['files'].items():assert sha(ROOT/'scores_private'/(key+'.pkl'))==h
    context=read(ROOT/'METADATA_PRIVATE.pkl');meta=context['meta'];archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert sha(archive)==context['archive_sha256']
    with np.load(archive,allow_pickle=True) as q:y_nez=q['y'].copy()
    assert len(y_nez)==7635 and int((1-y_nez).sum())==1743
    rows=[];pooled={a:dict(y=[],p=[],v=[]) for a in ARMS}
    for r in fr['models']:
        arm,fold=r['arm'],r['fold'];q=read(ROOT/'scores_private'/f'{arm}_{fold}.pkl');ix=q['indices'];pn=q['score_nez']
        assert q['freeze']==fh and q['checkpoint']==r['checkpoint_sha256'] and q['threshold']==r['threshold']
        pred=pn<q['threshold'];y=1-y_nez[ix];p=1-pn.astype(np.float64)
        assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        for pid in sorted(set(meta['patient'][ix])):
            j=meta['patient'][ix]==pid
            values=dict(**EV.classification(y[j],p[j],pred[j]),**EV.ranking(y[j],p[j],meta['channel'][ix][j]))
            assert all(np.isfinite(values[k]) for k in METRICS)
            rows.append(dict(arm=arm,fold=fold,patient=str(pid),center=str(meta['center'][ix][j][0]),channels=int(j.sum()),**{k:float(values[k]) for k in METRICS}))
        for k,v in [('y',y),('p',p),('v',pred)]:pooled[arm][k].append(v)
    for arm in ARMS:
        rr=[r for r in rows if r['arm']==arm]
        assert len(rr)==len({r['patient'] for r in rr})==80 and sum(r['channels'] for r in rr)==7635
    return rows,pooled

def main():
    rows,pooled=load_rows();main=[];folds=[];sd=[];source=[];coverage=[];matrix={};ids=sorted({r['patient'] for r in rows})
    for arm in ARMS:
        rr=sorted([r for r in rows if r['arm']==arm],key=lambda r:r['patient'])
        assert [r['patient'] for r in rr]==ids
        v=vector(rr);matrix[arm]=v;main.append(dict(arm=arm,patients=80,**dict(zip(METRICS,v.mean(0).tolist()))))
        ff=[]
        for fold in range(1,6):
            z=[r for r in rr if r['fold']==fold];vv=vector(z).mean(0);ff.append(vv)
            folds.append(dict(arm=arm,fold=fold,patients=len(z),**dict(zip(METRICS,vv.tolist()))))
        sd.append(dict(arm=arm,**dict(zip(METRICS,np.std(ff,axis=0,ddof=1).tolist()))))
        for center in sorted({r['center'] for r in rr}):
            z=[r for r in rr if r['center']==center]
            source.append(dict(arm=arm,center=center,patients=len(z),suppressed=len(z)<5,**dict(zip(METRICS,vector(z).mean(0).tolist() if len(z)>=5 else [float('nan')]*9))))
        coverage.extend(dict(arm=arm,metric=m,finite_patients=int(np.isfinite(v[:,j]).sum()),total_patients=80) for j,m in enumerate(METRICS))
    for name,data in [('MAIN_RESULTS',main),('FOLD_RESULTS',folds),('FOLD_SD',sd),('SOURCE_RESULTS',source),('METRIC_COVERAGE',coverage)]:csv_out(name,data)
    draw=np.random.default_rng(42).integers(0,80,(10000,80));ci=[];family=[]
    groups=[('absolute',a,'') for a in ARMS]+[('paired',a,b) for a,b in PAIRS]
    for kind,left,right in groups:
        values=matrix[left]-(matrix[right] if right else 0);mean,lim=intervals(values,draw)
        ci.extend(dict(kind=kind,left=left,right=right,metric=m,mean=float(mean[j]),low=float(lim[0,j]),high=float(lim[1,j]),patients=80,draws=10000) for j,m in enumerate(METRICS))
        if right=='C2_REFERENCE':
            _,lim2=intervals(values,draw,(.0025,.9975))
            family.append(dict(left=left,right=right,metric='macro_f1',mean=float(mean[0]),low=float(lim2[0,0]),high=float(lim2[1,0]),comparisons=10,draws=10000,interval='Bonferroni_percentile_99.5pct_each_exploratory'))
        print('OUTER_INTERVALS_COMPLETE',kind,left,right,flush=True)
    assert len(ci)==234 and len(family)==10
    csv_out('INTERVALS',ci);csv_out('MACRO_FAMILY_INTERVALS',family)
    pp=[]
    for arm in ARMS:
        z=pooled[arm];y=np.concatenate(z['y']);p=np.concatenate(z['p']);v=np.concatenate(z['v'])
        stats=EV.classification(y,p,v)
        pp.append(dict(arm=arm,channels=len(y),aggregation='POOLED_CHANNELS_NOT_PRIMARY',**{m:float(stats[m]) for m in METRICS if m not in ['mrr','hit_at_1']},
            tp=int((v&(y==1)).sum()),tn=int((~v&(y==0)).sum()),fp=int((v&(y==0)).sum()),fn=int((~v&(y==1)).sum())))
    csv_out('POOLED_RESULTS',pp)
    old=list(csv.DictReader((BASE_ROOT/'final_stage_v1/public/MAIN_RESULTS.csv').open(encoding='utf-8')))
    b=next(r for r in old if r['arm']=='C2_INTERACTION')
    for j,m in enumerate(METRICS):assert abs(matrix['C2_REFERENCE'][:,j].mean()-float(b[m]))<1e-12
    save(ROOT/'PATIENT_METRICS_PRIVATE.pkl',rows)
    observed=[dict(arm=a,patient_equal_Macro_F1=float(matrix[a][:,0].mean()),above_0_64=bool(matrix[a][:,0].mean()>.64)) for a in ARMS]
    write(PUBLIC/'OBSERVED_TARGETS.json',dict(status='ALL11_FIXED_ARMS_REPORTED_NO_SELECTION',arms=observed,selected_arm=None,
        target_strictly_greater_than=.64,scientific_status='EXPLORATORY_REPEATED_TEST_MULTI_CANDIDATE_AUDIT',independent_confirmation=False,
        old_development_gates_unchanged=True,no_post_Outer_tuning=True))
    print('FROZEN_CANDIDATE_OUTER_REPORT_COMPLETE',flush=True)

if __name__=='__main__':main()
