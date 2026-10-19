"""Fixed14-arm report: no excluded rows or unidentified factorial interaction."""
from common import *

def load_rows():
    fr=frozen();fh=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json');seal=jr(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')
    assert seal['freeze']==fh and len(seal['files'])==70 and seal['all70_cells_sealed_before_metric_reporting']
    for key,h in seal['files'].items():assert sha(ROOT/'scores_private'/(key+'.pkl'))==h
    ctx=read(ROOT/'METADATA_PRIVATE.pkl');meta=ctx['meta']
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz');assert sha(archive)==ctx['archive_sha256']
    with np.load(archive,allow_pickle=True) as q:y=1-q['y']
    assert len(y)==7635 and int(y.sum())==1743
    rows=[];pool={a:dict(y=[],p=[],v=[]) for a in ARMS}
    for r in fr['cells']:
        arm,fold=r['arm'],r['fold'];q=read(ROOT/'scores_private'/f'{arm}_{fold}.pkl');ii=q['indices']
        p=1-q['score_nez'].astype(float);v=q['score_nez']<r['threshold']
        assert q['freeze']==fh and q['checkpoint']==r['checkpoint_sha256'] and q['threshold']==r['threshold']
        for pid in sorted(set(meta['patient'][ii])):
            j=meta['patient'][ii]==pid
            values=dict(**EV.classification(y[ii][j],p[j],v[j]),**EV.ranking(y[ii][j],p[j],meta['channel'][ii][j]))
            assert all(np.isfinite(values[m]) for m in METRICS)
            rows.append(dict(arm=arm,fold=fold,patient=str(pid),center=str(meta['center'][ii][j][0]),channels=int(j.sum()),**{m:float(values[m]) for m in METRICS}))
        for k,a in [('y',y[ii]),('p',p),('v',v)]:pool[arm][k].append(a)
    assert len(rows)==1120;return rows,pool

def contrast_definitions():
    return [('absolute',a,{a:1}) for a in ARMS]+[('paired','FULL_minus_'+a,{'FULL_F0':1,a:-1}) for a in ARMS[1:]]

def main():
    rows,pool=load_rows();ids=sorted({r['patient'] for r in rows});matrix={};main=[];folds=[];sd=[];source=[];coverage=[]
    for arm in ARMS:
        rr=sorted([r for r in rows if r['arm']==arm],key=lambda r:r['patient'])
        assert [r['patient'] for r in rr]==ids and len(ids)==80 and sum(r['channels'] for r in rr)==7635
        vv=vector(rr);matrix[arm]=vv;main.append(dict(arm=arm,patients=80,**dict(zip(METRICS,vv.mean(0).tolist()))));ff=[]
        for fold in range(1,6):
            q=[r for r in rr if r['fold']==fold];v=vector(q).mean(0);ff.append(v)
            folds.append(dict(arm=arm,fold=fold,patients=len(q),**dict(zip(METRICS,v.tolist()))))
        sd.append(dict(arm=arm,**dict(zip(METRICS,np.std(ff,axis=0,ddof=1).tolist()))))
        for center in sorted({r['center'] for r in rr}):
            q=[r for r in rr if r['center']==center]
            source.append(dict(arm=arm,center=center,patients=len(q),suppressed=len(q)<5,**dict(zip(METRICS,vector(q).mean(0).tolist() if len(q)>=5 else [float('nan')]*9))))
        coverage.extend(dict(arm=arm,metric=m,finite_patients=int(np.isfinite(vv[:,j]).sum()),patients=80) for j,m in enumerate(METRICS))
    for name,data in [('MAIN_RESULTS',main),('FOLD_RESULTS',folds),('FOLD_SD',sd),('SOURCE_RESULTS',source),('METRIC_COVERAGE',coverage)]:csv_out(name,data)
    draws=np.random.default_rng(42).integers(0,80,(10000,80));ci=[];family=[]
    for kind,label,weights in contrast_definitions():
        v=sum(matrix[a]*w for a,w in weights.items());mean,lim=interval(v,draws)
        ci.extend(dict(kind=kind,contrast=label,metric=m,mean=float(mean[j]),low=float(lim[0,j]),high=float(lim[1,j]),patients=80,draws=10000) for j,m in enumerate(METRICS))
        if kind=='paired':
            tail=.05/(2*13);_,limits=interval(v,draws,(tail,1-tail))
            family.append(dict(kind=kind,contrast=label,metric='macro_f1',mean=float(mean[0]),low=float(limits[0,0]),high=float(limits[1,0]),comparisons=13,draws=10000,tail=tail))
    assert len(ci)==243 and len(family)==13;csv_out('INTERVALS',ci);csv_out('MACRO_FAMILY_INTERVALS',family)
    pooled=[]
    for arm in ARMS:
        q=pool[arm];y=np.concatenate(q['y']);p=np.concatenate(q['p']);v=np.concatenate(q['v']);s=EV.classification(y,p,v)
        pooled.append(dict(arm=arm,channels=len(y),aggregation='POOLED_CHANNELS_NOT_PRIMARY',**{m:float(s[m]) for m in METRICS if m not in ['mrr','hit_at_1']},
            tp=int((v&(y==1)).sum()),tn=int((~v&(y==0)).sum()),fp=int((v&(y==0)).sum()),fn=int((~v&(y==1)).sum())))
    csv_out('POOLED_RESULTS',pooled);save(ROOT/'PATIENT_METRICS_PRIVATE.pkl',rows)
    path=PREVROOT/'public/MAIN_RESULTS.csv'
    with path.open(encoding='utf-8') as f:oldrows=list(csv.DictReader(f))
    for arm,name in REF_NAMES.items():
        r=next(r for r in oldrows if r['arm']==name)
        for j,m in enumerate(METRICS):assert abs(float(r[m])-matrix[arm][:,j].mean())<1e-12
    classifiers=set(MODEL.BASE.TABULAR_ARMS);oldclass=[r for r in oldrows if r['arm'] in classifiers];assert len(oldclass)==14
    csv_out('HISTORICAL_CLASSIFIER_REFERENCE',oldclass)
    write(PUB/'HISTORICAL_CLASSIFIER_AUDIT.json',dict(status='REUSED_UNCHANGED_ANONYMOUS_CONTEXT',source_commit=jr(HERE/'PROTOCOL_LOCK.json')['base_commit'],source_table_sha256=sha(path),
        source_independent_verification_sha256=sha(PREVROOT/'public/INDEPENDENT_OUTER_VERIFICATION.json'),groups=14,new_fits=0,in_current_14_group_CI_family=False))
    write(PUB/'REPORT_STATUS.json',dict(status='ALL14_FIXED_GROUPS_REPORTED',patient_rows=1120,nominal_intervals=243,macro_family_intervals=13,selected_arm=None,
        scientific_status='EXPLORATORY_REPEATED_TEST',independent_confirmation=False,no_post_Outer_tuning=True,early_late_factorial_interaction_identifiable=False))
    print('NI_ALL14_FULL80_REPORT_COMPLETE',flush=True)

if __name__=='__main__':main()
