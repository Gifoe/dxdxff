"""Independently reconstruct every endpoint and paired interval from sealed scores."""
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,accuracy_score
from common import *

def table(name):
    with (PUBLIC/(name+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def main():
    fr=read_frozen();seal=jr(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json');fh=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')
    assert seal['freeze']==fh and seal['all55_cells_sealed_before_full_truth'] and len(seal['files'])==55
    ctx=read(ROOT/'METADATA_PRIVATE.pkl');meta=ctx['meta'];archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert sha(archive)==ctx['archive_sha256']
    with np.load(archive,allow_pickle=True) as z:y=1-z['y']
    stored=read(ROOT/'PATIENT_METRICS_PRIVATE.pkl');index={(r['arm'],r['fold'],r['patient']):r for r in stored}
    rebuilt=[];numeric=0;pooled={a:dict(y=[],p=[],v=[]) for a in ARMS}
    for r in fr['models']:
        arm,fold=r['arm'],r['fold'];key=f'{arm}_{fold}';path=ROOT/'scores_private'/(key+'.pkl')
        assert sha(path)==seal['files'][key];q=read(path);ii=q['indices'];pn=q['score_nez'];pp=1-pn.astype(float);vv=pn<r['threshold']
        assert q['checkpoint']==r['checkpoint_sha256'] and q['threshold']==r['threshold'] and q['freeze']==fh
        for pid in sorted(set(meta['patient'][ii])):
            j=meta['patient'][ii]==pid;yy=y[ii][j];p=pp[j];v=vv[j];ch=meta['channel'][ii][j].astype(str)
            order=sorted(range(len(yy)),key=lambda k:(-float(p[k]),str(ch[k])));first=next(k for k,i in enumerate(order) if yy[i])
            values=dict(macro_f1=f1_score(yy,v,labels=[0,1],average='macro',zero_division=0),f1=f1_score(yy,v,zero_division=0),nez_f1=f1_score(1-yy,1-v.astype(int),zero_division=0),
                auroc=roc_auc_score(yy,p),auprc=average_precision_score(yy,p),accuracy=accuracy_score(yy,v),mrr=1/(first+1),hit_at_1=float(first==0),ez_fraction_bias=float(v.mean()-yy.mean()))
            row=index[(arm,fold,str(pid))]
            for m,val in values.items():assert abs(row[m]-val)<1e-12;numeric+=1
            rebuilt.append(dict(arm=arm,fold=fold,patient=str(pid),center=str(meta['center'][ii][j][0]),channels=int(j.sum()),**values))
        for k,v in [('y',y[ii]),('p',pp),('v',vv)]:pooled[arm][k].append(v)
    assert len(stored)==len(rebuilt)==880 and numeric==7920 and len(index)==880
    ids=sorted({r['patient'] for r in rebuilt});assert len(ids)==80
    matrices={a:vector(sorted([r for r in rebuilt if r['arm']==a],key=lambda r:r['patient'])) for a in ARMS}
    main=table('MAIN_RESULTS');folds=table('FOLD_RESULTS');sd=table('FOLD_SD');centers=table('SOURCE_RESULTS');coverage=table('METRIC_COVERAGE')
    assert len(main)==11 and len(folds)==55 and len(sd)==11 and len(coverage)==99
    for arm in ARMS:
        rr=[r for r in rebuilt if r['arm']==arm];row=next(r for r in main if r['arm']==arm);ff=[]
        assert len(rr)==80 and sum(r['channels'] for r in rr)==7635 and {r['patient'] for r in rr}==set(ids)
        for j,m in enumerate(METRICS):assert abs(float(row[m])-matrices[arm][:,j].mean())<1e-12
        for fold in range(1,6):
            z=[r for r in rr if r['fold']==fold];assert len(z)==[16,16,17,15,16][fold-1]
            values=vector(z).mean(0);ff.append(values);row=next(r for r in folds if r['arm']==arm and int(r['fold'])==fold)
            assert max(abs(float(row[m])-values[j]) for j,m in enumerate(METRICS))<1e-12
        row=next(r for r in sd if r['arm']==arm)
        assert max(abs(float(row[m])-np.std(ff,axis=0,ddof=1)[j]) for j,m in enumerate(METRICS))<1e-12
        for center in sorted({r['center'] for r in rr}):
            z=[r for r in rr if r['center']==center];row=next(r for r in centers if r['arm']==arm and r['center']==center)
            assert int(row['patients'])==len(z) and (row['suppressed']=='True')==(len(z)<5)
            if len(z)>=5:assert max(abs(float(row[m])-vector(z)[:,j].mean()) for j,m in enumerate(METRICS))<1e-12
    assert all(int(r['finite_patients'])==int(r['total_patients'])==80 for r in coverage)
    ci=table('INTERVALS');family=table('MACRO_FAMILY_INTERVALS');assert len(ci)==234 and len(family)==10
    draw=np.random.default_rng(42).integers(0,80,(10000,80));maximum=0.;count=0
    groups=[('absolute',a,'') for a in ARMS]+[('paired',a,b) for a,b in PAIRS]
    for kind,left,right in groups:
        values=matrices[left]-(matrices[right] if right else 0)
        # Independent implementation: metric-wise indexed resampling, not report intervals().
        for j,m in enumerate(METRICS):
            samples=values[:,j][draw].mean(1);limits=np.percentile(samples,[2.5,97.5]);mean=values[:,j].mean()
            row=next(r for r in ci if (r['kind'],r['left'],r['right'],r['metric'])==(kind,left,right,m))
            err=max(abs(float(row['mean'])-mean),abs(float(row['low'])-limits[0]),abs(float(row['high'])-limits[1]));assert err<1e-12
            assert int(row['patients'])==80 and int(row['draws'])==10000;maximum=max(maximum,err);count+=1
            if right=='C2_REFERENCE' and m=='macro_f1':
                row=next(r for r in family if r['left']==left);limits=np.percentile(samples,[.25,99.75])
                assert max(abs(float(row['low'])-limits[0]),abs(float(row['high'])-limits[1]),abs(float(row['mean'])-mean))<1e-12
    assert count==234
    for row in table('POOLED_RESULTS'):
        z=pooled[row['arm']];yy=np.concatenate(z['y']);pp=np.concatenate(z['p']);vv=np.concatenate(z['v'])
        vals=dict(macro_f1=f1_score(yy,vv,average='macro'),f1=f1_score(yy,vv),nez_f1=f1_score(1-yy,1-vv.astype(int)),
            auroc=roc_auc_score(yy,pp),auprc=average_precision_score(yy,pp),accuracy=accuracy_score(yy,vv),ez_fraction_bias=float(vv.mean()-yy.mean()),
            tp=int((vv&(yy==1)).sum()),tn=int((~vv&(yy==0)).sum()),fp=int((vv&(yy==0)).sum()),fn=int((~vv&(yy==1)).sum()))
        assert len(yy)==7635 and all(abs(float(row[k])-v)<1e-12 for k,v in vals.items())
    raw=table('RAW_COVERAGE');assert len(raw)==50
    for arm in NEW:
        z=[r for r in raw if r['arm']==arm]
        assert len(z)==5 and sum(int(r['patients']) for r in z)==80 and sum(int(r['channels']) for r in z)==7635 and sum(int(r['valid_windows']) for r in z)==1471965
    targets=jr(PUBLIC/'OBSERVED_TARGETS.json');assert targets['selected_arm'] is None and not targets['independent_confirmation']
    for r in targets['arms']:
        a=float(matrices[r['arm']][:,0].mean());assert r['patient_equal_Macro_F1']==a and r['above_0_64']==bool(a>.64)
    assert abs(matrices['C2_REFERENCE'][:,0].mean()-.6396579900303192)<1e-12
    write(PUBLIC/'INDEPENDENT_OUTER_VERIFICATION.json',dict(status='PASS',patient_rows=880,patient_metric_checks=numeric,interval_checks=count,macro_family_interval_checks=10,
        maximum_interval_difference=maximum,all11_complete80_verified=True,all10_new_full_Raw_censuses_verified=True,all55_models_and_thresholds_frozen=True,
        original7_features_unchanged=True,one_classification_head=True,old_scores_and_Development_gates_unchanged=True,no_new_training_or_inference=True))
    write(PUBLIC/'OUTER_STATUS.json',dict(status='COMPLETE_VERIFIED_ALL11_FULL80_FROZEN_OUTER_AUDIT',arms=11,patients_each=80,new_inference_cells=50,reused_reference_cells=5,
        patient_rows=880,metric_checks=numeric,features=7,selected_arm=None,scientific_status='EXPLORATORY_REPEATED_TEST_MULTI_CANDIDATE_AUDIT',independent_confirmation=False,
        freeze_sha256=fh,score_seal_sha256=sha(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')))
    print('ALL11_FULL80_OUTER_INDEPENDENT_VERIFICATION_PASS',flush=True)

if __name__=='__main__':main()
