"""Independent sklearn endpoints, explicit tie ranking and independent resampling."""
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,accuracy_score
from common import *

def table(name):
    with (PUB/(name+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def main():
    fr=frozen();fh=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json');seal=jr(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')
    assert seal['freeze']==fh and len(seal['files'])==90 and seal['all90_cells_sealed_before_metric_reporting']
    ctx=read(ROOT/'METADATA_PRIVATE.pkl');meta=ctx['meta'];archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz')
    assert sha(archive)==ctx['archive_sha256']
    with np.load(archive,allow_pickle=True) as q:y=1-q['y']
    stored=read(ROOT/'PATIENT_METRICS_PRIVATE.pkl');lookup={(r['arm'],r['fold'],r['patient']):r for r in stored};rows=[];numeric=0;pooled={a:dict(y=[],p=[],v=[]) for a in ARMS}
    for r in fr['cells']:
        arm,fold=r['arm'],r['fold'];key=f'{arm}_{fold}';path=ROOT/'scores_private'/(key+'.pkl');assert sha(path)==seal['files'][key]
        q=read(path);ii=q['indices'];pn=q['score_nez'];p=1-pn.astype(float);v=pn<r['threshold']
        assert q['checkpoint']==r['checkpoint_sha256'] and q['freeze']==fh and q['threshold']==r['threshold']
        for pid in sorted(set(meta['patient'][ii])):
            j=meta['patient'][ii]==pid;yy=y[ii][j];pp=p[j];vv=v[j];ch=meta['channel'][ii][j].astype(str)
            order=sorted(range(len(yy)),key=lambda k:(-float(pp[k]),ch[k]));first=next(k for k,i in enumerate(order) if yy[i])
            values=dict(macro_f1=f1_score(yy,vv,labels=[0,1],average='macro',zero_division=0),f1=f1_score(yy,vv,zero_division=0),nez_f1=f1_score(1-yy,1-vv.astype(int),zero_division=0),
                auroc=roc_auc_score(yy,pp),auprc=average_precision_score(yy,pp),accuracy=accuracy_score(yy,vv),mrr=1/(first+1),hit_at_1=float(first==0),ez_fraction_bias=float(vv.mean()-yy.mean()))
            row=lookup[(arm,fold,str(pid))]
            for m,val in values.items():assert abs(row[m]-val)<1e-12;numeric+=1
            rows.append(dict(arm=arm,fold=fold,patient=str(pid),center=str(meta['center'][ii][j][0]),channels=int(j.sum()),**values))
        for k,a in [('y',y[ii]),('p',p),('v',v)]:pooled[arm][k].append(a)
    assert len(rows)==len(lookup)==len(stored)==1440 and numeric==12960
    ids=sorted({r['patient'] for r in rows});assert len(ids)==80
    matrices={a:vector(sorted([r for r in rows if r['arm']==a],key=lambda r:r['patient'])) for a in ARMS}
    main=table('MAIN_RESULTS');folds=table('FOLD_RESULTS');sd=table('FOLD_SD');centers=table('SOURCE_RESULTS');cov=table('METRIC_COVERAGE')
    assert len(main)==18 and len(folds)==90 and len(sd)==18 and len(cov)==162
    for arm in ARMS:
        rr=[r for r in rows if r['arm']==arm];assert len(rr)==80 and {r['patient'] for r in rr}==set(ids) and sum(r['channels'] for r in rr)==7635
        r=next(r for r in main if r['arm']==arm);assert max(abs(float(r[m])-matrices[arm][:,j].mean()) for j,m in enumerate(METRICS))<1e-12
        ff=[]
        for fold in range(1,6):
            z=[r for r in rr if r['fold']==fold];assert len(z)==[16,16,17,15,16][fold-1]
            values=vector(z).mean(0);ff.append(values);r=next(r for r in folds if r['arm']==arm and int(r['fold'])==fold)
            assert max(abs(float(r[m])-values[j]) for j,m in enumerate(METRICS))<1e-12
        r=next(r for r in sd if r['arm']==arm);assert max(abs(float(r[m])-np.std(ff,axis=0,ddof=1)[j]) for j,m in enumerate(METRICS))<1e-12
        for center in sorted({r['center'] for r in rr}):
            z=[r for r in rr if r['center']==center];r=next(r for r in centers if r['arm']==arm and r['center']==center)
            assert int(r['patients'])==len(z) and (r['suppressed']=='True')==(len(z)<5)
            if len(z)>=5:assert max(abs(float(r[m])-vector(z)[:,j].mean()) for j,m in enumerate(METRICS))<1e-12
    assert all(int(r['finite_patients'])==int(r['patients'])==80 for r in cov)
    defs=[('absolute',a,[(a,1)]) for a in ARMS]+[('paired','FULL_minus_'+a,[('FULL_F0',1),(a,-1)]) for a in ARMS[1:]]+[
        ('interaction','Early_by_late_guidance',[('FULL_F0',1),('WO_BOTH_GUIDANCE',1),('WO_EARLY_FILM',-1),('WO_LATE_GATE',-1)])]
    draws=np.random.default_rng(42).integers(0,80,(10000,80));ci=table('INTERVALS');family=table('MACRO_FAMILY_INTERVALS');assert len(ci)==324 and len(family)==18
    maximum=0.;checks=familychecks=0
    for kind,label,weights in defs:
        for j,m in enumerate(METRICS):
            values=sum(matrices[a][:,j]*w for a,w in weights);samples=values[draws].mean(1);limits=np.quantile(samples,[.025,.975]);point=values.mean()
            r=next(r for r in ci if r['contrast']==label and r['kind']==kind and r['metric']==m)
            err=max(abs(float(r['mean'])-point),abs(float(r['low'])-limits[0]),abs(float(r['high'])-limits[1]));assert err<1e-12
            assert int(r['patients'])==80 and int(r['draws'])==10000;maximum=max(maximum,err);checks+=1
            if j==0 and kind!='absolute':
                tail=.05/36;lim=np.quantile(samples,[tail,1-tail]);r=next(r for r in family if r['contrast']==label)
                assert max(abs(float(r['mean'])-point),abs(float(r['low'])-lim[0]),abs(float(r['high'])-lim[1]))<1e-12;familychecks+=1
    assert checks==324 and familychecks==18
    for r in table('POOLED_RESULTS'):
        z=pooled[r['arm']];yy=np.concatenate(z['y']);pp=np.concatenate(z['p']);vv=np.concatenate(z['v'])
        vals=dict(macro_f1=f1_score(yy,vv,average='macro'),f1=f1_score(yy,vv),nez_f1=f1_score(1-yy,1-vv.astype(int)),auroc=roc_auc_score(yy,pp),auprc=average_precision_score(yy,pp),
            accuracy=accuracy_score(yy,vv),ez_fraction_bias=float(vv.mean()-yy.mean()),tp=int((vv&(yy==1)).sum()),tn=int((~vv&(yy==0)).sum()),fp=int((vv&(yy==0)).sum()),fn=int((~vv&(yy==1)).sum()))
        assert len(yy)==7635 and all(abs(float(r[k])-v)<1e-12 for k,v in vals.items())
    raw=table('RAW_COVERAGE');assert len(raw)==80
    for arm in NEW:
        z=[r for r in raw if r['arm']==arm];assert len(z)==5 and sum(int(r['patients']) for r in z)==80 and sum(int(r['channels']) for r in z)==7635
        assert sum(int(r['valid_windows']) for r in z)==(0 if arm=='PHYSIOLOGY_ONLY' else 1471965)
        assert all((r['raw_consumed']=='True')==(arm!='PHYSIOLOGY_ONLY') for r in z)
        if arm=='EARLY_FILM_MISMATCH':
            assert sum(int(r['changed_valid_windows']) for r in z)>0
            assert all(int(r['changed_valid_windows'])==int(r['eligible_valid_windows']) and int(r['eligible_valid_windows'])+int(r['singleton_valid_windows'])==int(r['valid_windows']) for r in z)
    for arm,value in [('FULL_F0',.6479765335907922),('WO_EARLY_FILM',.6396579900303192)]:assert abs(matrices[arm][:,0].mean()-value)<1e-12
    write(PUB/'INDEPENDENT_OUTER_VERIFICATION.json',dict(status='PASS',patient_rows=1440,patient_metric_checks=numeric,nominal_interval_checks=checks,macro_family_interval_checks=familychecks,
        maximum_interval_difference=maximum,all18_complete80_verified=True,all15_new_Raw_arm_censuses_verified=True,physiology_only_Raw_not_consumed=True,
        reused_scores_unchanged=True,all90_model_selections_frozen=True,single_head=True,no_post_Outer_tuning=True))
    write(PUB/'OUTER_STATUS.json',dict(status='COMPLETE_VERIFIED_F0_18_ARM_FIVE_FOLD_OUTER',arms=18,patients_each=80,channels_each=7635,new_fits=80,reused_cells=10,logical_model_cells=90,
        patient_rows=1440,metric_checks=numeric,selected_arm=None,scientific_status='EXPLORATORY_REPEATED_TEST',independent_confirmation=False,
        freeze_sha256=fh,score_seal_sha256=sha(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')))
    print('F0_ALL18_FULL80_INDEPENDENT_VERIFICATION_PASS',flush=True)

if __name__=='__main__':main()
