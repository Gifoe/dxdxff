"""Independent sklearn endpoints, deterministic rank ties and resampling audit."""
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,accuracy_score
from common import *

def table(name):
    with (PUB/(name+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def main():
    fr=frozen();fh=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json');seal=jr(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')
    assert seal['freeze']==fh and len(seal['files'])==45 and seal['all45_cells_sealed_before_metric_reporting']
    ctx=read(ROOT/'METADATA_PRIVATE.pkl');meta=ctx['meta']
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz');assert sha(archive)==ctx['archive_sha256']
    with np.load(archive,allow_pickle=True) as q:y=1-q['y']
    stored=read(ROOT/'PATIENT_METRICS_PRIVATE.pkl');lookup={(r['arm'],r['fold'],r['patient']):r for r in stored}
    rows=[];numeric=0;pooled={a:dict(y=[],p=[],v=[]) for a in ARMS}
    for r in fr['cells']:
        arm,fold=r['arm'],r['fold'];path=ROOT/'scores_private'/f'{arm}_{fold}.pkl';assert sha(path)==seal['files'][f'{arm}_{fold}']
        q=read(path);ii=q['indices'];p=1-q['score_nez'].astype(float);v=q['score_nez']<r['threshold']
        assert q['checkpoint']==r['checkpoint_sha256'] and q['freeze']==fh and q['threshold']==r['threshold']
        state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)['model']
        assert not any('interaction' in name for name in state)
        assert state['keep7'][4]==0
        if r['reused']:
            key=f'{REF_NAMES[arm]}_{fold}';src=PREVROOT/'scores_private'/(key+'.pkl')
            assert sha(src)==jr(PREVROOT/'OUTER_SCORE_SEAL_PRIVATE.json')['files'][key]
            z=read(src);assert np.array_equal(q['score_nez'],z['score_nez']) and q['threshold']==z['threshold'] and np.array_equal(ii,z['indices'])
        for pid in sorted(set(meta['patient'][ii])):
            j=meta['patient'][ii]==pid;yy=y[ii][j];pp=p[j];vv=v[j];ch=meta['channel'][ii][j].astype(str)
            order=sorted(range(len(yy)),key=lambda k:(-float(pp[k]),ch[k]));first=next(k for k,i in enumerate(order) if yy[i])
            values=dict(macro_f1=f1_score(yy,vv,labels=[0,1],average='macro',zero_division=0),f1=f1_score(yy,vv,zero_division=0),nez_f1=f1_score(1-yy,1-vv.astype(int),zero_division=0),
                auroc=roc_auc_score(yy,pp),auprc=average_precision_score(yy,pp),accuracy=accuracy_score(yy,vv),mrr=1/(first+1),hit_at_1=float(first==0),ez_fraction_bias=float(vv.mean()-yy.mean()))
            row=lookup[(arm,fold,str(pid))]
            for m,val in values.items():assert abs(row[m]-val)<1e-12;numeric+=1
            rows.append(dict(arm=arm,fold=fold,patient=str(pid),center=str(meta['center'][ii][j][0]),channels=int(j.sum()),**values))
        for k,a in [('y',y[ii]),('p',p),('v',v)]:pooled[arm][k].append(a)
    assert len(rows)==len(lookup)==len(stored)==720 and numeric==6480
    ids=sorted({r['patient'] for r in rows});assert len(ids)==80
    matrices={a:vector(sorted([r for r in rows if r['arm']==a],key=lambda r:r['patient'])) for a in ARMS}
    main=table('MAIN_RESULTS');folds=table('FOLD_RESULTS');sd=table('FOLD_SD');centers=table('SOURCE_RESULTS');cov=table('METRIC_COVERAGE')
    assert len(main)==9 and len(folds)==45 and len(sd)==9 and len(cov)==81
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
            else:assert all(np.isnan(float(r[m])) for m in METRICS)
    assert all(int(r['finite_patients'])==int(r['patients'])==80 for r in cov)
    weights={
        'FULL_MINUS_B0':[('FULL',1),('B0_ORIGINAL',-1)],
        'FEATURE_PR':[('FULL',1),('WO_FEATURE_PR',-1)],
        'RAW_PR':[('FULL',1),('WO_RAW_PR',-1)],
        'BOTH_PR':[('FULL',1),('WO_BOTH_PR',-1)],
        'WINDOW_PR':[('FULL',1),('WO_WINDOW_PR',-1)],
        'EARLY_FILM':[('FULL',1),('WO_EARLY_FILM',-1)],
        'LATE_GATE':[('FULL',1),('WO_LATE_GATE',-1)],
        'PRE_REFERENCE':[('FULL',1),('WO_PRE_REFERENCE',-1)],
        'PR_INTERACTION':[('FULL',1),('WO_FEATURE_PR',-1),('WO_RAW_PR',-1),('WO_BOTH_PR',1)]}
    assert {k:dict(v) for k,v in weights.items()}==CONTRASTS
    primary=['FULL_MINUS_B0','FEATURE_PR','RAW_PR','WINDOW_PR','EARLY_FILM','LATE_GATE','PRE_REFERENCE']
    assert primary==PRIMARY
    defs=[('absolute',a,[(a,1)]) for a in ARMS]+[('paired',name,w) for name,w in weights.items()]
    draws=np.random.default_rng(42).integers(0,80,(10000,80));ci=table('INTERVALS');family=table('PRIMARY_FAMILY_INTERVALS');assert len(ci)==162 and len(family)==14
    maximum=0.;checks=familychecks=0
    for kind,label,coefficients in defs:
        for j,m in enumerate(METRICS):
            values=sum(matrices[a][:,j]*w for a,w in coefficients);samples=values[draws].mean(1);limits=np.quantile(samples,[.025,.975]);point=values.mean()
            r=next(r for r in ci if r['contrast']==label and r['kind']==kind and r['metric']==m)
            err=max(abs(float(r['mean'])-point),abs(float(r['low'])-limits[0]),abs(float(r['high'])-limits[1]));assert err<1e-12
            assert int(r['patients'])==80 and int(r['draws'])==10000;maximum=max(maximum,err);checks+=1
            if m in ['macro_f1','auroc'] and kind=='paired' and label in primary:
                tail=.05/28;lim=np.quantile(samples,[tail,1-tail]);r=next(r for r in family if r['contrast']==label and r['metric']==m)
                assert int(r['comparisons'])==14 and int(r['draws'])==10000 and abs(float(r['tail'])-tail)<1e-15
                assert max(abs(float(r['mean'])-point),abs(float(r['low'])-lim[0]),abs(float(r['high'])-lim[1]))<1e-12;familychecks+=1
    assert checks==162 and familychecks==14
    for r in table('POOLED_RESULTS'):
        z=pooled[r['arm']];yy=np.concatenate(z['y']);pp=np.concatenate(z['p']);vv=np.concatenate(z['v'])
        vals=dict(macro_f1=f1_score(yy,vv,average='macro'),f1=f1_score(yy,vv),nez_f1=f1_score(1-yy,1-vv.astype(int)),auroc=roc_auc_score(yy,pp),auprc=average_precision_score(yy,pp),
            accuracy=accuracy_score(yy,vv),ez_fraction_bias=float(vv.mean()-yy.mean()),tp=int((vv&(yy==1)).sum()),tn=int((~vv&(yy==0)).sum()),fp=int((vv&(yy==0)).sum()),fn=int((~vv&(yy==1)).sum()))
        assert len(yy)==7635 and all(abs(float(r[k])-val)<1e-12 for k,val in vals.items())
    raw=table('RAW_COVERAGE');assert len(raw)==45
    for arm in ARMS:
        z=[r for r in raw if r['arm']==arm];assert len(z)==5 and sum(int(r['patients']) for r in z)==80 and sum(int(r['channels']) for r in z)==7635
        assert sum(int(r['valid_windows']) for r in z)==1471965
        assert all(r['raw_consumed']=='True' and (r['reused_scientific_cell']=='True')==(arm in REF_NAMES) for r in z)
        uses_context=arm not in ['B0_ORIGINAL','WO_FEATURE_PR','WO_BOTH_PR','WO_WINDOW_PR']
        for row in z:
            active,used,fallback=[int(row[k]) for k in ['relative_active_coordinates','relative_used_coordinates','global_fallback_coordinates']]
            if uses_context:
                assert active==6*int(row['valid_windows']) and used+fallback==active and used>0
                assert 0<=int(row['singleton_valid_windows'])<=int(row['valid_windows'])
            else:
                assert active==used==fallback==int(row['singleton_valid_windows'])==0
    assert abs(matrices['B0_ORIGINAL'][:,0].mean()-.6538278218104221)<1e-12
    assert abs(matrices['B0_ORIGINAL'][:,1].mean()-.7522551973793955)<1e-12
    with (PREVROOT/'public/MAIN_RESULTS.csv').open(encoding='utf-8',newline='') as f:
        old=next(r for r in csv.DictReader(f) if r['arm']=='FULL_F0')
    assert all(abs(float(old[m])-matrices['B0_ORIGINAL'][:,j].mean())<1e-12 for j,m in enumerate(METRICS))
    componentrows=table('COMPONENT_EVIDENCE');assert len(componentrows)==7
    componentpass={}
    for name in primary:
        w=weights[name];means=sum(matrices[a]*x for a,x in w).mean(0)
        positive_folds=0
        for fold in range(1,6):
            point=sum(vector([r for r in rows if r['arm']==a and r['fold']==fold]).mean(0)*x for a,x in w)
            positive_folds+=int(point[0]>0 and point[1]>0)
        lows={m:next(float(r['low']) for r in family if r['contrast']==name and r['metric']==m) for m in ['macro_f1','auroc']}
        pointpositive=bool(means[0]>0 and means[1]>0);familypositive=min(lows.values())>0
        passed=pointpositive and familypositive and positive_folds>=4;componentpass[name]=passed
        r=next(r for r in componentrows if r['contrast']==name)
        assert abs(float(r['macro_f1_difference'])-means[0])<1e-12 and abs(float(r['auroc_difference'])-means[1])<1e-12
        assert int(r['folds_both_positive'])==positive_folds and int(r['folds'])==5
        assert (r['both_point_estimates_positive']=='True')==pointpositive
        assert (r['both_family_lower_bounds_positive']=='True')==familypositive
        assert (r['stable_within_this_seed_protocol']=='True')==passed
        assert abs(float(r['macro_family_low'])-lows['macro_f1'])<1e-12 and abs(float(r['auroc_family_low'])-lows['auroc'])<1e-12
    full,original=matrices['FULL'].mean(0),matrices['B0_ORIGINAL'].mean(0)
    expected=dict(main_macro_f1_above_0p64=bool(full[0]>.64),macro_f1_within_1pp_of_original=bool(full[0]-original[0]>=-.01),
        main_auroc_above_original=bool(full[1]>original[1]),
        all_retained_components_have_stable_two_metric_evidence=all(componentpass[k] for k in primary if k!='FULL_MINUS_B0'))
    goal=jr(PUB/'GOAL_AUDIT.json')
    assert goal['conditions']==expected and goal['status']==('PASS' if all(expected.values()) else 'FAIL')
    assert goal['selected_arm'] is None and not goal['seed_robustness_established']
    for key,value in [('full_macro_f1',full[0]),('full_auroc',full[1]),('original_macro_f1',original[0]),('original_auroc',original[1])]:
        assert abs(goal[key]-value)<1e-12
    write(PUB/'INDEPENDENT_OUTER_VERIFICATION.json',dict(status='PASS',patient_rows=720,patient_metric_checks=numeric,
        nominal_interval_checks=checks,primary_family_interval_checks=familychecks,maximum_interval_difference=maximum,
        all9_complete80_verified=True,all9_Raw_censuses_verified=True,reused_scores_unchanged=True,
        all45_model_selections_frozen=True,all_neural_models_single_head=True,all_head_interactions_absent=True,
        fixed_goal_audit_independently_recomputed=True,no_post_Outer_tuning=True))
    write(PUB/'OUTER_STATUS.json',dict(status='COMPLETE_VERIFIED_RGF0_NINE_GROUP_FIVE_FOLD_OUTER',arms=9,patients_each=80,channels_each=7635,
        new_neural_fits=40,reused_cells=5,logical_model_cells=45,patient_rows=720,metric_checks=numeric,selected_arm=None,
        scientific_status='EXPLORATORY_REPEATED_TEST',independent_confirmation=False,freeze_sha256=fh,
        score_seal_sha256=sha(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')))
    print('RGF0_ALL9_FULL80_INDEPENDENT_VERIFICATION_PASS',flush=True)

if __name__=='__main__':main()

