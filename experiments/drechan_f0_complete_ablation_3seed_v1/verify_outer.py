"""Independent sklearn endpoints, deterministic rank ties and resampling audit."""
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,accuracy_score
from common import *

def table(name):
    with (PUB/(name+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def main():
    fr=frozen();fh=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json');seal=jr(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')
    assert seal['freeze']==fh and len(seal['files'])==285 and seal['all285_cells_sealed_before_metric_reporting']
    ctx=read(ROOT/'METADATA_PRIVATE.pkl');meta=ctx['meta']
    archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz');assert sha(archive)==ctx['archive_sha256']
    with np.load(archive,allow_pickle=True) as q:y=1-q['y']
    stored=read(ROOT/'PATIENT_METRICS_PRIVATE.pkl');lookup={(r['arm'],r['seed'],r['fold'],r['patient']):r for r in stored}
    rows=[];numeric=0;pooled={(a,s):dict(y=[],p=[],v=[]) for a in ARMS for s in SEEDS}
    for r in fr['cells']:
        arm,seed,fold=r['arm'],r['seed'],r['fold'];key=key_for(arm,seed,fold);path=ROOT/'scores_private'/(key+'.pkl');assert sha(path)==seal['files'][key]
        q=read(path);ii=q['indices'];p=1-q['score_nez'].astype(float);v=q['score_nez']<r['threshold']
        assert q['checkpoint']==r['checkpoint_sha256'] and q['freeze']==fh and q['threshold']==r['threshold']
        state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)['model']
        assert not any('interaction' in name for name in state)
        if arm != 'RAW_ONLY': assert state['keep7'][4]==0
        if arm in MODEL.DROP: assert state['keep7'][MODEL.DROP[arm]]==0
        if r['reused']:
            z, source_hash, _ = reference_score(r)
            assert q['reused_source_sha256']==source_hash and np.array_equal(q['score_nez'],z['score_nez'])
            assert q['threshold']==z['threshold'] and np.array_equal(ii,z['indices'])
        for pid in sorted(set(meta['patient'][ii])):
            j=meta['patient'][ii]==pid;yy=y[ii][j];pp=p[j];vv=v[j];ch=meta['channel'][ii][j].astype(str)
            order=sorted(range(len(yy)),key=lambda k:(-float(pp[k]),ch[k]));first=next(k for k,i in enumerate(order) if yy[i])
            values=dict(macro_f1=f1_score(yy,vv,labels=[0,1],average='macro',zero_division=0),f1=f1_score(yy,vv,zero_division=0),nez_f1=f1_score(1-yy,1-vv.astype(int),zero_division=0),
                auroc=roc_auc_score(yy,pp),auprc=average_precision_score(yy,pp),accuracy=accuracy_score(yy,vv),mrr=1/(first+1),hit_at_1=float(first==0),ez_fraction_bias=float(vv.mean()-yy.mean()))
            row=lookup[(arm,seed,fold,str(pid))]
            for m,val in values.items():assert abs(row[m]-val)<1e-12;numeric+=1
            rows.append(dict(arm=arm,seed=seed,fold=fold,patient=str(pid),center=str(meta['center'][ii][j][0]),channels=int(j.sum()),**values))
        for k,a in [('y',y[ii]),('p',p),('v',v)]:pooled[(arm,seed)][k].append(a)

    assert len(rows)==len(lookup)==len(stored)==4560 and numeric==41040
    ids=sorted({r['patient'] for r in rows});assert len(ids)==80
    mat={}
    for arm in ARMS:
        for seed in SEEDS:
            rr=sorted([r for r in rows if r['arm']==arm and r['seed']==seed],key=lambda r:r['patient'])
            assert [r['patient'] for r in rr]==ids and sum(r['channels'] for r in rr)==7635
            mat[(arm,seed)]=vector(rr)
    matrices={a:sum(mat[(a,s)] for s in SEEDS)/3 for a in ARMS}
    def equal_metrics(row,values):
        assert all(abs(float(row[m])-values[j])<1e-12 for j,m in enumerate(METRICS))
    mainrows=table('MAIN_RESULTS');seedrows=table('SEED_RESULTS');seedsd=table('SEED_SD')
    foldrows=table('FOLD_RESULTS');foldsd=table('FOLD_SD')
    assert len(mainrows)==len(seedsd)==19 and len(seedrows)==len(foldsd)==57 and len(foldrows)==285
    for arm in ARMS:
        equal_metrics(next(r for r in mainrows if r['arm']==arm),matrices[arm].mean(0))
        equal_metrics(next(r for r in seedsd if r['arm']==arm),np.std([mat[(arm,s)].mean(0) for s in SEEDS],axis=0,ddof=1))
        for seed in SEEDS:
            equal_metrics(next(r for r in seedrows if r['arm']==arm and int(r['seed'])==seed),mat[(arm,seed)].mean(0))
            ff=[]
            for fold in range(1,6):
                rr=[r for r in rows if r['arm']==arm and r['seed']==seed and r['fold']==fold]
                assert len(rr)==[16,16,17,15,16][fold-1];point=vector(rr).mean(0);ff.append(point)
                equal_metrics(next(r for r in foldrows if r['arm']==arm and int(r['seed'])==seed and int(r['fold'])==fold),point)
            equal_metrics(next(r for r in foldsd if r['arm']==arm and int(r['seed'])==seed),np.std(ff,axis=0,ddof=1))
    for name,perseed in [('SOURCE_RESULTS',False),('SOURCE_SEED_RESULTS',True)]:
        for r in table(name):
            rr=[x for x in rows if x['arm']==r['arm'] and x['center']==r['center'] and (not perseed or x['seed']==int(r['seed']))]
            n=len(rr) if perseed else len(rr)//3
            assert n==int(r['patients']) and (r['suppressed']=='True')==(n<5)
            if n>=5:equal_metrics(r,vector(rr).mean(0))
            else:assert all(np.isnan(float(r[m])) for m in METRICS)
    cov=table('METRIC_COVERAGE');assert len(cov)==513 and all(int(r['finite_patients'])==int(r['patients'])==80 for r in cov)
    coefficients={('absolute',a):{a:1} for a in ARMS}
    for a in ARMS:
        if a!='FULL':coefficients[('paired','FULL_MINUS_'+a)]={'FULL':1,a:-1}
    coefficients[('interaction','PR_INTERACTION')]={'FULL':1,'WO_FEATURE_PR':-1,'WO_RAW_PR':-1,'WO_BOTH_PR':1}
    coefficients[('interaction','GUIDANCE_INTERACTION')]={'FULL':1,'WO_EARLY_FILM':-1,'WO_LATE_GATE':-1,'WO_BOTH_GUIDANCE':1}
    assert len(coefficients)==39
    draws=np.random.default_rng(42).integers(0,80,(10000,80));ci=table('INTERVALS');family=table('PRIMARY_FAMILY_INTERVALS')
    assert len(ci)==351 and len(family)==36;checks=familychecks=0;maximum=0.
    for (kind,label),weights in coefficients.items():
        for j,m in enumerate(METRICS):
            values=sum(matrices[a][:,j]*w for a,w in weights.items());samples=values[draws].mean(1)
            lo,hi=np.quantile(samples,[.025,.975]);mean=values.mean()
            r=next(r for r in ci if r['kind']==kind and r['contrast']==label and r['metric']==m)
            error=max(abs(float(r['mean'])-mean),abs(float(r['low'])-lo),abs(float(r['high'])-hi))
            assert error<1e-12 and int(r['patients'])==80 and int(r['draws'])==10000
            maximum=max(maximum,error);checks+=1
            if kind=='paired' and m in ['macro_f1','auroc']:
                tail=.05/72;lo,hi=np.quantile(samples,[tail,1-tail])
                r=next(r for r in family if r['contrast']==label and r['metric']==m)
                assert int(r['comparisons'])==36 and abs(float(r['tail'])-tail)<1e-15
                assert max(abs(float(r['mean'])-mean),abs(float(r['low'])-lo),abs(float(r['high'])-hi))<1e-12
                familychecks+=1
    assert checks==351 and familychecks==36
    poolrows=table('POOLED_RESULTS');assert len(poolrows)==57
    for r in poolrows:
        z=pooled[(r['arm'],int(r['seed']))];yy,pp,vv=[np.concatenate(z[k]) for k in ['y','p','v']]
        vals=dict(macro_f1=f1_score(yy,vv,average='macro'),f1=f1_score(yy,vv),nez_f1=f1_score(1-yy,1-vv.astype(int)),
            auroc=roc_auc_score(yy,pp),auprc=average_precision_score(yy,pp),accuracy=accuracy_score(yy,vv),
            ez_fraction_bias=float(vv.mean()-yy.mean()),tp=int((vv&(yy==1)).sum()),tn=int((~vv&(yy==0)).sum()),
            fp=int((vv&(yy==0)).sum()),fn=int((~vv&(yy==1)).sum()))
        assert len(yy)==7635 and all(abs(float(r[k])-v)<1e-12 for k,v in vals.items())
    for r in table('POOLED_SEED_MEAN'):
        for m in METRICS:
            if m not in ['mrr','hit_at_1']:
                assert abs(float(r[m])-np.mean([float(z[m]) for z in poolrows if z['arm']==r['arm']]))<1e-12
    raw=table('RAW_COVERAGE');assert len(raw)==285
    for arm in ARMS:
        for seed in SEEDS:
            z=[r for r in raw if r['arm']==arm and int(r['seed'])==seed]
            assert len(z)==5 and sum(int(r['patients']) for r in z)==80 and sum(int(r['channels']) for r in z)==7635
            assert sum(int(r['valid_windows']) for r in z)==(0 if arm=='PHYSIOLOGY_ONLY' else 1471965)
            uses=arm not in ['B0_ORIGINAL','WO_FEATURE_PR','WO_BOTH_PR','WO_WINDOW_PR','WO_BOTH_GUIDANCE','RAW_ONLY','PHYSIOLOGY_ONLY']
            for r in z:
                active,used,fallback=[int(r[k]) for k in ['relative_active_coordinates','relative_used_coordinates','global_fallback_coordinates']]
                assert (r['raw_consumed']=='True')==(arm!='PHYSIOLOGY_ONLY')
                if uses:
                    assert active==(5 if arm in MODEL.DROP else 6)*int(r['valid_windows']) and used+fallback==active and used>0
                else:assert active==used==fallback==int(r['relative_singleton_valid_windows'])==0
                if arm=='EARLY_FILM_MISMATCH':
                    changed,eligible,single=[int(r[k]) for k in ['mismatch_changed_valid_windows','mismatch_eligible_valid_windows','mismatch_singleton_valid_windows']]
                    assert changed==eligible and eligible+single==int(r['valid_windows'])
                    other=next(q for q in raw if q['arm']==arm and int(q['seed'])==42 and q['fold']==r['fold'])
                    assert all(r[k]==other[k] for k in ['mismatch_changed_valid_windows','mismatch_eligible_valid_windows','mismatch_singleton_valid_windows'])
    directions=table('FOLD_SEED_DIRECTIONS');evidence=table('COMPONENT_EVIDENCE')
    assert len(directions)==270 and len(evidence)==18
    for a in ARMS:
        if a=='FULL':continue
        name='FULL_MINUS_'+a;mp=matrices['FULL'].mean(0)-matrices[a].mean(0)
        sp=fp=0
        for seed in SEEDS:
            diff=mat[('FULL',seed)].mean(0)-mat[(a,seed)].mean(0);sp+=int(diff[0]>0 and diff[1]>0)
            for fold in range(1,6):
                diff=vector([r for r in rows if r['arm']=='FULL' and r['seed']==seed and r['fold']==fold]).mean(0)-vector([r for r in rows if r['arm']==a and r['seed']==seed and r['fold']==fold]).mean(0)
                r=next(r for r in directions if r['contrast']==name and int(r['seed'])==seed and int(r['fold'])==fold)
                equal_metrics(r,diff);fp+=int(diff[0]>0 and diff[1]>0)
        r=next(r for r in evidence if r['contrast']==name)
        assert abs(float(r['macro_f1_difference'])-mp[0])<1e-12 and abs(float(r['auroc_difference'])-mp[1])<1e-12
        assert int(r['seeds_both_positive'])==sp and int(r['fold_seed_cells_both_positive'])==fp
        assert (r['both_point_estimates_positive']=='True')==bool(mp[0]>0 and mp[1]>0)
        lows={m:next(float(q['low']) for q in family if q['contrast']==name and q['metric']==m) for m in ['macro_f1','auroc']}
        assert abs(float(r['macro_family_low'])-lows['macro_f1'])<1e-12 and abs(float(r['auroc_family_low'])-lows['auroc'])<1e-12
        assert (r['both_family_lower_bounds_positive']=='True')==(min(lows.values())>0)
    with (current.PUB/'MAIN_RESULTS.csv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):equal_metrics(r,mat[(r['arm'],42)].mean(0))
    goal=jr(PUB/'GOAL_AUDIT.json');full,b0=matrices['FULL'].mean(0),matrices['B0_ORIGINAL'].mean(0)
    assert goal['selected_arm'] is None and goal['status']=='DESCRIPTIVE_NOT_SELECTION_GATE'
    assert abs(goal['full_macro_f1']-full[0])<1e-12 and abs(goal['full_auroc']-full[1])<1e-12
    assert goal['main_macro_f1_above_0p64']==bool(full[0]>.64) and goal['macro_within1pp_B0']==bool(full[0]-b0[0]>=-.01)
    assert goal['main_auroc_above_B0']==bool(full[1]>b0[1])
    write(PUB/'INDEPENDENT_OUTER_VERIFICATION.json',dict(status='PASS',patient_rows=4560,patient_metric_checks=numeric,
        nominal_interval_checks=351,primary_family_interval_checks=36,maximum_interval_difference=maximum,
        all19_3seed_complete80_verified=True,reused_scores_unchanged=True,all285_model_selections_frozen=True,
        metric_seed_averaging_not_ensemble=True,unique_patient_resampling=True,no_post_Outer_tuning=True))
    write(PUB/'OUTER_STATUS.json',dict(status='COMPLETE_VERIFIED_CFA3_19_GROUP_3SEED_FIVE_FOLD_OUTER',arms=19,seeds=3,
        patients_each=80,channels_each=7635,logical_model_cells=285,patient_rows=4560,metric_checks=numeric,selected_arm=None,
        scientific_status='EXPLORATORY_REPEATED_TEST',independent_confirmation=False,freeze_sha256=fh,
        score_seal_sha256=sha(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')))
    print('CFA3_ALL19_3SEED_INDEPENDENT_VERIFICATION_PASS',flush=True)

if __name__=='__main__':main()
