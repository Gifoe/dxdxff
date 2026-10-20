"""Independent sklearn endpoints, deterministic rank ties and resampling audit."""
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,accuracy_score
from common import *

def table(name):
    with (PUB/(name+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def main():
    fr=frozen();fh=sha(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json');seal=jr(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')
    assert seal['freeze']==fh and len(seal['files'])==25 and seal['all25_cells_sealed_before_metric_reporting']
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
        if arm not in MODEL.REFERENCES:assert not any(n.startswith('gate.') for n in state)
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


    assert len(rows)==len(lookup)==len(stored)==400 and numeric==3600 and all(r['seed']==42 for r in rows)
    ids=sorted({r['patient'] for r in rows});assert len(ids)==80
    matrices={}
    for arm in ARMS:
        rr=sorted([r for r in rows if r['arm']==arm],key=lambda r:r['patient'])
        assert [r['patient'] for r in rr]==ids and sum(r['channels'] for r in rr)==7635
        matrices[arm]=vector(rr)
    def equal_metrics(r,v):
        assert all(abs(float(r[m])-v[j])<1e-12 for j,m in enumerate(METRICS))
    mainrows=table('MAIN_RESULTS');foldrows=table('FOLD_RESULTS');sds=table('FOLD_SD')
    assert len(mainrows)==len(sds)==5 and len(foldrows)==25
    for arm in ARMS:
        equal_metrics(next(r for r in mainrows if r['arm']==arm),matrices[arm].mean(0));ff=[]
        for fold in range(1,6):
            z=[r for r in rows if r['arm']==arm and r['fold']==fold];assert len(z)==[16,16,17,15,16][fold-1]
            point=vector(z).mean(0);ff.append(point)
            equal_metrics(next(r for r in foldrows if r['arm']==arm and int(r['fold'])==fold),point)
        equal_metrics(next(r for r in sds if r['arm']==arm),np.std(ff,axis=0,ddof=1))
    for r in table('SOURCE_RESULTS'):
        z=[q for q in rows if q['arm']==r['arm'] and q['center']==r['center']]
        assert int(r['patients'])==len(z) and (r['suppressed']=='True')==(len(z)<5)
        if len(z)>=5:equal_metrics(r,vector(z).mean(0))
        else:assert all(np.isnan(float(r[m])) for m in METRICS)
    cov=table('METRIC_COVERAGE');assert len(cov)==45 and all(int(r['finite_patients'])==int(r['patients'])==80 for r in cov)
    weights={('absolute',a):{a:1} for a in ARMS}
    for a in ARMS:
        if a!='FULL':weights[('paired','FULL_MINUS_'+a)]={'FULL':1,a:-1}
    weights[('paired','A2_MINUS_UNMATCHED_NO_EARLY')]={'A2_CAPACITY_MATCHED':1,'WO_EARLY_FILM':-1}
    assert len(weights)==10 and not INTERACTIONS
    draws=np.random.default_rng(42).integers(0,80,(10000,80));ci=table('INTERVALS');family=table('PRIMARY_FAMILY_INTERVALS')
    assert len(ci)==90 and len(family)==10;checks=familychecks=0;maximum=0.
    for (kind,label),w in weights.items():
        for j,m in enumerate(METRICS):
            v=sum(matrices[a][:,j]*x for a,x in w.items());samples=v[draws].mean(1);mean=v.mean();lo,hi=np.quantile(samples,[.025,.975])
            r=next(r for r in ci if r['kind']==kind and r['contrast']==label and r['metric']==m)
            error=max(abs(float(r['mean'])-mean),abs(float(r['low'])-lo),abs(float(r['high'])-hi))
            assert error<1e-12 and int(r['patients'])==80 and int(r['draws'])==10000;maximum=max(maximum,error);checks+=1
            if kind=='paired' and m in ['macro_f1','auroc']:
                tail=.05/20;lo,hi=np.quantile(samples,[tail,1-tail])
                r=next(r for r in family if r['contrast']==label and r['metric']==m)
                assert int(r['comparisons'])==10 and abs(float(r['tail'])-tail)<1e-15
                assert max(abs(float(r['mean'])-mean),abs(float(r['low'])-lo),abs(float(r['high'])-hi))<1e-12;familychecks+=1
    assert checks==90 and familychecks==10
    for r in table('POOLED_RESULTS'):
        z=pooled[(r['arm'],42)];yy,pp,vv=[np.concatenate(z[k]) for k in ['y','p','v']]
        values=dict(macro_f1=f1_score(yy,vv,average='macro'),f1=f1_score(yy,vv),nez_f1=f1_score(1-yy,1-vv.astype(int)),
            auroc=roc_auc_score(yy,pp),auprc=average_precision_score(yy,pp),accuracy=accuracy_score(yy,vv),
            ez_fraction_bias=float(vv.mean()-yy.mean()),tp=int((vv&(yy==1)).sum()),tn=int((~vv&(yy==0)).sum()),
            fp=int((vv&(yy==0)).sum()),fn=int((~vv&(yy==1)).sum()))
        assert len(yy)==7635 and all(abs(float(r[k])-v)<1e-12 for k,v in values.items())
    raw=table('RAW_COVERAGE');assert len(raw)==25
    for arm in ARMS:
        z=[r for r in raw if r['arm']==arm];assert len(z)==5 and sum(int(r['patients']) for r in z)==80 and sum(int(r['channels']) for r in z)==7635
        assert sum(int(r['valid_windows']) for r in z)==(0 if arm=='PHYSIOLOGY_ONLY' else 1471965)
        uses=arm not in ['WO_EARLY_FILM','A2_CAPACITY_MATCHED']
        for r in z:
            active,used,fallback=[int(r[k]) for k in ['relative_active_coordinates','relative_used_coordinates','global_fallback_coordinates']]
            if uses:assert active==(5 if arm in MODEL.DROP else 6)*int(r['valid_windows']) and used+fallback==active and used>0
            else:assert active==used==fallback==int(r['relative_singleton_valid_windows'])==0
            if arm=='EARLY_FILM_MISMATCH':
                changed,eligible,single=[int(r[k]) for k in ['mismatch_changed_valid_windows','mismatch_eligible_valid_windows','mismatch_singleton_valid_windows']]
                assert changed==eligible and eligible+single==int(r['valid_windows'])
    details=table('FOLD_DIRECTIONS');evidence=table('COMPONENT_EVIDENCE');assert len(details)==25 and len(evidence)==5
    for label,w in {label:w for (kind,label),w in weights.items() if kind=='paired'}.items():
        mean=sum(matrices[a].mean(0)*x for a,x in w.items());positive=0
        for fold in range(1,6):
            point=sum(vector([r for r in rows if r['arm']==a and r['fold']==fold]).mean(0)*x for a,x in w.items())
            equal_metrics(next(r for r in details if r['contrast']==label and int(r['fold'])==fold),point)
            positive+=int(point[0]>0 and point[1]>0)
        r=next(r for r in evidence if r['contrast']==label)
        assert abs(float(r['macro_f1_difference'])-mean[0])<1e-12 and abs(float(r['auroc_difference'])-mean[1])<1e-12
        assert int(r['folds_both_positive'])==positive and (r['both_point_estimates_positive']=='True')==bool(mean[0]>0 and mean[1]>0)
        lows={m:next(float(q['low']) for q in family if q['contrast']==label and q['metric']==m) for m in ['macro_f1','auroc']}
        assert abs(float(r['macro_family_low'])-lows['macro_f1'])<1e-12 and abs(float(r['auroc_family_low'])-lows['auroc'])<1e-12
        assert (r['both_family_lower_bounds_positive']=='True')==(min(lows.values())>0)
    with (parent.PUB/'MAIN_RESULTS.csv').open(encoding='utf-8',newline='') as f:old=[r for r in csv.DictReader(f) if int(r['seed'])==42]
    for a,original in MODEL.REUSE.items():equal_metrics(next(r for r in old if r['arm']==original),matrices[a].mean(0))
    assert abs(matrices['FULL'][:,0].mean()-.6534651290238689)<1e-12 and abs(matrices['FULL'][:,1].mean()-.7652863490156572)<1e-12
    goal=jr(PUB/'GOAL_AUDIT.json');assert goal['selected_arm'] is None and goal['status']=='DESCRIPTIVE_NOT_SELECTION_GATE'
    assert abs(goal['full_macro_f1']-matrices['FULL'][:,0].mean())<1e-12 and abs(goal['full_auroc']-matrices['FULL'][:,1].mean())<1e-12
    write(PUB/'INDEPENDENT_OUTER_VERIFICATION.json',dict(status='PASS',patient_rows=400,patient_metric_checks=numeric,
        nominal_interval_checks=90,primary_family_interval_checks=10,maximum_interval_difference=maximum,
        all5_complete80_verified=True,reused_scores_unchanged=True,all25_model_selections_frozen=True,
        all5_arms_without_late_gate=True,no_extra_aliases=True,seed42_only=True,no_post_Outer_tuning=True))
    write(PUB/'OUTER_STATUS.json',dict(status='COMPLETE_VERIFIED_PCA_5_GROUP_SEED42_FIVE_FOLD_OUTER',arms=5,seed=42,
        patients_each=80,channels_each=7635,logical_model_cells=25,patient_rows=400,metric_checks=numeric,selected_arm=None,
        scientific_status='EXPLORATORY_REPEATED_TEST',independent_confirmation=False,freeze_sha256=fh,
        score_seal_sha256=sha(ROOT/'OUTER_SCORE_SEAL_PRIVATE.json')))
    print('PCA_ALL5_SEED42_INDEPENDENT_VERIFICATION_PASS',flush=True)

if __name__=='__main__':main()
