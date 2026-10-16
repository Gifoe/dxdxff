"""Independent score-statistic, BH, coverage and recurrence verification."""
import json
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control,wilcoxon
from sklearn.metrics import roc_auc_score
import atlas as q


def rank_effect(x,y):
    good=np.isfinite(x);a=np.asarray(x)[good];b=np.asarray(y)[good]
    return float(2*roc_auc_score(b,a)-1) if len(np.unique(b))==2 else np.nan


def independently_interval(v):
    a=np.asarray(v,float);a=a[np.isfinite(a)]
    if not len(a):return [np.nan]*3
    ids=np.random.default_rng(42).integers(len(a),size=(10000,len(a)))
    return [float(np.mean(a)),*np.quantile(np.mean(a[ids],axis=1),[.025,.975]).tolist()]


def main():
    root=q.ROOT/'rolecount_checked';pub=root/'public';status=q.jr(pub/'ATLAS_STATUS.json');binding=q.jr(root/'INPUT_FREEZE_PRIVATE.json')
    assert status['status']=='COMPLETE_DESCRIPTIVE_PHYSIOLOGICAL_EVIDENCE_ATLAS'
    assert q.sha(root/'INPUT_FREEZE_PRIVATE.json')==status['input_freeze_sha256']
    for section in ['code','admission_guard_amendment']:
        for name,digest in binding[section].items():assert q.sha(q.HERE/name)==digest
    assert binding['exact_original_FIT_counts']=={'1':51,'2':51,'3':50,'4':52,'5':51}
    assert q.sha(root/'NEW_COORDINATE_PATIENT_EFFECTS_PRIVATE.pkl')==status['private_new_effects_sha256']
    rec=q.read(q.BASE/'RECONSTRUCTION_PRIVATE.pkl');labels,fit=q.legal_primary(rec,binding)
    primary=set(rec['meta']['split_patient'][(rec['meta']['split_fold']==1)&(rec['meta']['split_role']=='fit')]);assert fit==primary and len(fit)==51
    data=q.read(root/'NEW_COORDINATE_PATIENT_EFFECTS_PRIVATE.pkl');frame=data['patient_effects'];runs=data['record_effects']
    assert set(frame.patient)==fit and not frame.duplicated(['patient','coordinate']).any()
    assert not runs.duplicated(['patient','coordinate','seizure']).any()
    measurements=q.read(q.QUAL/'MEASUREMENTS_PRIVATE.pkl');checked=run_checked=0;maximum=0.
    for record in measurements['records']:
        ix=record['archive_indices'];pid=np.unique(rec['meta']['patient'][ix])[0]
        if pid not in fit:continue
        y=np.array([labels[int(i)] for i in ix]);values=measurements['values'][ix]
        for c,name in enumerate(q.COORDS):
            expected=rank_effect(values[:,c],y);row=frame[(frame.patient==pid)&(frame.coordinate==name)].iloc[0]
            np.testing.assert_allclose(expected,row.effect,atol=1e-12,rtol=0,equal_nan=True)
            if np.isfinite(expected):maximum=max(maximum,abs(expected-row.effect))
            assert row.finite_EZ_channels==((y==1)&np.isfinite(values[:,c])).sum()
            assert row.finite_NEZ_channels==((y==0)&np.isfinite(values[:,c])).sum();checked+=1
            j=c//2
            if c%2 and j not in [2,3]:continue
            for s in range(record['present'].shape[0]):
                if j in [2,3]:
                    k=j-2
                    x=record['observed'][s,:,k].astype(float) if c%2 else record['fraction'][s,:,k].copy()
                    if c%2:x[~record['eligible'][s,:,k]]=np.nan
                else:x=record['scalar'][s,:,j].copy()
                x[~record['present'][s]]=np.nan
                expected=rank_effect(x,y);row=runs[(runs.patient==pid)&(runs.coordinate==name)&(runs.seizure==s)].iloc[0]
                np.testing.assert_allclose(expected,row.effect,atol=1e-12,rtol=0,equal_nan=True);run_checked+=1
    table=pd.read_csv(pub/'NEW12_PATIENT_ASSOCIATIONS.csv').set_index('coordinate')
    source=pd.read_csv(pub/'NEW12_SOURCE_ASSOCIATIONS.csv');repeat=pd.read_csv(pub/'NEW12_REPEAT_SEIZURE_SUPPORT.csv').set_index('coordinate')
    for name,g in frame.groupby('coordinate'):
        row=table.loc[name];np.testing.assert_allclose(independently_interval(g.effect),[row.mean_effect,row.ci_low,row.ci_high],atol=1e-12,rtol=0,equal_nan=True)
        values=g.effect.to_numpy();values=values[np.isfinite(values)]
        p=1. if not len(values) or np.max(np.abs(values))==0 else wilcoxon(values,zero_method='wilcox',alternative='two-sided',method='auto').pvalue
        assert abs(p-row.pvalue)<=1e-12 and len(values)==row.patients
        eligible=same=0
        for center,h in g.groupby('center'):
            actual=source[(source.coordinate==name)&(source.center==center)].iloc[0]
            np.testing.assert_allclose(independently_interval(h.effect),[actual.mean_effect,actual.ci_low,actual.ci_high],atol=1e-12,rtol=0,equal_nan=True)
            assert h.effect.notna().sum()==actual.finite_patients
            if actual.finite_patients>=5:eligible+=1;same+=np.sign(actual.mean_effect)==np.sign(row.mean_effect)
        assert eligible==row.eligible_source_groups and same==row.same_direction_source_groups
        actual=repeat.loc[name];own=runs[runs.coordinate==name];directed=[];legacy=[];finite=zero=0
        for pid,h in own.groupby('patient'):
            a=h.effect.to_numpy();a=a[np.isfinite(a)];finite+=len(a);zero+=int((a==0).sum());nonzero=a[a!=0]
            if len(a)>=2:directed.append(float(((a>0) if row.mean_effect>0 else (a<0) if row.mean_effect<0 else np.zeros(len(a),bool)).mean()))
            if len(nonzero)>=2:legacy.append(float(max((nonzero>0).mean(),(nonzero<0).mean())))
        assert finite==actual.finite_record_effects and zero==actual.zero_record_effects
        assert len(directed)==actual.repeated_finite_seizure_patients and len(legacy)==actual.repeated_nonzero_seizure_patients
        np.testing.assert_allclose([np.median(directed) if directed else np.nan,np.median(legacy) if legacy else np.nan],
            [actual.median_inclusive_directed_agreement,actual.median_legacy_zero_excluding_majority],atol=1e-12,rtol=0,equal_nan=True)
        association=row.BH_q_new12<.05 and (row.ci_low>0 or row.ci_high<0) and row.patient_direction_fraction>=.6 and same>=3
        recurrence=actual.record_analogue_applicable and len(directed)>=10 and actual.median_inclusive_directed_agreement>=.7
        assert association==row.association_support and recurrence==actual.repeated_direction_support
    np.testing.assert_allclose(false_discovery_control(table.pvalue.to_numpy(),method='bh'),table.BH_q_new12,atol=1e-12,rtol=0)
    atlas=pd.read_csv(pub/'PRIMARY172_ASSOCIATION_ATLAS.csv');assert len(atlas)==172
    np.testing.assert_allclose(false_discovery_control(atlas.pvalue.to_numpy(),method='bh'),atlas.exploratory_joint172_BH_q,atol=1e-12,rtol=0)
    old=q.jr(pub/'OLD_EFFECT_STATISTICAL_AUDIT.json');assert old['status']=='PASS' and old['effect_rows_reaggregated']==800 and old['source_rows_reaggregated']==3200
    assert checked==len(frame)==612 and run_checked==len(runs)==1296
    assert table.association_support.sum()==status['new_association_support_count']==10
    assert repeat.record_analogue_applicable.sum()==8 and repeat.repeated_direction_support.sum()==status['new_repeated_direction_support_count']==3
    assert q.sha(q.OLD/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')==binding['parent_Outer_seal']
    result=dict(status='PASS',new_patient_effects_independently_replayed=checked,new_record_effects_independently_replayed=run_checked,
        independent_effect_engine='sklearn_ROC_concordance_vs_MannWhitney_U',maximum_effect_error=maximum,
        independent_BH_engine='scipy_false_discovery_control',new12_and_joint172_families_checked=True,
        all_source_and_patient_bootstrap_intervals_checked=True,all_recurrence_counts_and_flags_checked=True,
        exact_original_role_counts_verified=True,primary_patients=51,model_fits=0,new_Outer_access=False,
        original_failed_admission_preserved=True,source_sha256=q.sha(q.HERE/'verify_atlas.py'))
    q.write(pub/'INDEPENDENT_ATLAS_VERIFICATION.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
