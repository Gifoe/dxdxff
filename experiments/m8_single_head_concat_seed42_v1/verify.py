"""Independent sealed-score, grouped-statistic, all10000-draw and decision replay."""
from report import *

def main():
    fr=read(ROOT/'MODEL_FREEZE_PRIVATE.pkl');binding=fr['binding'];prot=jr(HERE/'PROTOCOL_LOCK.json')
    assert jr(PUB/'TRAINING_VERIFICATION.json')['status']=='PASS' and jr(PUB/'ADMISSION.json')['status']=='PASS'
    for name,h in binding['source'].items():assert sha(HERE/name)==h
    choice=jr(PUB/'CANDIDATE_FROZEN.json');dev=read(ROOT/'DEVELOPMENT_PATIENT_METRICS_PRIVATE.pkl')
    selected,rows=choose(dev,prot);assert selected==choice['selected_model'] and clean(rows)==choice['scores']
    checked=0;grouped=0;absolute=0;adjusted=0;predrows=0
    for stage in ['development','outer']:
        frame=frames(stage);p=metrics(frame);stored=read(ROOT/(stage.upper()+'_PATIENT_METRICS_PRIVATE.pkl'))
        assert np.array_equal(p[METRICS].to_numpy(),stored[METRICS].to_numpy(),equal_nan=True);predrows+=len(p)
        out=PUB/'development' if stage=='development' else PUB
        tab=pd.read_csv(out/'MAIN_RESULTS.csv');diag=pd.read_csv(out/'DIAGNOSTIC_RESULTS.csv') if (out/'DIAGNOSTIC_RESULTS.csv').exists() else pd.DataFrame()
        if len(diag):tab=pd.concat([tab,diag],ignore_index=True)
        ftab=pd.read_csv(out/'FOLD_RESULTS.csv');ctab=pd.read_csv(out/'SOURCE_RESULTS.csv');stab=pd.read_csv(out/'FOLD_MEAN_SD.csv');cov=pd.read_csv(out/'METRIC_COVERAGE.csv')
        base=p[p.model=='REF_M8'].sort_values(['fold','patient']);ids=base.patient.to_numpy();unique=sorted(set(ids))
        draws=np.random.default_rng(42).integers(0,len(unique),(10000,len(unique)));mult=np.array([(draws==j).sum(1) for j in range(len(unique))]).T
        matrix={}
        for name,g in p.groupby('model',sort=True):
            g=g.sort_values(['fold','patient']);matrix[name]=g[METRICS].to_numpy(dtype=float)
            same=tab[tab.model==name].iloc[0];assert np.allclose(same[METRICS].to_numpy(dtype=float),np.nanmean(matrix[name],axis=0),atol=1e-12,rtol=0)
            foldmeans=[]
            for col,t in [('fold',ftab),('center',ctab)]:
                for key,h in g.groupby(col):
                    rr=t[(t.model==name)&(t[col]==key)].iloc[0];values=np.nanmean(h[METRICS].to_numpy(),axis=0)
                    assert np.allclose(values,rr[METRICS].to_numpy(dtype=float),atol=1e-12,rtol=0);grouped+=len(METRICS)
                    if col=='fold':foldmeans.append(values)
            values=np.array(foldmeans)
            for j,m in enumerate(METRICS):
                rr=stab[(stab.model==name)&(stab.metric==m)].iloc[0]
                assert np.allclose([np.nanmean(values[:,j]),np.nanstd(values[:,j],ddof=1)],[rr.fold_mean,rr.fold_sample_sd],atol=1e-12,rtol=0)
                rr=cov[(cov.model==name)&(cov.metric==m)].iloc[0];assert rr.finite_appearances==np.isfinite(matrix[name][:,j]).sum();grouped+=3
        def independent(a,b):
            valid=np.isfinite(a)&np.isfinite(b);d=np.where(valid,a-b,np.nan)
            num=np.array([np.nansum(d[ids==pid],axis=0) for pid in unique]);den=np.array([np.isfinite(d[ids==pid]).sum(0) for pid in unique])
            n=mult@den;dist=np.divide(mult@num,n,out=np.full(n.shape,np.nan),where=n>0)
            return np.nanmean(d,axis=0),dist,valid.sum(0)
        intervals=pd.read_csv(out/'PAIRED_COMPARISONS.csv')
        for (a,b),q in intervals.groupby(['model','reference']):
            mean,dist,n=independent(matrix[a],matrix[b]);lohi=np.nanquantile(dist,[.025,.975],axis=0)
            corrected=np.nanquantile(dist,[.05/16,1-.05/16],axis=0)
            for j,m in enumerate(METRICS):
                rr=q[q.metric==m].iloc[0]
                assert np.allclose([mean[j],*lohi[:,j]],[rr.delta,rr.ci_low,rr.ci_high],atol=1e-12,rtol=0,equal_nan=True)
                assert rr.valid_appearances==n[j]
                if 'selected8endpoint_bonferroni_ci_low' in rr and np.isfinite(rr.selected8endpoint_bonferroni_ci_low):
                    assert np.allclose(corrected[:,j],[rr.selected8endpoint_bonferroni_ci_low,rr.selected8endpoint_bonferroni_ci_high],atol=1e-12,rtol=0);adjusted+=1
                checked+=1
        intervals=pd.read_csv(out/'ABSOLUTE_INTERVALS.csv')
        for name,q in intervals.groupby('model'):
            mean,dist,n=independent(matrix[name],np.zeros_like(matrix[name]));lohi=np.nanquantile(dist,[.025,.975],axis=0)
            for j,m in enumerate(METRICS):
                rr=q[q.metric==m].iloc[0];assert np.allclose([mean[j],*lohi[:,j]],[rr['mean'],rr.ci_low,rr.ci_high],atol=1e-12,rtol=0,equal_nan=True);absolute+=1
    assert (checked,adjusted,absolute,predrows)==(369,16,216,1800)
    result=pd.read_csv(PUB/'MAIN_RESULTS.csv').set_index('model');decision=jr(PUB/'FINAL_DECISION.json')
    rows,allpass,primary=check_target(result.loc[selected],result.loc['REF_M8'])
    assert decision['selected_model']==selected and decision['all8_point_estimate_pass']==allpass and decision['all4_primary_classification_pass']==bool(primary)
    # CSV parsers may shift an unrounded endpoint by1ULP; authoritative decision comes from private score rows.
    pp=read(ROOT/'OUTER_PATIENT_METRICS_PRIVATE.pkl')
    aa={m:EV.nanmean(pp[pp.model==selected][m]) for m in METRICS};bb={m:EV.nanmean(pp[pp.model=='REF_M8'][m]) for m in METRICS};rr,ap,pr=check_target(aa,bb)
    assert clean(rr)==decision['metrics'] and bool(ap)==decision['all8_point_estimate_pass']
    prior=pd.read_csv(Path(r'D:\m8_joint_physiology_ablation_seed42_v1_runtime\public\outer\MAIN_RESULTS.csv'),dtype={'config':str})
    for name,code,head in [('REF_D1','000','D1'),('REF_M8','000','M8'),('REF_M8_NO_VAR','100','M8')]:
        old=prior[(prior.config==code)&(prior['head']==head)].iloc[0]
        assert np.allclose(result.loc[name,METRICS].to_numpy(dtype=float),old[METRICS].to_numpy(dtype=float),atol=1e-12,rtol=0)
    prior=pd.read_csv(Path(r'D:\m8_mechanism_ablations_seed42_v1_runtime\public\main_results.csv'))
    for name,oldname in [('REF_RAW_ONLY','M8_RAW_ONLY'),('REF_LATE_FUSION','A6_LATE_FUSION')]:
        old=prior[prior.model==oldname].iloc[0]
        assert np.allclose(result.loc[name,METRICS].to_numpy(dtype=float),old[METRICS].to_numpy(dtype=float),atol=1e-12,rtol=0)
    complexity=pd.read_csv(PUB/'COMPLEXITY.csv').set_index('model');cost=pd.read_csv(PUB/'OUTER_FORWARD_COST.csv')
    for i in range(5):
        arm=f'S{i}';assert complexity.loc[arm,'parameters']==(10377 if i>=3 else 9329)
        assert int(cost[cost.model==arm].valid_windows.sum())==1471965
    for path,h in {**binding['protected'],**binding['prior']['protected']}.items():assert sha(path)==h
    assert jr(ROOT/'FROZEN_BEFORE_OUTER_PRIVATE.json')['candidate_sha256']==sha(PUB/'CANDIDATE_FROZEN.json')
    files={str(p.relative_to(PUB)).replace('\\','/'):sha(p) for p in PUB.rglob('*') if p.is_file() and p.name!='INDEPENDENT_VERIFICATION.json'}
    write(PUB/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',Outer_patient_rows=1280,development_appearances=520,actual_new_models=25,
        paired_interval_metrics=checked,adjusted_intervals=adjusted,absolute_intervals=absolute,all10000_draws_independently_recomputed=True,
        grouped_numeric_fields=grouped,all5_reference_metrics_reproduced=True,all5_full80_window_MAC_coverage_verified=True,
        candidate_reconstructed_from_validation_only=selected,strict_final_decision_reproduced=True,protected_files_unchanged=True,
        source_sha256=sha(__file__),files=files,training_or_new_inference=False))
    print('SINGLE_HEAD_FULL_INDEPENDENT_VERIFICATION_PASS',selected,decision['status'],flush=True)

if __name__=='__main__':main()
