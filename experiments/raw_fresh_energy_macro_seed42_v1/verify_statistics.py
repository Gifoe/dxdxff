"""Independent all-CI/fold/source/coverage replay on already verified metrics."""
import numpy as np
import pandas as pd
import final_stage as f
c=f.c


def main():
    f.setup();base=c.jr(f.PUBLIC/'INDEPENDENT_FINAL_VERIFICATION.json');assert base['status']=='PASS' and base['patient_metric_rows']==560
    assert base['score_seal_sha256']==c.sha(f.FINAL/'OUTER_SCORE_SEAL_PRIVATE.json')
    frame=pd.DataFrame(c.read(f.FINAL/'OUTER_PATIENT_METRICS_PRIVATE.pkl'));ids=sorted(frame.patient.unique());assert len(ids)==80
    draw=np.random.default_rng(42).integers(0,80,(10000,80));values={};maximum=0.;checks=0
    for procedure in f.PROCEDURES:
        g=frame[frame.procedure==procedure].sort_values('patient');assert g.patient.tolist()==ids
        values[procedure]=g[c.METRICS].to_numpy(float);assert np.isfinite(values[procedure]).all()
    def compare(reported,x):
        nonlocal maximum,checks
        limits=np.quantile(x[draw].mean(1),[.025,.975],axis=0);mean=x.mean(0)
        for j,m in enumerate(c.METRICS):
            row=reported[reported.metric==m];assert len(row)==1;r=row.iloc[0]
            error=max(abs(float(r['mean'])-mean[j]),abs(float(r.low)-limits[0,j]),abs(float(r.high)-limits[1,j]))
            maximum=max(maximum,error);assert error<1e-12 and int(r.draws)==10000 and int(r.unique_patients)==80;checks+=1
    absolute=pd.read_csv(f.PUBLIC/'OUTER_ABSOLUTE_CI.csv')
    for p in f.PROCEDURES:compare(absolute[absolute.procedure==p],values[p])
    for (left,right),reported in pd.read_csv(f.PUBLIC/'OUTER_PAIRED_CI.csv').groupby(['left','right']):compare(reported,values[left]-values[right])
    interaction=values['PRE_LOGENERGY']-values['PRE_MEAN']-values['WINDOW_LOGENERGY']+values['WINDOW_MEAN']
    compare(pd.read_csv(f.PUBLIC/'OUTER_FACTORIAL_INTERACTION_CI.csv'),interaction)
    actual_folds=frame.groupby(['procedure','fold'])[c.METRICS].mean().sort_index()
    public_folds=pd.read_csv(f.PUBLIC/'OUTER_FOLDS.csv').set_index(['procedure','fold']).sort_index()
    assert np.max(np.abs(actual_folds.to_numpy()-public_folds[c.METRICS].to_numpy()))<1e-12
    actual_sd=actual_folds.groupby('procedure').std(ddof=1).sort_index()
    public_sd=pd.read_csv(f.PUBLIC/'OUTER_FOLD_SD.csv').set_index('procedure').sort_index()
    assert np.max(np.abs(actual_sd.to_numpy()-public_sd[c.METRICS].to_numpy()))<1e-12
    source=pd.read_csv(f.PUBLIC/'OUTER_SOURCE.csv');source_checks=0
    for (p,center),g in frame.groupby(['procedure','center']):
        r=source[(source.procedure==p)&(source.center==center)].iloc[0];n=g.patient.nunique()
        assert int(r.patients)==n and bool(r.suppressed)==(n<5)
        if n<5:assert r[c.METRICS].isna().all()
        else:
            for m in c.METRICS:assert abs(float(r[m])-float(g[m].mean()))<1e-12
        source_checks+=1
    coverage=pd.read_csv(f.PUBLIC/'OUTER_METRIC_COVERAGE.csv')
    for _,r in coverage.iterrows():
        g=frame[frame.procedure==r.procedure];assert int(r.total_patients)==80
        assert int(r.finite_patients)==int(np.isfinite(g[r.metric]).sum())==80
    c.write(f.PUBLIC/'INDEPENDENT_STATISTICS_VERIFICATION.json',dict(status='PASS',independent_bootstrap_metric_checks=checks,
        all_absolute_paired_and_factorial_CIs=True,maximum_bootstrap_difference=maximum,fold_means_and_sampleSD_exact=True,
        source_groups_checked=source_checks,small_source_suppression_verified=True,all9_metric_coverage_80perprocedure=True,
        metric_artifact_sha256=c.sha(f.FINAL/'OUTER_PATIENT_METRICS_PRIVATE.pkl'),score_seal_sha256=base['score_seal_sha256'],
        verifier_source_sha256=c.sha(c.Path(__file__)),no_training_or_prediction=True,scope='already_independently_verified_patient_metrics'))
    print('FRESH_RAW_ALL_STATISTICS_INDEPENDENT_PASS',checks,'CI metric checks',flush=True)


if __name__=='__main__':main()
