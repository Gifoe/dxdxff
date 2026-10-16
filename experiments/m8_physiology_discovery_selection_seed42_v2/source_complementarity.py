"""Exploratory fixed-score source contrasts and Raw compensation diagnostics.

Registered model pairs only. This added reporting decomposition cannot select,
train, rerun predictions, change thresholds, or define a new winning model.
"""
from common import *


def intervals(a):
    a=np.asarray(a,float);n=len(a);draw=np.random.default_rng(42).integers(n,size=(10000,n))
    dist=np.nanmean(a[draw],axis=1)
    return np.nanmean(a,0),np.nanpercentile(dist,[2.5,97.5],axis=0)


def main():
    assert jr(PUB/'INDEPENDENT_RESULTS_VERIFICATION.json')['status']=='PASS'
    frame=read(ROOT/'OUTER_PATIENT_METRICS_PRIVATE.pkl');sources=[];compensation=[]
    pairs=pd.read_csv(PUB/'outer/PAIRED_COMPARISONS.csv')[['model','reference']].drop_duplicates()
    for (center,g) in frame.groupby('center'):
        for _,r in pairs.iterrows():
            a=g[g.model==r.model].sort_values('patient');b=g[g.model==r.reference].sort_values('patient')
            assert list(a.patient)==list(b.patient)
            d=a[METRICS].to_numpy(float)-b[METRICS].to_numpy(float);mu,ci=intervals(d)
            for j,m in enumerate(METRICS):sources.append(dict(center=center,model=r.model,reference=r.reference,metric=m,
                patients=len(a),delta=mu[j],ci_low=ci[0,j],ci_high=ci[1,j],draws=10000,exploratory=True,multiple_source_tests_not_corrected=True))
    mat={name:g.sort_values('patient')[METRICS].to_numpy(float) for name,g in frame.groupby('model')}
    for family in ['M8','Concat']:
        for reference in ['OLD7','BEST_SWAP','RANDOM_A','RANDOM_B']:
            raw=mat[family+'__SELECTED']-mat[family+'__'+reference]
            physiology=mat['D1__SELECTED']-mat['D1__'+reference]
            mu,ci=intervals(raw-physiology)
            for j,m in enumerate(METRICS):compensation.append(dict(family=family,reference=reference,metric=m,
                difference_of_differences=mu[j],ci_low=ci[0,j],ci_high=ci[1,j],patients=80,draws=10000,
                interpretation='Raw-family selected-minus-reference advantage minus D1 advantage; negative means attenuation, not causal proof.',
                added_after_frozen_primary_results=True))
    pd.DataFrame(sources).to_csv(PUB/'outer/SOURCE_PAIRED_COMPARISONS.csv',index=False)
    pd.DataFrame(compensation).to_csv(PUB/'outer/RAW_COMPENSATION_DIAGNOSTIC.csv',index=False)
    write(PUB/'COMPLEMENTARITY_REPORT_STATUS.json',dict(status='PASS',source_contrast_rows=len(sources),compensation_rows=len(compensation),
        score_only_posthoc_diagnostic=True,no_predictions_training_or_selection=True,main_primary_tables_unchanged=True,
        source_sha256=sha(HERE/'source_complementarity.py')))
    print('SCORE_ONLY_SOURCE_AND_RAW_COMPENSATION_REPORT_COMPLETE',flush=True)


if __name__=='__main__':main()
