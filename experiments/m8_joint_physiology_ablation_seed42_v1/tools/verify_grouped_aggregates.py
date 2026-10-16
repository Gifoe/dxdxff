"""Independent verification of fold/source means, sampleSD, stability and coverage."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from report import ROOT,PUB,read,write,sha
import numpy as np
import pandas as pd
METRICS=['auroc','auprc','macro_f1','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']

def same(a,b):
    assert np.allclose(np.asarray(a,dtype=float),np.asarray(b,dtype=float),rtol=0,atol=1e-12,equal_nan=True)

def main():
    n=0
    for stage in ['development','outer']:
        p=read(ROOT/(stage.upper()+'_PATIENT_METRICS_PRIVATE.pkl'))
        tables={name:pd.read_csv(PUB/stage/(name+'.csv'),dtype={'config':str}) for name in ['FOLD_METRICS','SOURCE_METRICS','FOLD_MEAN_SD','FOLD_DELETION_STABILITY','METRIC_COVERAGE']}
        for head in ['D1','M8']:
            reference=p[(p.config=='000')&(p['head']==head)].sort_values(['fold','patient'])
            for code in [f'{i:03b}' for i in range(8)]:
                g=p[(p.config==code)&(p['head']==head)].sort_values(['fold','patient'])
                fold=[]
                for group,name in [('fold','FOLD_METRICS'),('center','SOURCE_METRICS')]:
                    for key in sorted(set(g[group])):
                        rows=g[g[group]==key]
                        match=tables[name];r=match[(match.config==code)&(match['head']==head)&(match[group]==key)].iloc[0]
                        expected=np.nanmean(rows[METRICS].to_numpy(),axis=0)
                        same(expected,r[METRICS]);n+=len(METRICS)
                        assert r.appearances==len(rows) and r.unique_patients==rows.patient.nunique()
                        if group=='fold':fold.append(expected)
                fv=np.asarray(fold)
                for j,m in enumerate(METRICS):
                    match=tables['FOLD_MEAN_SD'];r=match[(match.config==code)&(match['head']==head)&(match.metric==m)].iloc[0]
                    same([np.nanmean(fv[:,j]),np.nanstd(fv[:,j],ddof=1)],[r.fold_mean,r.fold_sample_sd]);n+=2
                    match=tables['METRIC_COVERAGE'];r=match[(match.config==code)&(match['head']==head)&(match.metric==m)].iloc[0]
                    assert r.finite_appearances==np.isfinite(g[m]).sum() and r.total_appearances==len(g);n+=2
                    if code=='000':continue
                    delta=g[m].to_numpy()-reference[m].to_numpy()
                    values=np.array([np.nanmean(delta[g.fold.to_numpy()==f]) for f in range(1,6)])
                    match=tables['FOLD_DELETION_STABILITY'];r=match[(match.config==code)&(match['head']==head)&(match.metric==m)].iloc[0]
                    same([np.median(values),np.min(values),np.max(values),np.std(values,ddof=1)],
                        [r.median_fold_delta,r.worst_fold_delta,r.best_fold_delta,r.fold_delta_sample_sd]);n+=4
                    assert r.positive_folds==(values>0).sum() and r.negative_folds==(values<0).sum() and r.zero_folds==(values==0).sum();n+=3
    write(PUB/'INDEPENDENT_GROUPED_AGGREGATES_AUDIT.json',dict(status='PASS',verified_numeric_fields=n,
        all_fold_source_means_sampleSD_signed_stability_coverage_recomputed=True,training_or_inference=False,source_sha256=sha(__file__)))
    print('JOINT_ALL_GROUPED_AGGREGATES_INDEPENDENTLY_VERIFIED',n)

if __name__=='__main__':main()
