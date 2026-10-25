"""Independent pairwise-concordance, rank-sum, bootstrap and FDR checks."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm,rankdata,false_discovery_control
from sklearn.metrics import roc_auc_score
import run_analysis as a


def direct_effect(x,y):
    good=np.isfinite(x);x=x[good];y=y[good]
    if not y.any() or not (~y).any():return np.nan
    return float(np.sign(x[y,None]-x[~y][None,:]).mean())


def signed_rank_p(x):
    x=np.asarray(x);x=x[np.isfinite(x)&(x!=0)];n=len(x)
    if not n:return 1.,0.
    ranks=rankdata(abs(x),method='average');pos=ranks[x>0].sum();neg=ranks[x<0].sum()
    _,t=np.unique(abs(x),return_counts=True)
    var=(n*(n+1)*(2*n+1)-.5*np.sum(t**3-t))/24
    z=max(0,abs(pos-n*(n+1)/4)-.5)/np.sqrt(var)
    return float(2*norm.sf(z)),float(min(pos,neg))


def check_summary(values,row,test=True):
    v=np.asarray(values);v=v[np.isfinite(v)];n=len(v)
    assert n==row['valid_patients']
    ids=np.random.default_rng(42).integers(n,size=(10000,n));counts=np.zeros((10000,n))
    np.add.at(counts,(np.arange(10000)[:,None],ids),1.)
    ci=np.percentile(counts@v/n,[2.5,97.5])
    expected=[np.mean(v),np.median(v),*ci,100*np.mean(v>0),100*np.mean(v<0),100*np.mean(v==0)]
    keys=['mean_effect','median_effect','ci_low','ci_high','positive_percent','negative_percent','zero_percent']
    np.testing.assert_allclose(expected,[row[k] for k in keys],atol=1e-12,rtol=0)
    if test:
        p,stat=signed_rank_p(v)
        np.testing.assert_allclose([p,stat],[row['pvalue'],row['wilcoxon_statistic']],atol=1e-12,rtol=0)


def main(runtime):
    import psutil
    psutil.Process().cpu_affinity([0])
    runtime=Path(runtime);pub=runtime/'public';private=runtime/'private'
    binding=json.loads((private/'INPUT_FREEZE.json').read_text())
    for n,h in binding['source'].items():assert a.sha(a.HERE/n)==h
    rec,x,ez,protocol=a.inputs();meta=rec['meta'];ids=sorted(set(meta['patient']))
    actual=pd.read_csv(private/'patient_effect_summary.csv',dtype={'patient':str});assert len(actual)==960
    assert not actual.duplicated(['patient','coordinate']).any()
    matrix=[];source=[];rng=np.random.default_rng(42);nsum=np.zeros((10000,12));ncount=np.zeros((10000,12),int)
    maximum=0.
    for pid in ids:
        ix=np.flatnonzero(meta['patient']==pid);xx=x[ix];yy=ez[ix]
        row=[];source.append(str(meta['center'][ix[0]]))
        permuted=rng.permuted(np.broadcast_to(yy,(10000,len(yy))),axis=1)
        assert np.all(permuted.sum(1)==yy.sum())
        for j,c in enumerate(protocol['coordinates']):
            good=np.isfinite(xx[:,j]);v=xx[good,j];y=yy[good]
            expected=direct_effect(xx[:,j],yy);other=2*roc_auc_score(y,v)-1 if y.any() and (~y).any() else np.nan
            observed=actual[(actual.patient==pid)&(actual.coordinate==c['coordinate'])].iloc[0]
            np.testing.assert_allclose([expected,other],[observed.effect]*2,atol=1e-12,rtol=0,equal_nan=True)
            if np.isfinite(expected):maximum=max(maximum,abs(expected-observed.effect))
            assert observed.finite_EZ_channels==int(y.sum()) and observed.finite_NEZ_channels==int((~y).sum())
            assert observed.missing_channels==int((~good).sum());row.append(expected)
            # Pairwise dominance scores, not the primary rank-sum formula.
            dominance=np.sign(v[:,None]-v[None,:]).sum(1);labels=permuted[:,good]
            n1=labels.sum(1);den=n1*(len(v)-n1);ok=den>0
            null=labels.astype(float)@dominance
            nsum[ok,j]+=null[ok]/den[ok];ncount[ok,j]+=1
        matrix.append(row)
    matrix=np.asarray(matrix);source=np.asarray(source)
    np.testing.assert_allclose(matrix,np.load(private/'PATIENT_EFFECT_MATRIX.npy'),atol=1e-12,rtol=0,equal_nan=True)
    null=np.divide(nsum,ncount,out=np.full_like(nsum,np.nan),where=ncount>0)
    np.testing.assert_allclose(null,np.load(private/'PERMUTATION_NULL_MEANS.npy'),atol=1e-12,rtol=0,equal_nan=True)
    overall=pd.read_csv(pub/'feature_association.csv');sources=pd.read_csv(pub/'source_association.csv')
    tests=pd.read_csv(pub/'statistical_tests.csv');hist=pd.read_csv(pub/'patient_effect_histogram.csv')
    distribution=pd.read_csv(pub/'patient_effect_summary.csv');control=pd.read_csv(pub/'permutation_control.csv')
    assert len(overall)==len(tests)==len(distribution)==len(control)==12 and len(sources)==48
    for j,c in enumerate(protocol['coordinates']):
        key=c['coordinate'];r=overall[overall.coordinate==key].iloc[0];check_summary(matrix[:,j],r)
        for z in protocol['sources']:
            sr=sources[(sources.coordinate==key)&(sources.source==z)].iloc[0]
            assert sr.cohort_patients==int((source==z).sum());check_summary(matrix[source==z,j],sr,False)
        h=hist[hist.coordinate==key];counts,edges=np.histogram(matrix[:,j][np.isfinite(matrix[:,j])],bins=np.linspace(-1,1,41))
        assert np.array_equal(h.patients,counts) and counts.sum()==r.valid_patients
        d=distribution[distribution.coordinate==key].iloc[0];v=matrix[:,j];v=v[np.isfinite(v)]
        np.testing.assert_allclose([d.q025,d.q25,d['median'],d.q75,d.q975],np.quantile(v,[.025,.25,.5,.75,.975]),atol=1e-12,rtol=0)
        ct=control[control.coordinate==key].iloc[0];v=null[:,j];v=v[np.isfinite(v)]
        np.testing.assert_allclose([ct.null_mean,ct.null_sd,ct.null_ci_low,ct.null_ci_high],
            [np.mean(v),np.std(v,ddof=1),*np.quantile(v,[.025,.975])],atol=1e-12,rtol=0)
        expected=(1+np.sum(abs(v)>=abs(r.mean_effect)-1e-14))/(len(v)+1)
        assert abs(expected-ct.empirical_two_sided_p)<1e-12
    q=false_discovery_control(overall.pvalue.to_numpy(),method='bh')
    np.testing.assert_allclose(q,overall.BH_q,atol=1e-12,rtol=0)
    np.testing.assert_allclose(tests.pvalue,overall.pvalue,atol=1e-12,rtol=0)
    np.testing.assert_allclose(tests.BH_q,q,atol=1e-12,rtol=0)
    assert a.sha(a.REC)==binding['reconstruction'] and a.sha(a.ARCHIVE)==binding['label_archive']
    verification=dict(status='PASS',patient_effects_independently_verified=960,overall_coordinates=12,
        source_coordinate_rows=48,bootstrap_draws_each=10000,patient_bootstrap_intervals_checked=60,
        primary_Wilcoxon_and_BH_tests_checked=12,label_permutations_per_patient=10000,
        independently_checked_permuted_effects=80*10000*12,maximum_effect_error=maximum,
        effect_engines=['rank_sum','pairwise_sign_concordance','sklearn_ROC_AUC'],
        independent_BH='scipy_false_discovery_control',independent_Wilcoxon='tie_corrected_rank_variance_normal_tail',
        original_inputs_unchanged=True,individual_effects_exported=False,models_trained=0)
    a.write(pub/'INDEPENDENT_VERIFICATION.json',verification)
    a.write(pub/'STATUS.json',dict(status='COMPLETE_INDEPENDENTLY_VERIFIED_SIXFEATURE_EZ_ASSOCIATION',
        patients=80,channels=7635,seizures=256,coordinates=12,patient_effect_rows=960,models_trained=0,
        exploratory_post_historical_feature_selection=True))
    print('INDEPENDENT_ASSOCIATION_VERIFICATION_PASS',json.dumps(verification),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',type=Path,required=True)
    main(parser.parse_args().runtime)
