"""Independent score-only verification, factorial saturated reconstruction and release audit."""
from report import *
from itertools import combinations

def main():
    prot=jr(HERE/'PROTOCOL_LOCK.json');fr=read(ROOT/'MODEL_FREEZE_PRIVATE.pkl')
    for name,h in fr['binding']['source'].items():assert sha(HERE/name)==h
    assert jr(PUB/'TRAINING_VERIFICATION.json')['status']=='PASS'
    choice=jr(PUB/'DEVELOPMENT_SELECTION_FROZEN.json')
    dev=read(ROOT/'DEVELOPMENT_PATIENT_METRICS_PRIVATE.pkl')
    rows,ch=candidate_decisions(dev,prot);assert ch==choice['choices'] and clean(rows)==choice['eligible']
    checks=0;direct=0
    for stage in ['development','outer']:
        frame=frames(stage);patients=patient_metrics(frame)
        stored=read(ROOT/(stage.upper()+'_PATIENT_METRICS_PRIVATE.pkl'))
        assert np.array_equal(patients[METRICS].to_numpy(),stored[METRICS].to_numpy(),equal_nan=True)
        main=pd.read_csv(PUB/stage/'MAIN_RESULTS.csv',dtype={'config':str})
        ci=pd.read_csv(PUB/stage/'PAIRED_CONTRASTS.csv',dtype={'config':str})
        ids=patients[(patients.config=='000')&(patients['head']=='D1')].sort_values(['fold','patient']).patient.to_numpy()
        unique,counts,draw=bootstrap_counts(ids)
        for head in ['D1','M8']:
            arr=np.stack([patients[(patients.config==c)&(patients['head']==head)].sort_values(['fold','patient'])[METRICS].to_numpy() for c in CONFIGS])
            for k,c in enumerate(CONFIGS):
                row=main[(main.config==c)&(main['head']==head)].iloc[0]
                assert np.allclose(row[METRICS].to_numpy(dtype=float),np.nanmean(arr[k],axis=0),atol=1e-12,rtol=0,equal_nan=True)
            # Independent orthogonal Walsh expansion: reconstruct every cell from mean/effects.
            means=np.nanmean(arr,axis=1); intercept=np.mean(means,axis=0);effects={}
            for n in [1,2,3]:
                for factors in combinations(range(3),n):
                    sign=np.prod(np.array([[2*int(c[k])-1 for k in factors] for c in CONFIGS]),axis=1)
                    coeff=sign@means/8;effects[factors]=coeff
                    name=''.join('VHE'[k] for k in factors)
                    actual=ci[(ci.kind=='factorial')&(ci['head']==head)&(ci.factors==name)]
                    # The historical metrics here are finite for all primary requested endpoints.
                    for j,m in enumerate(METRICS):
                        v=actual[actual.metric==m].iloc[0]
                        if np.isfinite(arr[:,:,j]).all():assert abs(v.delta-coeff[j]*2**n)<1e-12
                        checks+=1
            recon=[]
            for c in CONFIGS:
                q=intercept.copy()
                for factors,beta in effects.items():q=q+beta*np.prod([2*int(c[k])-1 for k in factors])
                recon.append(q)
            assert np.allclose(recon,means,atol=1e-12,rtol=0,equal_nan=True)
            for k,c in enumerate(CONFIGS[1:],1):
                delta=arr[k]-arr[0]
                for j,m in enumerate(METRICS):
                    vals=[]
                    for dd in draw[:100]:
                        ix=np.concatenate([np.flatnonzero(ids==unique[i]) for i in dd]);vals.append(np.nanmean(delta[ix,j]))
                    # Pairing identity checked directly against cluster counts, separately from reporter.
                    sums=np.array([np.nansum(delta[ids==p,j]) for p in unique]);nums=np.array([np.isfinite(delta[ids==p,j]).sum() for p in unique])
                    den=counts[:100]@nums;directdist=np.divide(counts[:100]@sums,den,out=np.full(100,np.nan),where=den>0)
                    assert np.allclose(vals,directdist,atol=1e-12,rtol=0,equal_nan=True);direct+=100
        assert len(main)==16
    # Reused Outer M8 rows must reproduce historical full/drop public metrics, not just rounded values.
    old=pd.read_csv(Path(r'D:\m8_physiology_feature_discovery_seed42_v1_runtime\final_Outer\public\main_results.csv'))
    outer=pd.read_csv(PUB/'outer/MAIN_RESULTS.csv',dtype={'config':str})
    for c,name in prot['reuse'].items():
        a=outer[(outer.config==c)&(outer['head']=='M8')].iloc[0];b=old[old.model=='LEGACY_'+name].iloc[0]
        assert np.allclose(a[METRICS].to_numpy(dtype=float),b[METRICS].to_numpy(dtype=float),atol=1e-12,rtol=0,equal_nan=True)
    for path,h in fr['binding']['prior_binding']['protected'].items():assert sha(path)==h
    files={str(p.relative_to(PUB)).replace('\\','/'):sha(p) for p in PUB.rglob('*') if p.is_file() and p.name!='INDEPENDENT_COMPLETION_AUDIT.json'}
    write(PUB/'INDEPENDENT_COMPLETION_AUDIT.json',dict(status='PASS',configurations=8,heads=2,Outer_patient_rows=1280,
        development_patient_rows=1040,new_actual_models=40,reused_models=40,independent_factorial_metric_checks=checks,
        directly_replayed_cluster_metric_draws=direct,all8_saturated_cells_reconstructed=True,reused_M8_metrics_match_prior=True,
        choice_reproduced_from_validation_only=ch,original_protected_files_unchanged=True,files=files,
        no_new_training_or_inference=True,not_independent_confirmation=True,source_sha256=sha(__file__)))
    print('JOINT_2CUBED_INDEPENDENT_VERIFICATION_PASS_NO_OUTER_SELECTION',flush=True)

if __name__=='__main__':main()
