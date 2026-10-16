"""Independent private reaggregation and metadata-only numeric checks."""
import json
import numpy as np
import pandas as pd
import qualify as q


def independently_z(x,patients):
    z=np.full(x.shape,np.nan,float)
    for pid in np.unique(patients):
        ix=np.flatnonzero(patients==pid)
        for c in range(x.shape[1]):
            ids=ix[np.isfinite(x[ix,c])];a=x[ids,c]
            if not len(a):continue
            z[ids,c]=0 if np.max(a)==np.min(a) else (a-a.mean())/a.std()
    return z


def main():
    root=q.ROOT/'constant_safe';pub=root/'public';status=q.jr(pub/'QUALIFICATION_STATUS.json')
    binding=q.jr(root/'INPUT_FREEZE_PRIVATE.json')
    assert q.sha(root/'MEASUREMENTS_PRIVATE.pkl')==status['private_measurements_sha256']
    assert q.sha(root/'INPUT_FREEZE_PRIVATE.json')==status['input_freeze_sha256']
    for key in ['code','numeric_amendment']:
        for name,digest in binding[key].items():assert q.sha(q.HERE/name)==digest
    for key,digest in binding['identity'].items():
        if key.startswith('bank'):path=q.Path(r'C:\a0_source_posterior_decoder_seed42_runtime')/f'fold{key[4:]}/BANK_PRIVATE.pt'
        else:
            f,s=key[5:].split('_');path=q.PARENT/'screens_private'/(f'fold{f}_screen{s}.pkl' if s!='0' else f'fold{f}_final_path.pkl')
        assert q.sha(path)==digest
    assert q.sha(q.PARENT/'OUTER_PREDICTIONS_SEAL_PRIVATE.json')==binding['parent_Outer_seal']
    data=q.read(root/'MEASUREMENTS_PRIVATE.pkl');x=data['values'];replayed=np.full(x.shape,np.nan,float)
    support=np.zeros((7635,2),int);observed=np.zeros((7635,2),int);finite=np.zeros(6,int)
    for record in data['records']:
        ix=record['archive_indices'];p=record['present'];a=record['scalar'];e=record['eligible'];o=record['observed'];f=record['fraction']
        assert not (o&~e).any() and not (e&~p[:,:,None]).any()
        assert np.array_equal(np.isfinite(f),o)
        finite+=np.isfinite(a[p]).sum(0)
        for c,global_index in enumerate(ix):
            for j in [0,1,4,5]:
                values=a[p[:,c],c,j]
                replayed[global_index,2*j]=sum(values)/len(values)
                replayed[global_index,2*j+1]=np.sqrt(sum((v-replayed[global_index,2*j])**2 for v in values)/len(values))
            for k,j in enumerate([2,3]):
                count=int(e[:,c,k].sum());obs=int(o[:,c,k].sum());support[global_index,k]=count;observed[global_index,k]=obs
                if obs:replayed[global_index,2*j]=sum(f[o[:,c,k],c,k])/obs
                if count:replayed[global_index,2*j+1]=obs/count
    np.testing.assert_allclose(replayed,x,rtol=5e-15,atol=0,equal_nan=True)
    rec=q.read(q.BASE/'RECONSTRUCTION_PRIVATE.pkl');patients=np.asarray(rec['meta']['patient']);z=independently_z(replayed,patients)
    from constant_safe import patient_z
    np.testing.assert_allclose(z,patient_z(x,patients),atol=1e-12,rtol=1e-12,equal_nan=True)
    numeric=pd.read_csv(pub/'INNER_NUMERIC_QUALIFICATION.csv');checked=0
    for r in data['roles']:
        train=r['indices']['train'];impute=np.array([np.mean(v[np.isfinite(v)]) if np.isfinite(v).any() else 0. for v in z[train].T])
        filled=np.where(np.isnan(z),impute,z);center=filled[train].mean(0);sd=filled[train].std(0);sd[sd==0]=1.
        fitted=(filled-center)/sd
        for key,v in [('imputer',impute),('mean',center),('scale',sd)]:np.testing.assert_allclose(v,r['pre'][key],atol=1e-12,rtol=1e-12)
        for role,ix in r['indices'].items():
            for j,name in enumerate(q.GROUPS):
                entry=numeric[(numeric.fold==r['fold'])&(numeric.screen==r['screen'])&(numeric.role==role)&(numeric.feature==name)].iloc[0]
                assert entry.patients==len(set(patients[ix])) and entry.channels==len(ix)
                assert abs(entry.unit_equivariant_max_abs_fitted_input-np.max(np.abs(fitted[ix,2*j:2*j+2])))<=1e-10
                rates=[]
                for c in [2*j,2*j+1]:
                    count=0
                    for pid in set(patients[train]):
                        v=x[train[patients[train]==pid],c];v=v[np.isfinite(v)]
                        count+=len(v)>1 and np.max(v)-np.min(v)>64*np.finfo(np.float32).eps*np.max(np.abs(v))
                    rates.append(count/len(set(patients[train])))
                np.testing.assert_allclose(rates,[entry.coordinate0_nonconstant_train_patient_fraction,entry.coordinate1_nonconstant_train_patient_fraction],atol=1e-15)
                checked+=1
    coverage=pd.read_csv(pub/'SOURCE_COVERAGE.csv');decisions=pd.read_csv(pub/'QUALIFICATION_DECISIONS.csv')
    for j,name in enumerate(q.GROUPS):
        overall=coverage[(coverage.center=='ALL')&(coverage.feature==name)].iloc[0]
        assert overall.source_finite_units==finite[j] and overall.channels==7635 and overall.patients==80
        assert overall.primary0_finite_channels==np.isfinite(x[:,2*j]).sum()
        assert overall.primary1_finite_channels==np.isfinite(x[:,2*j+1]).sum()
        own=numeric[(numeric.feature==name)&(numeric.role=='train')]
        minimum=own[['coordinate0_nonconstant_train_patient_fraction','coordinate1_nonconstant_train_patient_fraction']].max(axis=1).min()
        eligible=overall.source_finite_rate>=.99 and overall.baseline_eligible_rate>=.99 and overall.primary1_coverage>=.99 and (j in [2,3] or overall.primary0_coverage>=.99) and minimum>=.5
        assert eligible==decisions[decisions.feature==name].iloc[0].eligible_for_future_preregistered_supervised_comparison
    stress=pd.read_csv(pub/'UNIT_STRESS.csv');assert (stress.unit_equivariant_max_abs_error<=1e-9).all()
    assert checked==450 and len(data['roles'])==20 and len(data['records'])==80
    result=dict(status='PASS',independently_reaggregated_coordinate_cells=int(x.size),inner_numeric_cells_checked=checked,
        censored_time_channels_HG=int((observed[:,0]==0).sum()),censored_time_channels_LL=int((observed[:,1]==0).sum()),
        baseline_unavailable_channels_HG=int((support[:,0]==0).sum()),baseline_unavailable_channels_LL=int((support[:,1]==0).sum()),
        unit_stress_max_abs=float(stress.unit_equivariant_max_abs_error.max()),
        eligible_groups=int(decisions.eligible_for_future_preregistered_supervised_comparison.sum()),
        labels_indexed=False,classifier_fits=0,new_Outer_access=False,original_failure_preserved=True)
    q.write(pub/'INDEPENDENT_QUALIFICATION_VERIFICATION.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
